# Spec — Phase 4 (M9): Video-mode FlowPackager tile scheduler

Date: 2026-06-03
Status: Design (pre-plan)
Branch: `dev` (long-lived integration branch; commit only when asked)
Roadmap: `docs/superpowers/specs/2026-06-01-roadmap.md` Phase 4; expands
`docs/superpowers/specs/2026-06-01-detection-core-yolo-obb-tiling-design.md` §14.3.

## Goal

Add a video-mode scheduler that, per frame, uses `FlowPackagerCalculator`'s
`TrackingData` (motion) plus the previous frame's final detections to decide
**DETECT** (run tiled inference) vs **SKIP** (track-only), to **prioritize /
shrink** the tile list, and on SKIP to **propagate** the prior detections so the
frame still emits a result through the single global NMS. It is a scheduling /
caching signal only — it assigns no long-lived track IDs (BoTSORT remains M7)
and never batches tiles across frames.

This first slice targets the **axis-aligned `Detection`** type. `TrackingData`
is the primary driver but is **optional**: without it the scheduler degrades to
a cadence + confidence + prior-detection-overlap policy.

## Scope

In scope:
1. A new `VideoTileSchedulerCalculator` (CPU, api2) that emits scheduled tile
   rects + propagated detections per frame.
2. Motion-driven decision + tile prioritization decoded from `TrackingData` via
   the existing `MotionVectorFrameFromTrackingData`.
3. Graceful fallback when `TRACKING` is not connected.
4. Unit tests (calculator-level) + one small example/integration graph test
   (loopback + concat + single global NMS).

Out of scope (YAGNI / deferred):
- OBB (`OrientedDetection`) scheduling — trivial follow-up once this lands.
- Long-lived cross-frame track association / re-identification — that is M7
  (BoTSORT). The SKIP "candidate hold" here is a simple re-emit of the prior
  final detections, NOT association.
- Owning the tiling algorithm (tiles still come from any upstream source) or the
  motion pipeline (`MotionAnalysisCalculator` → `FlowPackagerCalculator` are
  pre-existing and wired by the caller).
- `FlowPackagerCalculator`'s offline `CACHE_DIR`/`TRACKING_CHUNK` path — live
  mode consumes `TRACKING` directly.

## Non-negotiable rules (inherited from the detection-core invariants)

- **Exactly one global NMS per frame.** Held (propagated) candidates and freshly
  inferred detections meet only at that single frame-level NMS.
- **No cross-frame input batches**, even in video mode.
- **No track IDs** assigned here.
- **Defaults reproduce image mode:** with all options at default and `TRACKING`
  unconnected, every frame is a DETECT of all tiles with no propagation.
- Every frame emits exactly one detection result (possibly empty) at its source
  timestamp — including SKIP frames and empty-tile frames.

## Current-state findings (verified)

- `FlowPackagerCalculator` (`mediapipe/calculators/video/flow_packager_calculator.cc`):
  input `FLOW:RegionFlowFeatureList` (+ optional `CAMERA:CameraMotion`); output
  `TRACKING:TrackingData` per frame (+ optional `TRACKING_CHUNK`/`COMPLETE`).
- `TrackingData` (`mediapipe/util/tracking/flow_packager.proto`): `frame_flags`
  (`FLAG_BACKGROUND_UNSTABLE`, `FLAG_DUPLICATED`, `FLAG_CHUNK_BOUNDARY`, …),
  `average_motion_magnitude`, `global_feature_count`, `background_model`,
  and `motion_data` (CSC-sparse per-feature vectors).
- **Decode helper** `void MotionVectorFrameFromTrackingData(const TrackingData&,
  MotionVectorFrame*)` (`mediapipe/util/tracking/tracking.h:233`, lib
  `//mediapipe/util/tracking:tracking`) yields a `MotionVectorFrame`:
  - `std::vector<MotionVector> motion_vectors`, each with `Vector2_f pos`
    (normalized, longest-side-normalized domain), `background` (camera motion),
    `object` (foreground motion), `track_id`, and helpers `Location()`,
    `Motion()` (= background+object).
  - booleans `valid_background_model`, `is_duplicated`, `is_chunk_boundary`,
    and `float aspect_ratio` (w/h).
  - Domain note: positions are in the **aspect-preserving** domain (longest
    dimension = 1). Mapping to frame-normalized `[0,1]²` uses `aspect_ratio`
    (see "Aspect mapping" below). Implementation MUST confirm the helper's exact
    normalization against `tracking.cc` before relying on the mapping.
- `NormalizedRect` (`mediapipe/framework/formats/rect.proto`): `x_center`,
  `y_center`, `width`, `height`, `rotation` (all normalized) — the input the
  shipped `TileSpecToTilePlanCalculator` consumes.
- `Detection` (`mediapipe/framework/formats/detection.proto`): `score`,
  `label`, and `location_data` (`RELATIVE_BOUNDING_BOX{xmin,ymin,width,height}`).
- `PreviousLoopbackCalculator` (`mediapipe/calculators/core/`): standard
  "previous frame's output aligned to the current timestamp" — used to feed
  prior final detections back to the scheduler without an Open-time cycle.
- Shipped tiling family (this fork, `mediapipe/calculators/tensor/`):
  `TileSpecToTilePlanCalculator` (`std::vector<NormalizedRect>` → `TilePlan`,
  validates + indexes + optional plan cache), `StreamingTilesToTensorBatch`,
  `MergeTileDetectionsAccumulator` (OBB today). The scheduler emits rects into
  `TileSpecToTilePlanCalculator` — it does NOT build a `TilePlan` itself (DRY).

## Architecture

```
                              ┌────────────────────────────────────────────┐
 Video frame ─► MotionAnalysis ─► FlowPackagerCalculator ─► TRACKING:TrackingData (optional)
                              └────────────────────────────────────────────┘ │
 base tiles (rects) ───────────────────────────────────────────────┐         │
 final DETECTIONS ─► PreviousLoopbackCalculator ─► PRIOR_DETECTIONS@t│         │
                                                                    ▼         ▼
                                              VideoTileSchedulerCalculator  [NEW]
                                                │  ├─► TILES (scheduled rects; empty on SKIP)
                                                │  │      └─► TileSpecToTilePlan ─► tiled detect path ─┐
                                                │  └─► PROPAGATED_DETECTIONS (priors on SKIP; else ∅) ─┤
                                                ▼                                                       ▼
                                                          Concatenate(vector<Detection>) ─► global NMS ─► final DETECTIONS
                                                                                                          │ (loops back) ▲
                                                                                                          └──────────────┘
```

The scheduler is a pure, stateful-counter function of
`(frame_index, base tiles, prior detections, optional TrackingData, options)`.

### Calculator contract (`VideoTileSchedulerCalculator`, api2, CPU)

Inputs:
- `TILES` : `std::vector<NormalizedRect>` — base candidate tiles this frame.
- `PRIOR_DETECTIONS` : `std::vector<Detection>` — previous frame's final
  detections (via `PreviousLoopbackCalculator`); an empty/absent packet = cache
  miss.
- `TRACKING` : `TrackingData` — **optional** motion signal.

Outputs:
- `TILES` : `std::vector<NormalizedRect>` — scheduled rects on DETECT; **empty**
  vector on SKIP.
- `PROPAGATED_DETECTIONS` : `std::vector<Detection>` — **empty** on DETECT;
  copy of prior detections on SKIP.

Both outputs are emitted **every** frame at the input timestamp (default
timestamp offset; no `Arbitrary`), so downstream stays in lockstep.

State: `int frame_index_` (incremented per `Process`).

### Options (`VideoTileSchedulerCalculatorOptions`, proto2)

```proto
message VideoTileSchedulerCalculatorOptions {
  // Use the next free MediaPipe calculator-options extension id. The shipped
  // tiling protos use 471230004 (tile_spec) and 471230005 (streaming); pick an
  // unused id such as 471230006 and verify it is unused before committing.
  extend mediapipe.CalculatorOptions { optional VideoTileSchedulerCalculatorOptions ext = 471230006; }

  // Run a full DETECT at least every N frames (staleness bound). 1 = always
  // detect (image-mode default). 0 = never on cadence alone.
  optional int32 detect_every_n_frames = 1 [default = 1];

  // Force DETECT when the prior frame's strongest detection score is below this
  // (confidence-drop). 0 = disabled.
  optional float min_confidence = 2 [default = 0.0];

  // Cap on scheduled tiles on a DETECT frame; keep the top-K by priority
  // (DROP_LOW_PRIORITY). 0 = no cap (emit all base tiles → full refresh).
  optional int32 max_scheduled_tiles = 3 [default = 0];

  // Force DETECT when aggregate foreground (object) motion energy exceeds this
  // (scene change). 0 = disabled. Requires TRACKING.
  optional float motion_refresh_threshold = 4 [default = 0.0];

  // Allow SKIP (override cadence) when aggregate foreground motion is below this
  // (near-static). 0 = disabled. Requires TRACKING.
  optional float motion_skip_threshold = 5 [default = 0.0];

  // Force DETECT when decoded flow feature count is below this (unreliable
  // flow). 0 = disabled. Requires TRACKING.
  optional int32 min_features = 6 [default = 0];
}
```

### Decision logic (deterministic precedence)

Let `priors_empty = PRIOR_DETECTIONS empty/absent`. If `TRACKING` present, decode
once via `MotionVectorFrameFromTrackingData`; compute foreground motion energy
`E = mean over features of object.Norm()` (0 when no features), and read
`valid_background_model`, `is_duplicated`, `is_chunk_boundary`,
`feature_count = motion_vectors.size()`.

Evaluate in order; first match wins:
1. **DETECT (hard refresh)** if any: `frame_index_ == 0` (first frame);
   `priors_empty` (cache miss); `min_confidence > 0 && (priors_empty ||
   max prior score < min_confidence)`; or `TRACKING present &&
   (!valid_background_model || is_chunk_boundary ||
   (motion_refresh_threshold > 0 && E > motion_refresh_threshold) ||
   (min_features > 0 && feature_count < min_features))`.
2. **SKIP (motion says nothing changed)** if `TRACKING present &&
   (is_duplicated || (motion_skip_threshold > 0 && E < motion_skip_threshold))`.
3. **DETECT (cadence/staleness)** if `detect_every_n_frames > 0 &&
   frame_index_ % detect_every_n_frames == 0`.
4. **SKIP** otherwise.

With defaults (`detect_every_n_frames=1`, thresholds 0, no TRACKING): rule 1
(first/cache-miss) then rule 3 (every frame) ⇒ always DETECT — image mode.

### DETECT-frame tile output

- If `max_scheduled_tiles == 0` **or** `T <= max_scheduled_tiles`: emit **all**
  base tiles, original order preserved (full refresh; no prioritization needed).
- Else compute a priority per base tile and keep the top-`max_scheduled_tiles`
  (stable: ties and zero-priority broken by original index, so output is
  deterministic):
  - **Motion priority** (TRACKING present): for each decoded feature, map `pos`
    to frame-normalized `[0,1]²` (aspect mapping below); a feature lies in a tile
    if its point is inside the tile's axis-aligned rect (`x0..x0+w`, `y0..y0+h`).
    Tile priority = Σ `object.Norm()` over its features (foreground motion only,
    so camera pan doesn't inflate every tile).
  - **Fallback priority** (no TRACKING, or a tile has zero in-tile features):
    number of prior detection boxes whose center lies in the tile.
- `PROPAGATED_DETECTIONS` = empty.

### SKIP-frame output
- `TILES` = empty vector (⇒ `TileSpecToTilePlan` → empty `TilePlan` → no
  inference for this frame).
- `PROPAGATED_DETECTIONS` = copy of `PRIOR_DETECTIONS`. These flow to the
  Concatenate→global-NMS step and become the frame's result at the current
  timestamp.

### Aspect mapping (MotionVectorFrame → frame-normalized)

`MotionVectorFrame` positions are in the longest-side-normalized domain with
`aspect_ratio = w/h`:
- if `aspect_ratio >= 1` (landscape): `x' = x`, `y' = y * aspect_ratio`;
- else (portrait): `x' = x / aspect_ratio`, `y' = y`.

The implementation MUST verify this against `tracking.cc`'s denormalization in a
unit test (full-frame feature at domain center maps to `(0.5, 0.5)`).

## Integration (example graph + its test)

- `PreviousLoopbackCalculator`: `LOOP` = final `DETECTIONS`; `MAIN` = a
  per-frame tick (e.g. the frame stream); `PREV_LOOP` = `PRIOR_DETECTIONS` for
  the scheduler. This breaks the scheduler→…→NMS→scheduler cycle (back-edge).
- Concatenate `PROPAGATED_DETECTIONS` with the tiled path's merged detections
  into one `std::vector<Detection>`, then run **one** NMS
  (`NonMaxSuppressionCalculator`). The plan pins the exact concat calculator
  (`ConcatenateVectorCalculator<Detection>` / `ConcatenateDetectionVectorCalculator`
  — confirm which is registered) and the NMS wiring.
- The full end-to-end with real inference is not required for this slice; the
  integration test may stub the tiled-detect path (feed canned merged detections
  at DETECT timestamps) to exercise loopback + concat + single NMS + propagation.

## File structure

- **Create** `mediapipe/calculators/tensor/video_tile_scheduler_calculator.proto`
  — the options message (placed with the tiling family for discoverability with
  `TileSpecToTilePlanCalculator`; the calculator depends on `util/tracking` for
  decoding).
- **Create** `mediapipe/calculators/tensor/video_tile_scheduler_calculator.cc`
  — the calculator (decision logic + prioritization + propagation). One file,
  one responsibility.
- **Create** `mediapipe/calculators/tensor/video_tile_scheduler_calculator_test.cc`
  — unit tests with hand-built inputs.
- **Create** `mediapipe/calculators/tensor/video_tile_scheduler_pipeline_test.cc`
  — the loopback + concat + single-NMS integration test (stubbed detect path).
- **Modify** `mediapipe/calculators/tensor/BUILD` — proto + calculator + test
  targets; deps `//mediapipe/framework/formats:rect_cc_proto`,
  `//mediapipe/framework/formats:detection_cc_proto`,
  `//mediapipe/util/tracking:tracking`,
  `//mediapipe/util/tracking:flow_packager_cc_proto`, api2 node.

## Testing (CPU)

Unit (`CalculatorRunner`, hand-built inputs):
- Defaults + no TRACKING ⇒ every frame DETECT, all tiles, empty propagation
  (image-mode parity).
- Cadence: `detect_every_n_frames=3` ⇒ DETECT on 0,3,6…, SKIP between, SKIP
  frames emit empty TILES + propagated priors.
- Confidence-drop: prior max score < `min_confidence` ⇒ DETECT even off-cadence.
- Cache miss: empty priors ⇒ DETECT.
- TRACKING flags: `FLAG_DUPLICATED` ⇒ SKIP (off a non-first frame);
  `FLAG_BACKGROUND_UNSTABLE` ⇒ DETECT; `FLAG_CHUNK_BOUNDARY` ⇒ DETECT.
- Motion thresholds: `E > motion_refresh_threshold` ⇒ DETECT;
  `E < motion_skip_threshold` ⇒ SKIP overriding cadence.
- `min_features`: low feature count ⇒ DETECT.
- Prioritization: `max_scheduled_tiles=2` with 4 tiles and motion features in 2
  of them ⇒ those 2 tiles kept, deterministic order; fallback to
  prior-detection-center containment when no features.
- Aspect mapping: landscape + portrait `aspect_ratio`, center feature → (0.5,0.5).
- Propagation: SKIP frame's `PROPAGATED_DETECTIONS` equals the priors;
  DETECT frame's is empty.

Integration: loopback feeds prior dets; SKIP frame's propagated dets reach NMS
and produce the frame result; DETECT frame's stubbed merged dets reach NMS; both
pass through exactly one NMS; output emitted once per frame at its timestamp.

## Done criteria
- `VideoTileSchedulerCalculator` shipped with the deterministic precedence above.
- Motion-driven via decoded `TrackingData`; graceful cadence/confidence fallback
  when `TRACKING` is unconnected.
- DETECT/SKIP, prioritization+`DROP_LOW_PRIORITY` cap, and SKIP-frame propagation
  all unit-tested; one integration test proves single-global-NMS + loopback.
- Defaults reproduce image mode. No track IDs, no cross-frame batches, one NMS.
- All tests pass under `--define MEDIAPIPE_DISABLE_GPU=1`.
