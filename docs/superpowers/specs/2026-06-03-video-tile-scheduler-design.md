# Spec - Phase 4 (M9): FlowPackager-driven video tile scheduler

Date: 2026-06-03
Status: Design (pre-plan)
Branch: `dev` (long-lived integration branch; commit only when asked)
Roadmap: `docs/superpowers/specs/2026-06-01-roadmap.md` Phase 4; expands
`docs/superpowers/specs/2026-06-01-detection-core-yolo-obb-tiling-design.md`
section 14.3.

## Goal

Add a video-mode scheduler that decides whether tiled detection should refresh
on a frame by consuming `FlowPackagerCalculator`'s per-frame `TrackingData` and
its tracking flags. The scheduler decides **DETECT** vs **SKIP**, and on DETECT
it may prioritize/cap the tile list before it reaches `TileSpecToTilePlan`.

This design intentionally removes `detect_every_n_frames` and any private frame
counter from the refresh policy. `FlowPackagerCalculator` already emits
`TRACKING:TrackingData` at the source timestamp for each processed flow packet,
and its internal `frame_idx_` is only chunk metadata. A second frame-cadence
clock in the tile scheduler conflicts with that model.

The scheduler does not propagate prior detections by copying them. SKIP-frame
results must come from the existing tracking path (`BoxTrackerCalculator` plus
`TrackedDetectionManagerCalculator`) and then pass through the same frame-level
global NMS as fresh tiled detections.

This slice targets axis-aligned `Detection`. OBB scheduling can follow after
the relative-coordinate and NMS contracts are stable.

## Scope

In scope:
1. `VideoTileSchedulerCalculator` (CPU, api2) in `mediapipe/calculators/tensor/`.
2. Refresh decisions driven by decoded `TrackingData` flags and motion energy.
3. DETECT-frame tile prioritization/capping over externally supplied tiles.
4. Integration contract with the existing video tracking graph pattern:
   `MotionAnalysisCalculator -> FlowPackagerCalculator -> BoxTrackerCalculator
   -> TrackedDetectionManagerCalculator`.
5. A frame-level suppression stage over fresh detections plus tracker-updated
   detections, with a safe single-tile/no-tracker pass-through bypass.

Out of scope:
- A new tracking algorithm, track IDs, re-identification, or BoTSORT-like
  association.
- Owning the tiling algorithm. Tiles remain an upstream input and can be SAHI,
  fixed-grid, ROI-driven, or any other external strategy.
- `FlowPackagerCalculator`'s offline `CACHE_DIR`/`TRACKING_CHUNK` cache path.
  Live scheduling consumes `TRACKING` packets directly.
- Running multiple video frames in one model input batch.

## Integration boundary

M9 is an integrated video graph milestone, not a standalone scheduler-only
calculator change. The scheduler, frame-local tile batching, cache behavior,
tile-local candidate reduction, tracker-updated SKIP results, and final
frame-level suppression share the same timestamp, coordinate-system, cache-key,
and candidate-lifetime contracts. They must be designed and verified together.

Implementation may proceed in small commits, but the graph contract is accepted
only end-to-end. Do not move multi-batch streaming, cache interaction,
tile-local candidate reduction, tracker integration, or frame-level suppression
out of this design unless a replacement contract proves the adjacent calculator
interfaces still compose.

## Non-negotiable rules

- **Refresh is tracking-driven.** No `detect_every_n_frames`, no local
  `frame_index_ % N`, and no private staleness clock in the scheduler.
- **At most one global NMS per source frame.** Fresh tiled detections and
  tracker-updated detections normally meet at the frame-level global NMS. The
  NMS call may be bypassed only for a provably single-source frame: one valid
  tile row, no tracker-updated detections, and the row has already run tile-local
  NMS or the model output is known to be post-NMS.
- **No copied-prior SKIP result.** A SKIP frame emits no scheduled tiles; the
  frame result comes from the tracker path.
- **No cross-frame input batches.** Tiling may batch tiles from one frame, but
  never tiles from multiple frames.
- **Tiles are streamed in frame-local batches.** If a DETECT frame has more
  scheduled tiles than the model input batch capacity, the tiled input path emits
  multiple inference batches for that same source frame and sends each batch as
  soon as it is prepared. It must not wait until every tile for the frame has
  been tensorized.
- **Candidate reduction is tile-local before merge, global after merge.** Each
  tile row should apply score/confidence filtering before local candidate
  reduction. Optional tile-local NMS/top-K is allowed for efficiency, but the
  frame-level global NMS remains the only place that may compare candidates
  across tiles or against tracker-updated detections. A safe single-tile bypass
  may pass through without invoking NMS when there is nothing global to compare.
- **Coordinate system before public Tasks conversion is relative.** Scheduler,
  tracker, merge, and NMS streams use `LocationData::RELATIVE_BOUNDING_BOX`.
  Pixel `BOUNDING_BOX` conversion belongs after final NMS when producing public
  Tasks results that require pixel units.
- **Defaults reproduce image mode.** If `TRACKING` is not connected/present, the
  scheduler DETECTs every frame and emits all base tiles.
- **Every frame emits a result.** SKIP frames emit an empty scheduled tile vector
  and rely on `TrackedDetectionManagerCalculator`/empty-vector materialization
  to keep downstream streams timestamp-aligned.

## Verified codebase facts

- `FlowPackagerCalculator` reads `FLOW` plus optional `CAMERA`, calls
  `PackFlow`, and emits `TRACKING:TrackingData` at
  `flow_stream->Value().Timestamp()`. Its `frame_idx_` is stored only in
  `TrackingDataChunk::Item` when chunking is enabled.
- `TrackingData` exposes `frame_flags`:
  `FLAG_BACKGROUND_UNSTABLE`, `FLAG_DUPLICATED`, `FLAG_CHUNK_BOUNDARY`, plus
  `global_feature_count` and `average_motion_magnitude`.
- `MotionVectorFrameFromTrackingData(const TrackingData&, MotionVectorFrame*)`
  decodes to `MotionVectorFrame` with:
  - `valid_background_model`
  - `is_duplicated`
  - `is_chunk_boundary`
  - `aspect_ratio`
  - `motion_vectors`, each carrying `pos`, `background`, and `object`.
- The mature mobile object detection graph uses `FlowLimiterCalculator` around
  expensive detection work to keep in-flight inference bounded.
- The mature tracking test graph wires:
  `MotionAnalysisCalculator -> FlowPackagerCalculator -> BoxTrackerCalculator`.
- `TrackedDetectionManagerCalculator` converts input `Detection` relative boxes
  to `NormalizedRect`, updates them with `TRACKING_BOXES`, and outputs
  `Detection` with `RELATIVE_BOUNDING_BOX`.
- `NonMaxSuppressionCalculator` without an `IMAGE` input computes overlap in
  the detection coordinate domain. Weighted NMS reads/writes
  `relative_bounding_box`, so the integrated video graph must keep detections
  relative until after NMS.
- `PassThroughOrEmptyDetectionVectorCalculator` can convert an empty detection
  stream packet into an empty `std::vector<Detection>` at a frame tick.
- `PreviousLoopbackCalculator` requires the `LOOP` input to be marked
  `back_edge: true` when feeding previous final detections into the current
  frame.
- `StreamingTilesToTensorBatchCalculator` already models frame-local multi-batch
  output: `T > batch_capacity` emits multiple `TENSORS`/`BATCH_INFO` packets at
  synthetic monotonically increasing batch timestamps while preserving
  `source_frame_timestamp` in `TensorBatchInfo`.
- `TensorBatchInfo` carries `batch_index`, `total_batches`, `valid_count`,
  `batch_capacity`, `tile_indices`, and per-batch geometry. The last batch may
  be underfilled; fixed-batch models pad to batch capacity while dynamic-batch
  models emit `N == valid_count`.
- `MergeTileDetectionsAccumulatorCalculator` accumulates per-batch detections by
  `source_frame_timestamp` and emits one merged frame result only after all
  `total_batches` for that source frame arrive.

## Architecture

```
VIDEO frame
  |-------------------------> MotionAnalysisCalculator
                                |
                                v
                           FlowPackagerCalculator
                                |
                                | TRACKING:TrackingData @ timestamp t
                                +---------------------------+
                                                            |
base tiles --------------------------------------------------|----+
previous final detections -- PreviousLoopbackCalculator -----+    |
                                                                 v
                                                     VideoTileSchedulerCalculator
                                                       | REFRESH:bool
                                                       | TILES:scheduled tiles
                                                       v
                                     TileSpecToTilePlan -> TilePlan
                                                       |
                                                       v
                               StreamingTilesToTensorBatchCalculator
                                  | TENSORS/BATCH_INFO batch 0..K-1
                                  v
                               Inference -> batch decoder/filter/local NMS
                                  -> merge accumulator
                                                       |
                                                       v
                                           fresh merged detections (relative)
                                                       |
                                                       v
                                     PassThroughOrEmptyDetectionVectorCalculator

TRACKING:TrackingData -> BoxTrackerCalculator -> TRACKING_BOXES
fresh detector updates -------------------------> TrackedDetectionManagerCalculator
TRACKING_BOXES ---------------------------------> TrackedDetectionManagerCalculator
                                                       |
                                                       v
                                           tracked detections (relative)

fresh detections (relative) ----+
tracked detections (relative) --+--> ConcatenateDetectionVectorCalculator
                                      -> NonMaxSuppressionCalculator
                                      -> final detections (relative)
                                      -> public pixel conversion if needed
                                      -> PreviousLoopbackCalculator LOOP
```

The scheduler owns only the refresh decision and the scheduled tile list. It
does not own tracker state, does not emit detection hold packets, and does not
perform NMS.

## Calculator contract

`VideoTileSchedulerCalculator` is an api2 CPU calculator.

Inputs:
- `TILES`: `std::vector<NormalizedRect>`. Base candidate tiles for the current
  frame. These are already in frame-normalized coordinates.
- `PRIOR_DETECTIONS`: `std::vector<Detection>`. Previous final detections from
  `PreviousLoopbackCalculator`, relative boxes only. Empty means no usable
  detector/tracker cache.
- `TRACKING`: optional `TrackingData`. When absent/unconnected, the scheduler
  falls back to image-mode parity and DETECTs every frame.

Outputs:
- `TILES`: `std::vector<NormalizedRect>`. On DETECT, the scheduled subset of
  base tiles. On SKIP, an empty vector.
- `REFRESH`: `bool`. `true` for DETECT, `false` for SKIP. This is for tests,
  graph gating, and observability; it is not a second clock.

Both outputs are emitted every `Process()` at the input timestamp. The scheduler
uses default timestamp offset and must not declare `TimestampChange::Arbitrary`.

State:
- No frame counter.
- No private detection cache.
- No image or tensor cache inside the scheduler. Cache state lives in graph
  streams, tracker calculators, and tiling/inference calculators.

## Options

```proto
message VideoTileSchedulerCalculatorOptions {
  // Use the next free MediaPipe calculator-options extension id. The shipped
  // tiling protos use 471230004 and 471230005; verify 471230006 remains unused
  // before committing code.
  extend mediapipe.CalculatorOptions {
    optional VideoTileSchedulerCalculatorOptions ext = 471230006;
  }

  // Cap scheduled tiles on a DETECT frame. 0 = no cap, emit all base tiles.
  optional int32 max_scheduled_tiles = 1 [default = 0];

  // Force DETECT when mean foreground/object motion exceeds this threshold.
  // 0 = disabled.
  optional float motion_refresh_threshold = 2 [default = 0.0];

  // Allow SKIP when mean foreground/object motion is below this threshold.
  // 0 = disabled. FLAG_DUPLICATED can still skip when skip_on_duplicated=true.
  optional float motion_skip_threshold = 3 [default = 0.0];

  // Force DETECT when decoded/global feature count is below this threshold.
  // 0 = disabled.
  optional int32 min_global_features = 4 [default = 0];

  // Force DETECT when FlowPackager marks the background model unstable.
  optional bool refresh_on_background_unstable = 5 [default = true];

  // Force DETECT at FlowPackager chunk boundaries.
  optional bool refresh_on_chunk_boundary = 6 [default = true];

  // Allow SKIP on FLAG_DUPLICATED when there are prior detections to track.
  optional bool skip_on_duplicated = 7 [default = true];

  // When TRACKING is absent, DETECT every frame to preserve image-mode parity.
  optional bool detect_without_tracking = 8 [default = true];

  // Conservative default: if tracking is present but no hard DETECT or explicit
  // SKIP condition matches, DETECT rather than silently holding stale boxes.
  optional bool refresh_on_uncertain_tracking = 9 [default = true];
}
```

No option may reintroduce frame-count cadence. If a product needs wall-clock
freshness, it should be expressed outside this scheduler as a graph-level
policy that produces a tracking refresh signal, not as `frame_index_ % N`.

## Refresh decision

Let:
- `tracking_present` = `TRACKING` connected and non-empty.
- `priors_empty` = `PRIOR_DETECTIONS` packet empty or vector empty.
- `mvf` = decoded `MotionVectorFrame` when tracking is present.
- `feature_count` = `tracking.global_feature_count()` when set, otherwise
  `mvf.motion_vectors.size()`.
- `E` = mean `Norm(object)` over decoded motion vectors. If there are no
  vectors, `E = 0`.

Decision order:

1. **DETECT: tracking unavailable.** If `!tracking_present`, return
   `detect_without_tracking` (default `true`).
2. **DETECT: no cache to track.** If `priors_empty`, DETECT. SKIP is only legal
   when there is a previous detector result/tracker state to carry forward.
3. **DETECT: FlowPackager hard-refresh flags.**
   - `refresh_on_background_unstable && !mvf.valid_background_model`
   - `refresh_on_chunk_boundary && mvf.is_chunk_boundary`
4. **DETECT: unreliable or large motion.**
   - `min_global_features > 0 && feature_count < min_global_features`
   - `motion_refresh_threshold > 0 && E > motion_refresh_threshold`
5. **SKIP: FlowPackager says the frame is duplicated.**
   - `skip_on_duplicated && mvf.is_duplicated`
6. **SKIP: near-static foreground motion.**
   - `motion_skip_threshold > 0 && E < motion_skip_threshold`
7. **Fallback.**
   - If `refresh_on_uncertain_tracking`, DETECT.
   - Else SKIP.

This order makes hard refresh conditions dominate skip conditions. For example,
a chunk boundary with duplicated content still DETECTs, because tracker ids and
history cannot be safely mapped across the chunk boundary.

## DETECT-frame tile output

- If `max_scheduled_tiles == 0` or `base_tiles.size() <= max_scheduled_tiles`,
  emit all base tiles in original order.
- Otherwise compute a priority for each base tile and keep the top
  `max_scheduled_tiles`.
- Motion priority is per tile: sum `Norm(object)` for decoded features whose
  frame-normalized position falls inside that tile.
- Fallback priority is also per tile: if a tile's motion priority is zero, use
  the number of prior relative detection centers inside that tile. Do not wait
  for all tiles to have zero motion before falling back.
- Sort by descending priority, break ties by original tile index, then emit kept
  tiles in original order to maximize tile-plan cache hits.

## Frame-local multi-batch streaming

Let:
- `T` = number of scheduled tiles emitted by the scheduler for a DETECT frame.
- `B` = model input batch capacity from `InferenceMetadata.batch_capacity()`.
- `K = ceil(T / B)` = number of inference calls required for this source frame.

For example, if `T = 5` and `B = 2`, the tiled detect path must produce:

| Batch | Tile rows | `valid_count` | Fixed-batch tensor `N` | Dynamic-batch tensor `N` |
| --- | --- | ---: | ---: | ---: |
| 0 | tiles 0, 1 | 2 | 2 | 2 |
| 1 | tiles 2, 3 | 2 | 2 | 2 |
| 2 | tile 4 | 1 | 2 with one padded row | 1 |

Rules:
- Batch rows may contain tiles from exactly one source frame.
- Each batch gets a synthetic monotonic batch timestamp for graph ordering, but
  every `BATCH_INFO` keeps the original `source_frame_timestamp`.
- The first full batch must be sent downstream immediately after its `B` rows
  are tensorized. The implementation must not first crop/resize all `T` tiles
  and then bulk-send `K` batches.
- The final underfilled batch is emitted immediately after the last scheduled
  tile for that frame is prepared.
- The per-batch decoder projects detections with the corresponding
  `TensorBatchInfo.geometry`, then the merge accumulator waits for all `K`
  batches and emits exactly one fresh detection vector at the source frame
  timestamp.
- Global NMS runs only after the merge accumulator emits the single source-frame
  fresh detection vector and after tracker-updated detections for that same
  source timestamp are available.
- If MediaPipe scheduling cannot run downstream inference until the tensorizer's
  `Process()` returns, the tensorizer must be refactored to a true loop/yield
  shape, such as a BeginLoop-style per-batch producer. It is not enough to call
  `Send()` several times from one long monolithic `Process()` if inference still
  waits for all tile work to finish.

## Candidate filtering and NMS order

For each tile row in an inference batch, candidate processing must follow this
order:

1. Decode model outputs for that tile row.
2. Apply `conf_threshold` / `min_score_thresh` immediately. This is mandatory;
   low-score candidates should not enter local NMS, merge accumulation, or
   global NMS.
3. Optionally apply a pre-NMS top-K cap to bound local NMS cost.
4. Optionally run tile-local NMS inside that single tile row.
5. Optionally apply a post-local-NMS top-K cap for transport/merge limits.
6. Project surviving candidates to full-frame-relative coordinates with that
   row's `TensorBatchInfo.geometry`.
7. Merge all tile rows for the source frame.
8. Concatenate the merged fresh detections with tracker-updated detections.
9. Run the frame-level global NMS, or take the safe pass-through bypass below.

Rules:
- Tile-local NMS may compare candidates only within one tile row. It must not
  compare candidates from different tiles, different source frames, or the
  tracker path.
- Tile-local NMS is a performance optimization, not a correctness boundary.
  Overlapping tiles can still produce duplicate boxes for the same object, and
  those duplicates must survive until global NMS.
- Local NMS thresholds should be no more aggressive than the product's recall
  target allows. If weighted global NMS is used, local NMS/top-K can change the
  weighted box by removing candidates early; this must be covered by accuracy
  tests and should be configurable.
- If a model already emits post-NMS detections, the local NMS stage should be a
  no-op unless tests show additional local suppression is safe.
- Global NMS still runs once per source frame when there are multiple tile rows,
  tracker-updated detections, or no proof that the single tile row was already
  locally NMSed/post-NMS.
- Safe final-NMS bypass is allowed only when all of these are true:
  - `valid_tile_rows == 1`
  - tracker-updated detections for the source timestamp are empty
  - no propagated/cached detector candidates are concatenated
  - tile-local NMS ran for that row, or the model is declared post-NMS
  - no weighted global NMS is required for box fusion
- When bypassed, the graph must still emit exactly one final detection vector at
  the source timestamp and should expose a test-observable flag/counter so tests
  can prove the NMS calculator was not invoked.

## SKIP-frame output

- `TILES` = empty vector.
- `REFRESH` = false.
- No detection vector is emitted by the scheduler. `TrackedDetectionManager`
  provides tracker-updated detections for the frame. If the detector branch
  produces no packet on SKIP, use `PassThroughOrEmptyDetectionVectorCalculator`
  to materialize an empty vector before concat/NMS.

## Coordinate-system contract

- Scheduler inputs and tracker/NMS integration streams use
  `LocationData::RELATIVE_BOUNDING_BOX`.
- `TrackedDetectionManagerCalculator` already consumes and outputs relative
  boxes, so it is compatible with the scheduler loopback.
- `NonMaxSuppressionCalculator` must run without `IMAGE` for this graph segment,
  keeping overlap calculations in relative coordinates.
- If the public Tasks API result must be pixel units, run
  `DetectionTransformationCalculator` after final NMS and outside this scheduler
  loop.

## Cache and zero-copy contract

The scheduler is metadata-only and must not break GPU zero-copy paths.

- Tile cache: keep tile rect order stable so `TileSpecToTilePlanCalculator` can
  reuse plan/cache entries for repeated tile sets.
- Batch geometry cache: `StreamingTilesToTensorBatchCalculator` caches
  per-batch matrices/ROIs keyed by frame size, model input shape, and the exact
  tile rows in that batch. For `T = 5, B = 2`, the cache sees three row groups:
  `[0,1]`, `[2,3]`, and `[4]`.
- Input tensor/workspace cache: tensor buffer pooling belongs in the
  tiling/inference calculators. A streaming implementation needs enough pooled
  CPU/GPU workspaces to send batch 0 to inference while preparing batch 1 without
  mutating a buffer still owned by downstream inference.
- Pixel cache: pixels are not cached by default. Reusing cached tile tensors or
  detector outputs is only legal when the cache key includes a valid source
  image identity/content version, model metadata, tile geometry, preprocessing
  options, and batch row. A duplicated-frame signal should normally SKIP
  inference and use tracker-updated detections instead of replaying stale tile
  tensors.
- Output cache: per-batch detector outputs are accumulated by
  `source_frame_timestamp` until `total_batches` arrive, then released. Previous
  final detections are carried by `PreviousLoopbackCalculator` and tracker
  calculators, not by copying vectors into scheduler-owned state.
- OpenGL zero-copy: GPU paths must pass `Image`/`GpuBuffer`/tensor handles
  through the existing GPU calculators. A SKIP frame should avoid tile tensor
  generation by emitting an empty tile vector; it should not force CPU readback
  just to decide refresh.

## Integration requirements

- Keep the motion/tracker path alive for every video frame. Do not put
  `FlowLimiterCalculator` in front of `MotionAnalysisCalculator` or
  `FlowPackagerCalculator`.
- Use `FlowLimiterCalculator` only around the expensive tiled detect branch when
  the product graph needs real-time backpressure, mirroring
  `object_detection_mobile_gpu.pbtxt`.
- The tiled detect branch must preserve the streaming multi-batch contract:
  `scheduled tiles -> TilePlan -> per-batch tensors -> inference per batch ->
  decoder/conf-filter/tile-local candidate reduction -> merge accumulator ->
  one source-frame fresh detection vector`.
- Backpressure must not cause batch 0 of a frame to wait for batch `K-1` of the
  same frame before inference can begin. If in-flight inference is limited to 1,
  the queue boundary should still be between prepared batches, not after all
  tile preprocessing for the frame.
- Use `PreviousLoopbackCalculator` with `input_stream_info { tag_index: "LOOP"
  back_edge: true }`.
- Use `PassThroughOrEmptyDetectionVectorCalculator` so concat/NMS receives a
  vector at every frame timestamp even when the detect branch is skipped.
- Keep tile-local NMS/top-K inside the tiled detector branch before merge. The
  final frame-level suppression stage remains after concat with tracker-updated
  detections; it may choose pass-through only for the safe single-tile/no-tracker
  case.
- The integration test may stub the tiled detector and tracker-updated stream,
  but it must prove that SKIP frames do not copy prior detections in the
  scheduler and that final output still passes through exactly one NMS.

## Testing

Unit tests:
- No `TRACKING` input: every frame DETECTs and emits all tiles.
- Empty priors with tracking present: DETECT.
- `FLAG_BACKGROUND_UNSTABLE`: DETECT.
- `FLAG_CHUNK_BOUNDARY`: DETECT, even if duplicated.
- `FLAG_DUPLICATED`: SKIP when priors are non-empty.
- Low `global_feature_count`/decoded feature count: DETECT.
- High foreground motion: DETECT.
- Low foreground motion: SKIP when `motion_skip_threshold` is enabled.
- Uncertain tracking with no matching hard-refresh/skip rule: DETECT by default.
- `max_scheduled_tiles`: top-K is deterministic and restores original order.
- Per-tile fallback: a tile with zero motion priority can still be selected
  from prior detection centers even when other tiles have motion features.
- Aspect mapping: landscape and portrait decoded positions map to frame-normalized
  tile coordinates as verified against `tracking.cc`.
- Multi-batch streaming: `T = 5`, `batch_capacity = 2` emits three inference
  batches with `valid_count` 2, 2, 1, and the merge accumulator emits one fresh
  detection vector at the source frame timestamp before global NMS.
- Streaming latency: prove, with a spy/stub calculator or event log, that batch 0
  is observable by the inference stub before the tensorizer begins or completes
  the final underfilled batch. If the current graph execution model cannot prove
  this, refactor the tensorizer into a true per-batch producer.
- Cache behavior: repeated scheduled tile sets hit the tile-plan and per-batch
  geometry caches; tensor workspace pooling does not reuse a buffer before
  downstream inference releases it.
- Candidate reduction: per-tile `conf_threshold` removes low-score candidates
  before local NMS; optional tile-local NMS reduces duplicate candidates within a
  tile; overlapping-tile duplicates survive until the single global NMS.
- Final-NMS bypass: a single valid tile row with no tracker candidates and
  already-local-NMSed/post-NMS candidates bypasses the NMS call and emits the same
  vector; adding a second tile or any tracker candidate disables the bypass.

Integration test:
- Wire `PreviousLoopbackCalculator` with a real back edge.
- Stub fresh detections on DETECT frames and empty fresh detections on SKIP
  frames.
- Stub or wire tracker-updated detections on SKIP frames.
- Concatenate fresh and tracked detections, run the frame-level suppression stage
  (`NonMaxSuppressionCalculator` or safe bypass), and assert exactly one final
  detection vector packet per source timestamp.
- Assert the scheduler has no `PROPAGATED_DETECTIONS` output and no copied-prior
  SKIP result path.

## Done criteria

- `VideoTileSchedulerCalculator` has no frame-cadence option or frame counter.
- Refresh decisions are driven by `FlowPackager`/decoded tracking flags.
- SKIP frames are track-only through the tracker path, not copied priors.
- DETECT-frame tile priority/cap is deterministic and cache-friendly.
- DETECT frames with `tiles > batch_capacity` stream multiple frame-local
  inference batches and do not wait for all tile preprocessing before the first
  inference call.
- Per-tile score filtering and optional tile-local NMS/top-K reduce candidate
  volume before merge without replacing frame-level suppression. The final NMS
  call is skipped only for the safe single-tile/no-tracker/post-local-NMS case.
- Relative-coordinate graph contract matches existing detection/tracker/NMS
  calculators; pixel conversion is explicitly post-NMS/public-output only.
- CPU tests pass under `--define MEDIAPIPE_DISABLE_GPU=1`.
