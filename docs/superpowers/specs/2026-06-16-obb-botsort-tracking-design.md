# OBB BoTSORT Tracking (C++ core) — Design

**Date:** 2026-06-16
**Branch:** `dev` (MediaPipe fork; unmerged, no upstream PR)
**Status:** Approved design — ready for implementation plan
**Scope:** Sub-project 1 of 2 (C++ core). Sub-project 2 (C-API + Python bindings) is a follow-up.

## Goal

In the OrientedObjectDetector (OBB) tiled livestream path, optionally run BoTSORT to
assign stable `track_id`s to oriented detections, using **ID-association on
axis-aligned bounding boxes (AABBs)** while keeping output geometry as the fresh
rotated detections. Default OBB behavior is unchanged (no tracking). This is the OBB
analog of the shipped YOLO BoTSORT + track-id work, adapted to the oriented pipeline.

## Background / constraints

OBB today is effectively stateless per-frame: its graph
(`oriented_object_detector_graph.cc`) instantiates only `TiledObbMergeGraph`
(`MergeTileDetectionsAccumulatorCalculator` → `RotatedNonMaxSuppressionCalculator`),
with no stream-mode tracking branch. The wrapper accepts VIDEO/LIVE_STREAM but each
frame is independent.

Three constraints shape the design:
1. **OBB uses a separate proto/result type.** The pipeline flows
   `std::vector<OrientedDetection>` (`mediapipe/framework/formats/oriented_detection.proto`),
   NOT the framework `mediapipe::Detection` that the tracking machinery
   (`TiledTrackingGraph`, `BotsortTrackingCalculator`, `TiledFrameSuppressionCalculator`)
   consumes. The public result type is `OrientedObjectDetection`
   (`cx, cy, width, height, rotation`), separate from the shared `Detection`.
2. **Trackers are axis-aligned.** BoTSORT's Kalman state + association are on
   axis-aligned `tlwh`; BoxTracker propagates axis-aligned boxes via optical flow.
   Neither models rotation. → We track AABBs for **ID only**; geometry stays fresh.
   BoxTracker is excluded (its optical-flow boxes can't carry an angle).
3. **Favorable facts:** `OrientedDetection` proto ALREADY has `optional string
   track_id = 11` ("populated by tracking (later group)") + `label_id`/`score`/`rotation`
   — no proto change to the producer. The OBB graph already exposes the source frame
   (`image_in` → PassThrough IMAGE). `TiledObbMergeGraph` is a clean base.

## Architecture

### 1. `OrientedBotsortTrackingCalculator` (one new, self-contained)

api2 calculator in `mediapipe/calculators/tensor/`, reusing the existing
`BotsortTrackingCalculatorOptions` proto (same BoTSORT knobs — no new options message):
- **Inputs:** `IMAGE` (`ImageFrame`, BoTSORT's frame arg), `ORIENTED_DETECTIONS`
  (`std::vector<OrientedDetection>`, fresh, frame-merged).
- **Output:** `ORIENTED_DETECTIONS` (`std::vector<OrientedDetection>`) — the same fresh
  oriented detections, with `track_id` set on matched ones.
- **Open:** build a `botsort::BoTSORT` from the options (reid off; gmc per option),
  exactly as `BotsortTrackingCalculator::Open` does.
- **Process:** for each fresh `OrientedDetection`, compute its **AABB** (axis-aligned
  box enclosing the rotated rectangle defined by `cx,cy,width,height,rotation` in
  normalized space) → pixel `botsort::Detection` (`class_id = label_id(0)`,
  `confidence = score(0)`). Run `tracker_->track(aabbs, frame)`. For each returned
  `Track`, find the fresh oriented detection whose AABB best-IoU-matches the track's
  `tlwh` and `set_track_id(std::to_string(track->track_id))` on a copy of it (guard:
  only if not already set; best-IoU match). Emit the fresh oriented detections (rotated
  geometry untouched), ids attached where matched. **No gap-fill** — a Kalman/AABB box
  has no angle, so tracker-only boxes are never emitted; rotation is never tracked.
- The AABB-of-rotated-rect helper uses the four corner points
  (`cx,cy` ± rotated half-extents) → min/max x,y. Unit-tested directly.

### 2. `TiledObbTrackMergeGraph` (one new subgraph)

Stream sibling of `TiledObbMergeGraph` in `graphs/tiled_detection/`:
```
ORIENTED_DETECTIONS + BATCH_INFO → MergeTileDetectionsAccumulatorCalculator
                                 → RotatedNonMaxSuppressionCalculator     (fresh oriented; reused)
fresh + IMAGE → OrientedBotsortTrackingCalculator → ORIENTED_DETECTIONS (with track_id)
```
Inputs: `ORIENTED_DETECTIONS`, `BATCH_INFO`, `IMAGE`. Output: `ORIENTED_DETECTIONS`.
Reuses `TiledObbMergeGraphOptions`, extended with an optional `tracking`
sub-message (carrying tracker_type + BoTSORT knobs), mirroring YOLO's
`TiledBoxMergeGraphOptions.tracking`. Simpler than YOLO's track-merge — no
`TiledFrameSuppression` fusion, because BoTSORT here is ID-only and the output set is
exactly the fresh rotated-NMS'd detections.

### 3. OBB graph stream branch

In `oriented_object_detector_graph.cc`, where `TiledObbMergeGraph` is added (~line 409),
branch: when `task_options.base_options().use_stream_mode() && TilingEnabled(tiling)
&& tracking.tracker_type == BOTSORT`, instantiate `TiledObbTrackMergeGraph` (wiring the
source frame to its `IMAGE` input + forwarding the tracking options); otherwise the
existing `TiledObbMergeGraph` (unchanged → **default OBB output byte-identical**).

### 4. `TrackingOptions` (BoTSORT-only)

OBB proto (`proto/oriented_object_detector_options.proto`) + public struct
(`oriented_object_detector.h`) + converter (`oriented_object_detector.cc`), mirroring
YOLO's `TrackingOptions` for cross-detector/binding parity:
- Enum `TrackerType { TRACKER_UNSPECIFIED = 0, BOX_TRACKER = 1, BOTSORT = 2 }`,
  **default `TRACKER_UNSPECIFIED`** which for OBB means *no tracking* (stateless
  per-frame — OBB's existing behavior). Same BoTSORT knobs as YOLO
  (track_high/low/new thresholds, track_buffer, match_threshold, enable_gmc).
- **Create-time validation:** `BOTSORT` requires `use_stream_mode() == true` (not
  IMAGE), `TilingEnabled(tiling) == true`, `!tiling.enable_motion_scheduling()`, and
  `num_classes ∈ [1, 256]` — each a distinct `InvalidArgument`, reusing YOLO's gate
  shape. `BOX_TRACKER` → `InvalidArgument` ("BoxTracker is not supported for oriented
  detection; use BOTSORT."). `TRACKER_UNSPECIFIED` (default) is exempt (no tracking).
- The confirmation-threshold-vs-`score_threshold` foot-gun (BoTSORT only emits a
  `track_id` for confirmed tracks) is documented at each OBB option site, matching the
  YOLO docs.

### 5. Surface `track_id` on the OBB result

OBB's result type is separate from the shared `Detection`:
- `OrientedObjectDetection` (`components/containers/oriented_object_detection_result.h`)
  gains `std::optional<std::string> track_id = std::nullopt;`.
- The oriented proto→container conversion (in
  `oriented_object_detection_result.cc`) reads the proto's existing `track_id` field
  (set when present). Additive; empty when untracked.

## Error handling / edge cases

- Default (`TRACKER_UNSPECIFIED`) or IMAGE mode → no tracking, output byte-identical to
  today.
- `BOTSORT` misconfigured (IMAGE / no tiling / motion-scheduling / >256 classes /
  BOX_TRACKER) → distinct `InvalidArgument` at `Create()` before model load.
- An oriented detection unmatched by any track → emitted with no `track_id` (rather than
  dropped) — geometry/recall preserved.
- `>256` classes rejected (BoTSORT's `uint8_t` class id), same as YOLO.

## Testing

- **Calculator unit test:** AABB-of-rotated-rect correctness (a 45°-rotated box →
  expected enclosing AABB); a track's `track_id` is set + stable across frames; output
  rotation/geometry equals the fresh input; empty-frame safe (no crash, empty output).
- **Subgraph test:** `TiledObbTrackMergeGraph` emits oriented detections carrying ids
  across frames; the default `TiledObbMergeGraph` path is unchanged.
- **Converter + validation tests (model-free):** `TrackingOptions` round-trip; reject
  BOTSORT+IMAGE, BOTSORT+no-tiling, BOX_TRACKER, >256 classes.
- **Container conversion test:** oriented proto with `track_id` → container optional
  set; absent → nullopt.
- **Real e2e:** OBB has a vendored fixture (`yolo_obb_test_model`, finds a ship on
  boats.jpg). A tiled VIDEO test selecting BOTSORT asserts oriented detections carry
  stable `track_id`s across consecutive frames (tracker thresholds tuned to the scene,
  test-only, like YOLO's e2e).

## Scope boundaries (YAGNI)

**This spec (C++ core):** the calculator, the stream merge subgraph, the graph branch,
`TrackingOptions` (BoTSORT-only) + validation, the result `track_id`, and the e2e.
YOLO is untouched; OBB default behavior is unchanged.

**Follow-up (sub-project 2):** OBB tracking C-API + Python bindings (the OBB
`TrackingOptions` + the oriented result's `track_id` through the C/Python oriented
result types).

**Out of scope:** rotation-aware tracking (rotated Kalman / rotated-IoU association);
BoxTracker for OBB; gap-fill / tracker-only oriented boxes; tracking on the non-tiled
OBB path.

## Verifiability

The calculator, subgraph, graph branch, options, validation, and container changes
build+run here under `--define MEDIAPIPE_DISABLE_GPU=1`; the OBB e2e runs against the
vendored `yolo_obb_test_model` fixture (OBB e2e tests already run here, unlike YOLO's
which skip). No Python in this sub-project.
