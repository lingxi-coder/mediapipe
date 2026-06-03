# Video Tile Scheduler (Phase 4 / M9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `VideoTileSchedulerCalculator`, a CPU, per-frame video scheduler
whose refresh policy is driven directly by `FlowPackagerCalculator`'s
`TrackingData` and decoded tracking flags. The scheduler emits scheduled tile
rects for DETECT frames and empty tile rects for SKIP frames. It does not copy
prior detections on SKIP.

**Architecture:** The graph keeps the mature MediaPipe video pattern:
`MotionAnalysisCalculator -> FlowPackagerCalculator -> BoxTrackerCalculator ->
TrackedDetectionManagerCalculator`. Fresh tiled detections and tracker-updated
detections are concatenated and sent through a frame-level suppression stage.
That stage invokes global NMS unless the safe single-tile/no-tracker bypass
applies. The tiled detect branch may use `FlowLimiterCalculator` for
backpressure, but the motion/tracker path remains per-frame.

**Spec:** `docs/superpowers/specs/2026-06-03-video-tile-scheduler-design.md`.

---

## Integration stance

This plan intentionally keeps the coupled video graph in one milestone. Do not
split the scheduler, frame-local batching, cache behavior, tile-local candidate
reduction, tracker-updated SKIP results, and frame-level suppression into
independent follow-up plans. These nodes share timestamps, cache keys, coordinate
systems, and candidate lifetimes; implementing them piecemeal risks incompatible
contracts between adjacent calculators.

The implementation should still be staged for reviewability, but every stage
must preserve the end-to-end graph contract defined here. A task is not complete
just because its local calculator works if it breaks downstream batching,
tracker concat, or final frame-level suppression semantics.

## Reference facts to preserve

- `FlowPackagerCalculator` emits `TRACKING:TrackingData` at the flow packet
  timestamp. Its internal `frame_idx_` is chunk metadata only.
- `TrackingData` exposes `FLAG_BACKGROUND_UNSTABLE`, `FLAG_DUPLICATED`,
  `FLAG_CHUNK_BOUNDARY`, `global_feature_count`, and
  `average_motion_magnitude`.
- `MotionVectorFrameFromTrackingData()` decodes to `MotionVectorFrame` with
  `valid_background_model`, `is_duplicated`, `is_chunk_boundary`,
  `aspect_ratio`, and foreground/object motion vectors.
- `TrackedDetectionManagerCalculator` consumes/outputs `Detection` with
  `LocationData::RELATIVE_BOUNDING_BOX`.
- `NonMaxSuppressionCalculator` without `IMAGE` must receive relative boxes;
  weighted NMS also reads/writes `relative_bounding_box`.
- `PreviousLoopbackCalculator` requires `input_stream_info { tag_index: "LOOP"
  back_edge: true }`.
- `PassThroughOrEmptyDetectionVectorCalculator` materializes empty detection
  vectors for skipped detector frames.
- `object_detection_mobile_gpu.pbtxt` uses `FlowLimiterCalculator` to bound
  expensive detection work; borrow this only for the detect branch, not for the
  per-frame motion/tracker branch.
- `StreamingTilesToTensorBatchCalculator` emits frame-local multi-batch
  `TENSORS`/`BATCH_INFO` packets with synthetic batch timestamps and original
  `source_frame_timestamp` preserved in `TensorBatchInfo`.
- `MergeTileDetectionsAccumulatorCalculator` waits for all
  `TensorBatchInfo.total_batches` for a source frame before emitting one merged
  fresh detection vector at the source frame timestamp.
- Decoder-side `conf_threshold`/`min_score_thresh` should run before any local
  candidate reduction. Optional tile-local NMS/top-K may run inside a single tile
  row, but cross-tile/tracker dedup remains the final global NMS.
- The final NMS call may be bypassed only when there is exactly one valid tile
  row, no tracker-updated detections, and that row already ran tile-local NMS or
  comes from a post-NMS model output.

## Cross-cutting constraints

- No `detect_every_n_frames`.
- No `frame_index_ % N` refresh logic.
- No `PROPAGATED_DETECTIONS` output from the scheduler.
- Defaults plus no `TRACKING` must reproduce image mode: every frame DETECTs and
  all base tiles are emitted.
- The scheduler is metadata-only. It must not inspect pixels, copy GPU buffers,
  or force OpenGL readback.
- Tile output order must be stable to help `TileSpecToTilePlanCalculator` cache
  repeated tile plans.
- A DETECT frame with more scheduled tiles than model batch capacity must stream
  multiple single-frame inference batches. Example: 5 scheduled tiles with
  batch capacity 2 means 3 inference calls with valid counts 2, 2, and 1.
- The first filled tile batch must be sent to inference as soon as it is ready;
  do not tensorize all tiles for the frame before sending batch 0.
- Per-tile candidate reduction order is fixed: decode, score/conf threshold,
  optional top-K, optional tile-local NMS, optional post-local top-K, projection,
  merge, then frame-level suppression. Tile-local NMS must not compare across
  tiles. The frame-level NMS call can be skipped only by the safe single-tile
  bypass.
- Coordinate system inside this graph segment is relative. Public pixel
  conversion is after final NMS only.
- If implementation commits are made, keep the project-requested
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  trailer. No extra commit-message protocol is part of this plan.

## File Structure

- **Create** `mediapipe/calculators/tensor/video_tile_scheduler_calculator.proto`
  for `VideoTileSchedulerCalculatorOptions`.
- **Create** `mediapipe/calculators/tensor/video_tile_scheduler_calculator.cc`
  for the api2 calculator.
- **Create** `mediapipe/calculators/tensor/video_tile_scheduler_calculator_test.cc`
  for unit tests.
- **Create** `mediapipe/calculators/tensor/video_tile_scheduler_pipeline_test.cc`
  for streaming multi-batch + loopback + tracker-stub + concat + single-NMS
  integration.
- **Modify** `mediapipe/calculators/tensor/BUILD`.

---

## Task 1: Proto + calculator skeleton, no frame counter

**Files:** create proto, calculator, unit test; modify BUILD.

This establishes the contract: required tile/prior streams, optional tracking,
scheduled tile output, and a `REFRESH` bool output. Default/no-tracking behavior
is image-mode parity.

- [ ] **Step 1: Write the proto** `video_tile_scheduler_calculator.proto`.

```proto
syntax = "proto2";

package mediapipe;

import "mediapipe/framework/calculator.proto";

message VideoTileSchedulerCalculatorOptions {
  extend mediapipe.CalculatorOptions {
    optional VideoTileSchedulerCalculatorOptions ext = 471230006;
  }

  optional int32 max_scheduled_tiles = 1 [default = 0];
  optional float motion_refresh_threshold = 2 [default = 0.0];
  optional float motion_skip_threshold = 3 [default = 0.0];
  optional int32 min_global_features = 4 [default = 0];
  optional bool refresh_on_background_unstable = 5 [default = true];
  optional bool refresh_on_chunk_boundary = 6 [default = true];
  optional bool skip_on_duplicated = 7 [default = true];
  optional bool detect_without_tracking = 8 [default = true];
  optional bool refresh_on_uncertain_tracking = 9 [default = true];
}
```

Before committing code, verify `471230006` is still unused:

```bash
rg -n "471230006|VideoTileSchedulerCalculatorOptions" mediapipe docs
```

- [ ] **Step 2: Add the skeleton unit test.**

Test name: `DefaultsWithoutTrackingDetectEveryFrame`.

Assertions:
- Feed two timestamps through one `CalculatorRunner`.
- Provide non-empty `TILES` and `PRIOR_DETECTIONS`.
- Do not connect `TRACKING`.
- Each output packet has `REFRESH == true`.
- Each output `TILES` packet equals the full base tile vector.
- There is no propagated detection output in the node config.

Use `Detection` helpers that create `LocationData::RELATIVE_BOUNDING_BOX` only.

- [ ] **Step 3: Write the calculator skeleton.**

Contract:

```cpp
class VideoTileSchedulerCalculator : public Node {
 public:
  static constexpr Input<std::vector<NormalizedRect>> kInTiles{"TILES"};
  static constexpr Input<std::vector<Detection>> kInPriorDets{
      "PRIOR_DETECTIONS"};
  static constexpr Input<TrackingData>::Optional kInTracking{"TRACKING"};
  static constexpr Output<std::vector<NormalizedRect>> kOutTiles{"TILES"};
  static constexpr Output<bool> kOutRefresh{"REFRESH"};
  MEDIAPIPE_NODE_CONTRACT(kInTiles, kInPriorDets, kInTracking, kOutTiles,
                          kOutRefresh);
};
```

Skeleton behavior:
- Validate non-negative numeric options in `Open()`.
- In `Process()`, if `TRACKING` is absent/unconnected and
  `detect_without_tracking` is true, emit all base tiles and `REFRESH=true`.
- Do not add `frame_index_`.
- Do not add scheduler-owned detection/image/tensor caches.

- [ ] **Step 4: Add BUILD targets.**

Mirror the existing `tile_spec_to_tile_plan_calculator` pattern. Expected deps:

```python
":video_tile_scheduler_calculator_cc_proto",
"//mediapipe/framework:calculator_framework",
"//mediapipe/framework/api2:node",
"//mediapipe/framework/formats:detection_cc_proto",
"//mediapipe/framework/formats:rect_cc_proto",
"//mediapipe/framework/port:ret_check",
"//mediapipe/util/tracking:flow_packager_cc_proto",
"@com_google_absl//absl/status",
```

The `//mediapipe/util/tracking:tracking` dep is introduced in Task 2 when the
decode helper is used.

- [ ] **Step 5: Verify.**

```bash
bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/calculators/tensor:video_tile_scheduler_calculator_test \
  --test_output=all
```

---

## Task 2: FlowPackager/tracking-flag refresh decision

**Files:** modify calculator and unit test.

This task implements the whole DETECT/SKIP policy without any cadence fallback.
The scheduler is conservative: hard refresh flags dominate skip flags, and
uncertain tracking DETECTs by default.

- [ ] **Step 1: Refactor decision logic into a testable helper.**

Use a small internal struct so tests can bypass the sparse `TrackingData`
encoding when needed:

```cpp
struct TrackingSignals {
  bool present = false;
  bool valid_background_model = true;
  bool is_duplicated = false;
  bool is_chunk_boundary = false;
  int feature_count = 0;
  float mean_foreground_motion = 0.0f;
};
```

`DecodeTrackingSignals(const TrackingData&)` should call
`MotionVectorFrameFromTrackingData()` and compute:
- `valid_background_model`
- `is_duplicated`
- `is_chunk_boundary`
- `feature_count` from `global_feature_count` when set, else decoded vector size
- mean foreground/object motion from decoded vectors

- [ ] **Step 2: Add failing tests.**

Add table-style tests for `DecideRefresh(priors, signals)`:

- `NoTrackingDefaultsToDetect`
- `EmptyPriorsForceDetect`
- `BackgroundUnstableForcesDetect`
- `ChunkBoundaryForcesDetectEvenWhenDuplicated`
- `DuplicatedSkipsWhenPriorsExist`
- `LowFeatureCountForcesDetect`
- `HighForegroundMotionForcesDetect`
- `LowForegroundMotionSkipsWhenThresholdEnabled`
- `UncertainTrackingDetectsByDefault`

Each test should assert both outputs:
- DETECT -> scheduled tiles non-empty, `REFRESH=true`.
- SKIP -> scheduled tiles empty, `REFRESH=false`.

- [ ] **Step 3: Implement the decision order.**

Decision pseudocode:

```cpp
bool ShouldRefresh(const std::vector<Detection>& priors,
                   const TrackingSignals& s) const {
  if (!s.present) return options_.detect_without_tracking();
  if (priors.empty()) return true;
  if (options_.refresh_on_background_unstable() &&
      !s.valid_background_model) return true;
  if (options_.refresh_on_chunk_boundary() && s.is_chunk_boundary) return true;
  if (options_.min_global_features() > 0 &&
      s.feature_count < options_.min_global_features()) return true;
  if (options_.motion_refresh_threshold() > 0.0f &&
      s.mean_foreground_motion > options_.motion_refresh_threshold()) {
    return true;
  }
  if (options_.skip_on_duplicated() && s.is_duplicated) return false;
  if (options_.motion_skip_threshold() > 0.0f &&
      s.mean_foreground_motion < options_.motion_skip_threshold()) {
    return false;
  }
  return options_.refresh_on_uncertain_tracking();
}
```

In `Process()`:
- Decode `TrackingSignals` once when `TRACKING` is connected and non-empty.
- DETECT emits scheduled tiles (full base list for now) and `REFRESH=true`.
- SKIP emits empty tile vector and `REFRESH=false`.
- Do not emit or store detections.

- [ ] **Step 4: Verify.**

```bash
bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/calculators/tensor:video_tile_scheduler_calculator_test \
  --test_output=all
```

---

## Task 3: DETECT-frame tile priority and cap

**Files:** modify calculator and unit test.

This task applies `max_scheduled_tiles` only on DETECT frames. SKIP frames still
emit an empty tile vector.

- [ ] **Step 1: Add tests.**

Tests:
- `NoCapEmitsAllTiles`
- `CapKeepsTopMotionTilesInStableOrder`
- `PerTilePriorFallbackCanSelectZeroMotionTile`
- `AllZeroPriorityFallsBackToOriginalOrder`
- `AspectMappingMatchesTrackingDomain`

The fallback test must cover the important case where some tiles have motion and
one tile has zero motion but a prior detection center. That zero-motion tile is
allowed to use prior fallback without requiring all tile motion sums to be zero.

- [ ] **Step 2: Implement priority helpers.**

Helpers:
- `PointInTile(float x, float y, const NormalizedRect& tile)`
- `FeatureFramePos(const Vector2_f& pos, float aspect, float* x, float* y)`
- `PriorCenterPriority(tile, priors)`
- `MotionPriority(tile, decoded_motion_vectors)`

Scheduling rules:
- If `max_scheduled_tiles <= 0`, return all tiles.
- If `base.size() <= max_scheduled_tiles`, return all tiles.
- Else score every tile:
  - Start with per-tile motion energy.
  - If that tile's motion score is zero, use prior-center count for that tile.
- Sort by descending score and ascending original index.
- Emit selected tiles in original input order.

- [ ] **Step 3: Verify.**

```bash
bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/calculators/tensor:video_tile_scheduler_calculator_test \
  --test_output=all
```

---

## Task 4: Frame-local multi-batch streaming and cache verification

**Files:** create or extend `video_tile_scheduler_pipeline_test.cc`; modify
BUILD only if new test deps are needed.

This task proves the tiled detect branch handles `tiles > batch_capacity`
correctly and quickly. The scheduler may choose 5 tiles for a DETECT frame while
the model accepts only 2 rows per inference input; that frame must run 3
inference batches, apply tile-local candidate reduction inside each row, and
still produce one source-frame detection vector before global NMS.

- [ ] **Step 1: Add the `T = 5, B = 2` streaming test.**

Graph shape:
- `VideoTileSchedulerCalculator` emits 5 scheduled tiles for one DETECT frame.
- `TileSpecToTilePlanCalculator` converts them to a `TilePlan`.
- `StreamingTilesToTensorBatchCalculator` uses `InferenceMetadata` with
  `batch_capacity: 2`.
- A lightweight inference spy/stub records the timestamp and order of every
  `TENSORS` packet it receives, then emits canned batch detections.
- Batch decoder/merge accumulator consumes `BATCH_INFO` and canned detections.

Assertions:
- exactly 3 inference input packets are observed for the source frame
- `TensorBatchInfo.batch_index` is 0, 1, 2
- `TensorBatchInfo.total_batches` is 3 for all three batches
- `valid_count` is 2, 2, 1
- all three `BATCH_INFO` packets carry the same `source_frame_timestamp`
- fixed-batch mode pads the last tensor to `N == 2`; dynamic-batch mode emits
  the last tensor with `N == 1`
- merge emits exactly one fresh detection vector at the source frame timestamp

- [ ] **Step 2: Prove streaming, not bulk materialization.**

Add a spy/event-log test that fails if inference cannot observe batch 0 until
after every tile in the source frame is tensorized. Acceptable evidence:
- a spy calculator receives batch 0 before the tensorizer logs preparation of
  the final underfilled batch, or
- the tensorizer is refactored into a true per-batch producer/loop where each
  `Process()` emits one prepared batch and yields to the graph scheduler.

Calling `Send()` multiple times from one long `Process()` is not sufficient if
downstream inference still cannot run until that `Process()` returns.

- [ ] **Step 3: Verify cache behavior.**

Enable cache/stat options and drive two frames with the same scheduled tile list.
Assertions:
- `TileSpecToTilePlanCalculator` hits the tile-plan cache on the second frame.
- `StreamingTilesToTensorBatchCalculator` hits the per-batch matrix/ROI cache for
  the same row groups: `[0,1]`, `[2,3]`, `[4]`.
- Tensor workspace pooling does not mutate or reuse a batch buffer while the
  inference stub still owns it. For CPU tests, use distinct packet ownership or
  a held-packet spy; for the GPU/OpenGL path, require resource-pool/fence
  semantics and no CPU readback.
- Pixel tensors or detector outputs are not cached by default. Any future
  content cache must key on source image identity/content version, model
  metadata, preprocessing options, tile geometry, and batch row.

- [ ] **Step 4: Verify per-tile candidate reduction order.**

Add decoder/merge tests with two overlapping tiles:
- In tile 0, include one candidate below `conf_threshold`, two duplicate
  candidates above threshold, and one separate object.
- In tile 1, include a duplicate of the same object seen in tile 0.

Assertions:
- the below-threshold candidate is dropped before local NMS
- optional tile-local NMS removes only the duplicate inside tile 0
- tile-local NMS does not compare tile 0 and tile 1 candidates
- both overlapping-tile candidates survive merge before global NMS
- the final global NMS runs once and removes the cross-tile duplicate
- disabling tile-local NMS still produces a correct final result, only with more
  candidates entering global NMS
- a separate single-tile/no-tracker case with local NMS enabled bypasses final
  NMS and emits the same candidate vector
- the same single-tile case with local NMS disabled does not bypass final NMS
- adding tracker-updated detections disables the bypass even when there is one
  tile, because fresh-vs-tracker dedup is global

Configuration rules:
- local NMS/top-K thresholds are configurable per decoder/task
- if weighted global NMS is enabled, local NMS/top-K must be accuracy-tested
  because removing candidates early can alter weighted boxes
- if weighted global NMS is required for the frame, the bypass is disabled
  because pass-through would skip box fusion
- if the model already emits post-NMS detections, local NMS should be disabled or
  tested as a no-op

- [ ] **Step 5: Verify.**

```bash
bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/calculators/tensor:video_tile_scheduler_pipeline_test \
  --test_output=all
```

---

## Task 5: Pipeline integration without copied-prior propagation

**Files:** create `video_tile_scheduler_pipeline_test.cc`; modify BUILD.

This proves the graph shape: scheduler controls the detect branch, tracker path
supplies SKIP-frame candidates, concat/NMS runs once per timestamp.

- [ ] **Step 1: Write the integration graph test.**

The test may stub fresh detections and tracker-updated detections, but the graph
must include the real wiring points:

- `PreviousLoopbackCalculator`
  - `MAIN:tick`
  - `LOOP:final_dets`
  - `input_stream_info { tag_index: "LOOP" back_edge: true }`
  - `PREV_LOOP:priors`
- `VideoTileSchedulerCalculator`
  - inputs `TILES:base_tiles`, `PRIOR_DETECTIONS:priors`,
    `TRACKING:tracking_data`
  - outputs `TILES:scheduled_tiles`, `REFRESH:refresh`
- detector stub or tiled detector branch
  - emits fresh detections on DETECT frames
  - supports the frame-local multi-batch contract from Task 4 when the scheduler
    emits more tiles than model batch capacity
  - emits no packet or empty packet on SKIP frames
- `PassThroughOrEmptyDetectionVectorCalculator`
  - turns skipped detector output into an empty vector at the frame timestamp
- tracker stub or real tracker branch
  - emits tracker-updated relative detections on SKIP frames
- `ConcatenateDetectionVectorCalculator`
- `NonMaxSuppressionCalculator`
  - `return_empty_detections: true`
  - no `IMAGE` input in this relative-coordinate segment

Drive at least three timestamps:
- Frame A: no priors -> DETECT; fresh detections reach NMS.
- Frame B: duplicated tracking with priors -> SKIP; fresh vector is empty;
  tracker-updated detections reach NMS.
- Frame C: chunk boundary -> DETECT even if duplicated.
- Frame D: DETECT with 5 scheduled tiles and batch capacity 2 -> 3 tiled
  inference batches merge into one fresh detection vector before NMS.
- Frame E: two overlapping tiles each keep a high-score duplicate after
  tile-local filtering -> global NMS deduplicates them once at frame level.

Assertions:
- one final detection vector packet per timestamp
- exactly one frame-level suppression stage in the test graph
- scheduler SKIP output has empty `scheduled_tiles`
- no scheduler `PROPAGATED_DETECTIONS` stream exists
- final detections use `relative_bounding_box`
- global NMS runs at most once per source frame, not once per tile batch
- tile-local NMS/top-K, when enabled, reduces only per-tile candidates and does
  not suppress cross-tile duplicates before merge
- single-tile/no-tracker/already-local-NMSed frames bypass the final NMS call;
  multi-tile or tracker-concat frames do not bypass it

- [ ] **Step 2: Add BUILD deps.**

Expected deps to confirm against actual target names:

```python
":video_tile_scheduler_calculator",
"//mediapipe/calculators/core:previous_loopback_calculator",
"//mediapipe/calculators/core:concatenate_detection_vector_calculator",
"//mediapipe/calculators/util:non_max_suppression_calculator",
"//mediapipe/calculators/util:pass_through_or_empty_detection_vector_calculator",
"//mediapipe/framework:calculator_framework",
"//mediapipe/framework/formats:detection_cc_proto",
"//mediapipe/framework/formats:rect_cc_proto",
"//mediapipe/framework/port:gtest_main",
"//mediapipe/framework/port:parse_text_proto",
"//mediapipe/framework/port:status_matchers",
```

- [ ] **Step 3: Verify.**

```bash
bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/calculators/tensor:video_tile_scheduler_pipeline_test \
  --test_output=all
```

---

## Task 6: Real video graph integration notes

**Files:** update the owning graph/task integration once this calculator is
ready. Do not hide these contracts inside the scheduler.

- [ ] Keep `MotionAnalysisCalculator -> FlowPackagerCalculator` per-frame.
- [ ] Wire `BoxTrackerCalculator` with `TRACKING:tracking_data` and the graph's
  frame time stream, following `mediapipe/calculators/video/testdata/tracker_graph.pbtxt`.
- [ ] Wire `TrackedDetectionManagerCalculator` to consume detector updates and
  `TRACKING_BOXES`, and to output tracker-updated relative detections.
- [ ] Use `FlowLimiterCalculator` only around expensive tiled detection work if
  the video graph needs mobile-style backpressure.
- [ ] Keep the tiled detect branch frame-local and streaming: when scheduled
  tiles exceed batch capacity, send each prepared batch to inference immediately
  and let the merge accumulator reconstruct one source-frame result.
- [ ] Keep per-tile candidate reduction inside the decoder/tiled detector branch:
  `conf_threshold` first, optional tile-local NMS/top-K second, projection/merge
  third, final global NMS after concat with tracker detections.
- [ ] Size CPU/GPU tensor workspace pools for streaming overlap. At minimum,
  never reuse a buffer still held by downstream inference; on OpenGL paths use
  resource ownership/fences rather than CPU readback.
- [ ] Keep OpenGL/GPU zero-copy intact: the scheduler emits only rect metadata
  and must not map image/tensor packets to CPU memory.
- [ ] Keep final public Tasks pixel conversion after global NMS, not before
  scheduler/tracker/NMS loopback.

---

## Self-review checklist

- [ ] `rg -n "detect_every_n_frames|frame_index_|PROPAGATED_DETECTIONS" \
  mediapipe/calculators/tensor/video_tile_scheduler*` returns no
  implementation usage.
- [ ] Defaults plus no `TRACKING` DETECT every frame.
- [ ] With tracking present, refresh is decided by `FlowPackager`/decoded flags.
- [ ] Hard refresh conditions dominate skip conditions.
- [ ] SKIP does not copy prior detections.
- [ ] Tile priority fallback is per tile, not all-or-nothing.
- [ ] `T = 5`, `batch_capacity = 2` produces 3 inference batches with
  `valid_count` 2, 2, 1 and one merged source-frame fresh detection vector.
- [ ] Batch 0 can reach the inference stub before the final underfilled batch is
  prepared; otherwise the tensorizer has been refactored into a true per-batch
  producer.
- [ ] Tile-plan cache, per-batch geometry cache, and tensor workspace pooling are
  tested separately.
- [ ] Per-tile `conf_threshold` runs before tile-local NMS/top-K.
- [ ] Tile-local NMS/top-K never compares candidates across tiles, and global NMS
  still deduplicates overlapping-tile candidates once per source frame.
- [ ] Final suppression is frame-level and relative-box only: global NMS runs for
  multi-source frames, while the safe single-tile/no-tracker/post-local-NMS case
  bypasses the NMS call.
- [ ] `PreviousLoopbackCalculator` `LOOP` input is a back edge.
- [ ] Empty detector output on SKIP becomes an empty detection vector.
- [ ] Scheduler remains metadata-only and does not break GPU zero-copy.

## Final verification

```bash
bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/calculators/tensor:video_tile_scheduler_calculator_test \
  //mediapipe/calculators/tensor:video_tile_scheduler_pipeline_test \
  --test_output=errors

git status --short
```

## Done criteria

- `VideoTileSchedulerCalculator` has no frame-cadence refresh logic.
- Refresh decisions are directly driven by `FlowPackager` tracking signals.
- Tracker path, not scheduler copied priors, supplies SKIP-frame detections.
- DETECT-frame tile capping is deterministic and cache-friendly.
- DETECT frames with more scheduled tiles than input batch capacity stream
  multiple single-frame inference calls without waiting for all tile
  preprocessing to finish before batch 0 enters inference.
- Per-tile score filtering and optional tile-local NMS/top-K reduce candidate
  volume before merge without replacing frame-level suppression. Final NMS is
  bypassed only for the safe single-tile/no-tracker/post-local-NMS case.
- Integration preserves one frame-level suppression stage, relative coordinates,
  and GPU zero-copy boundaries.
