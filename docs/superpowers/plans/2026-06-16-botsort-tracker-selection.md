# BoTSORT Scheduler-Aware Tracker Selection Implementation Plan

**Status:** Rebased on official MediaPipe `master` at `93954ac`. The mandatory
LiteRT namespace and Abseil status-macro migrations in Task 0 have been applied,
and the CPU baseline passes. Platform GPU baselines remain required before
tracker implementation continues.

**Reference design:**
`docs/superpowers/specs/2026-06-16-botsort-tracker-selection-design.md`

## Goal

Make the tiled YOLO `VIDEO` / `LIVE_STREAM` graph support either the existing
BoxTracker or motion-only BoTSORT while preserving dense per-source-frame
output. BoTSORT must distinguish full DETECT, partial-tile DETECT, and scheduler
SKIP frames, expose stable track IDs, and remain configurable through the axis
and OBB C++, C, and Python APIs.

The tasks below are implementation checkpoints, not independently releasable
features. The change is complete only when every task and the final integration
gate pass together.

## Fixed Decisions

- `REFRESH=false` calls `BoTSORT::predict_only(frame)` and never treats an empty
  detection vector as a negative observation.
- `REFRESH=true` calls `BoTSORT::track_observed(detections, frame,
  observed_rois)`. Association uses the complete active/unconfirmed/lost pool;
  observation ROIs only gate unmatched-track state transitions.
- Observation coverage is the exact source-pixel
  `TensorBatchInfo.geometry.effective_pixel_rois`, accumulated across every
  inference batch for the source frame.
- Track-center containment uses half-open pixel bounds `[x, x + width)` and
  `[y, y + height)`.
- `track()` remains a backward-compatible full-frame wrapper around
  `track_observed()`.
- `nominal_frame_rate` defaults to 30. Every source packet advances one tracker
  step; MediaPipe timestamps are not converted into a second tracker clock.
- BoxTracker remains the axis-aligned default. OBB does not gain BoxTracker or
  motion scheduling in this work.
- BoTSORT remains motion-only. ReID/TensorRT/ONNX are not introduced.
- Do not mix `@litert//tflite` and
  `@org_tensorflow//tensorflow/lite` targets in one binary.

## Task 0: Restore the Official-Master Build Baseline

**Purpose:** Separate upstream integration failures from BoTSORT behavior before
changing the tracker state machine.

**Files:**

- `mediapipe/tasks/cc/vision/{yolo_object_detector,oriented_object_detector}`
- `mediapipe/tasks/cc/vision/utils/{detection_label_resolution,tiled_detection_utils}`
- `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_*`
- `mediapipe/calculators/tensor/BUILD`

- [x] Replace custom `tensorflow/lite/...` includes with `tflite/...`.
- [x] Replace custom `@org_tensorflow//tensorflow/lite/...` dependencies with
  matching `@litert//tflite/...` targets.
- [x] Replace removed `MP_ASSIGN_OR_RETURN` / `MP_RETURN_IF_ERROR` usages in the
  custom detector, tiling, GL, and Metal sources with the official
  `ABSL_ASSIGN_OR_RETURN` / `ABSL_RETURN_IF_ERROR` macros and direct status-macro
  dependencies.
- [x] Verify no old TFLite references remain in those custom packages:

  ```bash
  rg '@org_tensorflow//tensorflow/lite|#include "tensorflow/lite' \
    mediapipe/tasks/cc/vision/yolo_object_detector \
    mediapipe/tasks/cc/vision/oriented_object_detector \
    mediapipe/tasks/cc/vision/utils \
    mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch* \
    mediapipe/calculators/tensor/BUILD
  ```

- [x] Let Bazel 7.7.0 finish Bzlmod/LiteRT dependency initialization, then run
  the unchanged baseline targets:

  ```bash
  HERMETIC_PYTHON_VERSION=3.12 bazel test \
    --define MEDIAPIPE_DISABLE_GPU=1 --cache_test_results=no \
    //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test \
    //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test \
    //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test
  ```

- [ ] Build one GPU-enabled target per supported platform after the CPU baseline
  passes, proving the migrated GL/Metal headers and labels resolve. Do not switch
  the detector to `delegate.litert`; this task only follows the official target
  namespace.

**Acceptance:** Relevant binaries contain only the official `@litert` TFLite
implementation, and existing tiled CPU behavior passes before BoTSORT changes.

## Task 1: Add Observation-Aware BoTSORT State Transitions

**Files:** `third_party/botsort/include/{BoTSORT.h,track.h}`,
`third_party/botsort/src/{BoTSORT.cpp,track.cpp}`,
`third_party/botsort/botsort_smoke_test.cc`.

- [ ] Add public APIs:

  ```cpp
  std::vector<std::shared_ptr<Track>> predict_only(const cv::Mat& frame);
  std::vector<std::shared_ptr<Track>> track_observed(
      const std::vector<Detection>& detections, const cv::Mat& frame,
      const std::vector<cv::Rect>& observed_rois);
  ```

- [ ] Keep `track(detections, frame)` and delegate to `track_observed()` with
  `{cv::Rect(0, 0, frame.cols, frame.rows)}`.
- [ ] Extract one shared prediction/GMC pass so `predict_only()` and
  `track_observed()` never predict the same track twice in one source frame.
- [ ] `predict_only()` increments `_frame_id`, predicts tracked and lost tracks,
  applies GMC, refreshes `frame_id` only for tracks that remain active, expires
  already-lost tracks using the existing retention rule, and returns activated
  tracked outputs. It must not associate, activate, mark lost, or remove
  unconfirmed tracks.
- [ ] `track_observed()` retains upstream two-stage association and new-track
  activation. An unmatched active track becomes lost only when its predicted
  center is observed; an unmatched unconfirmed track is removed only when its
  center is observed. Unmatched tracks outside coverage retain their state and
  predicted geometry.
- [ ] Lost-track aging remains global and independent of observed coverage.

**Tests:**

- DETECT establishes a confirmed track; multiple SKIPs emit predicted boxes
  with the same ID; re-detection keeps the ID.
- An already-lost track advances for later association but is not emitted.
- Long active SKIP runs do not consume the future lost-track retention window.
- Empty partial DETECT over the left ROI loses only the left track; the right
  track remains active.
- A detection can associate with a track whose prediction crossed an ROI
  boundary, proving association is not partitioned by tile.
- Empty full-frame DETECT applies normal loss/removal semantics to every
  unmatched track.

## Task 2: Make GMC and Track IDs Safe for Retained Tracks

**Files:** `third_party/botsort/include/GlobalMotionCompensation.h`,
`third_party/botsort/src/{BoTSORT.cpp,GlobalMotionCompensation.cpp,track.cpp}`,
and the vendored smoke test.

- [ ] Replace `Track::next_id()`'s function-local `static int` with
  `std::atomic<int>` and relaxed `fetch_add`; IDs must remain process-unique
  across concurrent detector instances.
- [ ] Build the SparseOptFlow foreground exclusion set from fresh detections
  plus predicted active, unconfirmed, and retained-lost boxes.
- [ ] Add a testable `botsort_internal::BuildForegroundMask` helper. Start with
  an all-255 downscaled mask, scale top/left with `floor`, bottom/right with
  `ceil`, clamp to the downscaled mask bounds, skip empty rectangles, and paint
  valid foreground regions zero.
- [ ] Pass the mask to `cv::goodFeaturesToTrack`. If no background features are
  available, return the existing identity/fallback transform without throwing.
- [ ] Use the same mask construction on DETECT and SKIP paths.

**Tests:** exact mask coordinates at `_downscale=2`, partially out-of-frame
boxes, overlapping fresh/predicted boxes, fully masked frames, and concurrent
ID allocation from at least two trackers.

## Task 3: Extend the BoTSORT Calculator Contract

**Files:**
`mediapipe/calculators/tensor/{botsort_tracking_calculator.cc,botsort_tracking_calculator.proto,botsort_tracking_calculator_test.cc,oriented_botsort_tracking_calculator.cc,BUILD}`.

- [ ] Add `nominal_frame_rate = 7 [default = 30]` to
  `BotsortTrackingCalculatorOptions` and pass it into `TrackerParams.frame_rate`.
- [ ] Axis calculator inputs become `IMAGE`, `DETECTIONS`, `REFRESH`, and
  `OBSERVED_ROIS`; output remains one `DETECTIONS` packet per source frame.
- [ ] `REFRESH=false` requires empty fresh detections and empty ROIs, then calls
  `predict_only()`.
- [ ] `REFRESH=true` requires non-empty valid coverage, converts
  `TilePixelRoi` to `cv::Rect`, and calls `track_observed()`; an empty detection
  vector is valid negative evidence inside those ROIs.
- [ ] Reject missing packets, non-positive image dimensions, non-positive or
  out-of-bounds ROIs, and inconsistent REFRESH/data combinations.
- [ ] Validate every input detection: `RELATIVE_BOUNDING_BOX`, exactly one
  `label_id` in `[0,255]`, exactly one finite score in `[0,1]`, finite box
  coordinates, and non-negative width/height. Return `InvalidArgument` instead
  of silently dropping malformed detections.
- [ ] Write `std::to_string(track_id)` to `Detection.track_id`; do not populate
  `detection_id`.
- [ ] OBB calculator receives the same nominal frame rate but keeps its existing
  ID-association-only contract and oriented output geometry.
- [ ] Calculator `Open()` repeats BoTSORT option validation for direct graph
  users: finite thresholds in `[0,1]`, low <= high, buffer in `[0,255]`, FPS in
  `[1,255]`, and
  `floor(nominal_frame_rate / 30.0 * track_buffer) <= 255`.

**Tests:** REFRESH truth table, partial observation, malformed detections/ROIs,
stable ID, no `detection_id`, GMC enabled, and 24/30/60 FPS propagation.

## Task 4: Accumulate Exact Observed ROIs Across Tile Batches

**Files:**
`mediapipe/calculators/tensor/merge_tile_box_detections_accumulator_calculator.cc`
and its test/BUILD target.

- [ ] Add source-timestamped output
  `OBSERVED_ROIS:std::vector<TilePixelRoi>`.
- [ ] For each `BATCH_INFO`, copy
  `geometry.effective_pixel_rois[0:valid_count]`; never reconstruct ROIs from
  normalized tile configuration.
- [ ] Use a second `TileFrameAccumulator<TilePixelRoi>` in lockstep with the
  detection accumulator. Both completion flags must match for every batch;
  mismatch is an internal error.
- [ ] On completion, deduplicate exact `(x,y,width,height)` tuples while
  preserving deterministic first-seen order, then emit detections and ROIs at
  `source_frame_timestamp`.
- [ ] `total_batches==0` emits empty detections and empty ROIs immediately.

**Tests:** `T=5,B=2` produces no output after batches 0 and 1, then one complete
output after batch 2; partial `max_scheduled_tiles=2`; padding; duplicate ROI
deduplication; missing detection packet; and scheduler SKIP.

## Task 5: Wire Scheduler Decisions Through the Graph

**Files:** `mediapipe/graphs/tiled_detection/{tiled_detection_graphs.proto,tiled_box_track_merge_graph.cc,tiled_tracking_graph.cc,tiled_tracking_graphs_test.cc,BUILD}`
and the YOLO graph builder.

- [ ] Add `nominal_frame_rate = 8 [default = 30]` to
  `TiledTrackingGraphOptions` and forward all BoTSORT options.
- [ ] For BoTSORT, `TiledBoxTrackMergeGraph` forwards `REFRESH` and the merge
  accumulator's `OBSERVED_ROIS` into `TiledTrackingGraph`.
- [ ] With scheduling enabled, connect the existing dense scheduler `REFRESH`
  output directly. Do not add another cadence or infer refresh from detection
  emptiness.
- [ ] With scheduling disabled and BoTSORT selected, use
  `PacketPresenceCalculator` on the source `ImageFrame` to produce dense true
  `REFRESH` packets.
- [ ] The BoxTracker branch must not declare or consume the new inputs and must
  retain its current encode/track/decode/gate path.
- [ ] Keep fresh-wins geometry in `TiledFrameSuppressionCalculator`; transfer
  the best-IoU tracker `track_id` to the fresh winner before global NMS.
- [ ] All tracker-facing streams use the original source-frame timestamp;
  per-batch synthetic inference timestamps terminate at the accumulator.

**Graph tests:** DETECT-SKIP-SKIP-DETECT packet density; scheduling-off dense
true refresh; partial observed coverage; complete empty DETECT; `T=5,B=2`
three-batch accumulation; ID propagation; and byte-equivalent BoxTracker output.

## Task 6: Update C++ Task Options and Validation

**Files:** axis and OBB detector option protos, public headers, converters,
Create methods, graph builders, and C++ tests.

- [ ] Add `nominal_frame_rate = 8 [default = 30]` to both detector tracking
  protos and `int nominal_frame_rate = 30` to both public C++ structs.
- [ ] Map the field through detector proto -> tiled graph options -> calculator.
- [ ] Remove the axis Create-time rejection for BoTSORT plus motion scheduling.
- [ ] Preserve existing requirements: stream mode, tiling enabled, and class
  count in `[1,256]` when BoTSORT is selected.
- [ ] Apply the same threshold/FPS/buffer validation as calculator `Open()`,
  before graph construction, with field-specific `InvalidArgument` messages.
- [ ] OBB keeps `TRACKER_UNSPECIFIED` as default, rejects BoxTracker, and applies
  BoTSORT validation only when BoTSORT is selected.
- [ ] Do not force the new official `delegate.litert`; tracker behavior must be
  inference-backend neutral.

**Tests:** converter round trips, each validation failure, BoTSORT plus scheduler
accepted, BoxTracker defaults unchanged, and OBB nominal-FPS parity.

## Task 7: Update C and Python Bindings Atomically

**Files:** axis/OBB C tracking structs and converters, ABI tests, Python ctypes
structures/dataclasses/builders, and Python tests.

- [ ] Append `int nominal_frame_rate` to `MpTrackingOptions` and
  `MpOrientedTrackingOptions`; do not insert it before existing fields.
- [ ] C converters interpret zero as the compatibility default 30. Positive
  values are copied verbatim; negative values reach Create validation and fail.
- [ ] Append `ctypes.c_int` in the matching Python structures and add dataclass
  default `nominal_frame_rate=30`.
- [ ] Update both Python builders and explicit-value/default tests.
- [ ] Pin the 64-bit ABI after the append:
  - tracking struct size 32, `nominal_frame_rate` offset 28;
  - parent tracking offset 184;
  - callback offset 216;
  - parent size 224.
- [ ] Retain the official `MpBaseOptions.file_descriptor` field in the Python
  mirror. It occupies previous alignment space, so the parent anchors above do
  not change, but the full-prefix ABI tests must prove this on the merged tree.

## Task 8: End-to-End Verification

- [ ] Run focused vendored/calculator tests:

  ```bash
  HERMETIC_PYTHON_VERSION=3.12 bazel test --cache_test_results=no \
    //third_party/botsort:botsort_smoke_test \
    //mediapipe/calculators/tensor:botsort_tracking_calculator_test \
    //mediapipe/calculators/tensor:oriented_botsort_tracking_calculator_test \
    //mediapipe/calculators/tensor:merge_tile_box_detections_accumulator_calculator_test
  ```

- [ ] Run graph and C++ task tests with GPU disabled.
- [ ] Run axis and OBB C converter plus ABI tests.
- [ ] Run axis and OBB Python tests under hermetic Python 3.12.
- [ ] Run fixture-backed tiled video e2e cases:
  - scheduling off with stable IDs;
  - scheduling on with non-empty predicted output on SKIP timestamps;
  - proof that SKIP has `REFRESH=false` and `total_batches==0`;
  - partial tile selection loses only observed-region misses;
  - full observed empty DETECT applies normal loss;
  - `T=5,B=2` invokes inference three times but updates the tracker once.
- [ ] Run at least one real GMC moving-camera sequence and one concurrent
  multi-detector ID test.
- [ ] Confirm the default BoxTracker result remains unchanged.
- [ ] Confirm `rg` finds no stale `@org_tensorflow//tensorflow/lite` or
  `tensorflow/lite` includes in the custom tiled detector packages.
- [ ] Record skipped fixture tests and platform-specific GPU gaps explicitly;
  dependency initialization is not test success.

## Completion Criteria

- Every source frame emits exactly one tracked output packet.
- SKIP frames perform no inference or association and retain active IDs.
- Partial DETECT frames apply negative evidence only inside exact observed ROIs.
- Multi-batch frames update the tracker only after all batches complete.
- Track IDs survive fresh-wins suppression and all public result layers.
- Axis/OBB C++, C, and Python options agree on nominal FPS and validation.
- No binary mixes official LiteRT TFLite targets with org_tensorflow TFLite.
- BoxTracker remains the default and passes regression coverage.
- All required Bazel targets complete under the merged official baseline.
