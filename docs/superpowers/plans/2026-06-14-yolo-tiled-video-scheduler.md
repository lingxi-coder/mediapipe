# YOLO Tiled Detection × Motion Scheduling (Video) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In VIDEO/LIVE_STREAM mode, gate the tiled TFLite inference with `VideoTileSchedulerCalculator` so low-motion frames SKIP detection and the BoxTracker (sub-project C) fills them; sub-project C is kept frozen and the scheduler runs its own optical-flow pass.

**Architecture:** A new pbtxt `OpticalFlowTrackingGraph` (IMAGE→TRACKING) and a new api2 `TiledDetectionStreamFrontGraph` (IMAGE + PRIOR_DETECTIONS → TENSORS + BATCH_INFO; internally flow + TileGrid + scheduler + TileSpecToTilePlan + batcher). The YOLO graph's stream branch swaps to the stream front and adds a `PreviousLoopbackCalculator` back-edge feeding the previous frame's merged detections as `PRIOR_DETECTIONS`. SKIP rides the already-wired empty-TILES path; `REFRESH` is unconnected. Public config: `enable_motion_scheduling` + `max_scheduled_tiles` on `TilingOptions`; rejected in IMAGE mode.

**Tech Stack:** MediaPipe Bazel (C++20, `--define MEDIAPIPE_DISABLE_GPU=1`), api2 Node/Subgraph + pbtxt `mediapipe_simple_subgraph`, proto2, the optical-flow stack (`MotionAnalysis`/`FlowPackager`) and `PreviousLoopbackCalculator`, GoogleTest.

**Spec:** `docs/superpowers/specs/2026-06-14-yolo-tiled-video-scheduler-design.md`

**Conventions every task:** English comments; build/test with `--define MEDIAPIPE_DISABLE_GPU=1`; commit messages end with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer; work on `dev`; **no edits to sub-project C's files or tests** (they must keep passing untouched); clangd "file not found" diagnostics are Bazel-include false-positives — the `bazel test` result is authoritative; TDD red→green.

---

## File Structure

**New files**
- `mediapipe/graphs/tiled_detection/optical_flow_tracking_graph.pbtxt` — `OpticalFlowTrackingGraph` (IMAGE→TRACKING), a `mediapipe_simple_subgraph`.
- `mediapipe/graphs/tiled_detection/tiled_detection_stream_front_graph.cc` — `TiledDetectionStreamFrontGraph` api2 subgraph.
- `mediapipe/graphs/tiled_detection/tiled_video_scheduler_graphs_test.cc` — graph tests for both new subgraphs.

**Modified files**
- `mediapipe/graphs/tiled_detection/tiled_detection_graphs.proto` — add `max_scheduled_tiles = 7` to `TiledDetectionFrontGraphOptions`.
- `mediapipe/graphs/tiled_detection/BUILD` — targets for the two new subgraphs + the new test.
- `mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.proto` — `TilingOptions` += `enable_motion_scheduling = 7`, `max_scheduled_tiles = 8`.
- `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h` — mirror the two fields in the C++ struct.
- `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.cc` — converter copies + Create()-time IMAGE-mode reject.
- `mediapipe/tasks/cc/vision/utils/tiled_detection_utils.h` — `SchedulingEnabled` predicate.
- `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc` — stream branch front swap + loopback.
- `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD` — deps on the stream front + `previous_loopback_calculator`.
- `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc` — converter unit test additions + reject e2e + scheduling e2e.

---

## Task 1: `OpticalFlowTrackingGraph` (pbtxt subgraph) + graph test

**Files:**
- Create: `mediapipe/graphs/tiled_detection/optical_flow_tracking_graph.pbtxt`
- Create: `mediapipe/graphs/tiled_detection/tiled_video_scheduler_graphs_test.cc`
- Modify: `mediapipe/graphs/tiled_detection/BUILD`

- [ ] **Step 1: Write the subgraph pbtxt**

Create `mediapipe/graphs/tiled_detection/optical_flow_tracking_graph.pbtxt` (the node blocks are copied verbatim from `mediapipe/graphs/tracking/subgraphs/box_tracking_cpu.pbtxt`, dropping the BoxTracker tail — this is the pure flow producer):

```
# Copyright 2026 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pure optical-flow producer: a single ImageFrame stream -> per-frame
# FlowPackager TrackingData. Mirrors the flow prefix of box_tracking_cpu.pbtxt
# (320x240 downscale -> MotionAnalysis -> FlowPackager) without the BoxTracker
# tail, so the TrackingData can drive the video tile scheduler.

type: "OpticalFlowTrackingGraph"

input_stream: "IMAGE:input_video"
output_stream: "TRACKING:tracking_data"

# Downscale to 320x240 for fast, robust motion analysis.
node: {
  calculator: "ImageTransformationCalculator"
  input_stream: "IMAGE:input_video"
  output_stream: "IMAGE:downscaled_input_video"
  node_options: {
    [type.googleapis.com/mediapipe.ImageTransformationCalculatorOptions] {
      output_width: 320
      output_height: 240
    }
  }
}

# Sparse optical flow + camera motion.
node: {
  calculator: "MotionAnalysisCalculator"
  input_stream: "VIDEO:downscaled_input_video"
  output_stream: "CAMERA:camera_motion"
  output_stream: "FLOW:region_flow"
  node_options: {
    [type.googleapis.com/mediapipe.MotionAnalysisCalculatorOptions]: {
      analysis_options {
        analysis_policy: ANALYSIS_POLICY_CAMERA_MOBILE
        flow_options {
          fast_estimation_min_block_size: 100
          top_inlier_sets: 1
          frac_inlier_error_threshold: 3e-3
          downsample_mode: DOWNSAMPLE_TO_INPUT_SIZE
          verification_distance: 5.0
          verify_long_feature_acceleration: true
          verify_long_feature_trigger_ratio: 0.1
          tracking_options {
            max_features: 500
            adaptive_extraction_levels: 2
            min_eig_val_settings {
              adaptive_lowest_quality_level: 2e-4
            }
            klt_tracker_implementation: KLT_OPENCV
          }
        }
      }
    }
  }
}

# Pack flow + camera motion into TrackingData (one per frame).
node: {
  calculator: "FlowPackagerCalculator"
  input_stream: "FLOW:region_flow"
  input_stream: "CAMERA:camera_motion"
  output_stream: "TRACKING:tracking_data"
  node_options: {
    [type.googleapis.com/mediapipe.FlowPackagerCalculatorOptions]: {
      flow_packager_options: {
        binary_tracking_data_support: false
      }
    }
  }
}
```

- [ ] **Step 2: Write the failing graph test**

Create `mediapipe/graphs/tiled_detection/tiled_video_scheduler_graphs_test.cc` (this file holds Task 1 + Task 4 tests; Task 1 adds the helpers + the flow test):

```cpp
// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/tracking/tracking.h"

namespace mediapipe {
namespace {

// A frame with a high-contrast moving square on a textured background, so the
// optical-flow tracker finds features to track (a flat/white frame yields no
// flow). `shift` translates the square to create real motion between frames.
std::unique_ptr<ImageFrame> TexturedFrame(int w, int h, int shift) {
  auto f = std::make_unique<ImageFrame>(ImageFormat::SRGB, w, h);
  uint8_t* p = f->MutablePixelData();
  const int stride = f->WidthStep();
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      // Checkerboard background (texture) + a dark square that moves.
      const bool square = (x > 10 + shift && x < 40 + shift && y > 10 && y < 40);
      const uint8_t v = square ? 0 : (((x / 4 + y / 4) % 2) ? 220 : 60);
      uint8_t* px = p + y * stride + x * 3;
      px[0] = px[1] = px[2] = v;
    }
  }
  return f;
}

TEST(OpticalFlowTrackingGraphTest, ProducesTrackingDataAndTerminates) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    output_stream: "tracking"
    node {
      calculator: "OpticalFlowTrackingGraph"
      input_stream: "IMAGE:image"
      output_stream: "TRACKING:tracking"
    }
  )pb");

  std::vector<Packet> tracking_packets;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("tracking", [&](const Packet& p) {
    tracking_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < 4; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image", Adopt(TexturedFrame(64, 64, /*shift=*/2 * i).release())
                     .At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  // The producer must terminate (no hang) and emit at least one TrackingData
  // after the first-frame warm-up (MotionAnalysis needs a previous frame).
  ASSERT_GE(tracking_packets.size(), 1u);
  for (const Packet& p : tracking_packets) {
    EXPECT_NO_THROW((void)p.Get<TrackingData>());
  }
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 3: Add BUILD targets**

In `mediapipe/graphs/tiled_detection/BUILD`, add the `mediapipe_simple_subgraph` load (alongside the existing `mediapipe_proto_library` load) and the targets:

```python
load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

mediapipe_simple_subgraph(
    name = "optical_flow_tracking_graph",
    graph = "optical_flow_tracking_graph.pbtxt",
    register_as = "OpticalFlowTrackingGraph",
    deps = [
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/video:flow_packager_calculator",
        "//mediapipe/calculators/video:motion_analysis_calculator",
    ],
)

cc_test(
    name = "tiled_video_scheduler_graphs_test",
    size = "medium",
    srcs = ["tiled_video_scheduler_graphs_test.cc"],
    deps = [
        ":optical_flow_tracking_graph",
        ":tiled_detection_stream_front_graph",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
        "//mediapipe/util/tracking:tracking",
        "@com_google_absl//absl/status",
    ],
)
```

(`:tiled_detection_stream_front_graph` is created in Task 4; this test compiles after Task 4. Run the OpticalFlow test alone in Step 4 via `--test_filter`.)

- [ ] **Step 4: Build the subgraph + run the flow test**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/graphs/tiled_detection:tiled_video_scheduler_graphs_test --test_filter=OpticalFlowTrackingGraphTest.*`
Expected: PASS. If it fails to LINK with undefined OpenCV symbols, add the missing `libopencv_*.a`/dylib to `third_party/opencv_macos.BUILD` (the optical-flow stack already links there from sub-project C; report any addition). If it hangs, MotionAnalysis is not terminating — verify `CloseAllPacketSources` is reached.

(Note: `:tiled_detection_stream_front_graph` does not exist yet, so the full test target won't build until Task 4. Build just the subgraph here first: `bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/graphs/tiled_detection:optical_flow_tracking_graph`, then run the filtered test after Task 4. If you want a green checkpoint now, temporarily drop `:tiled_detection_stream_front_graph` from the test deps and the `#include`-free Task-4 test body is not yet present, so the file compiles; restore the dep in Task 4.)

- [ ] **Step 5: Commit**

```bash
git add mediapipe/graphs/tiled_detection/optical_flow_tracking_graph.pbtxt \
        mediapipe/graphs/tiled_detection/tiled_video_scheduler_graphs_test.cc \
        mediapipe/graphs/tiled_detection/BUILD
git commit -m "feat(tiling): OpticalFlowTrackingGraph — IMAGE->TRACKING flow producer for the scheduler

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Public `TilingOptions` config — `enable_motion_scheduling` + `max_scheduled_tiles`

**Files:**
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.proto`
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h`
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.cc`
- Test: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc`

- [ ] **Step 1: Extend the converter unit test (failing)**

In `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc`, in the existing `TEST(YoloObjectDetectorOptionsTest, TilingOptionsConvertToProto)`, add before the closing brace (after the `max_detections_after_tile_nms` assertions):

```cpp
  // Sub-project B: motion-scheduling fields round-trip through the converter.
  options->tiling.enable_motion_scheduling = true;
  options->tiling.max_scheduled_tiles = 4;
  auto proto2 = ConvertYoloObjectDetectorOptionsToProto(options.get());
  EXPECT_TRUE(proto2->tiling().enable_motion_scheduling());
  EXPECT_EQ(proto2->tiling().max_scheduled_tiles(), 4);
```

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_filter=YoloObjectDetectorOptionsTest.TilingOptionsConvertToProto`
Expected: FAIL to compile (`enable_motion_scheduling`/`max_scheduled_tiles` don't exist yet).

- [ ] **Step 2: Add the proto fields**

In `mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.proto`, inside `message TilingOptions`, after `optional int32 max_detections_after_tile_nms = 6 [default = 0];`, add:

```proto
    // VIDEO/LIVE_STREAM only: gate per-frame tiled inference with a motion
    // scheduler; low-motion (near-duplicate) frames SKIP inference and the
    // BoxTracker fills them. Rejected in IMAGE mode.
    optional bool enable_motion_scheduling = 7 [default = false];
    // When scheduling, cap how many tiles are inferred per DETECT frame
    // (motion-prioritized). 0 = uncapped (all tiles).
    optional int32 max_scheduled_tiles = 8 [default = 0];
```

- [ ] **Step 3: Add the C++ struct fields**

In `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h`, inside `struct TilingOptions`, after `int max_detections_after_tile_nms = 0;`, add:

```cpp
    // VIDEO/LIVE_STREAM only: gate per-frame tiled inference with a motion
    // scheduler (near-duplicate frames SKIP; the tracker fills them).
    bool enable_motion_scheduling = false;
    // Per DETECT-frame cap on inferred tiles (motion-prioritized). 0 = all.
    int max_scheduled_tiles = 0;
```

- [ ] **Step 4: Add the converter copies**

In `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.cc`, in `ConvertYoloObjectDetectorOptionsToProto`, after `tiling->set_max_detections_after_tile_nms(...)` and before `return options_proto;`, add:

```cpp
  tiling->set_enable_motion_scheduling(options->tiling.enable_motion_scheduling);
  tiling->set_max_scheduled_tiles(options->tiling.max_scheduled_tiles);
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_filter=YoloObjectDetectorOptionsTest.TilingOptionsConvertToProto`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.proto \
        mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h \
        mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.cc \
        mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc
git commit -m "feat(tiling): TilingOptions gains enable_motion_scheduling + max_scheduled_tiles

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `SchedulingEnabled` predicate + IMAGE-mode reject

**Files:**
- Modify: `mediapipe/tasks/cc/vision/utils/tiled_detection_utils.h`
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.cc`
- Test: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc`

- [ ] **Step 1: Write the failing reject e2e test**

In `yolo_object_detector_test.cc`, add (mirrors `TiledExplicitTilesWithOverlapRejected` — the reject surfaces at `Create()`):

```cpp
// enable_motion_scheduling requires VIDEO/LIVE_STREAM; IMAGE mode is rejected
// at Create() (the scheduler is meaningless without temporal continuity).
TEST(YoloObjectDetectorTest, MotionSchedulingInImageModeRejected) {
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = ModelPath();
  options->running_mode = core::RunningMode::IMAGE;
  options->num_classes = 80;
  options->tiling.tile_cols = 2;
  options->tiling.enable_motion_scheduling = true;
  auto detector = YoloObjectDetector::Create(std::move(options));
  EXPECT_EQ(detector.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(detector.status().message(),
              testing::HasSubstr("motion scheduling"));
}
```

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_filter=*MotionSchedulingInImageModeRejected*`
Expected: FAIL (Create currently succeeds — no reject yet). Note: this test does not require a model fixture (Create validates options before loading the model is needed for this path); if Create needs the model and it's absent it may surface a different error — if so, gate with the standard `GTEST_SKIP` fixture check used by sibling tests.

- [ ] **Step 2: Add the `SchedulingEnabled` predicate**

In `mediapipe/tasks/cc/vision/utils/tiled_detection_utils.h`, immediately after the `TilingEnabled` template, add:

```cpp
// Returns true when motion scheduling should be active: scheduling is opted in
// AND tiling is enabled. (Running-mode gating — VIDEO/LIVE_STREAM only — is
// enforced separately at the wrapper's Create(), where running_mode lives.)
template <typename TilingProto>
bool SchedulingEnabled(const TilingProto& t) {
  return t.enable_motion_scheduling() && TilingEnabled(t);
}
```

- [ ] **Step 3: Add the IMAGE-mode reject in `Create()`**

In `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.cc`, in `Create()`, right after `const bool tiling_enabled = TilingEnabled(options_proto->tiling());`, add (uses the same `CreateStatusWithPayload` helper the file already uses for invalid-arg rejections):

```cpp
  if (options_proto->tiling().enable_motion_scheduling() &&
      options->running_mode == core::RunningMode::IMAGE) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiling.enable_motion_scheduling requires VIDEO or LIVE_STREAM running "
        "mode; motion scheduling is meaningless in IMAGE mode.",
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
```

(If `MediaPipeTasksStatus`/`CreateStatusWithPayload` are not already in scope, copy the exact include + payload-enum used by the existing invalid-arg rejection in this file — search for `CreateStatusWithPayload` in `yolo_object_detector.cc`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_filter=*MotionSchedulingInImageModeRejected*`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/tasks/cc/vision/utils/tiled_detection_utils.h \
        mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.cc \
        mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc
git commit -m "feat(tiling): SchedulingEnabled predicate + reject motion scheduling in IMAGE mode

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `TiledDetectionStreamFrontGraph` (api2 subgraph) + graph test

**Files:**
- Modify: `mediapipe/graphs/tiled_detection/tiled_detection_graphs.proto`
- Create: `mediapipe/graphs/tiled_detection/tiled_detection_stream_front_graph.cc`
- Modify: `mediapipe/graphs/tiled_detection/BUILD`
- Test: `mediapipe/graphs/tiled_detection/tiled_video_scheduler_graphs_test.cc`

- [ ] **Step 1: Add `max_scheduled_tiles` to the front options proto**

In `mediapipe/graphs/tiled_detection/tiled_detection_graphs.proto`, in `message TiledDetectionFrontGraphOptions`, after `optional bool is_dynamic_batch = 6 [default = false];`, add:

```proto
  // Stream-front only: forwarded to VideoTileSchedulerCalculator.max_scheduled_tiles.
  optional int32 max_scheduled_tiles = 7 [default = 0];
```

- [ ] **Step 2: Write the failing graph test (DETECT + no-stall)**

Append to `mediapipe/graphs/tiled_detection/tiled_video_scheduler_graphs_test.cc` (before the final `}  // namespace`). It drives the stream front through the REAL internal flow + scheduler, feeding non-empty priors so the scheduler's motion path is exercised, and asserts DETECT (non-empty batches) + termination (the [P1] no-stall arbiter):

```cpp
namespace {
// One full-frame detection (frame-normalized) used as a non-empty prior so the
// scheduler does not auto-DETECT via the priors-empty rule.
std::vector<Detection> OnePrior() {
  Detection d;
  d.add_score(0.9f);
  d.add_label_id(0);
  auto* bb = d.mutable_location_data()->mutable_relative_bounding_box();
  d.mutable_location_data()->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  bb->set_xmin(0.3f);
  bb->set_ymin(0.3f);
  bb->set_width(0.2f);
  bb->set_height(0.2f);
  return {d};
}
}  // namespace

TEST(TiledDetectionStreamFrontGraphTest, DetectsOnMotionAndTerminates) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    input_stream: "priors"
    output_stream: "batch_info"
    node {
      calculator: "mediapipe.tiled_detection.TiledDetectionStreamFrontGraph"
      input_stream: "IMAGE:image"
      input_stream: "PRIOR_DETECTIONS:priors"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:batch_info"
      options {
        [mediapipe.TiledDetectionFrontGraphOptions.ext] {
          tile_grid { cols: 2 }
          batch_capacity: 2
          input_height: 64
          input_width: 64
          input_channels: 3
        }
      }
    }
  )pb");

  std::vector<int> batches_per_frame;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("batch_info", [&](const Packet& p) {
    batches_per_frame.push_back(p.Get<TensorBatchInfo>().total_batches);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < 4; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image", Adopt(TexturedFrame(64, 64, /*shift=*/3 * i).release())
                     .At(Timestamp(i))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "priors",
        Adopt(new std::vector<Detection>(OnePrior())).At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  // Termination (no stall on the real, buffered/latent MotionAnalysis feed) is
  // the [P1] no-stall arbiter. On moving frames the scheduler DETECTs, so at
  // least one frame must carry inference batches (total_batches > 0).
  // (Deterministic SKIP-mechanics — empty TILES -> total_batches==0 -> empty
  // merged -> tracker fill — are already proven by
  // video_tile_scheduler_pipeline_test.SkipFramesEmitEmptyMergedResults.)
  ASSERT_GE(batches_per_frame.size(), 1u);
  int detect_frames = 0;
  for (int b : batches_per_frame) {
    if (b > 0) ++detect_frames;
  }
  EXPECT_GE(detect_frames, 1)
      << "moving frames with non-empty priors should DETECT at least once";
}
```

`#include "mediapipe/calculators/tensor/tiling_types.h"` at the top of the test file for `TensorBatchInfo` (and `LocationData` is in `location_data.pb.h` — add `#include "mediapipe/framework/formats/location_data.pb.h"`).

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/graphs/tiled_detection:tiled_video_scheduler_graphs_test --test_filter=TiledDetectionStreamFrontGraphTest.*`
Expected: FAIL to build (the subgraph / proto field don't exist yet).

- [ ] **Step 3: Write the stream front subgraph**

Create `mediapipe/graphs/tiled_detection/tiled_detection_stream_front_graph.cc` (mirrors `tiled_detection_front_graph.cc` PLUS the internal flow + scheduler; prepend the Apache header):

```cpp
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.pb.h"
#include "mediapipe/calculators/tensor/tile_grid_calculator.pb.h"
#include "mediapipe/calculators/tensor/video_tile_scheduler_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/graphs/tiled_detection/tiled_detection_graphs.pb.h"

namespace mediapipe {
namespace tiled_detection {

// Stream-mode sibling of TiledDetectionFrontGraph: runs its own optical-flow
// pass and a VideoTileSchedulerCalculator between TileGrid and
// TileSpecToTilePlan, so low-motion frames SKIP inference (empty TILES ->
// empty TilePlan -> BATCH_INFO-only frame). The scheduler's REFRESH output is
// intentionally left unconnected (SKIP rides the empty-TILES path).
//
// Inputs:
//   IMAGE - ImageFrame.
//   PRIOR_DETECTIONS - std::vector<Detection> (previous frame's merged
//     detections; the caller closes a PreviousLoopback back-edge).
// Outputs:
//   TENSORS - std::vector<Tensor> (one packet per batch; none on a SKIP frame).
//   BATCH_INFO - TensorBatchInfo (total_batches==0 on a SKIP frame).
class TiledDetectionStreamFrontGraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    const auto& options = sc->Options<TiledDetectionFrontGraphOptions>();
    api2::builder::Graph graph;
    auto image = graph.In("IMAGE").Cast<ImageFrame>();
    auto priors = graph.In("PRIOR_DETECTIONS").Cast<std::vector<Detection>>();

    // Own optical-flow pass (sub-project C's tracker keeps its own flow).
    auto& flow = graph.AddNode("OpticalFlowTrackingGraph");
    image >> flow.In("IMAGE");

    auto& grid = graph.AddNode("TileGridCalculator");
    grid.GetOptions<TileGridCalculatorOptions>() = options.tile_grid();
    image >> grid.In("TICK");

    auto& scheduler = graph.AddNode("VideoTileSchedulerCalculator");
    scheduler.GetOptions<VideoTileSchedulerCalculatorOptions>()
        .set_max_scheduled_tiles(options.max_scheduled_tiles());
    grid.Out("TILES") >> scheduler.In("TILES");
    priors >> scheduler.In("PRIOR_DETECTIONS");
    flow.Out("TRACKING") >> scheduler.In("TRACKING");
    // scheduler.Out("REFRESH") intentionally unconnected.

    auto& plan = graph.AddNode("TileSpecToTilePlanCalculator");
    scheduler.Out("TILES") >> plan.In("TILES");

    auto& batcher = graph.AddNode("StreamingTilesToTensorBatchCalculator");
    auto& bo =
        batcher.GetOptions<StreamingTilesToTensorBatchCalculatorOptions>();
    bo.set_metadata_batch_capacity(options.batch_capacity());
    bo.set_metadata_input_height(options.input_height());
    bo.set_metadata_input_width(options.input_width());
    bo.set_metadata_input_channels(options.input_channels());
    bo.set_metadata_is_dynamic_batch(options.is_dynamic_batch());
    image >> batcher.In("IMAGE");
    plan.Out("TILE_PLAN") >> batcher.In("TILE_PLAN");

    batcher.Out("TENSORS") >> graph.Out("TENSORS");
    batcher.Out("BATCH_INFO") >> graph.Out("BATCH_INFO");
    return graph.GetConfig();
  }
};

// NOTE: keep the fully-qualified type name on a single line. The
// REGISTER_MEDIAPIPE_GRAPH macro stringifies its argument with `#name`, so a
// line break here would inject a stray space into the registered name and the
// graph would never be found by lookup.
// clang-format off
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tiled_detection::TiledDetectionStreamFrontGraph);  // NOLINT(whitespace/line_length)
// clang-format on

}  // namespace tiled_detection
}  // namespace mediapipe
```

- [ ] **Step 4: Add the BUILD library target**

In `mediapipe/graphs/tiled_detection/BUILD`, add:

```python
cc_library(
    name = "tiled_detection_stream_front_graph",
    srcs = ["tiled_detection_stream_front_graph.cc"],
    deps = [
        ":optical_flow_tracking_graph",
        ":tiled_detection_graphs_cc_proto",
        "//mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator",
        "//mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tile_grid_calculator",
        "//mediapipe/calculators/tensor:tile_grid_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tile_spec_to_tile_plan_calculator",
        "//mediapipe/calculators/tensor:video_tile_scheduler_calculator",
        "//mediapipe/calculators/tensor:video_tile_scheduler_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)
```

(Confirm `video_tile_scheduler_calculator` + `..._cc_proto` target names in `//mediapipe/calculators/tensor:BUILD`; they follow the file name by convention.)

- [ ] **Step 5: Run both graph tests**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/graphs/tiled_detection:tiled_video_scheduler_graphs_test`
Expected: PASS (both `OpticalFlowTrackingGraphTest.*` and `TiledDetectionStreamFrontGraphTest.*`). If `DetectsOnMotionAndTerminates` hangs, the scheduler is stalling on the real-flow `TRACKING` bound — this is the [P1] risk; the fallback (per the spec) is to add a per-frame `TrackingData` materializer to `OpticalFlowTrackingGraph` (tick-gate driven by `IMAGE`). Report a hang as BLOCKED with that observation.

- [ ] **Step 6: Commit**

```bash
git add mediapipe/graphs/tiled_detection/tiled_detection_graphs.proto \
        mediapipe/graphs/tiled_detection/tiled_detection_stream_front_graph.cc \
        mediapipe/graphs/tiled_detection/tiled_video_scheduler_graphs_test.cc \
        mediapipe/graphs/tiled_detection/BUILD
git commit -m "feat(tiling): TiledDetectionStreamFrontGraph — flow + scheduler stream front

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `YoloObjectDetectorGraph` stream branch — front swap + loopback

**Files:**
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc`
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`

No new unit test here; correctness is covered by the existing tests (IMAGE/stream paths must still pass) + the Task 6 e2e. This is a graph-wiring integration task.

- [ ] **Step 1: Add the include + BUILD deps**

In `yolo_object_detector_graph.cc`, add to the includes:
```cpp
#include "mediapipe/tasks/cc/vision/utils/tiled_detection_utils.h"  // (already included; confirm)
```
In `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`, in `cc_library(name = "yolo_object_detector_graph")` deps, add:
```python
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/graphs/tiled_detection:tiled_detection_stream_front_graph",
```

- [ ] **Step 2: Make the front node conditional on scheduling + add the loopback**

In the tiled branch of `BuildYoloObjectDetectionTask`, find the current front-node creation:

```cpp
      auto& front =
          graph.AddNode("mediapipe.tiled_detection.TiledDetectionFrontGraph");
      auto& fo =
          front.GetOptions<::mediapipe::TiledDetectionFrontGraphOptions>();
      auto* tg = fo.mutable_tile_grid();
```

Replace the `auto& front = graph.AddNode("mediapipe.tiled_detection.TiledDetectionFrontGraph");` line (only that line) with:

```cpp
      // Stream mode + opted-in motion scheduling -> the scheduler-bearing
      // stream front (it also runs its own optical-flow pass and consumes a
      // PRIOR_DETECTIONS loopback); otherwise the plain front. Both use the
      // same TiledDetectionFrontGraphOptions.
      const bool scheduling_enabled =
          task_options.base_options().use_stream_mode() &&
          ::mediapipe::tasks::vision::SchedulingEnabled(tiling);
      auto& front = graph.AddNode(
          scheduling_enabled
              ? "mediapipe.tiled_detection.TiledDetectionStreamFrontGraph"
              : "mediapipe.tiled_detection.TiledDetectionFrontGraph");
```

Then, immediately AFTER the existing line `to_frame.Out(kImageCpuTag) >> front.In(kImageTag);`, add the scheduler-front extras (the `max_scheduled_tiles` option + the loopback node, captured for deferred back-edge closing):

```cpp
      // PreviousLoopbackCalculator feeds the previous frame's merged detections
      // as the scheduler's PRIOR_DETECTIONS (first frame: empty -> DETECT).
      // The back edge (LOOP) is closed after merged_dets is produced, below.
      ::mediapipe::api2::builder::GenericNode* loopback = nullptr;
      if (scheduling_enabled) {
        fo.set_max_scheduled_tiles(tiling.max_scheduled_tiles());
        auto& lb = graph.AddNode("PreviousLoopbackCalculator");
        image_in >> lb.In("MAIN");
        lb.Out("PREV_LOOP").Cast<std::vector<Detection>>() >>
            front.In("PRIOR_DETECTIONS");
        loopback = &lb;
      }
```

- [ ] **Step 3: Close the back-edge after `merged_dets`**

In the same function, find `*merged_dets >> label_id_to_text.In("");` and insert just BEFORE it:

```cpp
      // Close the scheduler's PRIOR_DETECTIONS loopback with this frame's
      // merged (frame-normalized, post-fusion) detections.
      if (loopback != nullptr) {
        *merged_dets >> loopback->In("LOOP").AsBackEdge();
      }
```

(`merged_dets` is the `std::optional<Source<std::vector<Detection>>>` already populated by both arms of the `use_stream_mode()` merge branch; in stream mode it is `TiledBoxTrackMergeGraph`'s output. The merge branch itself is UNCHANGED.)

- [ ] **Step 4: Build the graph + run the existing detector tests (regression)**

Run: `bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_graph`
Then: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test`
Expected: builds; all existing tests pass (IMAGE path, the C `TiledVideoTracksBoatsWhilePanning` with scheduling OFF, the reject test). If `GenericNode` is not the correct builder node type, check the return type of `graph.AddNode(...)` in this file (sub-project C used `auto& merge = graph.AddNode(...)`); the pointer type must match it.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc \
        mediapipe/tasks/cc/vision/yolo_object_detector/BUILD
git commit -m "feat(tiling): YoloObjectDetectorGraph inserts scheduler stream front + loopback

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Public `DetectForVideo` scheduling e2e

**Files:**
- Test: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc`

The `TranslateImage` helper + opencv test deps already exist (from sub-project C). No BUILD change expected.

- [ ] **Step 1: Add the scheduling e2e tests**

In `yolo_object_detector_test.cc`, add:

```cpp
// Tiled VIDEO with motion scheduling ON: panning boats. The cyclic
// scheduler<-loopback graph must run to completion (no deadlock) and boats
// must persist across frames (DETECT frames detect; any SKIP frames are
// filled by the tracker).
TEST(YoloObjectDetectorTest, TiledVideoSchedulingPanningKeepsBoats) {
  const std::string model_path = ModelPath();
  const std::string image_path = JoinPath("./", kTestDataDirectory, kBoatsImage);
  if (!mediapipe::file::Exists(model_path).ok() ||
      !mediapipe::file::Exists(image_path).ok()) {
    GTEST_SKIP() << "YOLO model or boats.jpg fixture not available.";
  }
  MP_ASSERT_OK_AND_ASSIGN(Image base_image, DecodeImageFromFile(image_path));

  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::VIDEO;
  options->max_results = 10;
  options->num_classes = 80;
  options->score_threshold = 0.09f;
  options->iou_threshold = 0.45f;
  options->tiling.tile_cols = 2;
  options->tiling.tile_overlap_fraction = 0.2f;
  options->tiling.enable_motion_scheduling = true;  // <-- the feature under test

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));

  int frames_with_boat = 0;
  const int kFrames = 8;
  for (int i = 0; i < kFrames; ++i) {
    Image frame = TranslateImage(base_image, /*dx=*/2 * i, /*dy=*/0);
    MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result,
                            detector->DetectForVideo(frame, /*timestamp_ms=*/i));
    for (const auto& d : result.detections) {
      ASSERT_EQ(d.categories.size(), 1u);
      if (d.categories[0].index == 8) {
        ++frames_with_boat;
        break;
      }
    }
  }
  MP_ASSERT_OK(detector->Close());
  EXPECT_GE(frames_with_boat, kFrames / 2)
      << "boats should persist across most frames with scheduling on";
}

// SKIP -> tracker-fill at the public level: after a first DETECT frame, feed
// IDENTICAL frames (no motion). Each DetectForVideo call returns that frame's
// result synchronously; boats must still be present (filled by the tracker on
// any SKIPped frame). The deterministic "inference actually skipped" proof is
// the graph-level video_tile_scheduler_pipeline_test.
TEST(YoloObjectDetectorTest, TiledVideoSchedulingStaticFramesKeepBoats) {
  const std::string model_path = ModelPath();
  const std::string image_path = JoinPath("./", kTestDataDirectory, kBoatsImage);
  if (!mediapipe::file::Exists(model_path).ok() ||
      !mediapipe::file::Exists(image_path).ok()) {
    GTEST_SKIP() << "YOLO model or boats.jpg fixture not available.";
  }
  MP_ASSERT_OK_AND_ASSIGN(Image base_image, DecodeImageFromFile(image_path));

  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::VIDEO;
  options->max_results = 10;
  options->num_classes = 80;
  options->score_threshold = 0.09f;
  options->iou_threshold = 0.45f;
  options->tiling.tile_cols = 2;
  options->tiling.tile_overlap_fraction = 0.2f;
  options->tiling.enable_motion_scheduling = true;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));

  const int kFrames = 6;
  int frames_with_boat = 0;
  for (int i = 0; i < kFrames; ++i) {
    // Identical frame every time (zero motion) -> near-duplicate -> SKIP after
    // the first DETECT; the tracker fills.
    Image frame = TranslateImage(base_image, /*dx=*/0, /*dy=*/0);
    MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result,
                            detector->DetectForVideo(frame, /*timestamp_ms=*/i));
    for (const auto& d : result.detections) {
      if (!d.categories.empty() && d.categories[0].index == 8) {
        ++frames_with_boat;
        break;
      }
    }
  }
  MP_ASSERT_OK(detector->Close());
  // The full public empty-TILES -> tracker-fill -> current-frame-output path
  // produces boats on (nearly) every frame, not just frame 0.
  EXPECT_GE(frames_with_boat, kFrames - 1)
      << "tracker should keep boats present across static (SKIPped) frames";
}
```

- [ ] **Step 2: Run the e2e tests**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 --cache_test_results=no //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_filter=*TiledVideoScheduling*`
Expected: PASS if fixtures present, else GTEST_SKIP. If `TiledVideoSchedulingStaticFramesKeepBoats` shows fewer boats than expected, the tracker may drop tracks across many SKIPs at this fixture scale — first confirm the graph does not deadlock (the primary goal); if the recall threshold is too strict for the tracker's behavior on identical frames, relax to `>= kFrames / 2` (do NOT remove the assertion) and note it. A deadlock/hang is BLOCKED (the loopback wiring is wrong — re-check `.AsBackEdge()` and that `merged_dets` feeds `LOOP`).

- [ ] **Step 3: Commit**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc
git commit -m "test(tiling): public DetectForVideo motion-scheduling e2e (panning + static SKIP fill)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Full regression

**Files:** none (verification only).

- [ ] **Step 1: New + adjacent graph/calculator tests**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 --cache_test_results=no \
  //mediapipe/graphs/tiled_detection:tiled_video_scheduler_graphs_test \
  //mediapipe/graphs/tiled_detection:tiled_tracking_graphs_test \
  //mediapipe/graphs/tiled_detection:tiled_detection_graphs_test \
  //mediapipe/calculators/tensor:video_tile_scheduler_pipeline_test \
  //mediapipe/calculators/tensor:video_tile_scheduler_e2e_test
```
Expected: all PASS. The two `video_tile_scheduler_*` tests are sub-project C/M9 tests that must remain green (regression guard for the frozen scheduler + the SKIP-mechanics proof).

- [ ] **Step 2: Both detector Tasks tests (C frozen + B new)**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 --cache_test_results=no \
  //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test \
  //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test
```
Expected: all PASS or GTEST_SKIP. The OBB test is untouched. C's `TiledVideoTracksBoatsWhilePanning` (scheduling OFF) must still pass — proving B did not regress C.

- [ ] **Step 3: Confirm clean tree + commit chain**

Run: `git status --short && git log --oneline -8`
Expected: clean tree; 6 feature/test commits (Tasks 1–6) on top of the spec commit.

---

## Self-Review notes (author checklist, completed)

- **Spec coverage:** `OpticalFlowTrackingGraph` (T1) ↔ spec §Component 1; config fields (T2) ↔ §Component 4; `SchedulingEnabled` + IMAGE reject (T3) ↔ §Component 4 + the two-gate decision; `TiledDetectionStreamFrontGraph` (T4) ↔ §Component 2 + the TRACKING density contract no-stall test; YOLO loopback wiring (T5) ↔ §Component 3; e2e (T6) ↔ §Testing #3(a)(b)(c) [(c) reject is in T3]; regression (T7) ↔ §Testing + "no edits to C". The TRACKING-density [P1] no-stall is the explicit arbiter in T4 Step 5, with the materializer fallback called out.
- **C frozen:** no task edits C's `TiledTrackingGraph`/`TiledBoxTrackMergeGraph`/suppression/codec/tick-gate/tests. The YOLO merge branch (TiledBoxTrackMergeGraph, IMAGE input) is left exactly as C wired it; B only swaps the FRONT node and adds the loopback.
- **Type/name consistency:** scheduler tags `TILES`/`PRIOR_DETECTIONS`/`TRACKING`/`REFRESH`; `OpticalFlowTrackingGraph` IO `IMAGE`/`TRACKING`; front options reuse `TiledDetectionFrontGraphOptions` (+`max_scheduled_tiles` field 7); `TilingOptions` fields 7/8; `SchedulingEnabled` = `enable_motion_scheduling && TilingEnabled`; loopback `MAIN`/`LOOP`/`PREV_LOOP` with `.AsBackEdge()`.
- **Known residual risks:** (1) the scheduler not stalling on real buffered MotionAnalysis flow — T4 Step 5 arbitrates, materializer fallback spec'd; (2) the cyclic loopback deadlock — T6 Step 2 arbitrates; (3) duplicate-frame SKIP reliability on identical frames — graph-level deterministic SKIP proof is the existing pipeline test, so the public test asserts recall (boats present), not skip-count.
