# BoTSORT Tracker Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users choose, in the YOLO tiled-livestream path, between the existing BoxTracker (optical-flow, default) and a newly-vendored motion-only BoTSORT (tracking-by-detection), selectable via `YoloObjectDetectorOptions`.

**Architecture:** Vendor BoTSORT-cpp under `third_party/botsort/` (motion-only, no CUDA/ONNX/TensorRT/INIReader) → wrap it in a `BotsortTrackingCalculator` (api2) → branch on a `tracker_type` inside the existing `TiledTrackingGraph` (so the downstream merge/suppression is untouched) → expose a `tracking` option on the detector proto + public C++ struct + converter, gated by Create-time validation.

**Tech Stack:** C++20, Bazel, MediaPipe api2 calculators/subgraphs, proto2, OpenCV + Eigen (both already vendored), googletest.

**Reference spec:** `docs/superpowers/specs/2026-06-16-botsort-tracker-selection-design.md`

**Standing constraints (every task):** stay on branch `dev`; never branch/merge; all comments in English; TDD; commit after each task; commit-message trailer MUST be exactly:
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```
Only desktop C++ (`--define MEDIAPIPE_DISABLE_GPU=1`) builds on this machine — that is the authoritative signal; ignore clangd "file not found" noise.

---

## File Structure

| File | Responsibility |
|---|---|
| `third_party/botsort/include/*.h`, `src/*.cpp` | Vendored motion-only BoTSORT sources (Task 1) |
| `third_party/botsort/include/ReID.h` | **Replaced** with a no-op stub (no TensorRT) (Task 1) |
| `third_party/botsort/BUILD` | `cc_library(name="botsort")` + smoke test (Task 1) |
| `third_party/botsort/LICENSE` | Upstream MIT license (Task 1) |
| `mediapipe/calculators/tensor/botsort_tracking_calculator.proto` | `BotsortTrackingCalculatorOptions` (Task 2) |
| `mediapipe/calculators/tensor/botsort_tracking_calculator.cc` | api2 calculator wrapping `BoTSORT` (Task 2) |
| `mediapipe/calculators/tensor/botsort_tracking_calculator_test.cc` | Calculator unit tests (Task 2) |
| `mediapipe/graphs/tiled_detection/tiled_detection_graphs.proto` | `TiledTrackingGraphOptions` + `tracking` field on `TiledBoxMergeGraphOptions` (Task 3) |
| `mediapipe/graphs/tiled_detection/tiled_tracking_graph.cc` | Branch BOX_TRACKER vs BOTSORT (Task 3) |
| `mediapipe/graphs/tiled_detection/tiled_box_track_merge_graph.cc` | Forward tracking options into `TiledTrackingGraph` (Task 3) |
| `mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.proto` | `TrackingOptions` + `tracking` field (Task 4) |
| `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h` | Public `TrackingOptions` struct (Task 4) |
| `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.cc` | Converter mapping (Task 4) + Create-time validation (Task 5) |
| `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc` | Wire `tracking` into the stream merge node (Task 4) |
| `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc` | Validation tests (Task 5) + e2e (Task 6) |

---

## Task 1: Vendor BoTSORT-cpp (motion-only) + BUILD + smoke test

**Files:**
- Create: `third_party/botsort/include/*.h`, `third_party/botsort/src/*.cpp`, `third_party/botsort/LICENSE`, `third_party/botsort/BUILD`, `third_party/botsort/botsort_smoke_test.cc`

Upstream: `https://github.com/viplix3/BoTSORT-cpp` (MIT). Raw base URL:
`https://raw.githubusercontent.com/viplix3/BoTSORT-cpp/main/`

- [ ] **Step 1: Write the failing smoke test**

Create `third_party/botsort/botsort_smoke_test.cc`:

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

#include <vector>

#include "BoTSORT.h"
#include "DataType.h"
#include "TrackerParams.h"
#include "gtest/gtest.h"
#include "opencv2/core.hpp"

namespace {

// Proves the vendored motion-only BoTSORT builds, links (no TensorRT/ONNX/
// INIReader), and tracks a moving box across two frames with GMC + ReID off.
TEST(BotsortSmokeTest, TracksAcrossTwoFrames) {
  TrackerParams params;  // defaults: gmc_enabled=false, reid_enabled=false
  BoTSORT tracker(params);

  cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));

  Detection d;
  d.bbox_tlwh = cv::Rect_<float>(50.f, 50.f, 20.f, 40.f);
  d.class_id = 1;
  d.confidence = 0.9f;
  tracker.track({d}, frame);

  Detection d2 = d;
  d2.bbox_tlwh = cv::Rect_<float>(52.f, 50.f, 20.f, 40.f);
  std::vector<std::shared_ptr<Track>> tracks = tracker.track({d2}, frame);

  ASSERT_FALSE(tracks.empty());
  EXPECT_EQ(tracks[0]->get_class_id(), 1);
}

}  // namespace
```

- [ ] **Step 2: Vendor the upstream sources (motion-only subset)**

Run (downloads only the motion-only files — NOT `ReID.cpp`, NOT `TRT_InferenceEngine/`):

```bash
cd /Users/luolingfeng/Projects/mediapipe
mkdir -p third_party/botsort/include third_party/botsort/src
BASE=https://raw.githubusercontent.com/viplix3/BoTSORT-cpp/main
for h in BoTSORT DataType GlobalMotionCompensation GmcParams KalmanFilter \
         KalmanFilterAccBased ReIDParams TrackerParams lapjv matching \
         profiler track utils; do
  curl -fsSL "$BASE/botsort/include/$h.h" -o "third_party/botsort/include/$h.h"
done
for c in BoTSORT GlobalMotionCompensation GmcParams KalmanFilter \
         KalmanFilterAccBased ReIDParams TrackerParams lapjv matching \
         track utils; do
  curl -fsSL "$BASE/botsort/src/$c.cpp" -o "third_party/botsort/src/$c.cpp"
done
curl -fsSL "$BASE/LICENSE" -o third_party/botsort/LICENSE
ls third_party/botsort/include third_party/botsort/src
```

Expected: `include/` has 14 `.h` files; `src/` has 11 `.cpp` files; `LICENSE` present. (`INIReader.h`, `ReID.h`/`ReID.cpp`, `TRT_InferenceEngine/` are intentionally NOT downloaded — `ReID.h` is created as a stub in Step 3.)

- [ ] **Step 3: Replace `ReID.h` with a no-op stub (removes the TensorRT include)**

Upstream `ReID.h` unconditionally includes `TRT_InferenceEngine/TensorRT_InferenceEngine.h`. `BoTSORT.h` includes `ReID.h` and `BoTSORT.cpp` calls `ReIDModel` methods (compiled, though only reached when ReID is enabled — which it never is here). Provide a complete no-op `ReIDModel` so `BoTSORT.{h,cpp}` compile unchanged without TensorRT.

Create `third_party/botsort/include/ReID.h`:

```cpp
// Motion-only stub for the vendored BoTSORT. Upstream ReID.h pulls in TensorRT;
// this CPU/desktop build runs motion-only (Kalman + IoU/lapjv + GMC), so ReID
// is never enabled at runtime. A complete no-op ReIDModel lets BoTSORT.{h,cpp}
// compile and link without CUDA/TensorRT/ONNXRuntime. See the design spec:
// docs/superpowers/specs/2026-06-16-botsort-tracker-selection-design.md
#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "DataType.h"
#include "ReIDParams.h"

class ReIDModel {
 public:
  ReIDModel(const ReIDParams& /*params*/,
            const std::string& /*onnx_model_path*/) {}
  ~ReIDModel() = default;

  void pre_process(cv::Mat& /*image*/) {}

  FeatureVector extract_features(cv::Mat& /*image*/) {
    return FeatureVector::Zero();
  }

  const std::string& get_distance_metric() const {
    static const std::string kMetric = "cosine";
    return kMetric;
  }
};
```

- [ ] **Step 4: Stub the three `load_config` methods (removes the INIReader dependency)**

`BoTSORT.cpp` takes the address of `TrackerParams::load_config`, `GMC_Params::load_config`, and `ReIDParams::load_config` (passed as `std::function` to `fetch_config`), so all three symbols must remain defined — but their upstream bodies use `INIReader`. We never pass a string config path (the calculator passes value alternatives), so the bodies are dead; replace them with default-returning stubs and drop the `INIReader` include.

Replace the entire contents of `third_party/botsort/src/TrackerParams.cpp` with:

```cpp
// Motion-only build: configuration comes from the calculator, never from .ini
// files, so load_config is a never-executed stub kept only so its symbol stays
// defined (BoTSORT.cpp takes its address). Drops the INIReader dependency.
#include "TrackerParams.h"

TrackerParams TrackerParams::load_config(const std::string& /*config_path*/) {
  return TrackerParams{};
}
```

Replace the entire contents of `third_party/botsort/src/ReIDParams.cpp` with:

```cpp
// Motion-only build: see TrackerParams.cpp. Never-executed stub.
#include "ReIDParams.h"

ReIDParams ReIDParams::load_config(const std::string& /*config_path*/) {
  return ReIDParams{};
}
```

Replace the entire contents of `third_party/botsort/src/GmcParams.cpp` with:

```cpp
// Motion-only build: see TrackerParams.cpp. Never-executed stub. Returns a
// default SparseOptFlow configuration for the requested method.
#include "GmcParams.h"

GMC_Params GMC_Params::load_config(GMC_Method method,
                                   const std::string& /*config_path*/) {
  GMC_Params params;
  params.method_ = method;
  params.method_params_ = SparseOptFlow_Params{};
  return params;
}
```

- [ ] **Step 5: Write the BUILD**

Create `third_party/botsort/BUILD`:

```python
# Vendored, motion-only BoTSORT-cpp (https://github.com/viplix3/BoTSORT-cpp).
# MIT-licensed. ReID/TensorRT/ONNXRuntime/INIReader paths are removed (see the
# design spec); this builds CPU-only against the already-vendored OpenCV+Eigen.

licenses(["notice"])  # MIT

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

cc_library(
    name = "botsort",
    srcs = glob(["src/*.cpp"]),
    hdrs = glob(["include/*.h"]),
    copts = ["-w"],  # silence third-party warnings (repo builds -Werror-ish)
    includes = ["include"],
    deps = [
        "//third_party:opencv",
        "@eigen//:eigen3",
    ],
)

cc_test(
    name = "botsort_smoke_test",
    srcs = ["botsort_smoke_test.cc"],
    deps = [
        ":botsort",
        "//third_party:opencv",
        "@com_google_googletest//:gtest_main",
    ],
)
```

- [ ] **Step 6: Run the smoke test (build + link + run)**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //third_party/botsort:botsort_smoke_test --test_output=errors
```
Expected: PASS. If linking fails on a missing OpenCV module (e.g. `video`/`features2d` used by `GlobalMotionCompensation.cpp`), those are already provided by `opencv_macos.BUILD` (added for the BoxTracker stack); confirm the error and add the missing module to `//third_party:opencv` only if genuinely absent. Do NOT re-introduce TensorRT/ONNX.

- [ ] **Step 7: Commit**

```bash
git add third_party/botsort
git commit -m "feat(botsort): vendor motion-only BoTSORT-cpp + smoke test

Vendors viplix3/BoTSORT-cpp (MIT) under third_party/botsort, motion-only:
ReID.h replaced with a no-op stub (no TensorRT), the three load_config methods
stubbed to drop the INIReader dependency. cc_library builds against the
already-vendored OpenCV+Eigen; a smoke test proves it links and tracks.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `BotsortTrackingCalculator` + options proto

**Files:**
- Create: `mediapipe/calculators/tensor/botsort_tracking_calculator.proto`
- Create: `mediapipe/calculators/tensor/botsort_tracking_calculator.cc`
- Create: `mediapipe/calculators/tensor/botsort_tracking_calculator_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

- [ ] **Step 1: Write the options proto**

Create `mediapipe/calculators/tensor/botsort_tracking_calculator.proto`:

```proto
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

syntax = "proto2";

package mediapipe;

import "mediapipe/framework/calculator.proto";

// Options for BotsortTrackingCalculator. Defaults equal upstream BoTSORT
// TrackerParams. Motion-only: there is no ReID/appearance knob.
message BotsortTrackingCalculatorOptions {
  extend mediapipe.CalculatorOptions {
    optional BotsortTrackingCalculatorOptions ext = 471230016;
  }
  optional float track_high_threshold = 1 [default = 0.6];
  optional float track_low_threshold = 2 [default = 0.1];
  optional float new_track_threshold = 3 [default = 0.7];
  optional int32 track_buffer = 4 [default = 30];
  optional float match_threshold = 5 [default = 0.7];
  // Global motion compensation (sparse optical flow). Off by default (upstream
  // parity); recommended on for moving-camera streams.
  optional bool enable_gmc = 6 [default = false];
}
```

- [ ] **Step 2: Add the proto + calculator BUILD targets**

In `mediapipe/calculators/tensor/BUILD`, add (near the other `mediapipe_proto_library` blocks):

```python
mediapipe_proto_library(
    name = "botsort_tracking_calculator_proto",
    srcs = ["botsort_tracking_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "botsort_tracking_calculator",
    srcs = ["botsort_tracking_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":botsort_tracking_calculator_cc_proto",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//third_party:opencv",
        "//third_party/botsort",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)
```

- [ ] **Step 3: Write the calculator**

Create `mediapipe/calculators/tensor/botsort_tracking_calculator.cc`:

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

#include <memory>
#include <variant>
#include <vector>

#include "BoTSORT.h"
#include "DataType.h"
#include "GmcParams.h"
#include "ReIDParams.h"
#include "TrackerParams.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/botsort_tracking_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "opencv2/core.hpp"

namespace mediapipe {
namespace api2 {

// Tracking-by-detection tracker (motion-only BoTSORT). Consumes the source
// frame + the per-frame fresh detections (frame-normalized, carrying
// label_id + score) and emits tracker-maintained detections, one packet per
// source frame, at the same output contract as the BoxTracker subgraph.
//
// Persistent track IDs are computed internally but NOT written to the output
// Detection (parity output; see the design spec). label_id is carried through
// BoTSORT's class_id, exact for label_id <= 255 (the graph rejects models with
// > 256 classes on this path).
//
// Inputs:
//   IMAGE      - ImageFrame (source video frame; required by BoTSORT's GMC).
//   DETECTIONS - std::vector<Detection> (frame-normalized RELATIVE_BOUNDING_BOX).
// Outputs:
//   DETECTIONS - std::vector<Detection> (tracker output, frame-normalized).
class BotsortTrackingCalculator : public Node {
 public:
  static constexpr Input<ImageFrame> kImage{"IMAGE"};
  static constexpr Input<std::vector<Detection>> kDetections{"DETECTIONS"};
  static constexpr Output<std::vector<Detection>> kOut{"DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kImage, kDetections, kOut);

  absl::Status Open(CalculatorContext* cc) override {
    const auto& opts = cc->Options<BotsortTrackingCalculatorOptions>();
    TrackerParams params;
    params.track_high_thresh = opts.track_high_threshold();
    params.track_low_thresh = opts.track_low_threshold();
    params.new_track_thresh = opts.new_track_threshold();
    params.track_buffer = opts.track_buffer();
    params.match_thresh = opts.match_threshold();
    params.gmc_enabled = opts.enable_gmc();
    params.reid_enabled = false;

    Config<GMC_Params> gmc_config = std::monostate{};
    if (opts.enable_gmc()) {
      GMC_Params gmc;
      gmc.method_ = GMC_Method::SparseOptFlow;
      gmc.method_params_ = SparseOptFlow_Params{};
      gmc_config = gmc;
    }
    tracker_ = std::make_unique<BoTSORT>(
        Config<TrackerParams>(params), gmc_config,
        Config<ReIDParams>(std::monostate{}), /*reid_onnx_model_path=*/"");
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const ImageFrame& image = *kImage(cc);
    cv::Mat frame = formats::MatView(&image);
    const float width = static_cast<float>(image.Width());
    const float height = static_cast<float>(image.Height());

    std::vector<::Detection> bs_dets;
    if (kDetections(cc).IsConnected() && !kDetections(cc).IsEmpty()) {
      for (const Detection& d : *kDetections(cc)) {
        if (!d.has_location_data() ||
            !d.location_data().has_relative_bounding_box()) {
          continue;
        }
        const auto& rbb = d.location_data().relative_bounding_box();
        ::Detection bd;
        bd.bbox_tlwh = cv::Rect_<float>(rbb.xmin() * width, rbb.ymin() * height,
                                        rbb.width() * width,
                                        rbb.height() * height);
        bd.class_id = d.label_id_size() > 0 ? d.label_id(0) : 0;
        bd.confidence = d.score_size() > 0 ? d.score(0) : 0.0f;
        bs_dets.push_back(bd);
      }
    }

    std::vector<std::shared_ptr<Track>> tracks =
        tracker_->track(bs_dets, frame);

    std::vector<Detection> out;
    out.reserve(tracks.size());
    for (const std::shared_ptr<Track>& t : tracks) {
      const std::vector<float> tlwh = t->get_tlwh();  // pixel space
      Detection det;
      det.add_score(t->get_score());
      det.add_label_id(static_cast<int>(t->get_class_id()));
      auto* loc = det.mutable_location_data();
      loc->set_format(LocationData::RELATIVE_BOUNDING_BOX);
      auto* box = loc->mutable_relative_bounding_box();
      box->set_xmin(tlwh[0] / width);
      box->set_ymin(tlwh[1] / height);
      box->set_width(tlwh[2] / width);
      box->set_height(tlwh[3] / height);
      // Note: track id (t->track_id) is intentionally NOT written to
      // det.track_id / detection_id (parity output contract).
      out.push_back(std::move(det));
    }
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<BoTSORT> tracker_;
};

MEDIAPIPE_REGISTER_NODE(BotsortTrackingCalculator);

}  // namespace api2
}  // namespace mediapipe
```

- [ ] **Step 4: Write the calculator unit test**

Create `mediapipe/calculators/tensor/botsort_tracking_calculator_test.cc`:

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

#include <memory>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

Detection MakeDetection(float xmin, float ymin, float w, float h, int label,
                        float score) {
  Detection d;
  d.add_score(score);
  d.add_label_id(label);
  auto* loc = d.mutable_location_data();
  loc->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* box = loc->mutable_relative_bounding_box();
  box->set_xmin(xmin);
  box->set_ymin(ymin);
  box->set_width(w);
  box->set_height(h);
  return d;
}

std::unique_ptr<CalculatorRunner> MakeRunner() {
  return std::make_unique<CalculatorRunner>(R"pb(
    calculator: "BotsortTrackingCalculator"
    input_stream: "IMAGE:image"
    input_stream: "DETECTIONS:dets"
    output_stream: "DETECTIONS:tracked"
  )pb");
}

void PushFrame(CalculatorRunner* runner, int ts,
               const std::vector<Detection>& dets) {
  auto image = std::make_unique<ImageFrame>(ImageFormat::SRGB, 200, 200);
  runner->MutableInputs()->Tag("IMAGE").packets.push_back(
      Adopt(image.release()).At(Timestamp(ts)));
  runner->MutableInputs()->Tag("DETECTIONS")
      .packets.push_back(
          MakePacket<std::vector<Detection>>(dets).At(Timestamp(ts)));
}

TEST(BotsortTrackingCalculatorTest, PreservesLabelAndScoreAndCarriesNoTrackId) {
  auto runner = MakeRunner();
  PushFrame(runner.get(), 0, {MakeDetection(0.25f, 0.25f, 0.1f, 0.2f, 7, 0.9f)});
  PushFrame(runner.get(), 1, {MakeDetection(0.26f, 0.25f, 0.1f, 0.2f, 7, 0.9f)});
  MP_ASSERT_OK(runner->Run());

  const auto& out = runner->Outputs().Tag("DETECTIONS").packets;
  ASSERT_EQ(out.size(), 2);
  const auto& last = out.back().Get<std::vector<Detection>>();
  ASSERT_FALSE(last.empty());
  EXPECT_EQ(last[0].label_id(0), 7);
  EXPECT_GT(last[0].score(0), 0.0f);
  // Parity output: no public track id surfaced.
  EXPECT_FALSE(last[0].has_track_id());
  EXPECT_FALSE(last[0].has_detection_id());
}

TEST(BotsortTrackingCalculatorTest, EmptyDetectionsProducesValidPacket) {
  auto runner = MakeRunner();
  PushFrame(runner.get(), 0, {});
  MP_ASSERT_OK(runner->Run());
  const auto& out = runner->Outputs().Tag("DETECTIONS").packets;
  ASSERT_EQ(out.size(), 1);
  EXPECT_TRUE(out[0].Get<std::vector<Detection>>().empty());
}

}  // namespace
}  // namespace mediapipe
```

Add to `mediapipe/calculators/tensor/BUILD`:

```python
cc_test(
    name = "botsort_tracking_calculator_test",
    srcs = ["botsort_tracking_calculator_test.cc"],
    deps = [
        ":botsort_tracking_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
    ],
)
```

- [ ] **Step 5: Run the test to verify it fails, then passes**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:botsort_tracking_calculator_test --test_output=errors
```
Before Step 3's calculator exists the build fails (unregistered calculator / missing file); after, expected: PASS (2 tests). `Detection.track_id`/`detection_id` exist in `detection.proto`, so `has_track_id()`/`has_detection_id()` compile.

- [ ] **Step 6: Commit**

```bash
git add mediapipe/calculators/tensor/botsort_tracking_calculator.proto \
        mediapipe/calculators/tensor/botsort_tracking_calculator.cc \
        mediapipe/calculators/tensor/botsort_tracking_calculator_test.cc \
        mediapipe/calculators/tensor/BUILD
git commit -m "feat(botsort): BotsortTrackingCalculator wrapping motion-only BoTSORT

api2 calculator: IMAGE + DETECTIONS -> DETECTIONS. Converts normalized
RELATIVE_BOUNDING_BOX <-> BoTSORT pixel tlwh, preserves label_id (<=255) +
score, and does NOT write track_id/detection_id (parity output). Defaults match
upstream TrackerParams.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Subgraph proto + `TiledTrackingGraph` tracker branch

**Files:**
- Modify: `mediapipe/graphs/tiled_detection/tiled_detection_graphs.proto`
- Modify: `mediapipe/graphs/tiled_detection/tiled_tracking_graph.cc`
- Modify: `mediapipe/graphs/tiled_detection/tiled_box_track_merge_graph.cc`
- Modify: `mediapipe/graphs/tiled_detection/BUILD`
- Test: `mediapipe/graphs/tiled_detection/tiled_tracking_graphs_test.cc`

- [ ] **Step 1: Add the proto messages**

In `mediapipe/graphs/tiled_detection/tiled_detection_graphs.proto`, append a new message and extend `TiledBoxMergeGraphOptions`.

Append at end of file (before nothing — it's the last message; add after it):

```proto
// Tracker selection + motion-only BoTSORT knobs for TiledTrackingGraph. Mirrors
// the detector-level TrackingOptions; the YOLO graph builder copies fields in.
message TiledTrackingGraphOptions {
  extend .mediapipe.CalculatorOptions {
    optional TiledTrackingGraphOptions ext = 471230017;
  }
  enum TrackerType {
    TRACKER_UNSPECIFIED = 0;  // -> BOX_TRACKER
    BOX_TRACKER = 1;          // optical-flow propagation (existing)
    BOTSORT = 2;              // tracking-by-detection, motion-only
  }
  optional TrackerType tracker_type = 1 [default = BOX_TRACKER];
  optional float track_high_threshold = 2 [default = 0.6];
  optional float track_low_threshold = 3 [default = 0.1];
  optional float new_track_threshold = 4 [default = 0.7];
  optional int32 track_buffer = 5 [default = 30];
  optional float match_threshold = 6 [default = 0.7];
  optional bool enable_gmc = 7 [default = false];
}
```

Add one field inside the existing `TiledBoxMergeGraphOptions` message (after `max_detections = 3`):

```proto
  // Tracker selection forwarded to TiledTrackingGraph (stream mode only).
  optional TiledTrackingGraphOptions tracking = 4;
```

- [ ] **Step 2: Branch `TiledTrackingGraph` on tracker type**

Replace the body of `TiledTrackingGraph::GetConfig` in `tiled_tracking_graph.cc`. Add includes at top:

```cpp
#include "mediapipe/calculators/tensor/botsort_tracking_calculator.pb.h"
#include "mediapipe/graphs/tiled_detection/tiled_detection_graphs.pb.h"
```

Replace the existing `GetConfig` body (from `api2::builder::Graph graph;` through `return graph.GetConfig();`) with:

```cpp
    api2::builder::Graph graph;
    auto image = graph.In("IMAGE").Cast<ImageFrame>();
    auto fresh = graph.In("DETECTIONS").Cast<std::vector<Detection>>();

    const auto& opts = sc->Options<TiledTrackingGraphOptions>();

    if (opts.tracker_type() == TiledTrackingGraphOptions::BOTSORT) {
      // Tracking-by-detection. BoTSORT carries label_id + score natively, so no
      // ENCODE/DECODE codec is needed. A tick gate keeps one packet per source
      // frame on the synchronized TRACKER_DETECTIONS input.
      auto& bot = graph.AddNode("BotsortTrackingCalculator");
      auto& bo = bot.GetOptions<BotsortTrackingCalculatorOptions>();
      bo.set_track_high_threshold(opts.track_high_threshold());
      bo.set_track_low_threshold(opts.track_low_threshold());
      bo.set_new_track_threshold(opts.new_track_threshold());
      bo.set_track_buffer(opts.track_buffer());
      bo.set_match_threshold(opts.match_threshold());
      bo.set_enable_gmc(opts.enable_gmc());
      image >> bot.In("IMAGE");
      fresh >> bot.In("DETECTIONS");

      auto& gate = graph.AddNode("DetectionsTickGateCalculator");
      fresh >> gate.In("TICK");
      bot.Out("DETECTIONS") >> gate.In("DATA");
      gate.Out("DETECTIONS") >> graph.Out("TRACKER_DETECTIONS");
      return graph.GetConfig();
    }

    // Default: BoxTracker (optical-flow) via ObjectTrackingSubgraphCpu, with the
    // label_id codec around it (the tracker carries only string ids).
    auto& encode = graph.AddNode("DetectionLabelIdCodecCalculator");
    encode.GetOptions<DetectionLabelIdCodecCalculatorOptions>().set_direction(
        DetectionLabelIdCodecCalculatorOptions::ENCODE);
    fresh >> encode.In("DETECTIONS");

    auto& tracker = graph.AddNode("ObjectTrackingSubgraphCpu");
    image >> tracker.In("VIDEO");
    encode.Out("DETECTIONS") >> tracker.In("DETECTIONS");

    auto& decode = graph.AddNode("DetectionLabelIdCodecCalculator");
    decode.GetOptions<DetectionLabelIdCodecCalculatorOptions>().set_direction(
        DetectionLabelIdCodecCalculatorOptions::DECODE);
    tracker.Out("DETECTIONS") >> decode.In("DETECTIONS");

    auto& gate = graph.AddNode("DetectionsTickGateCalculator");
    fresh >> gate.In("TICK");
    decode.Out("DETECTIONS") >> gate.In("DATA");

    gate.Out("DETECTIONS") >> graph.Out("TRACKER_DETECTIONS");
    return graph.GetConfig();
```

- [ ] **Step 3: Forward options in `TiledBoxTrackMergeGraph`**

In `tiled_box_track_merge_graph.cc`, where the tracking node is added:

```cpp
    auto& track = graph.AddNode("mediapipe.tiled_detection.TiledTrackingGraph");
    graph.In("IMAGE") >> track.In("IMAGE");
    merged_fresh >> track.In("DETECTIONS");
```

change to copy the tracking options through:

```cpp
    auto& track = graph.AddNode("mediapipe.tiled_detection.TiledTrackingGraph");
    if (options.has_tracking()) {
      track.GetOptions<TiledTrackingGraphOptions>().CopyFrom(options.tracking());
    }
    graph.In("IMAGE") >> track.In("IMAGE");
    merged_fresh >> track.In("DETECTIONS");
```

(`tiled_detection_graphs.pb.h` is already included in this file.)

- [ ] **Step 4: Update BUILD deps**

In `mediapipe/graphs/tiled_detection/BUILD`, add to the `cc_library` that builds `tiled_tracking_graph.cc` these deps:

```python
        "//mediapipe/calculators/tensor:botsort_tracking_calculator",
        "//mediapipe/calculators/tensor:botsort_tracking_calculator_cc_proto",
        ":tiled_detection_graphs_cc_proto",
```

And ensure the `tiled_box_track_merge_graph.cc` library depends on `:tiled_detection_graphs_cc_proto` (it already imports the pb.h — confirm the dep is present; add if missing).

- [ ] **Step 5: Write/extend the graph test**

In `mediapipe/graphs/tiled_detection/tiled_tracking_graphs_test.cc`, add a test that the BOTSORT branch produces one `TRACKER_DETECTIONS` packet per frame. Add near the existing `TiledTrackingGraph` tests:

```cpp
TEST(TiledTrackingGraphTest, BotsortBranchEmitsTrackerDetections) {
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "IMAGE:image"
        input_stream: "DETECTIONS:dets"
        output_stream: "TRACKER_DETECTIONS:out"
        node {
          calculator: "mediapipe.tiled_detection.TiledTrackingGraph"
          input_stream: "IMAGE:image"
          input_stream: "DETECTIONS:dets"
          output_stream: "TRACKER_DETECTIONS:out"
          node_options {
            [type.googleapis.com/mediapipe.TiledTrackingGraphOptions] {
              tracker_type: BOTSORT
            }
          }
        }
      )pb");

  std::vector<Packet> out_packets;
  tool::AddVectorSink("out", &config, &out_packets);
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  for (int ts = 0; ts < 2; ++ts) {
    auto image = std::make_unique<ImageFrame>(ImageFormat::SRGB, 200, 200);
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image", Adopt(image.release()).At(Timestamp(ts))));
    Detection d;
    d.add_score(0.9f);
    d.add_label_id(3);
    auto* box = d.mutable_location_data()->mutable_relative_bounding_box();
    d.mutable_location_data()->set_format(LocationData::RELATIVE_BOUNDING_BOX);
    box->set_xmin(0.25f);
    box->set_ymin(0.25f);
    box->set_width(0.1f);
    box->set_height(0.2f);
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "dets", MakePacket<std::vector<Detection>>(std::vector<Detection>{d})
                    .At(Timestamp(ts))));
  }
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_EQ(out_packets.size(), 2);
}
```

Ensure the test target in `BUILD` depends on `//mediapipe/calculators/tensor:botsort_tracking_calculator`, `:tiled_detection_graphs_cc_proto`, `//mediapipe/framework/formats:location_data_cc_proto`, and `//mediapipe/framework/tool:sink` (for `AddVectorSink`).

- [ ] **Step 6: Run the test, verify default path unchanged**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/graphs/tiled_detection:tiled_tracking_graphs_test --test_output=errors
```
Expected: PASS, including the pre-existing BOX_TRACKER tests (unchanged — when `tracker_type` is unset the default branch is byte-identical to the prior `GetConfig`).

- [ ] **Step 7: Commit**

```bash
git add mediapipe/graphs/tiled_detection/
git commit -m "feat(botsort): tracker_type branch in TiledTrackingGraph

Adds TiledTrackingGraphOptions (tracker_type + BoTSORT knobs) and a tracking
field on TiledBoxMergeGraphOptions. TiledTrackingGraph now branches: default
BOX_TRACKER path is byte-identical; BOTSORT routes through
BotsortTrackingCalculator + a tick gate (no label-id codec needed).
TiledBoxTrackMergeGraph forwards the options.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Detector proto `TrackingOptions` + public struct + converter + wiring

**Files:**
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.proto`
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h`
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.cc`
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc`
- Test: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc`

- [ ] **Step 1: Write the failing converter round-trip test**

In `yolo_object_detector_test.cc`, add (this calls the already-exposed `ConvertYoloObjectDetectorOptionsToProto`):

```cpp
TEST(YoloObjectDetectorOptionsConverterTest, MapsTrackingOptions) {
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->tracking.tracker_type =
      YoloObjectDetectorOptions::TrackingOptions::kBotsort;
  options->tracking.track_high_threshold = 0.55f;
  options->tracking.track_low_threshold = 0.15f;
  options->tracking.new_track_threshold = 0.65f;
  options->tracking.track_buffer = 25;
  options->tracking.match_threshold = 0.75f;
  options->tracking.enable_gmc = true;

  auto proto = ConvertYoloObjectDetectorOptionsToProto(options.get());

  EXPECT_EQ(proto->tracking().tracker_type(),
            YoloObjectDetectorOptionsProto::TrackingOptions::BOTSORT);
  EXPECT_FLOAT_EQ(proto->tracking().track_high_threshold(), 0.55f);
  EXPECT_FLOAT_EQ(proto->tracking().track_low_threshold(), 0.15f);
  EXPECT_FLOAT_EQ(proto->tracking().new_track_threshold(), 0.65f);
  EXPECT_EQ(proto->tracking().track_buffer(), 25);
  EXPECT_FLOAT_EQ(proto->tracking().match_threshold(), 0.75f);
  EXPECT_TRUE(proto->tracking().enable_gmc());
}
```

Run (expect FAIL to compile — `tracking` not defined yet):
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_output=errors
```
Expected: FAIL (no member `tracking`).

- [ ] **Step 2: Add the proto message + field**

In `yolo_object_detector_options.proto`, add a message (after `TilingOptions`, before the closing `}` of `YoloObjectDetectorOptions`) and a field:

```proto
  // Tracker selection for the tiled VIDEO/LIVE_STREAM path. tracker_type is
  // honored only when tiling is enabled and running mode is not IMAGE
  // (validated at Create()); otherwise there is no tracker node.
  message TrackingOptions {
    enum TrackerType {
      TRACKER_UNSPECIFIED = 0;  // -> BOX_TRACKER
      BOX_TRACKER = 1;          // optical-flow propagation (existing default)
      BOTSORT = 2;              // tracking-by-detection, motion-only
    }
    optional TrackerType tracker_type = 1 [default = BOX_TRACKER];
    optional float track_high_threshold = 2 [default = 0.6];
    optional float track_low_threshold = 3 [default = 0.1];
    optional float new_track_threshold = 4 [default = 0.7];
    optional int32 track_buffer = 5 [default = 30];
    optional float match_threshold = 6 [default = 0.7];
    optional bool enable_gmc = 7 [default = false];
  }
  optional TrackingOptions tracking = 11;
```

- [ ] **Step 3: Add the public C++ struct**

In `yolo_object_detector.h`, after the `TilingOptions` struct + its `tiling` member (around line 129), add:

```cpp
  // Tracker selection for the tiled VIDEO/LIVE_STREAM path (mirrors the proto
  // TrackingOptions). Honored only when tiling is enabled and running mode is
  // not IMAGE; validated at Create().
  struct TrackingOptions {
    // NOTE: values must stay numerically equal to the proto enum
    // TrackingOptions.TrackerType (BOX_TRACKER=1, BOTSORT=2) — the converter
    // static_casts between them.
    enum TrackerType {
      kBoxTracker = 1,  // optical-flow propagation (default)
      kBotsort = 2,     // tracking-by-detection, motion-only
    };
    TrackerType tracker_type = kBoxTracker;
    float track_high_threshold = 0.6f;
    float track_low_threshold = 0.1f;
    float new_track_threshold = 0.7f;
    int track_buffer = 30;
    float match_threshold = 0.7f;
    bool enable_gmc = false;
  };
  TrackingOptions tracking;
```

- [ ] **Step 4: Map it in the converter**

In `yolo_object_detector.cc`, in `ConvertYoloObjectDetectorOptionsToProto`, after the `tiling->set_max_scheduled_tiles(...)` line and before `return options_proto;`, add:

```cpp
  auto* tracking = options_proto->mutable_tracking();
  tracking->set_tracker_type(
      static_cast<YoloObjectDetectorOptionsProto::TrackingOptions::TrackerType>(
          options->tracking.tracker_type));
  tracking->set_track_high_threshold(options->tracking.track_high_threshold);
  tracking->set_track_low_threshold(options->tracking.track_low_threshold);
  tracking->set_new_track_threshold(options->tracking.new_track_threshold);
  tracking->set_track_buffer(options->tracking.track_buffer);
  tracking->set_match_threshold(options->tracking.match_threshold);
  tracking->set_enable_gmc(options->tracking.enable_gmc);
```

- [ ] **Step 5: Run the converter test — verify it passes**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_output=errors --test_filter='*MapsTrackingOptions*'
```
Expected: PASS.

- [ ] **Step 6: Wire `tracking` into the stream merge node**

In `yolo_object_detector_graph.cc`, in the `use_stream_mode()` branch where `TiledBoxTrackMergeGraph` is configured (after `mo.set_max_detections(task_options.max_results());`, around line 490), add:

```cpp
        auto* mtracking = mo.mutable_tracking();
        mtracking->set_tracker_type(
            static_cast<::mediapipe::TiledTrackingGraphOptions::TrackerType>(
                task_options.tracking().tracker_type()));
        mtracking->set_track_high_threshold(
            task_options.tracking().track_high_threshold());
        mtracking->set_track_low_threshold(
            task_options.tracking().track_low_threshold());
        mtracking->set_new_track_threshold(
            task_options.tracking().new_track_threshold());
        mtracking->set_track_buffer(task_options.tracking().track_buffer());
        mtracking->set_match_threshold(
            task_options.tracking().match_threshold());
        mtracking->set_enable_gmc(task_options.tracking().enable_gmc());
```

The proto `TiledTrackingGraphOptions::TrackerType` and `YoloObjectDetectorOptions::TrackingOptions::TrackerType` share numeric values (UNSPECIFIED=0/BOX_TRACKER=1/BOTSORT=2), so the `static_cast` is valid. Ensure the graph's `BUILD` cc_library depends on `//mediapipe/graphs/tiled_detection:tiled_detection_graphs_cc_proto` (it already references `TiledBoxMergeGraphOptions`; confirm present).

- [ ] **Step 7: Build the graph target**

Run:
```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_graph
```
Expected: builds clean.

- [ ] **Step 8: Commit**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/
git commit -m "feat(botsort): expose TrackingOptions on YOLO detector + wire to graph

Adds the TrackingOptions proto message + public C++ struct + converter mapping
(so users can actually select a tracker), and forwards tracking into the
stream-mode TiledBoxTrackMergeGraph node. Converter round-trip test included.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Create-time validation

**Files:**
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.cc`
- Test: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc`

- [ ] **Step 1: Write the failing validation tests (model-free)**

In `yolo_object_detector_test.cc`, add (mirrors the existing `MotionScheduling...Rejected` pattern — these fire before model load):

```cpp
TEST(YoloObjectDetectorTrackingValidationTest, BotsortInImageModeRejected) {
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->running_mode = core::RunningMode::IMAGE;
  options->num_classes = 80;
  options->tiling.tile_rows = 2;
  options->tiling.tile_cols = 2;
  options->tracking.tracker_type =
      YoloObjectDetectorOptions::TrackingOptions::kBotsort;
  auto result = YoloObjectDetector::Create(std::move(options));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(YoloObjectDetectorTrackingValidationTest, BotsortWithoutTilingRejected) {
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback = [](absl::StatusOr<YoloObjectDetectorResult>,
                                const Image&, int64_t) {};
  options->num_classes = 80;
  // tiling left at default 1x1 (disabled)
  options->tracking.tracker_type =
      YoloObjectDetectorOptions::TrackingOptions::kBotsort;
  auto result = YoloObjectDetector::Create(std::move(options));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(YoloObjectDetectorTrackingValidationTest,
     BotsortWithMotionSchedulingRejected) {
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->running_mode = core::RunningMode::VIDEO;
  options->num_classes = 80;
  options->tiling.tile_rows = 2;
  options->tiling.tile_cols = 2;
  options->tiling.enable_motion_scheduling = true;
  options->tracking.tracker_type =
      YoloObjectDetectorOptions::TrackingOptions::kBotsort;
  auto result = YoloObjectDetector::Create(std::move(options));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(YoloObjectDetectorTrackingValidationTest, BotsortWithTooManyClassesRejected) {
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->running_mode = core::RunningMode::VIDEO;
  options->num_classes = 300;  // > 256: uint8_t class-id limit
  options->tiling.tile_rows = 2;
  options->tiling.tile_cols = 2;
  options->tracking.tracker_type =
      YoloObjectDetectorOptions::TrackingOptions::kBotsort;
  auto result = YoloObjectDetector::Create(std::move(options));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}
```

Run (expect FAIL — no validation yet, so `Create` proceeds past these and fails later or succeeds-then-mismatches):
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_output=errors --test_filter='*TrackingValidation*'
```
Expected: FAIL.

- [ ] **Step 2: Add the validation block**

In `yolo_object_detector.cc`, in `Create()`, immediately after the existing `enable_motion_scheduling` IMAGE-mode check (the block ending at line ~199, before `auto detector = ...`), add:

```cpp
  if (options_proto->tracking().tracker_type() ==
      YoloObjectDetectorOptionsProto::TrackingOptions::BOTSORT) {
    if (options->running_mode == core::RunningMode::IMAGE) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "tracking.tracker_type=BOTSORT requires VIDEO or LIVE_STREAM running "
          "mode; tracking is not available in IMAGE mode.",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    if (!tiling_enabled) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "tracking.tracker_type=BOTSORT requires tiling to be enabled; the "
          "non-tiled path has no tracker stage.",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    if (options_proto->tiling().enable_motion_scheduling()) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "tracking.tracker_type=BOTSORT is incompatible with "
          "tiling.enable_motion_scheduling: BoTSORT cannot gap-fill SKIP "
          "frames (it emits no detections on a detection-less frame).",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    if (options_proto->num_classes() < 1 ||
        options_proto->num_classes() > 256) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "tracking.tracker_type=BOTSORT requires num_classes in [1, 256] "
          "(BoTSORT stores the class id as uint8).",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
  }
```

- [ ] **Step 3: Run the validation tests — verify they pass**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_output=errors --test_filter='*TrackingValidation*'
```
Expected: PASS (4 tests).

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.cc \
        mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc
git commit -m "feat(botsort): reject invalid BOTSORT selections at Create()

BOTSORT requires VIDEO/LIVE_STREAM + tiling + no motion-scheduling + num_classes
in [1,256], each a distinct InvalidArgument, so a selected BoTSORT can never be
silently ignored or truncate class ids. BOX_TRACKER/default unaffected.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Real tiled-stream e2e (gated on the YOLO fixture)

**Files:**
- Test: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc`

- [ ] **Step 1: Add the e2e test**

In `yolo_object_detector_test.cc`, add a test that mirrors the existing gated YOLO stream tests (find the helper that locates `yolov8n.tflite` and the `GTEST_SKIP()` guard already used by other YOLO model-gated tests; reuse the same constant/path and skip mechanism). Use this body, substituting the file's existing fixture-path constant for `kYoloModelFile` and the existing image-loading helper for `LoadTestImage`:

```cpp
TEST(YoloObjectDetectorBotsortE2ETest, TracksAcrossPanningFrames) {
  const std::string model_path =
      file::JoinPath("./", kTestDataDirectory, kYoloModelFile);
  if (!file::Exists(model_path).ok()) {
    GTEST_SKIP() << "YOLO tflite fixture not present; skipping BoTSORT e2e.";
  }

  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::VIDEO;
  options->num_classes = 80;
  options->max_results = 10;
  options->tiling.tile_rows = 2;
  options->tiling.tile_cols = 2;
  options->tracking.tracker_type =
      YoloObjectDetectorOptions::TrackingOptions::kBotsort;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));

  Image image = LoadTestImage("boats.jpg");
  for (int ts = 0; ts < 3; ++ts) {
    MP_ASSERT_OK_AND_ASSIGN(auto result, detector->DetectForVideo(image, ts));
    // BoTSORT runs on every inferred frame; by the last frame it must produce
    // tracked detections for a scene the tiled path detects.
    if (ts == 2) {
      EXPECT_FALSE(result.detections.empty());
    }
  }
  MP_ASSERT_OK(detector->Close());
}
```

- [ ] **Step 2: Run the e2e (skips without fixture)**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_output=all --test_filter='*BotsortE2E*'
```
Expected: PASS, or SKIPPED (logged "fixture not present") on a machine without `yolov8n.tflite`. If it runs, the last frame yields a non-empty tracked result.

- [ ] **Step 3: Full regression of the touched suites**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 \
  //third_party/botsort:botsort_smoke_test \
  //mediapipe/calculators/tensor:botsort_tracking_calculator_test \
  //mediapipe/graphs/tiled_detection:tiled_tracking_graphs_test \
  //mediapipe/graphs/tiled_detection:tiled_detection_graphs_test \
  //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test \
  --test_output=errors
```
Expected: all PASS (model-gated cases SKIP without the fixture). The pre-existing BOX_TRACKER graph + detector behavior must remain green (default path unchanged).

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc
git commit -m "test(botsort): tiled-stream e2e selecting BoTSORT (fixture-gated)

End-to-end VIDEO test selecting tracker_type=BOTSORT on a 2x2 tiled YOLO
pipeline; skips when the yolov8n tflite fixture is absent, like the other YOLO
model-gated tests.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final review (after all tasks)

Dispatch a final code reviewer over the full diff (`git diff master...dev` for this feature's commits) focusing on: the vendoring patch correctness (no TensorRT/INIReader symbols leaked; license present), the byte-identical BOX_TRACKER default path, the four validation gates, and the label_id/uint8 contract. Then use superpowers:finishing-a-development-branch.

---

## Self-Review

**Spec coverage:**
- Vendoring + motion-only patch + BUILD + smoke test → Task 1 ✓
- BotsortTrackingCalculator + options + label/score/no-track-id + empty-frame → Task 2 ✓
- TiledTrackingGraph branch + subgraph proto + byte-identical default → Task 3 ✓
- Detector proto + public struct + converter + graph wiring → Task 4 ✓
- Four Create-time validations (IMAGE / no-tiling / motion-scheduling / >256 classes) → Task 5 ✓
- Real e2e gated on fixture + regression → Task 6 ✓
- Follow-ups (bindings, track-id surfacing, OBB, non-tiled, ReID, SKIP gap-fill) → intentionally out of scope, not tasked ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code; bazel commands have expected output.

**Type consistency:** `YoloObjectDetectorOptions::TrackingOptions::kBotsort` (C++ struct, values 1/2) ↔ proto `...TrackingOptions::BOTSORT` (1/2) ↔ `TiledTrackingGraphOptions::BOTSORT` (1/2) ↔ `BotsortTrackingCalculatorOptions` fields — all enum values aligned for the `static_cast`s. Calculator option field names (`track_high_threshold`, …) match across the three protos and the `set_*`/`opts.*` calls. `BoTSORT`, `::Detection`, `Track`, `TrackerParams`, `GMC_Params`, `Config<T>` match the verified upstream headers.

**Known verification-time risks flagged in-plan:** OpenCV `video`/`features2d` linkage for GMC (Task 1 Step 6); exact YOLO fixture constant/skip-helper names to reuse (Task 6 Step 1).
