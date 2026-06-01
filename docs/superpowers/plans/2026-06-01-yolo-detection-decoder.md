# YOLO Detection Decoder Implementation Plan (Group 1, Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a batch-native `YoloTensorsToDetectionsCalculator` that decodes Ultralytics YOLOv8/v11 *detect-head* output tensors into MediaPipe `Detection`s (axis-aligned, normalized), with confidence thresholding and an optional pre-NMS top-K — leaving NMS to a downstream calculator.

**Architecture:** A brand-new api2 calculator placed beside the existing SSD `TensorsToDetectionsCalculator` in `mediapipe/calculators/tensor/`. It never touches the existing SSD calculator (which hard-asserts `batch==1`). It reads one float32 output tensor in `[N, 4+num_classes, A]` (CHANNELS_FIRST) or `[N, A, 4+num_classes]` (CHANNELS_LAST) layout, anchor-free, and emits `std::vector<std::vector<Detection>>` (outer index = batch row). No NMS here: the design's single frame-global NMS runs downstream (Plan 3).

**Tech Stack:** C++17, MediaPipe Framework api2 (`Node`, typed ports), protobuf2 options, Bazel (`mediapipe_proto_library`, `cc_library`, `cc_test`), GoogleTest via `CalculatorRunner`.

**Spec:** `docs/superpowers/specs/2026-06-01-detection-core-yolo-obb-tiling-design.md` (§7 M2; §4.1 shape notation; §5.4 batched-output convention).

**Assumptions (from spec §7, called out so they are explicit):**
- Single model output tensor, `float32`. Quantized outputs are out of scope (rejected with a clear error in a later plan; here we only accept `kFloat32`).
- Box coordinates are `xywh` **normalized to `[0,1]`** (`xywh_normalized`). Pixel-space export is out of scope for this plan.
- Class channels are **probabilities** (post-sigmoid, as Ultralytics TFLite export produces). No sigmoid is applied here.
- Detect head only — no objectness channel (v8/v11 detect has `4 + num_classes` channels, no separate objectness).

**TDD note:** Tasks 2, 5, and 6 are strict red-green (write a failing test, watch it fail, implement). Tasks 3 (CHANNELS_LAST) and 4 (multi-class) are *triangulation* tests — the Task 2 implementation is written general enough to satisfy them, so they pass on first run and lock the behavior in. Their steps say "Expected: PASS" deliberately; that is not a mistake.

---

## File structure

- Create: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.proto` — options message.
- Create: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.cc` — the calculator.
- Create: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc` — unit tests.
- Modify: `mediapipe/calculators/tensor/BUILD` — add proto, library, and test targets.

All commands run from the repo root with `--define MEDIAPIPE_DISABLE_GPU=1` (CPU-only; this calculator is CPU-only).

---

### Task 1: Options proto

**Files:**
- Create: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.proto`
- Modify: `mediapipe/calculators/tensor/BUILD`

- [ ] **Step 1: Pick a unique options extension id**

Calculator options extend `mediapipe.CalculatorOptions` and need a globally-unique extension number. Run:

```bash
grep -rhoE "ext = [0-9]+" mediapipe | sort -t= -k2 -n | tail -20
```

Expected: a list of in-use numbers (e.g., `335742639`, `246514968`). Pick a number **not** in the list. This plan uses `471230001`; if that already appears in the output, choose another unused value and use it consistently in Steps 2 and in Task 2.

- [ ] **Step 2: Write the proto**

Create `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.proto`:

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

syntax = "proto2";

package mediapipe;

import "mediapipe/framework/calculator.proto";

// Options for YoloTensorsToDetectionsCalculator: anchor-free decoding of
// Ultralytics YOLOv8/v11 detect-head output tensors.
message YoloTensorsToDetectionsCalculatorOptions {
  extend .mediapipe.CalculatorOptions {
    optional YoloTensorsToDetectionsCalculatorOptions ext = 471230001;
  }

  // Memory layout of the model's detect output tensor.
  enum Layout {
    LAYOUT_UNSPECIFIED = 0;
    // [N, 4 + num_classes, A]  (Ultralytics default export)
    CHANNELS_FIRST = 1;
    // [N, A, 4 + num_classes]
    CHANNELS_LAST = 2;
  }
  optional Layout layout = 1 [default = CHANNELS_FIRST];

  // [Required] Number of object classes (channels after the 4 box values).
  optional int32 num_classes = 2;

  // Minimum class probability for a candidate to be emitted.
  optional float conf_threshold = 3 [default = 0.25];

  // Per batch row, keep at most this many highest-scoring candidates before
  // NMS. -1 keeps all candidates above conf_threshold.
  optional int32 max_detections_before_nms = 4 [default = 300];
}
```

- [ ] **Step 3: Add the proto build target**

In `mediapipe/calculators/tensor/BUILD`, add (near the other `mediapipe_proto_library` targets):

```python
mediapipe_proto_library(
    name = "yolo_tensors_to_detections_calculator_proto",
    srcs = ["yolo_tensors_to_detections_calculator.proto"],
    deps = ["//mediapipe/framework:calculator_proto"],
)
```

- [ ] **Step 4: Build the proto to verify it compiles**

Run:

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_cc_proto
```

Expected: builds successfully (target name is the proto target + `_cc_proto`).

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.proto mediapipe/calculators/tensor/BUILD
git commit -m "feat(yolo): add YoloTensorsToDetectionsCalculator options proto"
```

---

### Task 2: Decode CHANNELS_FIRST, single batch row, single class

**Files:**
- Create: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.cc`
- Create: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

- [ ] **Step 1: Write the failing test**

Create `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc`:

```cpp
// Copyright 2026 The MediaPipe Authors. Apache-2.0.
#include <memory>
#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using BatchDetections = std::vector<std::vector<Detection>>;

// Builds a single-tensor input packet from row-major float data.
std::unique_ptr<std::vector<Tensor>> MakeTensor(
    const Tensor::Shape& shape, const std::vector<float>& data) {
  auto tensor = Tensor(Tensor::ElementType::kFloat32, shape);
  auto write = tensor.GetCpuWriteView();
  std::copy(data.begin(), data.end(), write.buffer<float>());
  auto v = std::make_unique<std::vector<Tensor>>();
  v->push_back(std::move(tensor));
  return v;
}

TEST(YoloTensorsToDetectionsCalculatorTest, ChannelsFirstSingleClass) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 1
        conf_threshold: 0.25
      }
    }
  )pb"));

  // Shape [N=1, C=5, A=2]; channels = cx,cy,w,h,score; row-major c*A + a.
  // Anchor0: cx .5 cy .5 w .2 h .4 score .9  -> kept
  // Anchor1: cx .25 cy .75 w .1 h .1 score .1 -> dropped (< 0.25)
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 5, 2},
                       {0.5f, 0.25f, 0.5f, 0.75f, 0.2f, 0.1f, 0.4f, 0.1f,
                        0.9f, 0.1f})
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Tag("DETECTIONS").packets;
  ASSERT_EQ(out.size(), 1);
  const BatchDetections& batch = out[0].Get<BatchDetections>();
  ASSERT_EQ(batch.size(), 1);          // one batch row
  ASSERT_EQ(batch[0].size(), 1);       // one kept detection
  const Detection& d = batch[0][0];
  ASSERT_EQ(d.score_size(), 1);
  EXPECT_NEAR(d.score(0), 0.9f, 1e-5);
  ASSERT_EQ(d.label_id_size(), 1);
  EXPECT_EQ(d.label_id(0), 0);
  const auto& bb = d.location_data().relative_bounding_box();
  EXPECT_NEAR(bb.xmin(), 0.4f, 1e-5);   // cx - w/2
  EXPECT_NEAR(bb.ymin(), 0.3f, 1e-5);   // cy - h/2
  EXPECT_NEAR(bb.width(), 0.2f, 1e-5);
  EXPECT_NEAR(bb.height(), 0.4f, 1e-5);
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 2: Write the calculator skeleton that compiles but fails the test**

Create `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.cc`. This first version decodes **only batch row 0** (Task 5 generalizes to N):

```cpp
// Copyright 2026 The MediaPipe Authors. Apache-2.0.
#include <algorithm>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// Decodes Ultralytics YOLOv8/v11 detect-head tensors into batched Detections.
// See docs/superpowers/specs/2026-06-01-detection-core-yolo-obb-tiling-design.md §7.
class YoloTensorsToDetectionsCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  static constexpr Output<std::vector<std::vector<Detection>>> kOutDetections{
      "DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInTensors, kOutDetections);

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<mediapipe::YoloTensorsToDetectionsCalculatorOptions>();
    RET_CHECK_GT(options_.num_classes(), 0)
        << "num_classes must be set and > 0";
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& tensors = *kInTensors(cc);
    RET_CHECK_EQ(tensors.size(), 1) << "expected exactly one output tensor";
    const Tensor& t = tensors[0];
    RET_CHECK(t.element_type() == Tensor::ElementType::kFloat32)
        << "only float32 output is supported";
    const auto& dims = t.shape().dims;
    RET_CHECK_EQ(dims.size(), 3) << "expected a rank-3 tensor";

    int num_classes = options_.num_classes();
    int channels = 4 + num_classes;
    int A;
    if (options_.layout() ==
        mediapipe::YoloTensorsToDetectionsCalculatorOptions::CHANNELS_LAST) {
      A = dims[1];
      RET_CHECK_EQ(dims[2], channels);
    } else {  // CHANNELS_FIRST (default)
      RET_CHECK_EQ(dims[1], channels);
      A = dims[2];
    }

    auto view = t.GetCpuReadView();
    const float* data = view.buffer<float>();

    auto out = std::make_unique<std::vector<std::vector<Detection>>>();
    out->resize(1);
    DecodeRow(data, /*n=*/0, channels, A, num_classes, &(*out)[0]);

    kOutDetections(cc).Send(std::move(out));
    return absl::OkStatus();
  }

 private:
  // Decodes batch row `n` into `dets`.
  void DecodeRow(const float* data, int n, int channels, int A, int num_classes,
                 std::vector<Detection>* dets) {
    const bool channels_last =
        options_.layout() ==
        mediapipe::YoloTensorsToDetectionsCalculatorOptions::CHANNELS_LAST;
    for (int a = 0; a < A; ++a) {
      auto at = [&](int c) -> float {
        return channels_last ? data[(n * A + a) * channels + c]
                             : data[(n * channels + c) * A + a];
      };
      const float cx = at(0), cy = at(1), w = at(2), h = at(3);
      int best = 0;
      float best_score = at(4);
      for (int c = 1; c < num_classes; ++c) {
        const float s = at(4 + c);
        if (s > best_score) {
          best_score = s;
          best = c;
        }
      }
      if (best_score < options_.conf_threshold()) continue;

      Detection d;
      auto* loc = d.mutable_location_data();
      loc->set_format(LocationData::RELATIVE_BOUNDING_BOX);
      auto* bb = loc->mutable_relative_bounding_box();
      bb->set_xmin(cx - w / 2.0f);
      bb->set_ymin(cy - h / 2.0f);
      bb->set_width(w);
      bb->set_height(h);
      d.add_score(best_score);
      d.add_label_id(best);
      dets->push_back(std::move(d));
    }
  }

  mediapipe::YoloTensorsToDetectionsCalculatorOptions options_;
};

MEDIAPIPE_REGISTER_NODE(YoloTensorsToDetectionsCalculator);

}  // namespace api2
}  // namespace mediapipe
```

> Note: this implementation already loops correctly for one row; Task 5's N>1 test will fail only because `Process` hard-codes `out->resize(1)` and decodes `n=0`. Keep it this way until Task 5.

- [ ] **Step 3: Add library and test build targets**

In `mediapipe/calculators/tensor/BUILD`, add:

```python
cc_library(
    name = "yolo_tensors_to_detections_calculator",
    srcs = ["yolo_tensors_to_detections_calculator.cc"],
    deps = [
        ":yolo_tensors_to_detections_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "yolo_tensors_to_detections_calculator_test",
    srcs = ["yolo_tensors_to_detections_calculator_test.cc"],
    deps = [
        ":yolo_tensors_to_detections_calculator",
        ":yolo_tensors_to_detections_calculator_cc_proto",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
    ],
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_test --test_output=all
```

Expected: `ChannelsFirstSingleClass` PASSES. (If the build fails on a missing dep like `location_data_cc_proto`, confirm the target name with `grep -n "name = \"location_data" mediapipe/framework/formats/BUILD`.)

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.cc mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "feat(yolo): decode CHANNELS_FIRST detect tensor (row 0, single class)"
```

---

### Task 3: CHANNELS_LAST layout

**Files:**
- Modify: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc`

(The calculator already supports both layouts via the `at()` indexer; this task adds the regression test that proves it.)

- [ ] **Step 1: Add the failing test**

Append inside the anonymous namespace of the test file:

```cpp
TEST(YoloTensorsToDetectionsCalculatorTest, ChannelsLastSameResult) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_LAST
        num_classes: 1
        conf_threshold: 0.25
      }
    }
  )pb"));

  // Shape [N=1, A=2, C=5]; row-major a*C + c. Same two anchors as Task 2.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 2, 5},
                       {0.5f, 0.5f, 0.2f, 0.4f, 0.9f,
                        0.25f, 0.75f, 0.1f, 0.1f, 0.1f})
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<
          std::vector<std::vector<Detection>>>();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 1);
  const auto& bb = batch[0][0].location_data().relative_bounding_box();
  EXPECT_NEAR(bb.xmin(), 0.4f, 1e-5);
  EXPECT_NEAR(bb.ymin(), 0.3f, 1e-5);
  EXPECT_NEAR(bb.width(), 0.2f, 1e-5);
  EXPECT_NEAR(bb.height(), 0.4f, 1e-5);
}
```

- [ ] **Step 2: Run the test**

Run:

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_test --test_output=all
```

Expected: both tests PASS. (This confirms the `CHANNELS_LAST` indexing path is correct.)

- [ ] **Step 3: Commit**

```bash
git add mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc
git commit -m "test(yolo): cover CHANNELS_LAST layout"
```

---

### Task 4: Multi-class argmax + confidence threshold

**Files:**
- Modify: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc`

(The `DecodeRow` argmax+threshold logic already supports multiple classes; this task proves it with a multi-class golden tensor.)

- [ ] **Step 1: Add the failing test**

Append inside the test file's anonymous namespace:

```cpp
TEST(YoloTensorsToDetectionsCalculatorTest, MultiClassArgmaxAndThreshold) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 3
        conf_threshold: 0.5
      }
    }
  )pb"));

  // Shape [N=1, C=7, A=2]; channels = cx,cy,w,h, s0,s1,s2 ; index c*A + a.
  // Anchor0: box(.5,.5,.2,.2) scores [.1,.8,.3] -> class 1, score .8 (kept)
  // Anchor1: box(.5,.5,.2,.2) scores [.4,.2,.1] -> max .4 < .5 (dropped)
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 7, 2},
                       {0.5f, 0.5f,   // cx
                        0.5f, 0.5f,   // cy
                        0.2f, 0.2f,   // w
                        0.2f, 0.2f,   // h
                        0.1f, 0.4f,   // s0
                        0.8f, 0.2f,   // s1
                        0.3f, 0.1f})  // s2
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<
          std::vector<std::vector<Detection>>>();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 1);
  EXPECT_EQ(batch[0][0].label_id(0), 1);
  EXPECT_NEAR(batch[0][0].score(0), 0.8f, 1e-5);
}
```

- [ ] **Step 2: Run the test**

Run:

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_test --test_output=all
```

Expected: all three tests PASS.

- [ ] **Step 3: Commit**

```bash
git add mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc
git commit -m "test(yolo): cover multi-class argmax and threshold"
```

---

### Task 5: Batch-native decode (N > 1)

**Files:**
- Modify: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.cc`
- Modify: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc`

- [ ] **Step 1: Write the failing test**

Append inside the test file's anonymous namespace:

```cpp
TEST(YoloTensorsToDetectionsCalculatorTest, BatchNativeTwoRows) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 1
        conf_threshold: 0.25
      }
    }
  )pb"));

  // Shape [N=2, C=5, A=1]; per row index ((n*C)+c)*A + a, A=1 so = n*5 + c.
  // Row0 score .9 (kept); Row1 score .1 (dropped) -> outer size 2, sizes {1,0}.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{2, 5, 1},
                       {0.5f, 0.5f, 0.2f, 0.4f, 0.9f,    // row 0
                        0.5f, 0.5f, 0.2f, 0.4f, 0.1f})   // row 1
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<
          std::vector<std::vector<Detection>>>();
  ASSERT_EQ(batch.size(), 2);       // one inner vector per batch row
  EXPECT_EQ(batch[0].size(), 1);
  EXPECT_EQ(batch[1].size(), 0);    // empty row still present
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_test --test_filter='*BatchNativeTwoRows' --test_output=all
```

Expected: FAIL — `batch.size()` is 1 (the skeleton hard-codes `out->resize(1)` and decodes only `n=0`).

- [ ] **Step 3: Generalize `Process` to loop over N**

In `yolo_tensors_to_detections_calculator.cc`, replace the batch-extraction block in `Process` (the lines computing `A`, then `out->resize(1)` and the single `DecodeRow` call) with a version that reads `N` and loops:

```cpp
    int N = dims[0];
    int num_classes = options_.num_classes();
    int channels = 4 + num_classes;
    int A;
    if (options_.layout() ==
        mediapipe::YoloTensorsToDetectionsCalculatorOptions::CHANNELS_LAST) {
      A = dims[1];
      RET_CHECK_EQ(dims[2], channels);
    } else {  // CHANNELS_FIRST (default)
      RET_CHECK_EQ(dims[1], channels);
      A = dims[2];
    }

    auto view = t.GetCpuReadView();
    const float* data = view.buffer<float>();

    auto out = std::make_unique<std::vector<std::vector<Detection>>>();
    out->resize(N);
    for (int n = 0; n < N; ++n) {
      DecodeRow(data, n, channels, A, num_classes, &(*out)[n]);
    }

    kOutDetections(cc).Send(std::move(out));
    return absl::OkStatus();
```

- [ ] **Step 4: Run the full test suite**

Run:

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_test --test_output=all
```

Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.cc mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc
git commit -m "feat(yolo): batch-native decode over N rows"
```

---

### Task 6: Pre-NMS top-K per row

**Files:**
- Modify: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.cc`
- Modify: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc`

- [ ] **Step 1: Write the failing test**

Append inside the test file's anonymous namespace:

```cpp
TEST(YoloTensorsToDetectionsCalculatorTest, TopKKeepsHighestScores) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 1
        conf_threshold: 0.1
        max_detections_before_nms: 2
      }
    }
  )pb"));

  // Shape [N=1, C=5, A=3]; index c*A + a. Three anchors, scores .3,.9,.6.
  // top-2 by score -> keep .9 and .6, drop .3.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 5, 3},
                       {0.5f, 0.5f, 0.5f,    // cx
                        0.5f, 0.5f, 0.5f,    // cy
                        0.2f, 0.2f, 0.2f,    // w
                        0.2f, 0.2f, 0.2f,    // h
                        0.3f, 0.9f, 0.6f})   // score
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<
          std::vector<std::vector<Detection>>>();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 2);
  EXPECT_NEAR(batch[0][0].score(0), 0.9f, 1e-5);  // sorted desc
  EXPECT_NEAR(batch[0][1].score(0), 0.6f, 1e-5);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_test --test_filter='*TopKKeepsHighestScores' --test_output=all
```

Expected: FAIL — without top-K the row has 3 detections (and unsorted).

- [ ] **Step 3: Apply top-K at the end of `DecodeRow`**

In `yolo_tensors_to_detections_calculator.cc`, add this just before the closing brace of `DecodeRow` (after the `for (int a ...)` loop that fills `dets`):

```cpp
    const int k = options_.max_detections_before_nms();
    if (k >= 0 && static_cast<int>(dets->size()) > k) {
      std::partial_sort(
          dets->begin(), dets->begin() + k, dets->end(),
          [](const Detection& l, const Detection& r) {
            return l.score(0) > r.score(0);
          });
      dets->resize(k);
    }
```

- [ ] **Step 4: Run the full test suite**

Run:

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_test --test_output=all
```

Expected: all five tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.cc mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc
git commit -m "feat(yolo): pre-NMS top-K per batch row"
```

---

## Done criteria (Plan 1)

- `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_test` is green (5 tests).
- `YoloTensorsToDetectionsCalculator` decodes v8/v11 detect tensors in both layouts, multi-class, batch-native (`N≥1`), with confidence threshold and pre-NMS top-K, emitting `std::vector<std::vector<Detection>>` and never touching the SSD calculator.

## Handoff to Plan 2 / Plan 3

- **Plan 2 (OBB)** reuses this calculator's structure for `YoloObbTensorsToOrientedDetectionsCalculator` (adds an angle channel and emits `OrientedDetection`), plus `RotatedNonMaxSuppressionCalculator`.
- **Plan 3 (tiling + metadata)** feeds this calculator from `StreamingTilesToTensorBatchCalculator`, regroups `BatchDetections` via `TensorBatchInfo`, projects to full-frame coords, and runs the single frame-global NMS. The `std::vector<std::vector<Detection>>` output type defined here is the `BatchDetections` of spec §5.4.
