# OrientedObjectDetector (Tasks API) Implementation Plan (Phase 2.1b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `OrientedObjectDetector` C++ Task exposing OBB detection through the high-level Tasks API, parallel to the untouched `ObjectDetector`/`YoloObjectDetector`, returning a new pixel-unit oriented result type.

**Architecture:** Three genuinely-TDD-able units first — a new `OrientedObjectDetectionResult` container + converter (proto→pixel struct), an oriented batch→single flatten calculator, and an `OrientedDetectionProjectionCalculator` (affine-project normalized OBB corners → refit, staying normalized). Then build-verify the graph (`decode → flatten → rotated NMS (unchanged) → projection`) and the Task class (mirrors `YoloObjectDetector`, but converts to the oriented result with image dimensions), plus a fixture-gated integration test.

**Tech Stack:** C++17, MediaPipe Tasks (`ModelTaskGraph`, `BaseVisionTaskApi`), api2 calculators, OpenCV (`cv::minAreaRect`), protobuf2, Bazel.

**Spec:** `docs/superpowers/specs/2026-06-02-tasks-api-oriented-object-detector-design.md`.

**Depends on (done, on `dev`):** Group-1 `YoloObbTensorsToOrientedDetectionsCalculator`, `RotatedNonMaxSuppressionCalculator` (used **unchanged**), `OrientedDetection` proto (normalized); 2.1a `YoloObjectDetector{,Graph,Options}` + `YoloBatchDetectionsToSingleCalculator` (the patterns to mirror).

## TESTABILITY NOTE (read first)

Tasks 1–3 are **real red-green TDD** (no model needed: pure proto→struct conversion + calculator geometry). Tasks 4–6 (options proto, graph, Task class) are **build-verify** (the Tasks layer needs a model to actually run). Task 7's `Detect()` integration test is **gated** on a `yolov8n-obb.tflite` fixture we don't have — it builds and **skips honestly** (no faked pass), like 2.1a's. Verified facts the plan relies on: `OrientedDetection` is normalized-only (`oriented_detection.proto`); the projection matrix maps a point `p` as `{p.x·m[0]+p.y·m[1]+m[3], p.x·m[4]+p.y·m[5]+m[7]}` (from `detection_projection_calculator.cc`); the result-converter pattern is `detection_result.{h,cc}`.

All commands run from repo root with `--define MEDIAPIPE_DISABLE_GPU=1`.

---

### Task 1: `OrientedObjectDetectionResult` container + converter

**Files:**
- Create: `mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h`
- Create: `mediapipe/tasks/cc/components/containers/oriented_object_detection_result.cc`
- Create: `mediapipe/tasks/cc/components/containers/oriented_object_detection_result_test.cc`
- Modify: `mediapipe/tasks/cc/components/containers/BUILD`

- [ ] **Step 1: Write the failing test** `oriented_object_detection_result_test.cc`:

```cpp
// Copyright 2026 The MediaPipe Authors. (full Apache 2.0 header)
#include "mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h"

#include <vector>

#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::tasks::components::containers {
namespace {

TEST(OrientedObjectDetectionResultTest, ConvertsNormalizedProtoToPixels) {
  mediapipe::OrientedDetection d;
  d.set_cx(0.5f); d.set_cy(0.25f);
  d.set_width(0.4f); d.set_height(0.2f);
  d.set_rotation(0.3f);
  d.add_score(0.9f); d.add_label_id(7);

  OrientedObjectDetectionResult result =
      ConvertToOrientedObjectDetectionResult({d}, /*image_size=*/{200, 100});

  ASSERT_EQ(result.detections.size(), 1);
  const OrientedObjectDetection& od = result.detections[0];
  EXPECT_NEAR(od.cx, 100.0f, 1e-4);     // 0.5 * 200
  EXPECT_NEAR(od.cy, 25.0f, 1e-4);      // 0.25 * 100
  EXPECT_NEAR(od.width, 80.0f, 1e-4);   // 0.4 * 200
  EXPECT_NEAR(od.height, 20.0f, 1e-4);  // 0.2 * 100
  EXPECT_NEAR(od.rotation, 0.3f, 1e-6); // unchanged
  ASSERT_EQ(od.categories.size(), 1);
  EXPECT_EQ(od.categories[0].index, 7);
  EXPECT_NEAR(od.categories[0].score, 0.9f, 1e-6);
  // Label enrichment deferred: names empty.
  EXPECT_FALSE(od.categories[0].category_name.has_value());
  EXPECT_FALSE(od.categories[0].display_name.has_value());
}

}  // namespace
}  // namespace mediapipe::tasks::components::containers
```

- [ ] **Step 2: Write the header** `oriented_object_detection_result.h`:

```cpp
// Copyright 2026 The MediaPipe Authors. (full Apache 2.0 header)
#ifndef MEDIAPIPE_TASKS_CC_COMPONENTS_CONTAINERS_ORIENTED_OBJECT_DETECTION_RESULT_H_
#define MEDIAPIPE_TASKS_CC_COMPONENTS_CONTAINERS_ORIENTED_OBJECT_DETECTION_RESULT_H_

#include <utility>
#include <vector>

#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/tasks/cc/components/containers/category.h"

namespace mediapipe::tasks::components::containers {

// One oriented (rotated) bounding-box detection in PIXEL units.
struct OrientedObjectDetection {
  std::vector<Category> categories;
  float cx = 0.0f;        // box center x, pixels
  float cy = 0.0f;        // box center y, pixels
  float width = 0.0f;     // pixels
  float height = 0.0f;    // pixels
  float rotation = 0.0f;  // radians, counter-clockwise
};

struct OrientedObjectDetectionResult {
  std::vector<OrientedObjectDetection> detections;
};

// Converts original-image-normalized OrientedDetection protos to the pixel-unit
// container. image_size is {width, height} in pixels. Category names/display
// names are left empty (label enrichment deferred for YOLO-family Tasks).
OrientedObjectDetectionResult ConvertToOrientedObjectDetectionResult(
    std::vector<mediapipe::OrientedDetection> detections_proto,
    std::pair<int, int> image_size);

}  // namespace mediapipe::tasks::components::containers
#endif  // MEDIAPIPE_TASKS_CC_COMPONENTS_CONTAINERS_ORIENTED_OBJECT_DETECTION_RESULT_H_
```

- [ ] **Step 3: Write the implementation** `oriented_object_detection_result.cc`:

```cpp
// Copyright 2026 The MediaPipe Authors. (full Apache 2.0 header)
#include "mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h"

#include <utility>
#include <vector>

#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/tasks/cc/components/containers/category.h"

namespace mediapipe::tasks::components::containers {

constexpr int kDefaultCategoryIndex = -1;

OrientedObjectDetectionResult ConvertToOrientedObjectDetectionResult(
    std::vector<mediapipe::OrientedDetection> detections_proto,
    std::pair<int, int> image_size) {
  const float w = static_cast<float>(image_size.first);
  const float h = static_cast<float>(image_size.second);
  OrientedObjectDetectionResult result;
  result.detections.reserve(detections_proto.size());
  for (const auto& d : detections_proto) {
    OrientedObjectDetection od;
    for (int i = 0; i < d.score_size(); ++i) {
      od.categories.push_back(
          {/* index= */ d.label_id_size() > i ? d.label_id(i)
                                              : kDefaultCategoryIndex,
           /* score= */ d.score(i),
           /* category_name= */ std::nullopt,
           /* display_name= */ std::nullopt});
    }
    od.cx = d.cx() * w;
    od.cy = d.cy() * h;
    od.width = d.width() * w;
    od.height = d.height() * h;
    od.rotation = d.rotation();
    result.detections.push_back(std::move(od));
  }
  return result;
}

}  // namespace mediapipe::tasks::components::containers
```
Note: `Category` aggregate-init order is `{index, score, category_name, display_name}` (see `category.h`). `category_name`/`display_name` are `std::optional<std::string>` — pass `std::nullopt`.

- [ ] **Step 4: Add BUILD targets** to `mediapipe/tasks/cc/components/containers/BUILD` (mirror the `detection_result` lib + a `cc_test`):

```python
cc_library(
    name = "oriented_object_detection_result",
    srcs = ["oriented_object_detection_result.cc"],
    hdrs = ["oriented_object_detection_result.h"],
    deps = [
        ":category",
        "//mediapipe/framework/formats:oriented_detection_cc_proto",
    ],
)

cc_test(
    name = "oriented_object_detection_result_test",
    srcs = ["oriented_object_detection_result_test.cc"],
    deps = [
        ":oriented_object_detection_result",
        "//mediapipe/framework/formats:oriented_detection_cc_proto",
        "//mediapipe/framework/port:gtest_main",
    ],
)
```

- [ ] **Step 5: Run the test.**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/components/containers:oriented_object_detection_result_test --test_output=all
```
Expected: PASS (1 test). Fix dep labels by grepping the sibling BUILD if needed.

- [ ] **Step 6: Commit.**

```bash
git add mediapipe/tasks/cc/components/containers/oriented_object_detection_result.* mediapipe/tasks/cc/components/containers/BUILD
git commit -m "feat(tasks-obb): OrientedObjectDetectionResult container + converter"
```

---

### Task 2: `YoloObbBatchDetectionsToSingleCalculator`

Oriented analog of 2.1a's `YoloBatchDetectionsToSingleCalculator`. Flattens batched `vector<vector<OrientedDetection>>` → `vector<OrientedDetection>` (N≤1).

**Files:**
- Create: `mediapipe/calculators/tensor/yolo_obb_batch_detections_to_single_calculator.cc`
- Create: `mediapipe/calculators/tensor/yolo_obb_batch_detections_to_single_calculator_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

- [ ] **Step 1: Write the failing test** (`..._test.cc`), mirroring `yolo_batch_detections_to_single_calculator_test.cc` but with `OrientedDetection`:

```cpp
// Copyright 2026 The MediaPipe Authors. (full Apache 2.0 header)
#include <memory>
#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

OrientedDetection MakeObb(int label) {
  OrientedDetection d;
  d.set_cx(0.5f); d.set_cy(0.5f); d.set_width(0.2f); d.set_height(0.2f);
  d.set_rotation(0.0f); d.add_score(0.9f); d.add_label_id(label);
  return d;
}

TEST(YoloObbBatchDetectionsToSingleCalculatorTest, FlattensSingleRow) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloObbBatchDetectionsToSingleCalculator"
    input_stream: "ORIENTED_DETECTIONS:batched"
    output_stream: "ORIENTED_DETECTIONS:flat"
  )pb"));
  auto in = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  in->push_back({MakeObb(0), MakeObb(1)});
  runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(in.release()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                        .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(out.size(), 2);
  EXPECT_EQ(out[0].label_id(0), 0);
  EXPECT_EQ(out[1].label_id(0), 1);
}

TEST(YoloObbBatchDetectionsToSingleCalculatorTest, EmptyBatchEmitsEmpty) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloObbBatchDetectionsToSingleCalculator"
    input_stream: "ORIENTED_DETECTIONS:batched"
    output_stream: "ORIENTED_DETECTIONS:flat"
  )pb"));
  runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(new std::vector<std::vector<OrientedDetection>>()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  EXPECT_TRUE(runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                  .Get<std::vector<OrientedDetection>>().empty());
}

TEST(YoloObbBatchDetectionsToSingleCalculatorTest, RejectsMultipleRows) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloObbBatchDetectionsToSingleCalculator"
    input_stream: "ORIENTED_DETECTIONS:batched"
    output_stream: "ORIENTED_DETECTIONS:flat"
  )pb"));
  auto in = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  in->push_back({MakeObb(0)});
  in->push_back({MakeObb(1)});
  runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(in.release()).At(Timestamp(0)));
  EXPECT_FALSE(runner.Run().ok());
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 2: Add BUILD targets, run → expect FAIL** (no calculator registered). Targets (mirror the axis-aligned flatten in `calculators/tensor/BUILD`):

```python
cc_library(
    name = "yolo_obb_batch_detections_to_single_calculator",
    srcs = ["yolo_obb_batch_detections_to_single_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:oriented_detection_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "yolo_obb_batch_detections_to_single_calculator_test",
    srcs = ["yolo_obb_batch_detections_to_single_calculator_test.cc"],
    size = "small",
    deps = [
        ":yolo_obb_batch_detections_to_single_calculator",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:oriented_detection_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
    ],
)
```

- [ ] **Step 3: Write the calculator** `yolo_obb_batch_detections_to_single_calculator.cc`:

```cpp
// Copyright 2026 The MediaPipe Authors. (full Apache 2.0 header)
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// Flattens batched oriented detections (one inner vector per batch row) to a
// single vector for the single-image Tasks API. Requires batch N <= 1.
class YoloObbBatchDetectionsToSingleCalculator : public Node {
 public:
  static constexpr Input<std::vector<std::vector<OrientedDetection>>> kIn{
      "ORIENTED_DETECTIONS"};
  static constexpr Output<std::vector<OrientedDetection>> kOut{
      "ORIENTED_DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Process(CalculatorContext* cc) override {
    const auto& batched = *kIn(cc);
    RET_CHECK_LE(batched.size(), 1u)
        << "single-image Task expects batch N<=1, got " << batched.size();
    std::vector<OrientedDetection> out;
    if (!batched.empty()) out = batched[0];
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(YoloObbBatchDetectionsToSingleCalculator);

}  // namespace api2
}  // namespace mediapipe
```

- [ ] **Step 4: Run the test.**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_obb_batch_detections_to_single_calculator_test --test_output=all
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit.**

```bash
git add mediapipe/calculators/tensor/yolo_obb_batch_detections_to_single_calculator.* mediapipe/calculators/tensor/BUILD
git commit -m "feat(obb): batch->single OrientedDetection flatten calculator"
```

---

### Task 3: `OrientedDetectionProjectionCalculator`

The core new geometry. Inputs `ORIENTED_DETECTIONS` (normalized, model-input space) + `PROJECTION_MATRIX` (`std::array<float,16>`); output `ORIENTED_DETECTIONS` (normalized, original-image space). Transforms the four oriented corners through the affine, refits via `cv::minAreaRect`. Stays normalized (no pixels).

**Files:**
- Create: `mediapipe/calculators/util/oriented_detection_projection_calculator.cc`
- Create: `mediapipe/calculators/util/oriented_detection_projection_calculator_test.cc`
- Modify: `mediapipe/calculators/util/BUILD`

- [ ] **Step 1: Write the failing test** `..._test.cc`. Uses an identity matrix (box preserved) and a uniform-scale+translate matrix on an axis-aligned box (center/size scale, rotation stays 0):

```cpp
// Copyright 2026 The MediaPipe Authors. (full Apache 2.0 header)
#include <array>
#include <memory>
#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

OrientedDetection Obb(float cx, float cy, float w, float h, float rot) {
  OrientedDetection d;
  d.set_cx(cx); d.set_cy(cy); d.set_width(w); d.set_height(h);
  d.set_rotation(rot); d.add_score(0.9f); d.add_label_id(0);
  return d;
}

// Row-major 4x4 affine; only m[0],m[1],m[3] (x) and m[4],m[5],m[7] (y) are used.
std::unique_ptr<std::array<float, 16>> Matrix(float sx, float sy, float tx,
                                              float ty) {
  auto m = std::make_unique<std::array<float, 16>>();
  m->fill(0.0f);
  (*m)[0] = sx;  (*m)[3] = tx;   // x' = sx*x + tx
  (*m)[5] = sy;  (*m)[7] = ty;   // y' = sy*y + ty
  (*m)[10] = 1; (*m)[15] = 1;
  return m;
}

std::unique_ptr<std::vector<OrientedDetection>> Dets(OrientedDetection d) {
  auto v = std::make_unique<std::vector<OrientedDetection>>();
  v->push_back(std::move(d));
  return v;
}

const std::vector<OrientedDetection>& Run(
    CalculatorRunner* runner, std::unique_ptr<std::vector<OrientedDetection>> in,
    std::unique_ptr<std::array<float, 16>> m) {
  runner->MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(in.release()).At(Timestamp(0)));
  runner->MutableInputs()->Tag("PROJECTION_MATRIX").packets.push_back(
      Adopt(m.release()).At(Timestamp(0)));
  MP_EXPECT_OK(runner->Run());
  return runner->Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
      .Get<std::vector<OrientedDetection>>();
}

CalculatorRunner MakeRunner() {
  return CalculatorRunner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "OrientedDetectionProjectionCalculator"
    input_stream: "ORIENTED_DETECTIONS:in"
    input_stream: "PROJECTION_MATRIX:matrix"
    output_stream: "ORIENTED_DETECTIONS:out"
  )pb"));
}

TEST(OrientedDetectionProjectionCalculatorTest, IdentityPreservesBox) {
  auto runner = MakeRunner();
  const auto& out = Run(&runner, Dets(Obb(0.5f, 0.5f, 0.4f, 0.2f, 0.0f)),
                        Matrix(1, 1, 0, 0));
  ASSERT_EQ(out.size(), 1);
  EXPECT_NEAR(out[0].cx(), 0.5f, 1e-4);
  EXPECT_NEAR(out[0].cy(), 0.5f, 1e-4);
  EXPECT_NEAR(out[0].width(), 0.4f, 1e-3);
  EXPECT_NEAR(out[0].height(), 0.2f, 1e-3);
}

TEST(OrientedDetectionProjectionCalculatorTest, ScaleTranslateAxisAligned) {
  auto runner = MakeRunner();
  // x' = 0.5*x + 0.1 ; y' = 0.5*y + 0.0
  const auto& out = Run(&runner, Dets(Obb(0.6f, 0.4f, 0.4f, 0.2f, 0.0f)),
                        Matrix(0.5f, 0.5f, 0.1f, 0.0f));
  ASSERT_EQ(out.size(), 1);
  EXPECT_NEAR(out[0].cx(), 0.4f, 1e-3);    // 0.5*0.6 + 0.1
  EXPECT_NEAR(out[0].cy(), 0.2f, 1e-3);    // 0.5*0.4
  EXPECT_NEAR(out[0].width(), 0.2f, 1e-3); // 0.4*0.5
  EXPECT_NEAR(out[0].height(), 0.1f, 1e-3);// 0.2*0.5
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 2: Add BUILD targets, run → expect FAIL.** Targets in `mediapipe/calculators/util/BUILD` (mirror `rotated_non_max_suppression_calculator`'s OpenCV deps):

```python
cc_library(
    name = "oriented_detection_projection_calculator",
    srcs = ["oriented_detection_projection_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:oriented_detection_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "oriented_detection_projection_calculator_test",
    srcs = ["oriented_detection_projection_calculator_test.cc"],
    size = "small",
    deps = [
        ":oriented_detection_projection_calculator",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:oriented_detection_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
    ],
)
```

- [ ] **Step 3: Write the calculator** `oriented_detection_projection_calculator.cc`:

```cpp
// Copyright 2026 The MediaPipe Authors. (full Apache 2.0 header)
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// Projects normalized OrientedDetections through a 4x4 affine (model-input ->
// original-image normalized coords) by transforming the four oriented corners
// and refitting a rotated rectangle. Output stays normalized.
class OrientedDetectionProjectionCalculator : public Node {
 public:
  static constexpr Input<std::vector<OrientedDetection>> kIn{
      "ORIENTED_DETECTIONS"};
  static constexpr Input<std::array<float, 16>> kMatrix{"PROJECTION_MATRIX"};
  static constexpr Output<std::vector<OrientedDetection>> kOut{
      "ORIENTED_DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kIn, kMatrix, kOut);

  absl::Status Process(CalculatorContext* cc) override {
    const auto& dets = *kIn(cc);
    const std::array<float, 16>& m = *kMatrix(cc);
    constexpr float kDegToRad = static_cast<float>(M_PI) / 180.0f;

    auto out = std::make_unique<std::vector<OrientedDetection>>();
    out->reserve(dets.size());
    for (const auto& d : dets) {
      const float c = std::cos(d.rotation()), s = std::sin(d.rotation());
      const float hw = d.width() / 2.0f, hh = d.height() / 2.0f;
      // Four corners (normalized model-input coords), projected through m.
      std::vector<cv::Point2f> pts;
      pts.reserve(4);
      for (const auto& off : {std::pair<float, float>(-hw, -hh),
                              std::pair<float, float>(hw, -hh),
                              std::pair<float, float>(hw, hh),
                              std::pair<float, float>(-hw, hh)}) {
        const float x = d.cx() + off.first * c - off.second * s;
        const float y = d.cy() + off.first * s + off.second * c;
        const float px = x * m[0] + y * m[1] + m[3];
        const float py = x * m[4] + y * m[5] + m[7];
        pts.emplace_back(px, py);
      }
      cv::RotatedRect rr = cv::minAreaRect(pts);
      OrientedDetection projected = d;  // copies scores/labels
      projected.set_cx(rr.center.x);
      projected.set_cy(rr.center.y);
      projected.set_width(rr.size.width);
      projected.set_height(rr.size.height);
      projected.set_rotation(rr.angle * kDegToRad);
      out->push_back(std::move(projected));
    }
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(OrientedDetectionProjectionCalculator);

}  // namespace api2
}  // namespace mediapipe
```
Notes for the implementer:
- `cv::minAreaRect` returns the angle in degrees; convention varies by OpenCV version (≥4.5 uses [0,90)). For the two golden tests (rotation 0, scale/translate) the recovered box is axis-aligned and `width`/`height`/`center` assertions are stable regardless of the angle convention. If a future rotated-input golden test is added, normalize the angle and assert modulo the box's 180° symmetry. Do NOT change the algorithm to chase a specific angle value.
- If `IdentityPreservesBox` shows width/height swapped (minAreaRect can return them transposed with a 90° angle), that is the known minAreaRect ambiguity — assert on the unordered {width,height} set, or keep the box non-square so the larger dimension is unambiguous (the test uses 0.4×0.2). If needed, add a small normalization in the calc that orders (width≥height, angle adjusted) and document it.

- [ ] **Step 4: Run the test.** (`opencv_core`/`opencv_imgproc` are already configured from Plan 2.)

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/util:oriented_detection_projection_calculator_test --test_output=all
```
Expected: PASS (2 tests). If width/height come back swapped on the identity case, apply the normalization noted above and re-run.

- [ ] **Step 5: Commit.**

```bash
git add mediapipe/calculators/util/oriented_detection_projection_calculator.* mediapipe/calculators/util/BUILD
git commit -m "feat(obb): OrientedDetectionProjectionCalculator (affine project + refit)"
```

---

### Task 4: Options proto

**Files:**
- Create: `mediapipe/tasks/cc/vision/oriented_object_detector/proto/oriented_object_detector_options.proto`
- Create: `mediapipe/tasks/cc/vision/oriented_object_detector/proto/BUILD`

- [ ] **Step 1: Pick a unique ext id** (`grep -rhoE "ext = [0-9]+" mediapipe | sort -t= -k2 -n | tail -25`; plan uses `471230011`).

- [ ] **Step 2: Write the proto.** Mirror `yolo_object_detector_options.proto` (2.1a), but per the spec drop `display_names_locale`/`category_allowlist`/`category_denylist` and add `class_agnostic_nms`:

```proto
// Copyright 2026 The MediaPipe Authors. (full Apache 2.0 header)
syntax = "proto2";
package mediapipe.tasks.vision.oriented_object_detector.proto;

import "mediapipe/framework/calculator.proto";
import "mediapipe/tasks/cc/core/proto/base_options.proto";

option java_package = "com.google.mediapipe.tasks.vision.orientedobjectdetector.proto";
option java_outer_classname = "OrientedObjectDetectorOptionsProto";

message OrientedObjectDetectorOptions {
  extend mediapipe.CalculatorOptions {
    optional OrientedObjectDetectorOptions ext = 471230011;
  }
  optional core.proto.BaseOptions base_options = 1;
  optional int32 max_results = 2 [default = -1];
  optional float score_threshold = 3 [default = 0.25];
  optional float iou_threshold = 4 [default = 0.45];
  optional bool class_agnostic_nms = 5 [default = false];
  enum Layout {
    LAYOUT_UNSPECIFIED = 0;
    CHANNELS_FIRST = 1;
    CHANNELS_LAST = 2;
  }
  optional Layout layout = 6 [default = CHANNELS_FIRST];
  optional int32 num_classes = 7 [default = 0];
}
```

- [ ] **Step 3: Write `proto/BUILD`** mirroring 2.1a's `yolo_object_detector/proto/BUILD` (deps: `calculator_options_proto`, `calculator_proto`, `base_options_proto`).

- [ ] **Step 4: Build.**

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector/proto:oriented_object_detector_options_cc_proto
```
Expected: success.

- [ ] **Step 5: Commit.**

```bash
git add mediapipe/tasks/cc/vision/oriented_object_detector/proto/
git commit -m "feat(tasks-obb): OrientedObjectDetector options proto"
```

---

### Task 5: `OrientedObjectDetectorGraph` (build-verify)

Mirror 2.1a's `yolo_object_detector_graph.cc` (read it first); swap the postprocessing chain.

**Files:**
- Create: `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_graph.cc`
- Create: `mediapipe/tasks/cc/vision/oriented_object_detector/BUILD`

- [ ] **Step 1: Read** `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc` and its `BUILD` in full.

- [ ] **Step 2: Write the graph** as a copy of `yolo_object_detector_graph.cc` with: namespace/options → `oriented_object_detector`; class `OrientedObjectDetectorGraph`; `REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tasks::vision::oriented_object_detector::OrientedObjectDetectorGraph)`. Replace the postprocessing block with:

```cpp
    // OBB decode -> batched oriented detections.
    auto& obb_decode =
        graph.AddNode("YoloObbTensorsToOrientedDetectionsCalculator");
    {
      auto& opts = obb_decode.GetOptions<
          ::mediapipe::YoloObbTensorsToOrientedDetectionsCalculatorOptions>();
      int num_classes = task_options.num_classes();
      RET_CHECK_GT(num_classes, 0)
          << "num_classes must be set in OrientedObjectDetectorOptions";
      opts.set_num_classes(num_classes);
      opts.set_conf_threshold(task_options.score_threshold());
      opts.set_layout(
          task_options.layout() == OrientedObjectDetectorOptions::CHANNELS_LAST
              ? ::mediapipe::YoloObbTensorsToOrientedDetectionsCalculatorOptions::
                    CHANNELS_LAST
              : ::mediapipe::YoloObbTensorsToOrientedDetectionsCalculatorOptions::
                    CHANNELS_FIRST);
    }
    model_output_tensors >> obb_decode.In(kTensorTag);

    // Flatten batch (N==1).
    auto& flatten =
        graph.AddNode("YoloObbBatchDetectionsToSingleCalculator");
    obb_decode.Out(kOrientedDetectionsTag) >> flatten.In(kOrientedDetectionsTag);

    // Rotated NMS in model-input-normalized space (Group-1 calc, UNCHANGED).
    auto& nms = graph.AddNode("RotatedNonMaxSuppressionCalculator");
    {
      auto& nms_opts =
          nms.GetOptions<::mediapipe::RotatedNonMaxSuppressionCalculatorOptions>();
      nms_opts.set_iou_threshold(task_options.iou_threshold());
      nms_opts.set_max_detections(task_options.max_results());
      nms_opts.set_class_agnostic(task_options.class_agnostic_nms());
    }
    flatten.Out(kOrientedDetectionsTag) >> nms.In(kOrientedDetectionsTag);

    // Project to original-image-normalized coords.
    auto& projection =
        graph.AddNode("OrientedDetectionProjectionCalculator");
    nms.Out(kOrientedDetectionsTag) >> projection.In(kOrientedDetectionsTag);
    preprocessing.Out(kMatrixTag) >> projection.In(kProjectionMatrixTag);
    auto oriented_detections = projection.Out(kOrientedDetectionsTag);
```
Define the tag constants used (`kOrientedDetectionsTag = "ORIENTED_DETECTIONS"`, `kProjectionMatrixTag = "PROJECTION_MATRIX"`; reuse `kTensorTag`, `kMatrixTag`, `kImageTag`, `kImageSizeTag`, `kNormRectTag` from the sibling). Output streams: `ORIENTED_DETECTIONS` (the `oriented_detections` source) and `IMAGE` (preprocessing image passthrough). Confirm the exact `RotatedNonMaxSuppressionCalculatorOptions` field names against `mediapipe/calculators/util/rotated_non_max_suppression_calculator.proto` (`iou_threshold`, `max_detections`, `class_agnostic`).

- [ ] **Step 3: Write `BUILD`** — copy 2.1a's `yolo_object_detector_graph` deps, swap: remove `yolo_tensors_to_detections_calculator*`; add `//mediapipe/calculators/tensor:yolo_obb_tensors_to_oriented_detections_calculator` (+_cc_proto), `:yolo_obb_batch_detections_to_single_calculator`, `//mediapipe/calculators/util:rotated_non_max_suppression_calculator` (+_cc_proto), `//mediapipe/calculators/util:oriented_detection_projection_calculator`, and `//mediapipe/tasks/cc/vision/oriented_object_detector/proto:oriented_object_detector_options_cc_proto`. Keep model_task_graph, image_preprocessing_graph, AddInference deps, framework deps. `alwayslink = 1`.

- [ ] **Step 4: Build (compile-verify; no model).**

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_graph
```
Expected: success. Fix compile/dep errors by mirroring the sibling.

- [ ] **Step 5: Commit.**

```bash
git add mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_graph.cc mediapipe/tasks/cc/vision/oriented_object_detector/BUILD
git commit -m "feat(tasks-obb): OrientedObjectDetectorGraph (decode->flatten->rotated NMS->project)"
```

---

### Task 6: `OrientedObjectDetector` Task class (build-verify)

Mirror 2.1a's `yolo_object_detector.{h,cc}`; the key difference is the result type + converting with image dimensions.

**Files:**
- Create: `oriented_object_detector.h`, `oriented_object_detector.cc`
- Modify: `BUILD`

- [ ] **Step 1: Read** `yolo_object_detector.{h,cc}` fully.

- [ ] **Step 2: Write `oriented_object_detector.h`** by copying `yolo_object_detector.h` and renaming to `OrientedObjectDetector`/`OrientedObjectDetectorOptions` (namespace `mediapipe::tasks::vision::oriented_object_detector`). Options struct = the proto fields (base_options, running_mode, max_results, score_threshold, iou_threshold, class_agnostic_nms, layout, num_classes, result_callback) — **no** display_names_locale/allowlist/denylist. Result alias: `using OrientedObjectDetectorResult = ::mediapipe::tasks::components::containers::OrientedObjectDetectionResult;`.

- [ ] **Step 3: Write `oriented_object_detector.cc`** by copying `yolo_object_detector.cc` with the renames, plus the two real differences:
  1. Graph type name → `"mediapipe.tasks.vision.oriented_object_detector.OrientedObjectDetectorGraph"`. Output stream tag `ORIENTED_DETECTIONS` (packet type `std::vector<OrientedDetection>`).
  2. Result conversion: in the packet handler, read the original image dimensions from the `IMAGE` output packet (`image.width()`, `image.height()`) — the same `IMAGE` packet the sibling already plumbs — and call `ConvertToOrientedObjectDetectionResult(packet.Get<std::vector<OrientedDetection>>(), {width, height})`. (This satisfies the spec's "graph surfaces image size to the Task": the size is read from the already-plumbed `IMAGE` output, so no separate `IMAGE_SIZE` stream is needed.)

> Implementer note: the sibling converts a pixel-`DetectionResult` directly via `ConvertToDetectionResult` with no size. Here the detections packet is **normalized** and conversion needs the image size — so the conversion must run where both the detections packet and the image are available (the result callback / sync path that already receives both). Wire it there.

- [ ] **Step 4: Add `cc_library` targets** to `BUILD`, mirroring 2.1a's `yolo_object_detector` lib, swapping: graph dep → `:oriented_object_detector_graph`; options proto → `.../oriented_object_detector/proto:oriented_object_detector_options_cc_proto`; result container dep → `//mediapipe/tasks/cc/components/containers:oriented_object_detection_result` (instead of `:detection_result`); add `//mediapipe/framework/formats:oriented_detection_cc_proto`.

- [ ] **Step 5: Build.**

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector
```
Expected: success. Fix by mirroring the sibling.

- [ ] **Step 6: Commit.**

```bash
git add mediapipe/tasks/cc/vision/oriented_object_detector/
git commit -m "feat(tasks-obb): OrientedObjectDetector Task class"
```

---

### Task 7: Gated integration test

**Files:**
- Create: `oriented_object_detector_test.cc`
- Modify: `BUILD`

- [ ] **Step 1: Recon** `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc` (the gated-skip pattern + the existing test image). Confirm no `yolov8n-obb.tflite` exists.

- [ ] **Step 2: Write the test** mirroring 2.1a's gated test: `GTEST_SKIP()` (via `mediapipe::file::Exists`) when `mediapipe/tasks/testdata/vision/yolov8n-obb.tflite` is absent. Real assertions (when the fixture lands): `Create` an `OrientedObjectDetector`, `Detect` on the existing `cats_and_dogs.jpg` test image, `EXPECT_FALSE(result.detections.empty())`, and per-detection sanity (positive width/height, finite rotation). Add `DetectForVideo` in the same gated style.

- [ ] **Step 3: Add the `cc_test`** to `BUILD`: real `data` dep on the existing test image (`//mediapipe/tasks/testdata/vision:test_images`); a **commented** `data` line for `yolov8n-obb.tflite` (add when the fixture lands — do NOT reference a nonexistent target); deps `:oriented_object_detector`, file_helpers, gtest_main, status_matchers, image/format deps.

- [ ] **Step 4: Build + run.**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test --test_output=all
```
Expected: target BUILDS; test RUNS and reports **SKIPPED** (fixture absent). Do NOT fake a pass or add a fake model.

- [ ] **Step 5: Commit.**

```bash
git add mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_test.cc mediapipe/tasks/cc/vision/oriented_object_detector/BUILD
git commit -m "test(tasks-obb): OrientedObjectDetector integration test (gated on model fixture)"
```

---

## Done criteria (2.1b)

- `oriented_object_detection_result_test` (1), `yolo_obb_batch_detections_to_single_calculator_test` (3), and `oriented_detection_projection_calculator_test` (2) are **green** — the locally-verifiable TDD core.
- `:oriented_object_detector_options_cc_proto`, `:oriented_object_detector_graph`, `:oriented_object_detector` all **build**.
- Integration test builds and **skips** (gated on `yolov8n-obb.tflite`).
- `ObjectDetector`, `YoloObjectDetector`, and the Group-1 `RotatedNonMaxSuppressionCalculator` are all **untouched**.

## Handoff

With 2.1a + 2.1b done, Phase 2's cc layer exposes both axis-aligned and oriented YOLO detection. Next Phase-2 follow-ons (own specs): 2.2 Python, 2.3 Web, 2.4 Java, 2.5 iOS bindings — each wrapping these cc Tasks. Full `Detect()` validation for both Tasks awaits the `yolov8n.tflite` / `yolov8n-obb.tflite` fixtures.
