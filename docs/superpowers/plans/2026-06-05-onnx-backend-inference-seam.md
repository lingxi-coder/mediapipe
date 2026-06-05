# ONNX Backend Inference Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `OnnxInferenceCalculator` (ONNX Runtime, CPU, float32) that reuses MediaPipe's existing `InferenceRunner` + `InferenceMetadata` seam, and prove it runs the real YOLO and OBB Tasks detectors against their existing oracles.

**Architecture:** A new api2 calculator `OnnxInferenceCalculator` backed by `OnnxInferenceRunner : InferenceRunner`. The shared `inference_calculator.{h,cc,proto}` are **not** modified. ONNX-specific NCHW↔NHWC layout adaptation is confined to the runner. The YOLO/OBB Tasks-detector graph builders gain an additive, default-off `backend` option that swaps the TFLite `AddInference(...)` for an `OnnxInferenceCalculator` node. onnxruntime is wired on macOS via a Homebrew `new_local_repository` cc_library wrap, mirroring the existing OpenCV precedent.

**Tech Stack:** Bazel (Bzlmod + legacy WORKSPACE), C++20, MediaPipe api2 calculators, ONNX Runtime C++ API (Homebrew `/opt/homebrew/opt/onnxruntime`, v1.24.3), ultralytics (fixture export).

**Spec:** `docs/superpowers/specs/2026-06-05-onnx-backend-inference-seam-design.md`

**Scope clarification (vs spec §1 wording):** the e2e proof runs through the **Tasks single-image detector graphs** (`YoloObjectDetector` on `cats_and_dogs.jpg`, `OrientedObjectDetector` on `boats.jpg`) — these are "the real YOLO/OBB detector" with existing oracles. Wiring ONNX through the *tiled* `StreamingTilesToTensorBatch` calculator pipeline is a deliberate later increment; the runner's `GetModelMetadata()` is still implemented and unit-tested here so the seam is ready for the tiler.

**Conventions for every commit in this plan:** end the commit message body with the trailer
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Work happens on branch `dev`.

**How to verify (this machine):** desktop CPU C++ only — prefix every bazel command with
`--define MEDIAPIPE_DISABLE_GPU=1`. The onnxruntime targets are macOS-only and build from the Homebrew
prefix; no extra config flags are needed beyond `MEDIAPIPE_DISABLE_GPU=1`.

---

## File Structure

**New files:**
- `third_party/onnxruntime_macos.BUILD` — cc_library wrapping the Homebrew onnxruntime dylib + headers.
- `mediapipe/calculators/tensor/onnx_inference_runner.h` — `OnnxInferenceRunner : InferenceRunner` decl.
- `mediapipe/calculators/tensor/onnx_inference_runner.cc` — session, `Run`, `GetModelMetadata`, `GetInputOutputTensorNames`.
- `mediapipe/calculators/tensor/onnx_inference_runner_test.cc` — runner unit test.
- `mediapipe/calculators/tensor/onnx_inference_calculator.proto` — `OnnxInferenceCalculatorOptions`.
- `mediapipe/calculators/tensor/onnx_inference_calculator.cc` — the api2 calculator.
- `mediapipe/calculators/tensor/onnx_inference_calculator_test.cc` — standalone calculator test.
- `mediapipe/calculators/tensor/onnxruntime_link_smoke_test.cc` — verifies the dep links.
- `mediapipe/tasks/testdata/vision/export_yolov8n_onnx.py` — fixture export script.

**Modified files:**
- `WORKSPACE` — add `macos_onnxruntime` `new_local_repository`.
- `mediapipe/calculators/tensor/BUILD` — proto_library + cc_library + cc_test targets (macOS-gated).
- `mediapipe/tasks/testdata/vision/BUILD` — add `*.onnx` to the `yolo_test_model` / `yolo_obb_test_model` globs.
- `mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.proto` — `Backend` msg + field 10.
- `mediapipe/tasks/cc/vision/oriented_object_detector/proto/oriented_object_detector_options.proto` — `Backend` msg + field 11.
- `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc` — ONNX branch.
- `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_graph.cc` — ONNX branch.
- `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.{h,cc}` — `onnx_model_path` public option + conversion.
- `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.{h,cc}` — `onnx_model_path` public option + conversion.
- `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc` — ONNX e2e test.
- `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_test.cc` — ONNX e2e test.
- `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD` and `.../oriented_object_detector/BUILD` — deps + data.

---

## Task 1: Wire onnxruntime into Bazel (macOS) + link smoke test

**Files:**
- Create: `third_party/onnxruntime_macos.BUILD`
- Create: `mediapipe/calculators/tensor/onnxruntime_link_smoke_test.cc`
- Modify: `WORKSPACE` (add repository near the `macos_opencv` block, ~line 534)
- Modify: `mediapipe/calculators/tensor/BUILD` (add the smoke-test target)

- [ ] **Step 1: Write the failing test**

Create `mediapipe/calculators/tensor/onnxruntime_link_smoke_test.cc`:

```cpp
// Verifies the Homebrew onnxruntime dependency compiles and links.
#include <string>

#include "mediapipe/framework/port/gtest.h"
#include "onnxruntime_cxx_api.h"  // from @macos_onnxruntime//:onnxruntime

namespace mediapipe {
namespace {

TEST(OnnxRuntimeLinkSmokeTest, VersionStringIsNonEmpty) {
  const std::string version = Ort::GetVersionString();
  EXPECT_FALSE(version.empty()) << "ONNX Runtime version string should be set";
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 2: Run the test to verify it fails (dependency missing)**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:onnxruntime_link_smoke_test`
Expected: FAIL — target does not exist yet / `onnxruntime_cxx_api.h` not found.

- [ ] **Step 3: Create the prebuilt-lib BUILD wrapper**

Create `third_party/onnxruntime_macos.BUILD`:

```python
# Description:
#   ONNX Runtime (CPU) for the OnnxInferenceCalculator on macOS, consumed from a
#   Homebrew install. Mirrors third_party/opencv_macos.BUILD.
#
# Setup: `brew install onnxruntime`. The version-independent Homebrew symlink
#   /opt/homebrew/opt/onnxruntime -> Cellar/onnxruntime/<version>
# means this needs no edit after `brew upgrade onnxruntime`.

load("@bazel_skylib//lib:paths.bzl", "paths")

licenses(["notice"])  # MIT

# Apple-Silicon Homebrew symlink prefix. Intel Homebrew would be "opt/onnxruntime"
# under "/usr/local" (set in the WORKSPACE new_local_repository path).
PREFIX = "opt/onnxruntime"

cc_library(
    name = "onnxruntime",
    srcs = glob([paths.join(PREFIX, "lib/libonnxruntime.dylib")]),
    hdrs = glob([paths.join(PREFIX, "include/onnxruntime/*.h")]),
    includes = [paths.join(PREFIX, "include/onnxruntime")],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
```

- [ ] **Step 4: Register the repository in WORKSPACE**

In `WORKSPACE`, immediately after the `macos_opencv` `new_local_repository(...)` block (~line 543), add:

```python
new_local_repository(
    name = "macos_onnxruntime",
    build_file = "@//third_party:onnxruntime_macos.BUILD",
    # Apple-Silicon Homebrew root; PREFIX "opt/onnxruntime" in the BUILD file
    # resolves through /opt/homebrew/opt/onnxruntime -> current Cellar version.
    # Intel Homebrew: "/usr/local".
    path = "/opt/homebrew",
)
```

- [ ] **Step 5: Add the smoke-test target**

In `mediapipe/calculators/tensor/BUILD`, append:

```python
cc_test(
    name = "onnxruntime_link_smoke_test",
    srcs = ["onnxruntime_link_smoke_test.cc"],
    tags = ["nomsan", "notsan"],
    deps = [
        "//mediapipe/framework/port:gtest_main",
    ] + select({
        "//mediapipe:macos": ["@macos_onnxruntime//:onnxruntime"],
        "//conditions:default": [],
    }),
)
```

Note: the `select` on `//mediapipe:macos` means non-macOS builds get no onnxruntime dep (the test then
has no header and would fail to compile there — that's acceptable because this target is only built on
macOS in this fork; if the config_setting `//mediapipe:macos` does not exist, use `//mediapipe:apple`
and confirm with `bazel query //mediapipe:macos`).

- [ ] **Step 6: Run the test to verify it passes**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:onnxruntime_link_smoke_test`
Expected: PASS (1 test). If linking fails with a missing dylib, confirm `brew --prefix onnxruntime`
points at `/opt/homebrew/opt/onnxruntime` and that `libonnxruntime.dylib` exists there.

- [ ] **Step 7: Commit**

```bash
git add third_party/onnxruntime_macos.BUILD WORKSPACE \
        mediapipe/calculators/tensor/onnxruntime_link_smoke_test.cc \
        mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'build(onnx): wire Homebrew onnxruntime via new_local_repository + link smoke test\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 2: ONNX fixture export script + testdata BUILD globs

**Files:**
- Create: `mediapipe/tasks/testdata/vision/export_yolov8n_onnx.py`
- Modify: `mediapipe/tasks/testdata/vision/BUILD:190-220` (the `yolo_test_model` and `yolo_obb_test_model` globs)

This task makes the `.onnx` fixtures buildable when present (and gracefully absent otherwise), exactly like
the existing `.tflite` fixtures. The fixtures themselves are gitignored.

- [ ] **Step 1: Create the export script**

Create `mediapipe/tasks/testdata/vision/export_yolov8n_onnx.py`:

```python
#!/usr/bin/env python3
"""Exports yolov8n (COCO) and yolov8n-obb (DOTA) to ONNX for the ONNX backend tests.

Mirrors export_yolov8n_tflite.py / export_yolov8n_obb_tflite.py. The produced
files are gitignored; they enable the (otherwise skipped) ONNX backend tests.

Usage:
    pip install ultralytics
    python3 export_yolov8n_onnx.py

Produces, alongside this script:
    yolov8n.onnx        (COCO, 80 classes, input [1,3,640,640] NCHW)
    yolov8n-obb.onnx    (DOTA, 15 classes + angle, input [1,3,640,640] NCHW)

Opset 17 is broadly supported by onnxruntime 1.24; bump only if export fails.
"""
from ultralytics import YOLO

for weights in ("yolov8n.pt", "yolov8n-obb.pt"):
    YOLO(weights).export(format="onnx", opset=17, imgsz=640, simplify=True)
```

- [ ] **Step 2: Extend the testdata globs**

In `mediapipe/tasks/testdata/vision/BUILD`, add `"yolov8n.onnx"` to the `yolo_test_model` filegroup glob
and `"yolov8n-obb.onnx"` to the `yolo_obb_test_model` filegroup glob. After editing, the two filegroups
read:

```python
filegroup(
    name = "yolo_test_model",
    srcs = glob(
        [
            "yolov8n.tflite",
            "yolov8n.onnx",
            "yolov8n_labels.txt",
        ],
        allow_empty = True,
    ),
)
```

```python
filegroup(
    name = "yolo_obb_test_model",
    srcs = glob(
        [
            "yolov8n-obb.tflite",
            "yolov8n-obb.onnx",
            "yolov8n_obb_labels.txt",
            "boats.jpg",
        ],
        allow_empty = True,
    ),
    visibility = [
        "//mediapipe/calculators/tensor:__pkg__",
        "//mediapipe/tasks:internal",
    ],
)
```

- [ ] **Step 3: Verify the globs still build (fixtures may be absent)**

Run: `bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/testdata/vision:yolo_test_model //mediapipe/tasks/testdata/vision:yolo_obb_test_model`
Expected: PASS (the `allow_empty=True` glob succeeds whether or not the `.onnx` blobs are present).

- [ ] **Step 4: (Optional, on this machine) generate the real fixtures so later tests actually run**

Run: `cd mediapipe/tasks/testdata/vision && python3 export_yolov8n_onnx.py`
Expected: `yolov8n.onnx` and `yolov8n-obb.onnx` appear in that directory. If ultralytics is unavailable,
skip — downstream tests will `GTEST_SKIP()` until the fixtures exist.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/tasks/testdata/vision/export_yolov8n_onnx.py \
        mediapipe/tasks/testdata/vision/BUILD
git commit -m "$(printf 'test(onnx): yolov8n/-obb ONNX export script + testdata globs (gitignored fixtures)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 3: `OnnxInferenceRunner` — construction + metadata introspection

**Files:**
- Create: `mediapipe/calculators/tensor/onnx_inference_runner.h`
- Create: `mediapipe/calculators/tensor/onnx_inference_runner.cc`
- Create: `mediapipe/calculators/tensor/onnx_inference_runner_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD` (cc_library + cc_test, macOS-gated)

The runner owns the `Ort::Session`, caches I/O names/specs, and implements the `InferenceRunner`
introspection methods. `Run()` is added in Task 4.

- [ ] **Step 1: Write the header**

Create `mediapipe/calculators/tensor/onnx_inference_runner.h`:

```cpp
#ifndef MEDIAPIPE_CALCULATORS_TENSOR_ONNX_INFERENCE_RUNNER_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_ONNX_INFERENCE_RUNNER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"  // InputOutputTensorNames
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "onnxruntime_cxx_api.h"

namespace mediapipe {

// InferenceRunner backed by ONNX Runtime (CPU execution provider, float32).
// v1 supports a single float32 input/output set; the detector runs N==1.
//
// The model is loaded from a path (typically a .onnx exported from ultralytics,
// whose input is NCHW [1,3,H,W]). GetModelMetadata() reports the logical NHWC
// geometry (tensor_layout "BHWC") so the tiler/preprocessing pack NHWC exactly
// as for TFLite; Run() transposes NHWC->NCHW internally. All ONNX-specific
// layout knowledge is confined to this class.
class OnnxInferenceRunner : public InferenceRunner {
 public:
  static absl::StatusOr<std::unique_ptr<OnnxInferenceRunner>> Create(
      const std::string& model_path, int intra_op_num_threads);

  absl::StatusOr<std::vector<Tensor>> Run(
      CalculatorContext* cc, const TensorSpan& tensor_span) override;

  const InputOutputTensorNames& GetInputOutputTensorNames() const override {
    return io_tensor_names_;
  }

  absl::StatusOr<InferenceMetadata> GetModelMetadata() const override;

 private:
  OnnxInferenceRunner() = default;

  // ONNX Runtime objects. env_ must outlive session_.
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;

  // Cached at Create(): I/O names (owned strings) + their C-string views.
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_names_c_;
  std::vector<const char*> output_names_c_;

  // Cached input[0] NCHW shape dims (batch, channels, height, width).
  std::vector<int64_t> input0_shape_;

  // InferenceRunner contract: signature -> input/output tensor names.
  InputOutputTensorNames io_tensor_names_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_ONNX_INFERENCE_RUNNER_H_
```

- [ ] **Step 2: Write the failing test (Create + GetModelMetadata)**

Create `mediapipe/calculators/tensor/onnx_inference_runner_test.cc`:

```cpp
#include "mediapipe/calculators/tensor/onnx_inference_runner.h"

#include <string>

#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::mediapipe::file::JoinPath;

constexpr char kTestDataDir[] = "/mediapipe/tasks/testdata/vision/";

std::string ModelPath() { return JoinPath("./", kTestDataDir, "yolov8n.onnx"); }

TEST(OnnxInferenceRunnerTest, MetadataReportsNhwcGeometry) {
  const std::string path = ModelPath();
  if (!mediapipe::file::Exists(path).ok()) {
    GTEST_SKIP() << "yolov8n.onnx fixture absent at " << path
                 << "; run export_yolov8n_onnx.py to enable.";
  }
  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          OnnxInferenceRunner::Create(path,
                                                      /*intra_op_num_threads=*/1));
  MP_ASSERT_OK_AND_ASSIGN(InferenceMetadata md, runner->GetModelMetadata());
  EXPECT_EQ(md.backend(), "onnx");
  EXPECT_EQ(md.tensor_layout(), "BHWC");
  EXPECT_EQ(md.input_height(), 640);
  EXPECT_EQ(md.input_width(), 640);
  EXPECT_EQ(md.input_channels(), 3);
  EXPECT_EQ(md.batch_capacity(), 1);
  EXPECT_GE(md.input_size(), 1);
  EXPECT_GE(md.output_size(), 1);
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:onnx_inference_runner_test`
Expected: FAIL — target/symbols not defined (link error: `OnnxInferenceRunner::Create` undefined).

- [ ] **Step 4: Implement Create + GetModelMetadata (Run is a stub for now)**

Create `mediapipe/calculators/tensor/onnx_inference_runner.cc`:

```cpp
#include "mediapipe/calculators/tensor/onnx_inference_runner.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "onnxruntime_cxx_api.h"

namespace mediapipe {

absl::StatusOr<std::unique_ptr<OnnxInferenceRunner>>
OnnxInferenceRunner::Create(const std::string& model_path,
                            int intra_op_num_threads) {
  auto runner = std::unique_ptr<OnnxInferenceRunner>(new OnnxInferenceRunner());
  try {
    runner->env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                               "mediapipe_onnx");
    Ort::SessionOptions opts;
    if (intra_op_num_threads > 0) {
      opts.SetIntraOpNumThreads(intra_op_num_threads);
    }
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    runner->session_ = std::make_unique<Ort::Session>(
        *runner->env_, model_path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t n_in = runner->session_->GetInputCount();
    const size_t n_out = runner->session_->GetOutputCount();
    RET_CHECK_GT(n_in, 0) << "ONNX model has no inputs: " << model_path;
    RET_CHECK_GT(n_out, 0) << "ONNX model has no outputs: " << model_path;

    for (size_t i = 0; i < n_in; ++i) {
      auto name = runner->session_->GetInputNameAllocated(i, allocator);
      runner->input_names_.emplace_back(name.get());
    }
    for (size_t i = 0; i < n_out; ++i) {
      auto name = runner->session_->GetOutputNameAllocated(i, allocator);
      runner->output_names_.emplace_back(name.get());
    }
    for (const auto& s : runner->input_names_)
      runner->input_names_c_.push_back(s.c_str());
    for (const auto& s : runner->output_names_)
      runner->output_names_c_.push_back(s.c_str());

    // Cache input[0] shape (NCHW for ultralytics exports).
    auto type_info = runner->session_->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    runner->input0_shape_ = tensor_info.GetShape();
    RET_CHECK_EQ(runner->input0_shape_.size(), 4u)
        << "v1 expects a 4-D NCHW input; got rank "
        << runner->input0_shape_.size();

    // InferenceRunner I/O-name contract: single default signature. The
    // standalone OnnxInferenceCalculator does not run the InferenceIoMapper in
    // v1, so the signature key is not load-bearing; "" denotes the default.
    SignatureInputOutputTensorNames sig;
    sig.input_tensor_names = runner->input_names_;
    sig.output_tensor_names = runner->output_names_;
    runner->io_tensor_names_[""] = std::move(sig);
  } catch (const Ort::Exception& e) {
    return absl::InternalError(
        absl::StrCat("Failed to create ONNX session for '", model_path,
                     "': ", e.what()));
  }
  return runner;
}

absl::StatusOr<InferenceMetadata> OnnxInferenceRunner::GetModelMetadata()
    const {
  InferenceMetadata md;
  md.set_backend("onnx");
  // ultralytics export is NCHW [N, C, H, W]; report logical NHWC geometry so the
  // tiler/preprocessing pack the same layout as for TFLite. Run() transposes.
  const int64_t n = input0_shape_[0];
  const int64_t c = input0_shape_[1];
  const int64_t h = input0_shape_[2];
  const int64_t w = input0_shape_[3];
  md.set_tensor_layout("BHWC");
  md.set_input_channels(static_cast<int>(c));
  md.set_input_height(static_cast<int>(h));
  md.set_input_width(static_cast<int>(w));
  md.set_is_dynamic_batch(n <= 0);
  md.set_batch_capacity(n > 0 ? static_cast<int>(n) : 1);

  try {
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < input_names_.size(); ++i) {
      auto* spec = md.add_input();
      spec->set_name(input_names_[i]);
      spec->set_dtype("float32");
      auto info = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
      for (int64_t d : info.GetShape()) spec->add_shape(static_cast<int>(d));
    }
    for (size_t i = 0; i < output_names_.size(); ++i) {
      auto* spec = md.add_output();
      spec->set_name(output_names_[i]);
      spec->set_dtype("float32");
      auto info = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
      for (int64_t d : info.GetShape()) spec->add_shape(static_cast<int>(d));
    }
  } catch (const Ort::Exception& e) {
    return absl::InternalError(
        absl::StrCat("ONNX metadata introspection failed: ", e.what()));
  }
  return md;
}

absl::StatusOr<std::vector<Tensor>> OnnxInferenceRunner::Run(
    CalculatorContext* cc, const TensorSpan& tensor_span) {
  return absl::UnimplementedError("OnnxInferenceRunner::Run added in Task 4");
}

}  // namespace mediapipe
```

Note: `SignatureInputOutputTensorNames` (fields `input_tensor_names` / `output_tensor_names`) and
`SignatureName` (= `std::string`) come from `mediapipe/util/tflite/tflite_signature_reader.h` (included
transitively via `inference_io_mapper.h`). There is **no** exported default-signature-key constant; the
`""` key above is fine because the standalone calculator does not consume IO-mapping in v1.

- [ ] **Step 5: Add BUILD targets (macOS-gated)**

In `mediapipe/calculators/tensor/BUILD`, add:

```python
cc_library(
    name = "onnx_inference_runner",
    srcs = select({
        "//mediapipe:macos": ["onnx_inference_runner.cc"],
        "//conditions:default": [],
    }),
    hdrs = select({
        "//mediapipe:macos": ["onnx_inference_runner.h"],
        "//conditions:default": [],
    }),
    deps = select({
        "//mediapipe:macos": [
            ":inference_io_mapper",
            ":inference_runner",
            ":tensor_span",
            "//mediapipe/framework:calculator_context",
            "//mediapipe/framework/formats:inference_metadata_cc_proto",
            "//mediapipe/framework/formats:tensor",
            "//mediapipe/framework/port:ret_check",
            "//mediapipe/framework/port:status",
            "//mediapipe/framework/port:statusor",
            "@com_google_absl//absl/status",
            "@com_google_absl//absl/status:statusor",
            "@com_google_absl//absl/strings",
            "@macos_onnxruntime//:onnxruntime",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "onnx_inference_runner_test",
    srcs = ["onnx_inference_runner_test.cc"],
    data = ["//mediapipe/tasks/testdata/vision:yolo_test_model"],
    tags = ["nomsan", "notsan"],
    deps = [
        "//mediapipe/framework/port:gtest_main",
    ] + select({
        "//mediapipe:macos": [
            ":onnx_inference_runner",
            "//mediapipe/framework/deps:file_path",
            "//mediapipe/framework/formats:inference_metadata_cc_proto",
            "//mediapipe/framework/port:file_helpers",
            "//mediapipe/framework/port:status_matchers",
        ],
        "//conditions:default": [],
    }),
)
```

Confirm the proto target name `//mediapipe/framework/formats:inference_metadata_cc_proto` by grepping the
formats BUILD (the generated cc_proto target for `inference_metadata.proto`); adjust if it differs.

- [ ] **Step 6: Run the test to verify it passes (or skips cleanly)**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:onnx_inference_runner_test`
Expected: PASS — with the `yolov8n.onnx` fixture present, the metadata assertions run; without it the test
`GTEST_SKIP()`s. Either way the target must build and the test binary run.

- [ ] **Step 7: Commit**

```bash
git add mediapipe/calculators/tensor/onnx_inference_runner.h \
        mediapipe/calculators/tensor/onnx_inference_runner.cc \
        mediapipe/calculators/tensor/onnx_inference_runner_test.cc \
        mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'feat(onnx): OnnxInferenceRunner Create + GetModelMetadata (NHWC-reported, NCHW model)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 4: `OnnxInferenceRunner::Run` — NHWC→NCHW transpose + inference

**Files:**
- Modify: `mediapipe/calculators/tensor/onnx_inference_runner.cc` (replace the `Run` stub)
- Modify: `mediapipe/calculators/tensor/onnx_inference_runner_test.cc` (add a Run test)

- [ ] **Step 1: Add the failing Run test**

Append to `mediapipe/calculators/tensor/onnx_inference_runner_test.cc` (inside the anonymous namespace),
adding includes `#include <vector>`, `#include "mediapipe/calculators/tensor/tensor_span.h"`,
`#include "mediapipe/framework/formats/tensor.h"`:

```cpp
TEST(OnnxInferenceRunnerTest, RunProducesDetectionOutput) {
  const std::string path = ModelPath();
  if (!mediapipe::file::Exists(path).ok()) {
    GTEST_SKIP() << "yolov8n.onnx fixture absent.";
  }
  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          OnnxInferenceRunner::Create(path, 1));

  // One NHWC [1,640,640,3] float32 input filled with 0.5 (mid-gray).
  std::vector<Tensor> inputs;
  inputs.emplace_back(Tensor::ElementType::kFloat32,
                      Tensor::Shape{1, 640, 640, 3});
  {
    auto view = inputs[0].GetCpuWriteView();
    float* p = view.buffer<float>();
    for (int i = 0; i < 1 * 640 * 640 * 3; ++i) p[i] = 0.5f;
  }
  TensorSpan span = MakeTensorSpan(inputs);

  MP_ASSERT_OK_AND_ASSIGN(auto outputs, runner->Run(/*cc=*/nullptr, span));
  ASSERT_EQ(outputs.size(), 1u);
  // yolov8n COCO detect head: [1, 84, 8400] (4 box + 80 classes, channels-first).
  const auto& dims = outputs[0].shape().dims;
  ASSERT_EQ(dims.size(), 3u);
  EXPECT_EQ(dims[0], 1);
  EXPECT_EQ(dims[1], 84);
  EXPECT_EQ(dims[2], 8400);
}
```

`MakeTensorSpan` is declared in `mediapipe/calculators/tensor/tensor_span.h` (the same helper
`InferenceCalculator` uses). The runner test BUILD target already depends on `:onnx_inference_runner`
which pulls in `:tensor_span`; add `"//mediapipe/framework/formats:tensor"` and `":tensor_span"` to the
macOS `select` deps of `onnx_inference_runner_test` if the test does not yet see them.

- [ ] **Step 2: Run the test to verify it fails**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:onnx_inference_runner_test`
Expected: FAIL — `Run` returns `UnimplementedError` (test asserts OK).

- [ ] **Step 3: Implement Run (replace the stub)**

In `mediapipe/calculators/tensor/onnx_inference_runner.cc`, add `#include "absl/strings/str_format.h"` and
replace the `Run` stub with:

```cpp
namespace {

// Transpose a single-batch NHWC [1,H,W,C] float buffer to NCHW [1,C,H,W].
void NhwcToNchw(const float* src, int h, int w, int c, float* dst) {
  for (int ch = 0; ch < c; ++ch) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        dst[(ch * h + y) * w + x] = src[(y * w + x) * c + ch];
      }
    }
  }
}

}  // namespace

absl::StatusOr<std::vector<Tensor>> OnnxInferenceRunner::Run(
    CalculatorContext* /*cc*/, const TensorSpan& tensor_span) {
  RET_CHECK_EQ(tensor_span.size(), 1)
      << "OnnxInferenceRunner v1 supports exactly one input tensor; got "
      << tensor_span.size();
  const Tensor& in = tensor_span[0];
  RET_CHECK(in.element_type() == Tensor::ElementType::kFloat32)
      << "OnnxInferenceRunner v1 supports float32 input only.";
  const auto& dims = in.shape().dims;
  RET_CHECK_EQ(dims.size(), 4u) << "Expected NHWC [1,H,W,C] input.";
  const int n = dims[0], h = dims[1], w = dims[2], c = dims[3];
  RET_CHECK_EQ(n, 1) << "OnnxInferenceRunner v1 supports batch N==1; got " << n;
  RET_CHECK_EQ(c, static_cast<int>(input0_shape_[1]))
      << "Input channels " << c << " != model channels " << input0_shape_[1];

  std::vector<Tensor> outputs;
  try {
    // NHWC -> NCHW into a contiguous staging buffer.
    std::vector<float> nchw(static_cast<size_t>(n) * c * h * w);
    {
      auto rv = in.GetCpuReadView();
      NhwcToNchw(rv.buffer<float>(), h, w, c, nchw.data());
    }
    const std::array<int64_t, 4> in_shape = {1, c, h, w};
    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_value = Ort::Value::CreateTensor<float>(
        mem, nchw.data(), nchw.size(), in_shape.data(), in_shape.size());

    auto ort_outputs = session_->Run(
        Ort::RunOptions{nullptr}, input_names_c_.data(), &input_value,
        /*input_count=*/1, output_names_c_.data(), output_names_c_.size());

    for (auto& val : ort_outputs) {
      auto info = val.GetTensorTypeAndShapeInfo();
      std::vector<int> shape;
      for (int64_t d : info.GetShape()) shape.push_back(static_cast<int>(d));
      Tensor out(Tensor::ElementType::kFloat32, Tensor::Shape{shape});
      const float* src = val.GetTensorMutableData<float>();
      auto wv = out.GetCpuWriteView();
      std::copy(src, src + out.shape().num_elements(), wv.buffer<float>());
      outputs.push_back(std::move(out));
    }
  } catch (const Ort::Exception& e) {
    return absl::InternalError(
        absl::StrFormat("ONNX inference failed: %s", e.what()));
  }
  return outputs;
}
```

Add `#include <algorithm>` and `#include <array>` to the file's includes.

- [ ] **Step 4: Run the test to verify it passes (or skips)**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:onnx_inference_runner_test`
Expected: PASS (or SKIP if the fixture is absent). With the fixture, both runner tests pass; the output
shape `[1,84,8400]` confirms the NCHW transpose fed the model correctly.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/onnx_inference_runner.cc \
        mediapipe/calculators/tensor/onnx_inference_runner_test.cc \
        mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'feat(onnx): OnnxInferenceRunner::Run — NHWC->NCHW transpose + CPU inference\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 5: `OnnxInferenceCalculator` + options proto

**Files:**
- Create: `mediapipe/calculators/tensor/onnx_inference_calculator.proto`
- Create: `mediapipe/calculators/tensor/onnx_inference_calculator.cc`
- Create: `mediapipe/calculators/tensor/onnx_inference_calculator_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD` (proto_library + cc_library + cc_test, macOS-gated)

A standalone api2 calculator: `TENSORS` in/out + optional `METADATA` side output. Reuses
`OnnxInferenceRunner`. Not routed through the `InferenceCalculator` selector.

- [ ] **Step 1: Write the options proto**

Create `mediapipe/calculators/tensor/onnx_inference_calculator.proto`:

```proto
syntax = "proto2";

package mediapipe;

import "mediapipe/framework/calculator.proto";
import "mediapipe/framework/calculator_options.proto";

message OnnxInferenceCalculatorOptions {
  extend mediapipe.CalculatorOptions {
    optional OnnxInferenceCalculatorOptions ext = 471230020;
  }
  // Path to the .onnx model.
  optional string model_path = 1;
  // ONNX Runtime intra-op thread count. <= 0 leaves the ORT default.
  optional int32 intra_op_num_threads = 2 [default = -1];
}
```

(The extension field number `471230020` is in the same private range the fork's detector options use —
`471230010`/`471230011`. If a registry collision is reported at build time, pick another unused number.)

- [ ] **Step 2: Write the failing calculator test**

Create `mediapipe/calculators/tensor/onnx_inference_calculator_test.cc`:

```cpp
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::mediapipe::file::JoinPath;

constexpr char kTestDataDir[] = "/mediapipe/tasks/testdata/vision/";
std::string ModelPath() { return JoinPath("./", kTestDataDir, "yolov8n.onnx"); }

TEST(OnnxInferenceCalculatorTest, RunsAndEmitsMetadata) {
  const std::string model_path = ModelPath();
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "yolov8n.onnx fixture absent.";
  }
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(absl::StrCat(R"(
    input_stream: "tensors"
    output_stream: "out"
    output_side_packet: "metadata"
    node {
      calculator: "OnnxInferenceCalculator"
      input_stream: "TENSORS:tensors"
      output_stream: "TENSORS:out"
      output_side_packet: "METADATA:metadata"
      options {
        [mediapipe.OnnxInferenceCalculatorOptions.ext] {
          model_path: ")", model_path, R"("
          intra_op_num_threads: 1
        }
      }
    }
  )"));

  std::vector<Packet> out_packets;
  tool::AddVectorSink("out", &config, &out_packets);
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  auto inputs = std::make_unique<std::vector<Tensor>>();
  inputs->emplace_back(Tensor::ElementType::kFloat32,
                       Tensor::Shape{1, 640, 640, 3});
  {
    auto wv = (*inputs)[0].GetCpuWriteView();
    float* p = wv.buffer<float>();
    for (int i = 0; i < 1 * 640 * 640 * 3; ++i) p[i] = 0.5f;
  }
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tensors", Adopt(inputs.release()).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseInputStream("tensors"));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  ASSERT_EQ(out_packets.size(), 1u);
  const auto& outs = out_packets[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(outs.size(), 1u);
  EXPECT_EQ(outs[0].shape().dims[1], 84);

  MP_ASSERT_OK_AND_ASSIGN(Packet md_packet,
                          graph.GetOutputSidePacket("metadata"));
  EXPECT_EQ(md_packet.Get<InferenceMetadata>().backend(), "onnx");
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace
}  // namespace mediapipe
```

(Confirm `tool::AddVectorSink` / `ParseTextProtoOrDie` include paths against a sibling tensor calculator
test such as `inference_calculator_test.cc`; add `#include "mediapipe/framework/tool/sink.h"` and
`#include "absl/strings/str_cat.h"` as needed.)

- [ ] **Step 3: Run the test to verify it fails**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:onnx_inference_calculator_test`
Expected: FAIL — `OnnxInferenceCalculator` not registered.

- [ ] **Step 4: Implement the calculator**

Create `mediapipe/calculators/tensor/onnx_inference_calculator.cc`:

```cpp
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/onnx_inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/onnx_inference_runner.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe {
namespace api2 {

// Standalone ONNX-Runtime inference node. Drop-in for InferenceCalculator's
// vector<Tensor> I/O, backed by OnnxInferenceRunner. CPU/float32, N==1.
class OnnxInferenceCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  static constexpr Output<std::vector<Tensor>> kOutTensors{"TENSORS"};
  static constexpr SideOutput<InferenceMetadata>::Optional kSideOutMetadata{
      "METADATA"};
  MEDIAPIPE_NODE_CONTRACT(kInTensors, kOutTensors, kSideOutMetadata);

  absl::Status Open(CalculatorContext* cc) override {
    const auto& options =
        cc->Options<mediapipe::OnnxInferenceCalculatorOptions>();
    RET_CHECK(!options.model_path().empty())
        << "OnnxInferenceCalculatorOptions.model_path is required.";
    MP_ASSIGN_OR_RETURN(
        runner_, OnnxInferenceRunner::Create(options.model_path(),
                                             options.intra_op_num_threads()));
    if (kSideOutMetadata(cc).IsConnected()) {
      MP_ASSIGN_OR_RETURN(InferenceMetadata md, runner_->GetModelMetadata());
      kSideOutMetadata(cc).Set(md);
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (kInTensors(cc).IsEmpty()) return absl::OkStatus();
    const auto& inputs = *kInTensors(cc);
    RET_CHECK(!inputs.empty());
    MP_ASSIGN_OR_RETURN(std::vector<Tensor> outputs,
                        runner_->Run(cc, MakeTensorSpan(inputs)));
    kOutTensors(cc).Send(std::move(outputs));
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<OnnxInferenceRunner> runner_;
};

MEDIAPIPE_REGISTER_NODE(OnnxInferenceCalculator);

}  // namespace api2
}  // namespace mediapipe
```

Confirm `MakeTensorSpan(const std::vector<Tensor>&)` is the overload exposed by `tensor_span.h` (it is the
one `InferenceCalculatorNodeImpl::Process` uses); if `SideOutput<...>::Optional` and `.Set(...)` differ in
this api2 version, mirror the exact form used by `InferenceCalculator::kSideOutMetadata` in
`inference_calculator.h` and its CPU impl's emission site.

- [ ] **Step 5: Add BUILD targets**

In `mediapipe/calculators/tensor/BUILD`, add a proto library, the calculator cc_library, and the test:

```python
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

mediapipe_proto_library(
    name = "onnx_inference_calculator_proto",
    srcs = ["onnx_inference_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "onnx_inference_calculator",
    srcs = select({
        "//mediapipe:macos": ["onnx_inference_calculator.cc"],
        "//conditions:default": [],
    }),
    deps = [
        ":onnx_inference_calculator_cc_proto",
    ] + select({
        "//mediapipe:macos": [
            ":onnx_inference_runner",
            ":tensor_span",
            "//mediapipe/framework/api2:node",
            "//mediapipe/framework/api2:port",
            "//mediapipe/framework/formats:inference_metadata_cc_proto",
            "//mediapipe/framework/formats:tensor",
            "//mediapipe/framework/port:ret_check",
            "//mediapipe/framework/port:status",
            "@com_google_absl//absl/status",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_test(
    name = "onnx_inference_calculator_test",
    srcs = ["onnx_inference_calculator_test.cc"],
    data = ["//mediapipe/tasks/testdata/vision:yolo_test_model"],
    tags = ["nomsan", "notsan"],
    deps = [
        "//mediapipe/framework/port:gtest_main",
    ] + select({
        "//mediapipe:macos": [
            ":onnx_inference_calculator",
            ":onnx_inference_calculator_cc_proto",
            "//mediapipe/framework:calculator_framework",
            "//mediapipe/framework:calculator_runner",
            "//mediapipe/framework/deps:file_path",
            "//mediapipe/framework/formats:inference_metadata_cc_proto",
            "//mediapipe/framework/formats:tensor",
            "//mediapipe/framework/port:file_helpers",
            "//mediapipe/framework/port:parse_text_proto",
            "//mediapipe/framework/port:status_matchers",
            "//mediapipe/framework/tool:sink",
            "@com_google_absl//absl/strings",
        ],
        "//conditions:default": [],
    }),
)
```

(`mediapipe_proto_library` generates both `:onnx_inference_calculator_proto` and the
`:onnx_inference_calculator_cc_proto` target referenced above — confirm the naming against an existing
`mediapipe_proto_library` in this BUILD, e.g. the one for `inference_calculator.proto`.)

- [ ] **Step 6: Run the test to verify it passes (or skips)**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:onnx_inference_calculator_test`
Expected: PASS (or SKIP without the fixture). The calculator runs, emits one `TENSORS` packet of shape
`[1,84,8400]`, and emits the `METADATA` side packet with `backend=="onnx"`.

- [ ] **Step 7: Commit**

```bash
git add mediapipe/calculators/tensor/onnx_inference_calculator.proto \
        mediapipe/calculators/tensor/onnx_inference_calculator.cc \
        mediapipe/calculators/tensor/onnx_inference_calculator_test.cc \
        mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'feat(onnx): standalone OnnxInferenceCalculator (TENSORS + optional METADATA)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 6: YOLO detector — backend option + ONNX graph branch + e2e

**Files:**
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.proto`
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc:213-217`
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h` (public options struct)
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.cc` (struct→proto conversion)
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc` (e2e)
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD` (deps + data)

- [ ] **Step 1: Add the `backend` field to the options proto**

In `yolo_object_detector_options.proto`, before the closing `}` of `YoloObjectDetectorOptions` (after
field 9), add:

```proto
  // Inference backend. Unset => TFLite (default, via base_options). When `onnx`
  // is set, the detector runs OnnxInferenceCalculator with the given .onnx file
  // instead of the TFLite inference subgraph. Additive and default-off.
  message Backend {
    message Tflite {}
    message Onnx {
      optional string model_path = 1;  // path to the .onnx model
    }
    oneof backend {
      Tflite tflite = 1;
      Onnx onnx = 2;
    }
  }
  optional Backend backend = 10;
```

- [ ] **Step 2: Add the failing tests (one fixture-free, one gated e2e)**

Append to `yolo_object_detector_test.cc` (inside the anonymous namespace). First add a using-declaration
near the top of the namespace so the conversion function is visible (mirrors the OBB test):

```cpp
using ::mediapipe::tasks::vision::yolo_object_detector::
    ConvertYoloObjectDetectorOptionsToProto;
```

A **fixture-free** conversion test that always runs here:

```cpp
TEST(YoloObjectDetectorOptionsTest, CopiesOnnxModelPathToProto) {
  auto opts = std::make_unique<YoloObjectDetectorOptions>();
  opts->onnx_model_path = "/tmp/yolov8n.onnx";
  auto proto = ConvertYoloObjectDetectorOptionsToProto(opts.get());
  ASSERT_TRUE(proto->has_backend());
  ASSERT_TRUE(proto->backend().has_onnx());
  EXPECT_EQ(proto->backend().onnx().model_path(), "/tmp/yolov8n.onnx");
}

TEST(YoloObjectDetectorOptionsTest, NoBackendWhenOnnxModelPathUnset) {
  auto opts = std::make_unique<YoloObjectDetectorOptions>();
  auto proto = ConvertYoloObjectDetectorOptionsToProto(opts.get());
  EXPECT_FALSE(proto->has_backend());  // default-off => TFLite path unchanged
}
```

(If `ConvertYoloObjectDetectorOptionsToProto` is not yet declared in `yolo_object_detector.h`, add the
declaration mirroring `oriented_object_detector.h:118`.)

Then the **gated e2e**, mirroring `DetectOnImage` but selecting the ONNX backend:

```cpp
constexpr char kYoloOnnxModel[] = "yolov8n.onnx";
std::string OnnxModelPath() {
  return JoinPath("./", kTestDataDirectory, kYoloOnnxModel);
}

TEST(YoloObjectDetectorTest, DetectOnImageOnnxBackend) {
  const std::string onnx_path = OnnxModelPath();
  const std::string tflite_path = ModelPath();
  if (!mediapipe::file::Exists(onnx_path).ok() ||
      !mediapipe::file::Exists(tflite_path).ok()) {
    GTEST_SKIP() << "yolov8n.onnx and/or yolov8n.tflite fixture absent.";
  }
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  // base_options still points at the .tflite so model metadata (labels) loads;
  // the ONNX path supplies the inference weights.
  options->base_options.model_asset_path = tflite_path;
  options->onnx_model_path = onnx_path;  // selects the ONNX backend
  options->running_mode = core::RunningMode::IMAGE;
  options->num_classes = 80;
  options->score_threshold = 0.25f;
  options->iou_threshold = 0.45f;
  options->max_results = 10;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));
  MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result,
                          detector->Detect(image));
  ASSERT_FALSE(result.detections.empty())
      << "ONNX backend produced no detections on " << kTestImage;
  std::set<int> labels;
  for (const auto& det : result.detections) {
    ASSERT_EQ(det.categories.size(), 1u);
    labels.insert(det.categories[0].index);
  }
  // Same oracle as the TFLite DetectOnImage test: a cat (15) or dog (16).
  EXPECT_TRUE(labels.count(15) || labels.count(16))
      << "expected a cat or dog from the ONNX backend";
  MP_ASSERT_OK(detector->Close());
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test`
Expected: FAIL — `YoloObjectDetectorOptions` (the public C++ struct) has no `onnx_model_path` member
(compile error).

- [ ] **Step 4: Add `onnx_model_path` to the public options struct**

In `yolo_object_detector.h`, in the `struct YoloObjectDetectorOptions { ... }`, add near the other simple
members (e.g. after `num_classes`):

```cpp
  // When set, runs the ONNX Runtime backend (OnnxInferenceCalculator) with this
  // .onnx model instead of the default TFLite inference path. base_options
  // still provides the model metadata (labels). Default: unset (TFLite).
  std::optional<std::string> onnx_model_path = std::nullopt;
```

Ensure `#include <optional>` and `#include <string>` are present in the header.

- [ ] **Step 5: Convert the struct field to the proto in the C++ wrapper**

In `yolo_object_detector.cc`, in `ConvertYoloObjectDetectorOptionsToProto(YoloObjectDetectorOptions*
options)` (defined at `yolo_object_detector.cc:97`; local proto variable is `options_proto`), after the
existing field copies (e.g. after `set_layout(...)`) add:

```cpp
  if (options->onnx_model_path.has_value()) {
    options_proto->mutable_backend()->mutable_onnx()->set_model_path(
        *options->onnx_model_path);
  }
```

- [ ] **Step 6: Add the ONNX branch in the graph builder**

In `yolo_object_detector_graph.cc`, replace the inference insertion at lines 213-217:

```cpp
    auto& inference = AddInference(
        model_resources, task_options.base_options().acceleration(), graph);
    preprocessing.Out(kTensorTag) >> inference.In(kTensorTag);
    TensorsSource model_output_tensors =
        inference.Out(kTensorTag).Cast<std::vector<Tensor>>();
```

with a backend-conditional (note `kTensorTag` == "TENSORS"):

```cpp
    TensorsSource model_output_tensors = [&]() {
      if (task_options.has_backend() && task_options.backend().has_onnx()) {
        // ONNX backend: standalone OnnxInferenceCalculator, no TFLite subgraph.
        auto& onnx = graph.AddNode("OnnxInferenceCalculator");
        onnx.GetOptions<::mediapipe::OnnxInferenceCalculatorOptions>()
            .set_model_path(task_options.backend().onnx().model_path());
        preprocessing.Out(kTensorTag) >> onnx.In(kTensorTag);
        return onnx.Out(kTensorTag).Cast<std::vector<Tensor>>();
      }
      auto& inference = AddInference(
          model_resources, task_options.base_options().acceleration(), graph);
      preprocessing.Out(kTensorTag) >> inference.In(kTensorTag);
      return inference.Out(kTensorTag).Cast<std::vector<Tensor>>();
    }();
```

Add `#include "mediapipe/calculators/tensor/onnx_inference_calculator.pb.h"` to the graph's includes.

- [ ] **Step 7: Update the YOLO detector BUILD**

In `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`:
- Add `"//mediapipe/calculators/tensor:onnx_inference_calculator"` (the registered calculator) and
  `"//mediapipe/calculators/tensor:onnx_inference_calculator_cc_proto"` to the `deps` of the
  `yolo_object_detector_graph` cc_library.
- Add `"//mediapipe/tasks/testdata/vision:yolo_test_model"` to the `data` of the
  `yolo_object_detector_test` target if not already present (it is required for the `.onnx` fixture).

Because the calculator is macOS-only, gate the graph dep with a `select`:

```python
    deps = [
        # ... existing deps ...
        "//mediapipe/calculators/tensor:onnx_inference_calculator_cc_proto",
    ] + select({
        "//mediapipe:macos": [
            "//mediapipe/calculators/tensor:onnx_inference_calculator",
        ],
        "//conditions:default": [],
    }),
```

(The `_cc_proto` is platform-independent and can be an unconditional dep; only the calculator
implementation is macOS-gated. On non-macOS, `backend.onnx` would fail at runtime with "OnnxInferenceCalculator
not registered" — acceptable per the spec.)

- [ ] **Step 8: Run the test to verify it passes (or skips)**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test`
Expected: PASS — the new ONNX test runs (fixtures present) or skips; the existing TFLite tests still pass.

- [ ] **Step 9: Commit**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/
git commit -m "$(printf 'feat(yolo-onnx): backend option + ONNX graph branch + e2e (cats_and_dogs oracle)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 7: OBB detector — backend option + ONNX graph branch + e2e

**Files:**
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/proto/oriented_object_detector_options.proto`
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_graph.cc:212` (the `AddInference` site)
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h` (public struct)
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.cc` (conversion)
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_test.cc` (e2e)
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/BUILD` (deps + data)

This mirrors Task 6 for the OBB detector. The proto field number is **11** (OBB options use fields 1-10).
The e2e runs on `boats.jpg` with the ships oracle (DOTA class 1).

- [ ] **Step 1: Add the `backend` field (field 11) to the OBB options proto**

In `oriented_object_detector_options.proto`, before the closing `}` of `OrientedObjectDetectorOptions`
(after field 10), add the SAME `Backend` message and field, numbered 11:

```proto
  // Inference backend. Unset => TFLite (default, via base_options). When `onnx`
  // is set, the detector runs OnnxInferenceCalculator with the given .onnx file
  // instead of the TFLite inference subgraph. Additive and default-off.
  message Backend {
    message Tflite {}
    message Onnx {
      optional string model_path = 1;  // path to the .onnx model
    }
    oneof backend {
      Tflite tflite = 1;
      Onnx onnx = 2;
    }
  }
  optional Backend backend = 11;
```

- [ ] **Step 2: Add the failing tests (one fixture-free, one gated e2e)**

In `oriented_object_detector_test.cc` (which already has `using ::mediapipe::...::
ConvertOrientedObjectDetectorOptionsToProto;` and the constants `kTestDataDirectory`,
`kOrientedModel`="yolov8n-obb.tflite", `kTestImage`="boats.jpg", `ModelPath()`, `ImagePath()`), add:

First, a **fixture-free** conversion test that always runs here (mirrors the existing
`CopiesCategoryFieldsToProto` test):

```cpp
TEST(OrientedObjectDetectorOptionsTest, CopiesOnnxModelPathToProto) {
  auto opts = std::make_unique<OrientedObjectDetectorOptions>();
  opts->onnx_model_path = "/tmp/yolov8n-obb.onnx";
  auto proto = ConvertOrientedObjectDetectorOptionsToProto(opts.get());
  ASSERT_TRUE(proto->has_backend());
  ASSERT_TRUE(proto->backend().has_onnx());
  EXPECT_EQ(proto->backend().onnx().model_path(), "/tmp/yolov8n-obb.onnx");
}

TEST(OrientedObjectDetectorOptionsTest, NoBackendWhenOnnxModelPathUnset) {
  auto opts = std::make_unique<OrientedObjectDetectorOptions>();
  auto proto = ConvertOrientedObjectDetectorOptionsToProto(opts.get());
  EXPECT_FALSE(proto->has_backend());  // default-off => TFLite path unchanged
}
```

Then the **gated e2e**, mirroring the existing `DetectOnImage` oracle exactly (DOTA ship = index 1,
`score_threshold` 0.25, `num_classes` 15):

```cpp
TEST(OrientedObjectDetectorTest, DetectOnImageOnnxBackend) {
  const std::string onnx_path =
      JoinPath("./", kTestDataDirectory, "yolov8n-obb.onnx");
  const std::string tflite_path = ModelPath();  // yolov8n-obb.tflite
  if (!mediapipe::file::Exists(onnx_path).ok() ||
      !mediapipe::file::Exists(tflite_path).ok()) {
    GTEST_SKIP() << "yolov8n-obb.onnx and/or .tflite fixture absent.";
  }
  auto options = std::make_unique<OrientedObjectDetectorOptions>();
  options->base_options.model_asset_path = tflite_path;  // supplies metadata/labels
  options->onnx_model_path = onnx_path;                  // selects ONNX backend
  options->running_mode = core::RunningMode::IMAGE;
  options->num_classes = 15;          // DOTA
  options->score_threshold = 0.25f;   // matches the existing DetectOnImage oracle
  options->iou_threshold = 0.45f;
  options->max_results = 10;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          OrientedObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));
  MP_ASSERT_OK_AND_ASSIGN(OrientedObjectDetectorResult result,
                          detector->Detect(image));
  ASSERT_FALSE(result.detections.empty())
      << "ONNX OBB backend produced no detections on boats.jpg";
  std::set<int> labels;
  for (const auto& det : result.detections) {
    ASSERT_EQ(det.categories.size(), 1u);
    labels.insert(det.categories[0].index);
  }
  EXPECT_TRUE(labels.count(1)) << "expected a ship (DOTA class 1) from ONNX backend";
  MP_ASSERT_OK(detector->Close());
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test`
Expected: FAIL — public `OrientedObjectDetectorOptions` struct has no `onnx_model_path` member.

- [ ] **Step 4: Add `onnx_model_path` to the public OBB options struct**

In `oriented_object_detector.h`, add to the `struct OrientedObjectDetectorOptions`:

```cpp
  // When set, runs the ONNX Runtime backend (OnnxInferenceCalculator) with this
  // .onnx model instead of the default TFLite inference path. base_options still
  // provides model metadata (labels). Default: unset (TFLite).
  std::optional<std::string> onnx_model_path = std::nullopt;
```

Ensure `#include <optional>` and `#include <string>` are present.

- [ ] **Step 5: Convert the struct field to the proto**

In `oriented_object_detector.cc`, in the struct→proto conversion function, add:

```cpp
  if (options->onnx_model_path.has_value()) {
    options_proto->mutable_backend()->mutable_onnx()->set_model_path(
        *options->onnx_model_path);
  }
```

- [ ] **Step 6: Add the ONNX branch in the OBB graph builder**

In `oriented_object_detector_graph.cc`, at the `auto& inference = AddInference(...)` site (~line 212),
apply the SAME backend-conditional pattern as Task 6 Step 6 (lambda returning `TensorsSource`, choosing
`OnnxInferenceCalculator` when `task_options.backend().has_onnx()`, else `AddInference`). Wire
`preprocessing.Out(kTensorTag) >> onnx.In(kTensorTag)` and return
`onnx.Out(kTensorTag).Cast<std::vector<Tensor>>()`. Add
`#include "mediapipe/calculators/tensor/onnx_inference_calculator.pb.h"`.

- [ ] **Step 7: Update the OBB detector BUILD**

In `mediapipe/tasks/cc/vision/oriented_object_detector/BUILD`, mirror Task 6 Step 7: add the
`onnx_inference_calculator_cc_proto` dep (unconditional) and the macOS-gated `onnx_inference_calculator`
dep to the `oriented_object_detector_graph` cc_library, and ensure
`"//mediapipe/tasks/testdata/vision:yolo_obb_test_model"` is in the test `data`.

- [ ] **Step 8: Run the test to verify it passes (or skips)**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test`
Expected: PASS — the ONNX OBB test runs (fixtures present) or skips; existing OBB tests still pass.

- [ ] **Step 9: Commit**

```bash
git add mediapipe/tasks/cc/vision/oriented_object_detector/
git commit -m "$(printf 'feat(obb-onnx): backend option + ONNX graph branch + e2e (boats/ship oracle)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 8: Regression sweep + verification

**Files:** none created; this task verifies the whole change is additive and green.

- [ ] **Step 1: Confirm the existing TFLite detector tests still pass (byte-identical path)**

Run:
```
bazel test --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test \
  //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test
```
Expected: PASS. The pre-existing TFLite tests (`backend` unset) behave exactly as before; the new ONNX
tests run or `GTEST_SKIP()` depending on fixture presence. Read the real gtest output — a piped `exit 0`
is not proof; confirm the per-test `OK`/`SKIPPED` lines.

- [ ] **Step 2: Confirm the shared inference path is untouched**

Run: `git diff --stat master..dev -- mediapipe/calculators/tensor/inference_calculator.cc mediapipe/calculators/tensor/inference_calculator.h mediapipe/calculators/tensor/inference_calculator.proto`
Expected: **empty** — this sub-project must not modify the shared `InferenceCalculator` selector/proto.

- [ ] **Step 3: Build the new tensor targets cleanly**

Run:
```
bazel test --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/calculators/tensor:onnxruntime_link_smoke_test \
  //mediapipe/calculators/tensor:onnx_inference_runner_test \
  //mediapipe/calculators/tensor:onnx_inference_calculator_test
```
Expected: PASS (runner/calculator tests SKIP if the `.onnx` fixtures were not generated; the smoke test
always runs).

- [ ] **Step 4: Final commit (only if any verification fixups were needed)**

If Steps 1-3 required small fixups, commit them:
```bash
git add -A
git commit -m "$(printf 'test(onnx): regression sweep fixups for the ONNX backend seam\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```
Otherwise, no commit is needed — the sub-project is complete.

---

## Notes for the implementer

- **Run real tests, read real output.** A piped `| cat` or `; echo done` masks failures. Confirm gtest
  `[  OK  ]` / `[ SKIPPED ]` lines.
- **Environmental clang false positives** (`absl/...h not found` in an editor/LSP) are not authoritative
  here — the bazel sandbox provides the include paths. Trust `bazel build`/`bazel test`.
- **If the `.onnx` fixtures are present on this machine** (Task 2 Step 4 generated them), the runner,
  calculator, and both detector e2e tests run for real and must pass. If absent, every gated test must at
  least build and `GTEST_SKIP()` cleanly.
- **ORT C++ API drift:** the snippets target ONNX Runtime 1.24 (`onnxruntime_cxx_api.h`). If a symbol name
  differs (e.g. `GetInputNameAllocated`), consult the installed header at
  `/opt/homebrew/opt/onnxruntime/include/onnxruntime/onnxruntime_cxx_api.h`.
