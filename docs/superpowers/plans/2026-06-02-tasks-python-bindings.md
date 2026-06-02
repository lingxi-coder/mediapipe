# Python Bindings for YOLO & Oriented Object Detectors — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the shipped `YoloObjectDetector` (axis-aligned) and `OrientedObjectDetector` (OBB) C++ Tasks to Python via the ctypes-over-C-API pattern, with all three running modes and best-effort label enrichment.

**Architecture:** Two new C API layers (`extern "C"`, compiled into `libmediapipe.so`) mechanically mirror `tasks/c/vision/object_detector/`. YOLO reuses the existing `MpDetectionResult` chain; OBB gets a brand-new pixel-unit oriented result container at the C, ctypes, and dataclass sub-layers. Two new Python ctypes task classes mirror `tasks/python/vision/object_detector.py`. Sequenced YOLO-first (Approach C) to prove the FFI/async/label harness before introducing the oriented container.

**Tech Stack:** Bazel (Bzlmod, C++20), `--define MEDIAPIPE_DISABLE_GPU=1`, ctypes, MediaPipe Tasks C API conventions.

**Spec:** `docs/superpowers/specs/2026-06-02-tasks-python-bindings-design.md`

---

## Cross-cutting conventions (read once)

- **Build verification, not classic TDD.** Real inference assertions are gated on absent model fixtures (`yolov8n.tflite` / `yolov8n-obb.tflite`), so the verification gate for each layer is: target builds, gated test runs and **skips cleanly**. This matches the shipped 2.1 cc tests.
- **`object_detector` (C + Python) and the shared `detections.py` / `MpDetectionResult` chain must stay untouched.** Label enrichment therefore happens in the new **task layer**, never inside the shared `DetectionResult.from_ctypes`.
- **Pixel units.** Both cc tasks already return pixel-unit results (`Detect` does the image-size conversion internally). The C API converters are pure struct→struct copies; never re-normalize.
- **ctypes field order must byte-match the C struct field order exactly.**
- `bool` in the C headers is safe: these headers are only ever compiled as C++ (via the `.cc`); Python consumes them through ctypes (`c_bool`), not a C compiler.
- Build command prefix used throughout: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 <target>` ; tests: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 <target> --test_output=all`.
- Clang/language-server "file not found" diagnostics in-editor are false positives (non-Bazel). Only Bazel results are authoritative.

## File Structure

**Phase A — YOLO (reuses existing containers):**
- Create `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h` — C API header.
- Create `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.cc` — C API impl.
- Create `mediapipe/tasks/c/vision/yolo_object_detector/BUILD` — C API targets.
- Create `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector_test.cc` — gated C test.
- Modify `mediapipe/tasks/c/BUILD` — register YOLO `_c_lib` in `mediapipe_source`.
- Create `mediapipe/tasks/python/vision/yolo_object_detector.py` — ctypes task class + label helper.
- Modify `mediapipe/tasks/python/vision/BUILD` — `yolo_object_detector` py_library.
- Create `mediapipe/tasks/python/test/vision/yolo_object_detector_test.py` — gated py test.
- Modify `mediapipe/tasks/python/test/vision/BUILD` — `yolo_object_detector_test`.

**Phase B — OBB (new oriented container):**
- Create `mediapipe/tasks/c/components/containers/oriented_detection_result.h` — C struct.
- Create `mediapipe/tasks/c/components/containers/oriented_detection_result_converter.{h,cc}` — converter.
- Modify `mediapipe/tasks/c/components/containers/BUILD` — converter targets.
- Create `mediapipe/tasks/c/vision/oriented_object_detector/{oriented_object_detector.h,.cc,BUILD,..._test.cc}`.
- Modify `mediapipe/tasks/c/BUILD` — register OBB `_c_lib`.
- Create `mediapipe/tasks/python/components/containers/oriented_detections_c.py` — ctypes mirror.
- Create `mediapipe/tasks/python/components/containers/oriented_detections.py` — dataclass.
- Modify `mediapipe/tasks/python/components/containers/BUILD` — both py_library targets.
- Create `mediapipe/tasks/python/vision/oriented_object_detector.py` — ctypes task class.
- Modify `mediapipe/tasks/python/vision/BUILD` — `oriented_object_detector` py_library.
- Create `mediapipe/tasks/python/test/vision/oriented_object_detector_test.py` — gated py test.
- Modify `mediapipe/tasks/python/test/vision/BUILD` — `oriented_object_detector_test`.

---

# Phase A — YOLO

### Task 1: YOLO C API header

**Files:**
- Create: `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h`

- [ ] **Step 1: Write the header**

```c
/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_YOLO_OBJECT_DETECTOR_H_
#define MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_YOLO_OBJECT_DETECTOR_H_

#include <cstdint>

#include "mediapipe/tasks/c/components/containers/detection_result.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"

#ifndef MP_EXPORT
#if defined(_MSC_VER)
#define MP_EXPORT __declspec(dllexport)
#else
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // _MSC_VER
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MpYoloObjectDetectorInternal* MpYoloObjectDetectorPtr;
typedef MpDetectionResult MpYoloObjectDetectorResult;

// The options for configuring a MediaPipe YOLO object detector task.
//
// NOTE: display_names_locale, category_allowlist, and category_denylist are
// passed through faithfully to the C++ task, but the 2.1a YOLO graph currently
// validates them only for mutual exclusivity and does NOT apply label mapping
// or allowlist/denylist filtering. They are therefore presently no-ops; only
// score_threshold / iou_threshold / max_results affect output today.
struct MpYoloObjectDetectorOptions {
  struct MpBaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  MpRunningMode running_mode;

  // Locale for display names in TFLite metadata, if any. Defaults to English.
  const char* display_names_locale;

  // Max number of top-scored results. < 0 returns all; 0 is invalid.
  int max_results;

  // Score threshold overriding the model metadata value. Default 0.0.
  float score_threshold;

  // Allowlist of category names (mutually exclusive with denylist).
  const char** category_allowlist;
  uint32_t category_allowlist_count;

  // Denylist of category names (mutually exclusive with allowlist).
  const char** category_denylist;
  uint32_t category_denylist_count;

  // IoU threshold for non-maximum suppression. Default 0.45.
  float iou_threshold;

  // Output tensor layout: CHANNELS_FIRST=1, CHANNELS_LAST=2 (numerically equal
  // to the C++ YoloObjectDetectorOptions::Layout enum).
  int layout;

  // Number of classes. If 0, derived from model metadata at graph build time.
  int num_classes;

  // Result callback for live-stream mode. Must be set iff running_mode is
  // MP_RUNNING_MODE_LIVE_STREAM. Passed arguments are valid only for the
  // lifetime of the callback.
  typedef void (*result_callback_fn)(MpStatus status,
                                     const MpYoloObjectDetectorResult* result,
                                     const MpImagePtr image,
                                     int64_t timestamp_ms);
  result_callback_fn result_callback;
};

// Creates a YoloObjectDetector from the provided `options`.
MP_EXPORT MpStatus
MpYoloObjectDetectorCreate(struct MpYoloObjectDetectorOptions* options,
                           MpYoloObjectDetectorPtr* detector_out,
                           char** error_msg);

// Performs detection on a single image.
MP_EXPORT MpStatus MpYoloObjectDetectorDetectImage(
    MpYoloObjectDetectorPtr detector, MpImagePtr image,
    const struct MpImageProcessingOptions* options,
    MpYoloObjectDetectorResult* result, char** error_msg);

// Performs detection on a video frame (monotonically increasing timestamps).
MP_EXPORT MpStatus MpYoloObjectDetectorDetectForVideo(
    MpYoloObjectDetectorPtr detector, MpImagePtr image,
    const struct MpImageProcessingOptions* options, int64_t timestamp_ms,
    MpYoloObjectDetectorResult* result, char** error_msg);

// Sends live image data; results delivered via the configured result_callback.
MP_EXPORT MpStatus MpYoloObjectDetectorDetectAsync(
    MpYoloObjectDetectorPtr detector, MpImagePtr image,
    const struct MpImageProcessingOptions* options, int64_t timestamp_ms,
    char** error_msg);

// Frees memory allocated inside a result. Does not free the result pointer.
MP_EXPORT void MpYoloObjectDetectorCloseResult(
    MpYoloObjectDetectorResult* result);

// Frees the detector.
MP_EXPORT MpStatus MpYoloObjectDetectorClose(MpYoloObjectDetectorPtr detector,
                                             char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_YOLO_OBJECT_DETECTOR_H_
```

- [ ] **Step 2: Commit** (header builds with the impl in Task 2; no standalone build step)

```bash
git add mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h
git commit -m "feat(tasks-c-yolo): YOLO object detector C API header"
```

---

### Task 2: YOLO C API implementation + BUILD

**Files:**
- Create: `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.cc`
- Create: `mediapipe/tasks/c/vision/yolo_object_detector/BUILD`

- [ ] **Step 1: Write the implementation**

```cpp
/* Copyright 2026 The MediaPipe Authors. Licensed under the Apache License,
Version 2.0. See object_detector.cc for the full header. */

#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/components/containers/detection_result_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options_converter.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

namespace YoloNs = ::mediapipe::tasks::vision::yolo_object_detector;

struct MpYoloObjectDetectorInternal {
  std::unique_ptr<YoloNs::YoloObjectDetector> instance;
};

namespace mediapipe::tasks::c::vision::yolo_object_detector {

namespace {

using ::mediapipe::Image;
using ::mediapipe::tasks::c::components::containers::CppCloseDetectionResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToDetectionResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
using ::mediapipe::tasks::vision::core::RunningMode;
using CppYoloResult = YoloNs::YoloObjectDetectorResult;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

YoloNs::YoloObjectDetector* GetCppDetector(MpYoloObjectDetectorPtr wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "YoloObjectDetector is null.";
  return wrapper->instance.get();
}

}  // namespace

void CppConvertToDetectorOptions(const MpYoloObjectDetectorOptions& in,
                                 YoloNs::YoloObjectDetectorOptions* out) {
  out->display_names_locale =
      in.display_names_locale ? std::string(in.display_names_locale) : "en";
  out->max_results = in.max_results;
  out->score_threshold = in.score_threshold;
  out->category_allowlist =
      std::vector<std::string>(in.category_allowlist_count);
  for (uint32_t i = 0; i < in.category_allowlist_count; ++i) {
    out->category_allowlist[i] = in.category_allowlist[i];
  }
  out->category_denylist = std::vector<std::string>(in.category_denylist_count);
  for (uint32_t i = 0; i < in.category_denylist_count; ++i) {
    out->category_denylist[i] = in.category_denylist[i];
  }
  out->iou_threshold = in.iou_threshold;
  out->layout =
      static_cast<YoloNs::YoloObjectDetectorOptions::Layout>(in.layout);
  out->num_classes = in.num_classes;
}

absl::Status CppYoloObjectDetectorCreate(
    const MpYoloObjectDetectorOptions& options,
    MpYoloObjectDetectorPtr* detector_out) {
  auto cpp_options = std::make_unique<YoloNs::YoloObjectDetectorOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToDetectorOptions(options, cpp_options.get());
  cpp_options->running_mode = static_cast<RunningMode>(options.running_mode);

  if (cpp_options->running_mode == RunningMode::LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      return absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
    }
    MpYoloObjectDetectorOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppYoloResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          MpImageInternal mp_image({.image = image});
          if (!cpp_result.ok()) {
            result_callback(ToMpStatus(cpp_result.status()), nullptr, &mp_image,
                            timestamp);
            return;
          }
          MpYoloObjectDetectorResult result;
          CppConvertToDetectionResult(*cpp_result, &result);
          result_callback(kMpOk, &result, &mp_image, timestamp);
          CppCloseDetectionResult(&result);
        };
  }

  auto detector = YoloNs::YoloObjectDetector::Create(std::move(cpp_options));
  if (!detector.ok()) {
    return detector.status();
  }
  *detector_out =
      new MpYoloObjectDetectorInternal{.instance = std::move(*detector)};
  return absl::OkStatus();
}

absl::Status CppYoloObjectDetectorDetect(
    MpYoloObjectDetectorPtr detector, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    MpYoloObjectDetectorResult* result) {
  auto cpp_detector = GetCppDetector(detector);
  std::optional<CppImageProcessingOptions> cpp_opts;
  if (image_processing_options) {
    CppImageProcessingOptions o;
    CppConvertToImageProcessingOptions(*image_processing_options, &o);
    cpp_opts = o;
  }
  auto cpp_result = cpp_detector->Detect(ToImage(image), cpp_opts);
  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToDetectionResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppYoloObjectDetectorDetectForVideo(
    MpYoloObjectDetectorPtr detector, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, MpYoloObjectDetectorResult* result) {
  auto cpp_detector = GetCppDetector(detector);
  std::optional<CppImageProcessingOptions> cpp_opts;
  if (image_processing_options) {
    CppImageProcessingOptions o;
    CppConvertToImageProcessingOptions(*image_processing_options, &o);
    cpp_opts = o;
  }
  auto cpp_result =
      cpp_detector->DetectForVideo(ToImage(image), timestamp_ms, cpp_opts);
  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToDetectionResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppYoloObjectDetectorDetectAsync(
    MpYoloObjectDetectorPtr detector, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  auto cpp_detector = GetCppDetector(detector);
  std::optional<CppImageProcessingOptions> cpp_opts;
  if (image_processing_options) {
    CppImageProcessingOptions o;
    CppConvertToImageProcessingOptions(*image_processing_options, &o);
    cpp_opts = o;
  }
  return cpp_detector->DetectAsync(ToImage(image), timestamp_ms, cpp_opts);
}

void CppYoloObjectDetectorCloseResult(MpYoloObjectDetectorResult* result) {
  CppCloseDetectionResult(result);
}

absl::Status CppYoloObjectDetectorClose(MpYoloObjectDetectorPtr detector) {
  auto cpp_detector = GetCppDetector(detector);
  auto result = cpp_detector->Close();
  if (!result.ok()) {
    return result;
  }
  delete detector;
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::vision::yolo_object_detector

extern "C" {

MpStatus MpYoloObjectDetectorCreate(struct MpYoloObjectDetectorOptions* options,
                                    MpYoloObjectDetectorPtr* detector_out,
                                    char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::yolo_object_detector::
      CppYoloObjectDetectorCreate(*options, detector_out);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MpStatus MpYoloObjectDetectorDetectImage(
    MpYoloObjectDetectorPtr detector, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    MpYoloObjectDetectorResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::yolo_object_detector::
      CppYoloObjectDetectorDetect(detector, image, image_processing_options,
                                  result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MpStatus MpYoloObjectDetectorDetectForVideo(
    MpYoloObjectDetectorPtr detector, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, MpYoloObjectDetectorResult* result,
    char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::yolo_object_detector::
      CppYoloObjectDetectorDetectForVideo(detector, image,
                                          image_processing_options,
                                          timestamp_ms, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MpStatus MpYoloObjectDetectorDetectAsync(
    MpYoloObjectDetectorPtr detector, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::yolo_object_detector::
      CppYoloObjectDetectorDetectAsync(detector, image,
                                       image_processing_options, timestamp_ms);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

void MpYoloObjectDetectorCloseResult(MpYoloObjectDetectorResult* result) {
  mediapipe::tasks::c::vision::yolo_object_detector::
      CppYoloObjectDetectorCloseResult(result);
}

MpStatus MpYoloObjectDetectorClose(MpYoloObjectDetectorPtr detector,
                                   char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::yolo_object_detector::
      CppYoloObjectDetectorClose(detector);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
```

- [ ] **Step 2: Write the BUILD** (mirrors `object_detector/BUILD`, swapping the cc task dep)

```python
# Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

licenses(["notice"])

YOLO_OBJECT_DETECTOR_DEPS = [
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/log:absl_check",
    "//mediapipe/framework/formats:image",
    "//mediapipe/tasks/c/components/containers:detection_result",
    "//mediapipe/tasks/c/components/containers:detection_result_converter",
    "//mediapipe/tasks/c/core:base_options",
    "//mediapipe/tasks/c/core:base_options_converter",
    "//mediapipe/tasks/c/core:common",
    "//mediapipe/tasks/c/core:mp_status",
    "//mediapipe/tasks/c/core:mp_status_converter",
    "//mediapipe/tasks/c/vision/core:image",
    "//mediapipe/tasks/c/vision/core:image_frame_util",
    "//mediapipe/tasks/c/vision/core:image_processing_options",
    "//mediapipe/tasks/c/vision/core:image_processing_options_converter",
    "//mediapipe/tasks/cc/vision/core:image_processing_options",
    "//mediapipe/tasks/cc/vision/core:running_mode",
    "//mediapipe/tasks/cc/vision/yolo_object_detector",
]

cc_library(
    name = "yolo_object_detector_lib",
    srcs = ["yolo_object_detector.cc"],
    hdrs = ["yolo_object_detector.h"],
    visibility = ["//visibility:public"],
    deps = YOLO_OBJECT_DETECTOR_DEPS,
)

cc_library(
    name = "yolo_object_detector_c_lib",
    srcs = ["yolo_object_detector.cc"],
    hdrs = ["yolo_object_detector.h"],
    visibility = ["//mediapipe/tasks/c:__subpackages__"],
    deps = YOLO_OBJECT_DETECTOR_DEPS,
    alwayslink = 1,
)

cc_test(
    name = "yolo_object_detector_test",
    srcs = ["yolo_object_detector_test.cc"],
    data = [
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/tasks/testdata/vision:test_images",
    ],
    linkstatic = 1,
    deps = [
        ":yolo_object_detector_lib",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/c/core:common",
        "//mediapipe/tasks/c/core:mp_status",
        "//mediapipe/tasks/c/vision/core:image",
        "//mediapipe/tasks/c/vision/core:image_processing_options",
        "//mediapipe/tasks/c/vision/core:image_test_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
```

- [ ] **Step 3: Build the library**

Run: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_lib`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.cc mediapipe/tasks/c/vision/yolo_object_detector/BUILD
git commit -m "feat(tasks-c-yolo): YOLO object detector C API impl + BUILD"
```

---

### Task 3: Register YOLO in the libmediapipe.so aggregator

**Files:**
- Modify: `mediapipe/tasks/c/BUILD` (the `mediapipe_source` cc_binary deps list, near line 60)

- [ ] **Step 1: Add the dep** immediately after the `object_detector:object_detector_c_lib` line:

```python
        "//mediapipe/tasks/c/vision/object_detector:object_detector_c_lib",
        "//mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_c_lib",
```

(The `mediapipe_source` target exports all default-visibility `MP_EXPORT` symbols; no per-symbol export edits are needed.)

- [ ] **Step 2: Build the shared library**

Run: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c:libmediapipe.so`
Expected: build succeeds and the `MpYoloObjectDetector*` symbols are linked in.

- [ ] **Step 3: Verify symbols are exported**

Run: `nm -D bazel-bin/mediapipe/tasks/c/libmediapipe.so | grep MpYoloObjectDetector`
Expected: lists `MpYoloObjectDetectorCreate`, `...DetectImage`, `...DetectForVideo`, `...DetectAsync`, `...CloseResult`, `...Close`.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/c/BUILD
git commit -m "build(tasks-c): link YOLO detector C API into libmediapipe.so"
```

---

### Task 4: YOLO C gated integration test

**Files:**
- Create: `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector_test.cc`

- [ ] **Step 1: Write the gated test** (build + skip when fixture absent)

```cpp
/* Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0.

Integration test for the YOLO object detector C API. Assertions are gated on a
yolov8n.tflite fixture; absent it, the test SKIPs cleanly. To enable: place the
model at mediapipe/tasks/testdata/vision/yolov8n.tflite, add it to that
package's BUILD (mediapipe_files + filegroup), add the data dep here, re-run. */

#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"

#include <cstdint>
#include <string>

#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_test_util.h"

namespace {

using ::mediapipe::file::JoinPath;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kYoloModel[] = "yolov8n.tflite";
constexpr char kTestImage[] = "cats_and_dogs.jpg";

std::string ModelPath() {
  return JoinPath("./", kTestDataDirectory, kYoloModel);
}
std::string ImagePath() {
  return JoinPath("./", kTestDataDirectory, kTestImage);
}

bool ModelAvailable() {
  return ::mediapipe::file::Exists(ModelPath()).ok();
}

TEST(YoloObjectDetectorCApiTest, ImageMode) {
  if (!ModelAvailable()) {
    GTEST_SKIP() << "YOLO model fixture not available at " << ModelPath()
                 << "; C API assertions gated until yolov8n.tflite is added.";
  }

  const std::string model_path = ModelPath();
  MpYoloObjectDetectorOptions options = {};
  options.base_options.model_asset_path = model_path.c_str();
  options.running_mode = MpRunningMode::MP_RUNNING_MODE_IMAGE;
  options.max_results = 10;
  options.score_threshold = 0.25f;
  options.iou_threshold = 0.45f;
  options.num_classes = 80;
  options.layout = 2;  // CHANNELS_LAST

  MpYoloObjectDetectorPtr detector = nullptr;
  char* error_msg = nullptr;
  MpStatus create_status =
      MpYoloObjectDetectorCreate(&options, &detector, &error_msg);
  ASSERT_EQ(create_status, kMpOk) << (error_msg ? error_msg : "");
  ASSERT_NE(detector, nullptr);

  MpImagePtr image = nullptr;
  ASSERT_EQ(MpImageCreateFromFile(ImagePath().c_str(), &image, &error_msg),
            kMpOk);

  MpYoloObjectDetectorResult result = {};
  ASSERT_EQ(MpYoloObjectDetectorDetectImage(detector, image, nullptr, &result,
                                            &error_msg),
            kMpOk);
  EXPECT_GT(result.detections_count, 0u);
  for (uint32_t i = 0; i < result.detections_count; ++i) {
    EXPECT_EQ(result.detections[i].categories_count, 1u);
    EXPECT_GT(result.detections[i].categories[0].score, 0.0f);
  }

  MpYoloObjectDetectorCloseResult(&result);
  EXPECT_EQ(MpYoloObjectDetectorClose(detector, &error_msg), kMpOk);
}

}  // namespace
```

> NOTE: confirm the exact helper name in `//mediapipe/tasks/c/vision/core:image_test_util` (`MpImageCreateFromFile` vs. the sibling object_detector_test usage) when implementing; mirror whatever `object_detector_test.cc` uses to load an image.

- [ ] **Step 2: Run the gated test (skips)**

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_test --test_output=all`
Expected: PASS with the `ImageMode` case reported as SKIPPED.

- [ ] **Step 3: Commit**

```bash
git add mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector_test.cc
git commit -m "test(tasks-c-yolo): gated YOLO C API integration test"
```

---

### Task 5: YOLO Python ctypes task class + label helper + BUILD

**Files:**
- Create: `mediapipe/tasks/python/vision/yolo_object_detector.py`
- Modify: `mediapipe/tasks/python/vision/BUILD`

- [ ] **Step 1: Write the Python task class**

```python
# Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0.
"""MediaPipe YOLO object detector task."""

import ctypes
import dataclasses
import enum
from typing import Callable, List, Optional, Tuple

from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import detections as detections_module
from mediapipe.tasks.python.components.containers import detections_c as detections_c_module
from mediapipe.tasks.python.core import async_result_dispatcher
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c as base_options_c_module
from mediapipe.tasks.python.core import mediapipe_c_bindings as mediapipe_c_bindings_c_module
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import image_processing_options_c as image_processing_options_c_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

YoloObjectDetectorResult = detections_module.DetectionResult
_BaseOptions = base_options_module.BaseOptions
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_AsyncResultDispatcher = async_result_dispatcher.AsyncResultDispatcher


class Layout(enum.IntEnum):
  """YOLO detect-head output tensor layout (numerically matches the C++ enum)."""

  CHANNELS_FIRST = 1
  CHANNELS_LAST = 2


_C_TYPES_RESULT_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32,  # MpStatus
    ctypes.POINTER(detections_c_module.MpDetectionResultC),
    ctypes.c_void_p,  # MpImage
    ctypes.c_int64,  # timestamp_ms
)


def _load_label_map(model_path: Optional[str]) -> Optional[List[str]]:
  """Best-effort: ordered category names from the model's TFLite metadata.

  Returns None when no label file is present (the common case for stock
  ultralytics exports) or on any parsing error. Display-only enrichment.
  """
  if not model_path:
    return None
  try:
    from mediapipe.tasks.python.metadata import metadata as _metadata  # pylint: disable=g-import-not-at-top

    displayer = _metadata.MetadataDisplayer.with_model_file(model_path)
    for name in displayer.get_packed_associated_file_list():
      if name.endswith('.txt') or 'label' in name.lower():
        buf = displayer.get_associated_file_buffer(name)
        text = buf.decode('utf-8') if isinstance(buf, bytes) else buf
        labels = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if labels:
          return labels
  except Exception:  # pylint: disable=broad-except
    return None
  return None


def _enrich_with_label_map(
    result: YoloObjectDetectorResult, label_map: Optional[List[str]]
) -> YoloObjectDetectorResult:
  """Sets category_name from label_map by index, in place. No-op if None."""
  if not label_map:
    return result
  for detection in result.detections:
    for category in detection.categories:
      idx = category.index
      if category.category_name is None and idx is not None and (
          0 <= idx < len(label_map)
      ):
        category.category_name = label_map[idx]
  return result


class MpYoloObjectDetectorOptionsC(ctypes.Structure):
  """The YOLO object detector options used in the C API.

  Field order MUST match struct MpYoloObjectDetectorOptions in
  mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h.
  """

  _fields_ = [
      ('base_options', base_options_c_module.MpBaseOptionsC),
      ('running_mode', ctypes.c_int),
      ('display_names_locale', ctypes.c_char_p),
      ('max_results', ctypes.c_int),
      ('score_threshold', ctypes.c_float),
      ('category_allowlist', ctypes.POINTER(ctypes.c_char_p)),
      ('category_allowlist_count', ctypes.c_uint32),
      ('category_denylist', ctypes.POINTER(ctypes.c_char_p)),
      ('category_denylist_count', ctypes.c_uint32),
      ('iou_threshold', ctypes.c_float),
      ('layout', ctypes.c_int),
      ('num_classes', ctypes.c_int),
      ('result_callback', _C_TYPES_RESULT_CALLBACK),
  ]


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpYoloObjectDetectorCreate',
        (
            ctypes.POINTER(MpYoloObjectDetectorOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpYoloObjectDetectorDetectImage',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.MpImageProcessingOptionsC
            ),
            ctypes.POINTER(detections_c_module.MpDetectionResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpYoloObjectDetectorDetectForVideo',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.MpImageProcessingOptionsC
            ),
            ctypes.c_int64,
            ctypes.POINTER(detections_c_module.MpDetectionResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpYoloObjectDetectorDetectAsync',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.MpImageProcessingOptionsC
            ),
            ctypes.c_int64,
        ),
    ),
    mediapipe_c_utils.CFunction(
        'MpYoloObjectDetectorCloseResult',
        [ctypes.POINTER(detections_c_module.MpDetectionResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpYoloObjectDetectorClose',
        (ctypes.c_void_p,),
    ),
)


@dataclasses.dataclass
class YoloObjectDetectorOptions:
  """Options for the YOLO object detector task.

  NOTE: display_names_locale / category_allowlist / category_denylist are
  accepted for parity with the C++ task, but the current 2.1a graph does not
  apply label mapping or allowlist/denylist filtering. Category names are
  populated best-effort in Python from TFLite metadata (None if unavailable).
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  display_names_locale: Optional[str] = None
  max_results: Optional[int] = -1
  score_threshold: Optional[float] = 0.0
  category_allowlist: Optional[List[str]] = None
  category_denylist: Optional[List[str]] = None
  iou_threshold: float = 0.45
  layout: Layout = Layout.CHANNELS_FIRST
  num_classes: int = 0
  result_callback: Optional[
      Callable[
          [detections_module.DetectionResult, image_module.Image, int], None
      ]
  ] = None


class YoloObjectDetector:
  """Performs YOLO (axis-aligned) object detection on images."""

  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p
  _dispatcher: _AsyncResultDispatcher
  _async_callback: _C_TYPES_RESULT_CALLBACK
  _label_map: Optional[List[str]]

  def __init__(self, lib, handle, dispatcher, async_callback, label_map):
    self._lib = lib
    self._handle = handle
    self._dispatcher = dispatcher
    self._async_callback = async_callback
    self._label_map = label_map

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'YoloObjectDetector':
    options = YoloObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RunningMode.IMAGE,
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: YoloObjectDetectorOptions
  ) -> 'YoloObjectDetector':
    running_mode_module.validate_running_mode(
        options.running_mode, options.result_callback
    )
    lib = mediapipe_c_bindings_c_module.load_shared_library(_CTYPES_SIGNATURES)
    label_map = _load_label_map(
        getattr(options.base_options, 'model_asset_path', None)
    )

    def convert_result(c_result_ptr, image_ptr, timestamp_ms):
      c_result = c_result_ptr[0]
      py_result = _enrich_with_label_map(
          YoloObjectDetectorResult.from_ctypes(c_result), label_map
      )
      py_image = image_module.Image.create_from_ctypes(image_ptr)
      return (py_result, py_image, timestamp_ms)

    dispatcher = _AsyncResultDispatcher(converter=convert_result)
    c_callback = dispatcher.wrap_callback(
        options.result_callback, _C_TYPES_RESULT_CALLBACK
    )

    allowlist_c = mediapipe_c_bindings_c_module.convert_strings_to_ctypes_array(
        options.category_allowlist
    )
    denylist_c = mediapipe_c_bindings_c_module.convert_strings_to_ctypes_array(
        options.category_denylist
    )
    ctypes_options = MpYoloObjectDetectorOptionsC(
        base_options=options.base_options.to_ctypes(),
        running_mode=options.running_mode.ctype,
        display_names_locale=options.display_names_locale,
        max_results=options.max_results,
        score_threshold=options.score_threshold,
        category_allowlist=allowlist_c,
        category_allowlist_count=(
            len(options.category_allowlist) if options.category_allowlist else 0
        ),
        category_denylist=denylist_c,
        category_denylist_count=(
            len(options.category_denylist) if options.category_denylist else 0
        ),
        iou_threshold=options.iou_threshold,
        layout=int(options.layout),
        num_classes=options.num_classes,
        result_callback=c_callback,
    )

    detector_handle = ctypes.c_void_p()
    lib.MpYoloObjectDetectorCreate(
        ctypes.byref(ctypes_options), ctypes.byref(detector_handle)
    )
    return YoloObjectDetector(
        lib=lib,
        handle=detector_handle,
        dispatcher=dispatcher,
        async_callback=c_callback,
        label_map=label_map,
    )

  def detect(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> YoloObjectDetectorResult:
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = detections_c_module.MpDetectionResultC()
    c_ipo = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpYoloObjectDetectorDetectImage(
        self._handle, c_image, c_ipo, ctypes.byref(c_result)
    )
    py_result = _enrich_with_label_map(
        YoloObjectDetectorResult.from_ctypes(c_result), self._label_map
    )
    self._lib.MpYoloObjectDetectorCloseResult(ctypes.byref(c_result))
    return py_result

  def detect_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> YoloObjectDetectorResult:
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = detections_c_module.MpDetectionResultC()
    c_ipo = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpYoloObjectDetectorDetectForVideo(
        self._handle, c_image, c_ipo, timestamp_ms, ctypes.byref(c_result)
    )
    py_result = _enrich_with_label_map(
        YoloObjectDetectorResult.from_ctypes(c_result), self._label_map
    )
    self._lib.MpYoloObjectDetectorCloseResult(ctypes.byref(c_result))
    return py_result

  def detect_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_ipo = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpYoloObjectDetectorDetectAsync(
        self._handle, c_image, c_ipo, timestamp_ms
    )

  def close(self):
    if self._handle:
      self._lib.MpYoloObjectDetectorClose(self._handle)
      self._handle = None
      self._dispatcher.close()
      self._lib.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    del exc_type, exc_value, traceback
    self.close()

  def __del__(self):
    self.close()
```

- [ ] **Step 2: Add the py_library to `mediapipe/tasks/python/vision/BUILD`** (mirror `object_detector`, add `category` + `metadata`):

```python
py_library(
    name = "yolo_object_detector",
    srcs = ["yolo_object_detector.py"],
    deps = [
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:detections",
        "//mediapipe/tasks/python/components/containers:detections_c",
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:base_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:mediapipe_c_utils",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//mediapipe/tasks/python/metadata",
        "//mediapipe/tasks/python/vision/core:image",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:image_processing_options_c",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)
```

> Confirm the exact Bazel label for the Python metadata module (`//mediapipe/tasks/python/metadata` target name) when wiring; adjust if it differs.

- [ ] **Step 3: Build the py_library**

Run: `bazel build //mediapipe/tasks/python/vision:yolo_object_detector`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/python/vision/yolo_object_detector.py mediapipe/tasks/python/vision/BUILD
git commit -m "feat(tasks-py-yolo): YOLO object detector Python ctypes task"
```

---

### Task 6: YOLO Python gated test + BUILD

**Files:**
- Create: `mediapipe/tasks/python/test/vision/yolo_object_detector_test.py`
- Modify: `mediapipe/tasks/python/test/vision/BUILD`

- [ ] **Step 1: Write the gated test**

```python
# Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0.
"""Tests for the YOLO object detector Python API (model assertions gated)."""

import os
import unittest

from absl.testing import absltest

from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import yolo_object_detector
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_YoloObjectDetector = yolo_object_detector.YoloObjectDetector
_YoloObjectDetectorOptions = yolo_object_detector.YoloObjectDetectorOptions
_RunningMode = running_mode_module.VisionTaskRunningMode

_MODEL_FILE = 'yolov8n.tflite'
_IMAGE_FILE = 'cats_and_dogs.jpg'


def _model_path():
  try:
    return test_utils.get_test_data_path(_MODEL_FILE)
  except Exception:  # pylint: disable=broad-except
    return None


def _model_available():
  p = _model_path()
  return bool(p) and os.path.exists(p)


class YoloObjectDetectorTest(absltest.TestCase):

  def test_options_construct_without_model(self):
    # Runs unconditionally: constructing options must not require a model.
    options = _YoloObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path='/nonexistent.tflite'),
        running_mode=_RunningMode.IMAGE,
        max_results=10,
        layout=yolo_object_detector.Layout.CHANNELS_LAST,
        num_classes=80,
    )
    self.assertEqual(options.iou_threshold, 0.45)
    self.assertEqual(int(options.layout), 2)

  @unittest.skipUnless(_model_available(), 'yolov8n.tflite fixture not present')
  def test_detect_image(self):
    image = image_module.Image.create_from_file(
        test_utils.get_test_data_path(_IMAGE_FILE)
    )
    options = _YoloObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=_model_path()),
        running_mode=_RunningMode.IMAGE,
        max_results=10,
        score_threshold=0.25,
        layout=yolo_object_detector.Layout.CHANNELS_LAST,
        num_classes=80,
    )
    with _YoloObjectDetector.create_from_options(options) as detector:
      result = detector.detect(image)
      self.assertNotEmpty(result.detections)
      for detection in result.detections:
        self.assertLen(detection.categories, 1)
        self.assertGreater(detection.categories[0].score, 0.0)
        # Pixel-unit bounding box (matches ObjectDetector semantics).
        self.assertGreaterEqual(detection.bounding_box.origin_x, 0)


if __name__ == '__main__':
  absltest.main()
```

> Confirm `test_utils.get_test_data_path` and `image_module.Image.create_from_file` names against `object_detector_test.py` when implementing; mirror exactly.

- [ ] **Step 2: Add the test target to `mediapipe/tasks/python/test/vision/BUILD`** (mirror `object_detector_test`, drop `test_models`, keep `test_images`):

```python
py_test(
    name = "yolo_object_detector_test",
    srcs = ["yolo_object_detector_test.py"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
    ],
    tags = ["not_run:arm"],
    deps = [
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:detections",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/vision:yolo_object_detector",
        "//mediapipe/tasks/python/vision/core:image",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
        "@mediapipe_pip_deps_absl_py//:pkg",
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)
```

- [ ] **Step 3: Run the gated test (skips inference, runs options test)**

Run: `bazel test //mediapipe/tasks/python/test/vision:yolo_object_detector_test --test_output=all`
Expected: PASS; `test_detect_image` reported SKIPPED, `test_options_construct_without_model` runs.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/python/test/vision/yolo_object_detector_test.py mediapipe/tasks/python/test/vision/BUILD
git commit -m "test(tasks-py-yolo): gated YOLO Python API test"
```

---

# Phase B — OBB

### Task 7: OBB C result container + converter + BUILD

**Files:**
- Create: `mediapipe/tasks/c/components/containers/oriented_detection_result.h`
- Create: `mediapipe/tasks/c/components/containers/oriented_detection_result_converter.h`
- Create: `mediapipe/tasks/c/components/containers/oriented_detection_result_converter.cc`
- Modify: `mediapipe/tasks/c/components/containers/BUILD`

- [ ] **Step 1: Write the C struct** `oriented_detection_result.h`

```c
/* Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0. */

#ifndef MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_ORIENTED_DETECTION_RESULT_H_
#define MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_ORIENTED_DETECTION_RESULT_H_

#include <stdint.h>

#include "mediapipe/tasks/c/components/containers/category.h"

#ifdef __cplusplus
extern "C" {
#endif

// One oriented (rotated) bounding-box detection, in original-image PIXEL units.
struct MpOrientedDetection {
  struct MpCategory* categories;
  uint32_t categories_count;
  float cx;        // box center x, pixels
  float cy;        // box center y, pixels
  float width;     // pixels
  float height;    // pixels
  float rotation;  // radians, counter-clockwise
};

struct MpOrientedDetectionResult {
  struct MpOrientedDetection* detections;
  uint32_t detections_count;
};

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_ORIENTED_DETECTION_RESULT_H_
```

- [ ] **Step 2: Write the converter header** `oriented_detection_result_converter.h`

```cpp
/* Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0. */

#ifndef MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_ORIENTED_DETECTION_RESULT_CONVERTER_H_
#define MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_ORIENTED_DETECTION_RESULT_CONVERTER_H_

#include "mediapipe/tasks/c/components/containers/oriented_detection_result.h"
#include "mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h"

namespace mediapipe::tasks::c::components::containers {

void CppConvertToOrientedDetectionResult(
    const mediapipe::tasks::components::containers::
        OrientedObjectDetectionResult& in,
    MpOrientedDetectionResult* out);

void CppCloseOrientedDetectionResult(MpOrientedDetectionResult* in);

}  // namespace mediapipe::tasks::c::components::containers

#endif  // MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_ORIENTED_DETECTION_RESULT_CONVERTER_H_
```

- [ ] **Step 3: Write the converter impl** `oriented_detection_result_converter.cc`

```cpp
/* Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0. */

#include "mediapipe/tasks/c/components/containers/oriented_detection_result_converter.h"

#include <cstddef>

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/category_converter.h"
#include "mediapipe/tasks/c/components/containers/oriented_detection_result.h"
#include "mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h"

namespace mediapipe::tasks::c::components::containers {

namespace {
using CppOrientedDetection =
    ::mediapipe::tasks::components::containers::OrientedObjectDetection;
using CppOrientedDetectionResult =
    ::mediapipe::tasks::components::containers::OrientedObjectDetectionResult;
}  // namespace

void CppConvertToOrientedDetection(const CppOrientedDetection& in,
                                   MpOrientedDetection* out) {
  out->categories_count = in.categories.size();
  out->categories = new MpCategory[out->categories_count];
  for (size_t i = 0; i < out->categories_count; ++i) {
    CppConvertToCategory(in.categories[i], &out->categories[i]);
  }
  out->cx = in.cx;
  out->cy = in.cy;
  out->width = in.width;
  out->height = in.height;
  out->rotation = in.rotation;
}

void CppConvertToOrientedDetectionResult(const CppOrientedDetectionResult& in,
                                         MpOrientedDetectionResult* out) {
  out->detections_count = in.detections.size();
  out->detections = new MpOrientedDetection[out->detections_count];
  for (size_t i = 0; i < out->detections_count; ++i) {
    CppConvertToOrientedDetection(in.detections[i], &out->detections[i]);
  }
}

void CppCloseOrientedDetection(MpOrientedDetection* in) {
  for (size_t i = 0; i < in->categories_count; ++i) {
    CppCloseCategory(&in->categories[i]);
  }
  delete[] in->categories;
  in->categories = nullptr;
}

void CppCloseOrientedDetectionResult(MpOrientedDetectionResult* in) {
  for (size_t i = 0; i < in->detections_count; ++i) {
    CppCloseOrientedDetection(&in->detections[i]);
  }
  delete[] in->detections;
  in->detections = nullptr;
}

}  // namespace mediapipe::tasks::c::components::containers
```

- [ ] **Step 4: Add BUILD targets** to `mediapipe/tasks/c/components/containers/BUILD` (after the `detection_result_converter` block):

```python
cc_library(
    name = "oriented_detection_result",
    hdrs = ["oriented_detection_result.h"],
    deps = [":category"],
)

cc_library(
    name = "oriented_detection_result_converter",
    srcs = ["oriented_detection_result_converter.cc"],
    hdrs = ["oriented_detection_result_converter.h"],
    deps = [
        ":category",
        ":category_converter",
        ":oriented_detection_result",
        "//mediapipe/tasks/cc/components/containers:oriented_object_detection_result",
    ],
)
```

- [ ] **Step 5: Build the converter**

Run: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/components/containers:oriented_detection_result_converter`
Expected: build succeeds.

- [ ] **Step 6: Commit**

```bash
git add mediapipe/tasks/c/components/containers/oriented_detection_result.h mediapipe/tasks/c/components/containers/oriented_detection_result_converter.h mediapipe/tasks/c/components/containers/oriented_detection_result_converter.cc mediapipe/tasks/c/components/containers/BUILD
git commit -m "feat(tasks-c-obb): oriented detection result C container + converter"
```

---

### Task 8: OBB C API header + impl + BUILD

**Files:**
- Create: `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h`
- Create: `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.cc`
- Create: `mediapipe/tasks/c/vision/oriented_object_detector/BUILD`

- [ ] **Step 1: Write the header** `oriented_object_detector.h`

```c
/* Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0. */

#ifndef MEDIAPIPE_TASKS_C_VISION_ORIENTED_OBJECT_DETECTOR_ORIENTED_OBJECT_DETECTOR_H_
#define MEDIAPIPE_TASKS_C_VISION_ORIENTED_OBJECT_DETECTOR_ORIENTED_OBJECT_DETECTOR_H_

#include <cstdint>

#include "mediapipe/tasks/c/components/containers/oriented_detection_result.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"

#ifndef MP_EXPORT
#if defined(_MSC_VER)
#define MP_EXPORT __declspec(dllexport)
#else
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // _MSC_VER
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MpOrientedObjectDetectorInternal* MpOrientedObjectDetectorPtr;
typedef MpOrientedDetectionResult MpOrientedObjectDetectorResult;

// Options for configuring a MediaPipe oriented (OBB) object detector task.
struct MpOrientedObjectDetectorOptions {
  struct MpBaseOptions base_options;
  MpRunningMode running_mode;

  // Max number of top-scored results. < 0 returns all; 0 is invalid.
  int max_results;

  // Score threshold overriding the model metadata value. Default 0.25.
  float score_threshold;

  // IoU threshold for rotated non-maximum suppression. Default 0.45.
  float iou_threshold;

  // If true, NMS is applied across all classes jointly.
  bool class_agnostic_nms;

  // Output tensor layout: CHANNELS_FIRST=1, CHANNELS_LAST=2.
  int layout;

  // Number of classes. If 0, derived from model metadata at graph build time.
  int num_classes;

  // Result callback for live-stream mode. Must be set iff running_mode is
  // MP_RUNNING_MODE_LIVE_STREAM. Valid only for the callback's lifetime.
  typedef void (*result_callback_fn)(
      MpStatus status, const MpOrientedObjectDetectorResult* result,
      const MpImagePtr image, int64_t timestamp_ms);
  result_callback_fn result_callback;
};

MP_EXPORT MpStatus MpOrientedObjectDetectorCreate(
    struct MpOrientedObjectDetectorOptions* options,
    MpOrientedObjectDetectorPtr* detector_out, char** error_msg);

MP_EXPORT MpStatus MpOrientedObjectDetectorDetectImage(
    MpOrientedObjectDetectorPtr detector, MpImagePtr image,
    const struct MpImageProcessingOptions* options,
    MpOrientedObjectDetectorResult* result, char** error_msg);

MP_EXPORT MpStatus MpOrientedObjectDetectorDetectForVideo(
    MpOrientedObjectDetectorPtr detector, MpImagePtr image,
    const struct MpImageProcessingOptions* options, int64_t timestamp_ms,
    MpOrientedObjectDetectorResult* result, char** error_msg);

MP_EXPORT MpStatus MpOrientedObjectDetectorDetectAsync(
    MpOrientedObjectDetectorPtr detector, MpImagePtr image,
    const struct MpImageProcessingOptions* options, int64_t timestamp_ms,
    char** error_msg);

MP_EXPORT void MpOrientedObjectDetectorCloseResult(
    MpOrientedObjectDetectorResult* result);

MP_EXPORT MpStatus MpOrientedObjectDetectorClose(
    MpOrientedObjectDetectorPtr detector, char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_ORIENTED_OBJECT_DETECTOR_ORIENTED_OBJECT_DETECTOR_H_
```

- [ ] **Step 2: Write the impl** `oriented_object_detector.cc` — identical structure to Task 2's YOLO impl, with these substitutions:
  - include `mediapipe/tasks/c/components/containers/oriented_detection_result_converter.h` and `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h`
  - `namespace ObbNs = ::mediapipe::tasks::vision::oriented_object_detector;`
  - `struct MpOrientedObjectDetectorInternal { std::unique_ptr<ObbNs::OrientedObjectDetector> instance; };`
  - result converters: `CppConvertToOrientedDetectionResult` / `CppCloseOrientedDetectionResult`
  - `CppConvertToDetectorOptions` sets ONLY: `max_results`, `score_threshold`, `iou_threshold`, `class_agnostic_nms`, `layout` (via `static_cast<ObbNs::OrientedObjectDetectorOptions::Layout>(in.layout)`), `num_classes` — NO display_names_locale/allowlist/denylist.
  - all function names `MpOrientedObjectDetector*` / `CppOrientedObjectDetector*`.

```cpp
/* Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0. */

#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/components/containers/oriented_detection_result_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options_converter.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h"

namespace ObbNs = ::mediapipe::tasks::vision::oriented_object_detector;

struct MpOrientedObjectDetectorInternal {
  std::unique_ptr<ObbNs::OrientedObjectDetector> instance;
};

namespace mediapipe::tasks::c::vision::oriented_object_detector {

namespace {

using ::mediapipe::Image;
using ::mediapipe::tasks::c::components::containers::
    CppCloseOrientedDetectionResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToOrientedDetectionResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
using ::mediapipe::tasks::vision::core::RunningMode;
using CppObbResult = ObbNs::OrientedObjectDetectorResult;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

ObbNs::OrientedObjectDetector* GetCppDetector(
    MpOrientedObjectDetectorPtr wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "OrientedObjectDetector is null.";
  return wrapper->instance.get();
}

}  // namespace

void CppConvertToDetectorOptions(
    const MpOrientedObjectDetectorOptions& in,
    ObbNs::OrientedObjectDetectorOptions* out) {
  out->max_results = in.max_results;
  out->score_threshold = in.score_threshold;
  out->iou_threshold = in.iou_threshold;
  out->class_agnostic_nms = in.class_agnostic_nms;
  out->layout =
      static_cast<ObbNs::OrientedObjectDetectorOptions::Layout>(in.layout);
  out->num_classes = in.num_classes;
}

absl::Status CppOrientedObjectDetectorCreate(
    const MpOrientedObjectDetectorOptions& options,
    MpOrientedObjectDetectorPtr* detector_out) {
  auto cpp_options = std::make_unique<ObbNs::OrientedObjectDetectorOptions>();
  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToDetectorOptions(options, cpp_options.get());
  cpp_options->running_mode = static_cast<RunningMode>(options.running_mode);

  if (cpp_options->running_mode == RunningMode::LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      return absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
    }
    MpOrientedObjectDetectorOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppObbResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          MpImageInternal mp_image({.image = image});
          if (!cpp_result.ok()) {
            result_callback(ToMpStatus(cpp_result.status()), nullptr, &mp_image,
                            timestamp);
            return;
          }
          MpOrientedObjectDetectorResult result;
          CppConvertToOrientedDetectionResult(*cpp_result, &result);
          result_callback(kMpOk, &result, &mp_image, timestamp);
          CppCloseOrientedDetectionResult(&result);
        };
  }

  auto detector = ObbNs::OrientedObjectDetector::Create(std::move(cpp_options));
  if (!detector.ok()) {
    return detector.status();
  }
  *detector_out =
      new MpOrientedObjectDetectorInternal{.instance = std::move(*detector)};
  return absl::OkStatus();
}

absl::Status CppOrientedObjectDetectorDetect(
    MpOrientedObjectDetectorPtr detector, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    MpOrientedObjectDetectorResult* result) {
  auto cpp_detector = GetCppDetector(detector);
  std::optional<CppImageProcessingOptions> cpp_opts;
  if (image_processing_options) {
    CppImageProcessingOptions o;
    CppConvertToImageProcessingOptions(*image_processing_options, &o);
    cpp_opts = o;
  }
  auto cpp_result = cpp_detector->Detect(ToImage(image), cpp_opts);
  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToOrientedDetectionResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppOrientedObjectDetectorDetectForVideo(
    MpOrientedObjectDetectorPtr detector, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, MpOrientedObjectDetectorResult* result) {
  auto cpp_detector = GetCppDetector(detector);
  std::optional<CppImageProcessingOptions> cpp_opts;
  if (image_processing_options) {
    CppImageProcessingOptions o;
    CppConvertToImageProcessingOptions(*image_processing_options, &o);
    cpp_opts = o;
  }
  auto cpp_result =
      cpp_detector->DetectForVideo(ToImage(image), timestamp_ms, cpp_opts);
  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToOrientedDetectionResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppOrientedObjectDetectorDetectAsync(
    MpOrientedObjectDetectorPtr detector, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  auto cpp_detector = GetCppDetector(detector);
  std::optional<CppImageProcessingOptions> cpp_opts;
  if (image_processing_options) {
    CppImageProcessingOptions o;
    CppConvertToImageProcessingOptions(*image_processing_options, &o);
    cpp_opts = o;
  }
  return cpp_detector->DetectAsync(ToImage(image), timestamp_ms, cpp_opts);
}

void CppOrientedObjectDetectorCloseResult(
    MpOrientedObjectDetectorResult* result) {
  CppCloseOrientedDetectionResult(result);
}

absl::Status CppOrientedObjectDetectorClose(
    MpOrientedObjectDetectorPtr detector) {
  auto cpp_detector = GetCppDetector(detector);
  auto result = cpp_detector->Close();
  if (!result.ok()) {
    return result;
  }
  delete detector;
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::vision::oriented_object_detector

extern "C" {

MpStatus MpOrientedObjectDetectorCreate(
    struct MpOrientedObjectDetectorOptions* options,
    MpOrientedObjectDetectorPtr* detector_out, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::oriented_object_detector::
      CppOrientedObjectDetectorCreate(*options, detector_out);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MpStatus MpOrientedObjectDetectorDetectImage(
    MpOrientedObjectDetectorPtr detector, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    MpOrientedObjectDetectorResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::oriented_object_detector::
      CppOrientedObjectDetectorDetect(detector, image, image_processing_options,
                                      result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MpStatus MpOrientedObjectDetectorDetectForVideo(
    MpOrientedObjectDetectorPtr detector, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, MpOrientedObjectDetectorResult* result,
    char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::oriented_object_detector::
      CppOrientedObjectDetectorDetectForVideo(detector, image,
                                              image_processing_options,
                                              timestamp_ms, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MpStatus MpOrientedObjectDetectorDetectAsync(
    MpOrientedObjectDetectorPtr detector, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::oriented_object_detector::
      CppOrientedObjectDetectorDetectAsync(detector, image,
                                           image_processing_options,
                                           timestamp_ms);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

void MpOrientedObjectDetectorCloseResult(
    MpOrientedObjectDetectorResult* result) {
  mediapipe::tasks::c::vision::oriented_object_detector::
      CppOrientedObjectDetectorCloseResult(result);
}

MpStatus MpOrientedObjectDetectorClose(MpOrientedObjectDetectorPtr detector,
                                       char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::oriented_object_detector::
      CppOrientedObjectDetectorClose(detector);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
```

- [ ] **Step 3: Write the BUILD** — identical to Task 2's YOLO BUILD with these renames: target prefix `oriented_object_detector`, the result container deps become `oriented_detection_result` + `oriented_detection_result_converter`, and the cc task dep becomes `//mediapipe/tasks/cc/vision/oriented_object_detector`.

- [ ] **Step 4: Build the library**

Run: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_lib`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.cc mediapipe/tasks/c/vision/oriented_object_detector/BUILD
git commit -m "feat(tasks-c-obb): oriented object detector C API + BUILD"
```

---

### Task 9: Register OBB in the libmediapipe.so aggregator

**Files:**
- Modify: `mediapipe/tasks/c/BUILD`

- [ ] **Step 1: Add the dep** after the YOLO line from Task 3:

```python
        "//mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_c_lib",
```

- [ ] **Step 2: Build + verify symbols**

Run: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c:libmediapipe.so && nm -D bazel-bin/mediapipe/tasks/c/libmediapipe.so | grep MpOrientedObjectDetector`
Expected: build succeeds; the 6 `MpOrientedObjectDetector*` symbols are listed.

- [ ] **Step 3: Commit**

```bash
git add mediapipe/tasks/c/BUILD
git commit -m "build(tasks-c): link oriented detector C API into libmediapipe.so"
```

---

### Task 10: OBB C gated integration test

**Files:**
- Create: `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector_test.cc`

- [ ] **Step 1: Write the gated test** — mirror Task 4 with these changes: `kOrientedModel = "yolov8n-obb.tflite"`, `num_classes = 15` (DOTA), `layout = 2`, result type `MpOrientedObjectDetectorResult`, and per-detection assertions:

```cpp
  // ... inside the (gated) test body, after DetectImage returns kMpOk:
  EXPECT_GT(result.detections_count, 0u);
  for (uint32_t i = 0; i < result.detections_count; ++i) {
    EXPECT_GT(result.detections[i].width, 0.0f);
    EXPECT_GT(result.detections[i].height, 0.0f);
    EXPECT_TRUE(std::isfinite(result.detections[i].rotation));
    EXPECT_EQ(result.detections[i].categories_count, 1u);
    EXPECT_GT(result.detections[i].categories[0].score, 0.0f);
  }
  MpOrientedObjectDetectorCloseResult(&result);
```

(Include `<cmath>` for `std::isfinite`. The skip guard, image loading, and create/close calls are structurally identical to Task 4.)

- [ ] **Step 2: Add the `cc_test` to the OBB BUILD from Task 8** (mirror Task 2's test rule with the oriented target names).

- [ ] **Step 3: Run the gated test (skips)**

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_test --test_output=all`
Expected: PASS with the test reported SKIPPED.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector_test.cc mediapipe/tasks/c/vision/oriented_object_detector/BUILD
git commit -m "test(tasks-c-obb): gated oriented detector C API test"
```

---

### Task 11: OBB Python ctypes container + dataclass + BUILD

**Files:**
- Create: `mediapipe/tasks/python/components/containers/oriented_detections_c.py`
- Create: `mediapipe/tasks/python/components/containers/oriented_detections.py`
- Modify: `mediapipe/tasks/python/components/containers/BUILD`

- [ ] **Step 1: Write the ctypes mirror** `oriented_detections_c.py` (field order matches `MpOrientedDetection`)

```python
# Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0.
"""MediaPipe Oriented Detection Result C API types."""

import ctypes

from mediapipe.tasks.python.components.containers import category_c


class MpOrientedDetectionC(ctypes.Structure):
  """CTypes for a single oriented detection (pixel units)."""

  _fields_ = [
      ('categories', ctypes.POINTER(category_c.MpCategoryC)),
      ('categories_count', ctypes.c_uint32),
      ('cx', ctypes.c_float),
      ('cy', ctypes.c_float),
      ('width', ctypes.c_float),
      ('height', ctypes.c_float),
      ('rotation', ctypes.c_float),
  ]


class MpOrientedDetectionResultC(ctypes.Structure):
  """CTypes for the oriented detection result."""

  _fields_ = [
      ('detections', ctypes.POINTER(MpOrientedDetectionC)),
      ('detections_count', ctypes.c_uint32),
  ]
```

- [ ] **Step 2: Write the dataclass** `oriented_detections.py`

```python
# Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0.
"""Oriented (rotated bounding box) detections data class."""

import dataclasses
from typing import Any, List

from mediapipe.tasks.python.components.containers import category as category_lib
from mediapipe.tasks.python.components.containers import category_c as category_c_lib
from mediapipe.tasks.python.components.containers import oriented_detections_c as oriented_detections_c_lib
from mediapipe.tasks.python.core.optional_dependencies import doc_controls


@dataclasses.dataclass
class OrientedDetection:
  """One oriented (rotated) bounding box detection, in original-image PIXELS.

  Attributes:
    categories: A list of Category objects.
    cx: Box center x, in pixels.
    cy: Box center y, in pixels.
    width: Box width, in pixels.
    height: Box height, in pixels.
    rotation: Rotation angle in radians, counter-clockwise.
  """

  categories: List[category_lib.Category]
  cx: float
  cy: float
  width: float
  height: float
  rotation: float

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, OrientedDetection):
      return False
    return (
        self.categories == other.categories
        and self.cx == other.cx
        and self.cy == other.cy
        and self.width == other.width
        and self.height == other.height
        and self.rotation == other.rotation
    )

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(
      cls, c_obj: oriented_detections_c_lib.MpOrientedDetectionC
  ) -> 'OrientedDetection':
    c_categories = category_c_lib.MpCategoriesC(
        categories=c_obj.categories, categories_count=c_obj.categories_count
    )
    py_categories = category_lib.create_list_of_categories_from_ctypes(
        c_categories
    )
    return OrientedDetection(
        categories=py_categories,
        cx=c_obj.cx,
        cy=c_obj.cy,
        width=c_obj.width,
        height=c_obj.height,
        rotation=c_obj.rotation,
    )


@dataclasses.dataclass
class OrientedObjectDetectionResult:
  """The list of detected oriented objects.

  Attributes:
    detections: A list of `OrientedDetection` objects.
  """

  detections: List[OrientedDetection]

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, OrientedObjectDetectionResult):
      return False
    return self.detections == other.detections

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(
      cls, c_obj: oriented_detections_c_lib.MpOrientedDetectionResultC
  ) -> 'OrientedObjectDetectionResult':
    return OrientedObjectDetectionResult(
        detections=[
            OrientedDetection.from_ctypes(c_obj.detections[i])
            for i in range(c_obj.detections_count)
        ]
    )
```

- [ ] **Step 3: Add py_library targets** to `mediapipe/tasks/python/components/containers/BUILD` (mirror `detections_c` and `detections`):

```python
py_library(
    name = "oriented_detections_c",
    srcs = ["oriented_detections_c.py"],
    deps = [
        "//mediapipe/tasks/python/components/containers:category_c",
    ],
)

py_library(
    name = "oriented_detections",
    srcs = ["oriented_detections.py"],
    deps = [
        ":oriented_detections_c",
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:category_c",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)
```

- [ ] **Step 4: Build both py_libraries**

Run: `bazel build //mediapipe/tasks/python/components/containers:oriented_detections`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/tasks/python/components/containers/oriented_detections_c.py mediapipe/tasks/python/components/containers/oriented_detections.py mediapipe/tasks/python/components/containers/BUILD
git commit -m "feat(tasks-py-obb): oriented detections ctypes + dataclass containers"
```

---

### Task 12: OBB Python ctypes task class + BUILD

**Files:**
- Create: `mediapipe/tasks/python/vision/oriented_object_detector.py`
- Modify: `mediapipe/tasks/python/vision/BUILD`

- [ ] **Step 1: Write the task class** — structurally identical to Task 5's YOLO class, with these substitutions:
  - import `oriented_detections` + `oriented_detections_c` instead of `detections`/`detections_c`
  - `OrientedObjectDetectorResult = oriented_detections_module.OrientedObjectDetectionResult`
  - `_C_TYPES_RESULT_CALLBACK` points to `ctypes.POINTER(oriented_detections_c_module.MpOrientedDetectionResultC)`
  - `MpOrientedObjectDetectorOptionsC._fields_` (order matches the OBB C header):
    ```python
    _fields_ = [
        ('base_options', base_options_c_module.MpBaseOptionsC),
        ('running_mode', ctypes.c_int),
        ('max_results', ctypes.c_int),
        ('score_threshold', ctypes.c_float),
        ('iou_threshold', ctypes.c_float),
        ('class_agnostic_nms', ctypes.c_bool),
        ('layout', ctypes.c_int),
        ('num_classes', ctypes.c_int),
        ('result_callback', _C_TYPES_RESULT_CALLBACK),
    ]
    ```
  - `_CTYPES_SIGNATURES` uses the `MpOrientedObjectDetector*` names and `MpOrientedDetectionResultC` pointers.
  - `OrientedObjectDetectorOptions` dataclass fields: `base_options`, `running_mode=IMAGE`, `max_results=-1`, `score_threshold=0.25`, `iou_threshold=0.45`, `class_agnostic_nms=False`, `layout=Layout.CHANNELS_FIRST`, `num_classes=0`, `result_callback=None` (NO display_names_locale/allowlist/denylist).
  - reuse the same `Layout` enum (define locally, identical to Task 5).
  - reuse the same `_load_label_map` and `_enrich_with_label_map` helpers (copy them; `_enrich_with_label_map` iterates `result.detections` → `detection.categories`, identical shape).
  - `create_from_options` builds `MpOrientedObjectDetectorOptionsC(...)` WITHOUT allowlist/denylist args; everything else mirrors Task 5.
  - `detect` / `detect_for_video` use `oriented_detections_c_module.MpOrientedDetectionResultC()` and the `MpOrientedObjectDetector*` lib calls; return `_enrich_with_label_map(OrientedObjectDetectorResult.from_ctypes(c_result), self._label_map)`.

- [ ] **Step 2: Add the py_library to `mediapipe/tasks/python/vision/BUILD`** (mirror Task 5's, swapping detections deps for the oriented ones):

```python
py_library(
    name = "oriented_object_detector",
    srcs = ["oriented_object_detector.py"],
    deps = [
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:oriented_detections",
        "//mediapipe/tasks/python/components/containers:oriented_detections_c",
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:base_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:mediapipe_c_utils",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//mediapipe/tasks/python/metadata",
        "//mediapipe/tasks/python/vision/core:image",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:image_processing_options_c",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)
```

- [ ] **Step 3: Build the py_library**

Run: `bazel build //mediapipe/tasks/python/vision:oriented_object_detector`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/python/vision/oriented_object_detector.py mediapipe/tasks/python/vision/BUILD
git commit -m "feat(tasks-py-obb): oriented object detector Python ctypes task"
```

---

### Task 13: OBB Python gated test + BUILD

**Files:**
- Create: `mediapipe/tasks/python/test/vision/oriented_object_detector_test.py`
- Modify: `mediapipe/tasks/python/test/vision/BUILD`

- [ ] **Step 1: Write the gated test** — mirror Task 6 with: `_MODEL_FILE = 'yolov8n-obb.tflite'`, `num_classes=15`, import `oriented_object_detector`, and the gated assertions:

```python
  @unittest.skipUnless(_model_available(), 'yolov8n-obb.tflite fixture absent')
  def test_detect_image(self):
    image = image_module.Image.create_from_file(
        test_utils.get_test_data_path(_IMAGE_FILE)
    )
    options = _OrientedObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=_model_path()),
        running_mode=_RunningMode.IMAGE,
        max_results=10,
        score_threshold=0.25,
        layout=oriented_object_detector.Layout.CHANNELS_LAST,
        num_classes=15,
    )
    with _OrientedObjectDetector.create_from_options(options) as detector:
      result = detector.detect(image)
      self.assertNotEmpty(result.detections)
      for detection in result.detections:
        self.assertGreater(detection.width, 0.0)   # pixel units
        self.assertGreater(detection.height, 0.0)
        self.assertTrue(math.isfinite(detection.rotation))
        self.assertLen(detection.categories, 1)
        self.assertGreater(detection.categories[0].score, 0.0)
```

(Add `import math`. The `test_options_construct_without_model` case mirrors Task 6 but asserts `options.score_threshold == 0.25` and `options.class_agnostic_nms is False`.)

- [ ] **Step 2: Add the test target to `mediapipe/tasks/python/test/vision/BUILD`** (mirror Task 6, swapping `:detections` for `:oriented_detections` and the vision dep for `:oriented_object_detector`).

- [ ] **Step 3: Run the gated test (skips inference)**

Run: `bazel test //mediapipe/tasks/python/test/vision:oriented_object_detector_test --test_output=all`
Expected: PASS; `test_detect_image` SKIPPED, options test runs.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/python/test/vision/oriented_object_detector_test.py mediapipe/tasks/python/test/vision/BUILD
git commit -m "test(tasks-py-obb): gated oriented detector Python API test"
```

---

## Final verification (after all tasks)

- [ ] Build the whole new surface + the shared library:

```bash
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/tasks/c:libmediapipe.so \
  //mediapipe/tasks/c/vision/yolo_object_detector:all \
  //mediapipe/tasks/c/vision/oriented_object_detector:all \
  //mediapipe/tasks/python/vision:yolo_object_detector \
  //mediapipe/tasks/python/vision:oriented_object_detector
```

- [ ] Run all four gated tests; confirm clean skips:

```bash
bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_test \
  //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_test \
  //mediapipe/tasks/python/test/vision:yolo_object_detector_test \
  //mediapipe/tasks/python/test/vision:oriented_object_detector_test
```

- [ ] Confirm `object_detector` (C + Python) and the shared `detection_result*` / `detections*` files are untouched: `git diff --stat master -- mediapipe/tasks/c/vision/object_detector mediapipe/tasks/python/vision/object_detector.py` shows no changes.
- [ ] `git status` clean.
```
