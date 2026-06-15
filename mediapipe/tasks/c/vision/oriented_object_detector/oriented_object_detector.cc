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

#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"

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
#include "mediapipe/tasks/c/components/containers/oriented_detection_result_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options_converter.h"
#include "mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_converter.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h"

struct MpOrientedObjectDetectorInternal {
  std::unique_ptr<
      ::mediapipe::tasks::vision::oriented_object_detector::OrientedObjectDetector>
      instance;
};

namespace mediapipe::tasks::c::vision::oriented_object_detector {

namespace ObbNs = ::mediapipe::tasks::vision::oriented_object_detector;

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

void CppConvertToDetectorOptions(const MpOrientedObjectDetectorOptions& in,
                                 ObbNs::OrientedObjectDetectorOptions* out) {
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
  out->class_agnostic_nms = in.class_agnostic_nms;
  out->layout =
      static_cast<ObbNs::OrientedObjectDetectorOptions::Layout>(in.layout);
  out->num_classes = in.num_classes;
  CppConvertToTilingOptions(in.tiling, &out->tiling);
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
