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
#include "mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.h"
#include "mediapipe/tasks/c/vision/yolo_object_detector/tracking_options_converter.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

struct MpYoloObjectDetectorInternal {
  std::unique_ptr<
      ::mediapipe::tasks::vision::yolo_object_detector::YoloObjectDetector>
      instance;
};

namespace mediapipe::tasks::c::vision::yolo_object_detector {

namespace YoloNs = ::mediapipe::tasks::vision::yolo_object_detector;

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
  CppConvertToTilingOptions(in.tiling, &out->tiling);
  CppConvertToTrackingOptions(in.tracking, &out->tracking);
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
