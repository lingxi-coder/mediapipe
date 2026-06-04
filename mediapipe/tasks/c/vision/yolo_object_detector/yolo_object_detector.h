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
// display_names_locale, category_allowlist, and category_denylist are applied
// by the YOLO graph: category names are read from the model metadata's label
// file and allow/deny filter results by class name (resolved to indices).
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
