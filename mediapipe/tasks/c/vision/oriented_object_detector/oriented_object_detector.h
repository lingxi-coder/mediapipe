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
  // MP_RUNNING_MODE_LIVE_STREAM. The arguments passed to the callback are valid
  // only for the duration of the callback invocation.
  typedef void (*result_callback_fn)(
      MpStatus status, const MpOrientedObjectDetectorResult* result,
      const MpImagePtr image, int64_t timestamp_ms);
  result_callback_fn result_callback;
};

// Creates an OrientedObjectDetector from the provided `options`.
MP_EXPORT MpStatus MpOrientedObjectDetectorCreate(
    struct MpOrientedObjectDetectorOptions* options,
    MpOrientedObjectDetectorPtr* detector_out, char** error_msg);

// Performs oriented detection on a single image.
MP_EXPORT MpStatus MpOrientedObjectDetectorDetectImage(
    MpOrientedObjectDetectorPtr detector, MpImagePtr image,
    const struct MpImageProcessingOptions* options,
    MpOrientedObjectDetectorResult* result, char** error_msg);

// Performs oriented detection on a video frame (monotonically increasing
// timestamps).
MP_EXPORT MpStatus MpOrientedObjectDetectorDetectForVideo(
    MpOrientedObjectDetectorPtr detector, MpImagePtr image,
    const struct MpImageProcessingOptions* options, int64_t timestamp_ms,
    MpOrientedObjectDetectorResult* result, char** error_msg);

// Sends live image data; results delivered via the configured result_callback.
MP_EXPORT MpStatus MpOrientedObjectDetectorDetectAsync(
    MpOrientedObjectDetectorPtr detector, MpImagePtr image,
    const struct MpImageProcessingOptions* options, int64_t timestamp_ms,
    char** error_msg);

// Frees memory allocated inside a result. Does not free the result pointer.
MP_EXPORT void MpOrientedObjectDetectorCloseResult(
    MpOrientedObjectDetectorResult* result);

// Frees the detector.
MP_EXPORT MpStatus MpOrientedObjectDetectorClose(
    MpOrientedObjectDetectorPtr detector, char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_ORIENTED_OBJECT_DETECTOR_ORIENTED_OBJECT_DETECTOR_H_
