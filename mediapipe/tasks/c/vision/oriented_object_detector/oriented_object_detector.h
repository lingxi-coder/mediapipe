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

// A frame-normalized tile given by its CENTER point and size. Mirrors
// OrientedObjectDetectorOptions::TilingOptions::TileRect. Named MpOrientedTileRect
// (not MpTileRect) to avoid colliding with the YOLO detector's distinct global
// extern "C" MpTileRect.
struct MpOrientedTileRect {
  float x_center;
  float y_center;
  float width;
  float height;
};

// Static tiling configuration.
// Mirrors the 6 logical OrientedObjectDetectorOptions::TilingOptions fields; the
// std::vector explicit_tiles is exposed as a pointer + count pair, so this struct
// has 7 members. OBB has no motion-scheduling knobs.
//
// Tiling is ENABLED when tile_rows * tile_cols > 1 or explicit_tiles_count > 0. A
// zero-initialized MpOrientedTilingOptions means tiling DISABLED. To tile with a
// grid, set BOTH tile_rows and tile_cols (each >= 1); a zero in either disables.
struct MpOrientedTilingOptions {
  int tile_rows;
  int tile_cols;
  float tile_overlap_fraction;
  // Explicit (non-grid) tiles. Caller owns this array; it is COPIED during Create
  // and need not outlive the MpOrientedObjectDetectorCreate call (mirrors the
  // category_allowlist pointer+count convention).
  const struct MpOrientedTileRect* explicit_tiles;
  uint32_t explicit_tiles_count;
  float tile_local_nms_iou_threshold;
  int max_detections_after_tile_nms;
};

// Tracker selection for the OBB tiled VIDEO/LIVE_STREAM path. Mirrors
// OrientedObjectDetectorOptions::TrackingOptions field-for-field.
//
// tracker_type: 0 = unspecified (-> no tracking, the OBB default), 1 =
// BOX_TRACKER (NOT supported for OBB -- rejected by Create()), 2 = BOTSORT
// (tracking-by-detection, motion-only). Numerically equal to the C++/proto enum.
// A zero-initialized MpOrientedTrackingOptions therefore means NO tracking --
// byte-identical to a C caller who never set tracking at all.
//
// The threshold/buffer knobs are honored only for BOTSORT. A plain C struct
// cannot distinguish "unset" from 0, so a BOTSORT caller MUST set them
// explicitly. BoTSORT only emits a track_id for CONFIRMED tracks, so
// track_high_threshold / new_track_threshold must be at/below the detector's
// score_threshold or no track_id is produced.
struct MpOrientedTrackingOptions {
  int tracker_type;
  float track_high_threshold;
  float track_low_threshold;
  float new_track_threshold;
  int track_buffer;
  float match_threshold;
  bool enable_gmc;
};

// Options for configuring a MediaPipe oriented (OBB) object detector task.
struct MpOrientedObjectDetectorOptions {
  struct MpBaseOptions base_options;
  MpRunningMode running_mode;

  // Locale for display names in TFLite metadata, if any. Defaults to English.
  const char* display_names_locale;

  // Max number of top-scored results. < 0 returns all; 0 is invalid.
  int max_results;

  // Score threshold overriding the model metadata value. The C binding copies
  // this verbatim; a zero-initialized struct yields 0.0 (NOT the C++/Python
  // default of 0.25), so set it explicitly.
  float score_threshold;

  // Allowlist of category names (mutually exclusive with denylist).
  const char** category_allowlist;
  uint32_t category_allowlist_count;

  // Denylist of category names (mutually exclusive with allowlist).
  const char** category_denylist;
  uint32_t category_denylist_count;

  // IoU threshold for rotated non-maximum suppression. Default 0.45.
  float iou_threshold;

  // If true, NMS is applied across all classes jointly.
  bool class_agnostic_nms;

  // Output tensor layout: CHANNELS_FIRST=1, CHANNELS_LAST=2.
  int layout;

  // Number of classes. If 0, derived from model metadata at graph build time.
  int num_classes;

  // Static tiling configuration. Zero-initialized => tiling disabled.
  struct MpOrientedTilingOptions tiling;

  // Tracker selection. Zero-initialized => no tracking (the OBB default).
  struct MpOrientedTrackingOptions tracking;

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
