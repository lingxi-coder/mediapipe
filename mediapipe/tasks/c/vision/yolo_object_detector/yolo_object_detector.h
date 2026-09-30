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

// A frame-normalized tile given by its CENTER point and size (NOT corner-based
// like RectF). Mirrors YoloObjectDetectorOptions::TilingOptions::TileRect.
struct MpTileRect {
  float x_center;
  float y_center;
  float width;
  float height;
};

// Static tiling configuration. Mirrors
// YoloObjectDetectorOptions::TilingOptions field-for-field.
//
// Tiling is ENABLED when tile_rows * tile_cols > 1 or explicit_tiles_count > 0.
// A zero-initialized MpTilingOptions (e.g. from `MpYoloObjectDetectorOptions
// options = {}`) therefore means tiling DISABLED -- byte-identical to a C caller
// that never set tiling at all. To tile with a grid, set BOTH tile_rows and
// tile_cols (each >= 1); a zero in either disables tiling.
struct MpTilingOptions {
  // Grid tiling: rows x cols of equal tiles. Mutually exclusive with
  // explicit_tiles.
  int tile_rows;
  int tile_cols;
  // Fractional overlap added around each grid tile. Default 0.0.
  float tile_overlap_fraction;

  // Explicit (non-grid) tiles. Caller owns this array; it is COPIED during
  // Create and need not outlive the MpYoloObjectDetectorCreate call. Mirrors
  // the category_allowlist (pointer + count) ownership convention.
  const struct MpTileRect* explicit_tiles;
  uint32_t explicit_tiles_count;

  // Per-tile (in-decoder) NMS IoU threshold; <= 0 disables.
  float tile_local_nms_iou_threshold;
  // Per-tile cap after tile-local NMS; <= 0 disables.
  int max_detections_after_tile_nms;

  // VIDEO/LIVE_STREAM only: gate per-frame tiled inference with a motion
  // scheduler. Setting this in IMAGE mode is rejected by the C++ Create()
  // (surfaced as a non-kMpOk MpStatus); the binding only passes it through.
  bool enable_motion_scheduling;
  // Per DETECT-frame cap on inferred tiles (motion-prioritized). 0 = all.
  int max_scheduled_tiles;
};

// Tracker selection for the tiled VIDEO/LIVE_STREAM path. Mirrors
// YoloObjectDetectorOptions::TrackingOptions field-for-field.
//
// tracker_type: 0 = unspecified (-> BOX_TRACKER), 1 = BOX_TRACKER (optical-flow,
// default), 2 = BOTSORT (tracking-by-detection, motion-only). Numerically equal
// to the C++/proto enum. A zero-initialized MpTrackingOptions therefore means
// BOX_TRACKER -- byte-identical to a C caller who never set tracking at all.
//
// The threshold/buffer knobs are honored only for BOTSORT. A plain C struct
// cannot distinguish "unset" from 0, so a BOTSORT caller MUST set the knobs
// explicitly (a zero-init struct yields 0.0 thresholds); the C++ struct defaults
// (0.6/0.1/0.7/30/0.7) are not reachable through a zero-init C struct.
// BOX_TRACKER ignores the knobs. (Same convention as score_threshold.)
//
// NOTE: BoTSORT only emits a track_id for confirmed tracks. Set
// track_high_threshold / new_track_threshold at or below your detection
// score_threshold, otherwise low-confidence detections never confirm and no
// track_id is ever produced.
struct MpTrackingOptions {
  int tracker_type;
  float track_high_threshold;
  float track_low_threshold;
  float new_track_threshold;
  int track_buffer;
  float match_threshold;
  bool enable_gmc;
  // Nominal input frame rate used to scale BoTSORT's retention window. Read
  // only by MpYoloObjectDetectorCreateV2; the legacy Create always uses 30 FPS
  // because this offset was padding in the previous public ABI.
  int nominal_frame_rate;
};

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

  // Static tiling configuration. Zero-initialized => tiling disabled.
  struct MpTilingOptions tiling;

  // Tracker selection. Zero-initialized => BOX_TRACKER (the existing default).
  struct MpTrackingOptions tracking;

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

// Creates a YoloObjectDetector and honors tracking.nominal_frame_rate. Use
// this entry point with the current options struct; zero means 30 FPS.
MP_EXPORT MpStatus
MpYoloObjectDetectorCreateV2(struct MpYoloObjectDetectorOptions* options,
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

// Frees the detector even if shutdown returns an error. The pointer is invalid
// after this call regardless of the returned status and must not be reused.
MP_EXPORT MpStatus MpYoloObjectDetectorClose(MpYoloObjectDetectorPtr detector,
                                             char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_YOLO_OBJECT_DETECTOR_H_
