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

#ifndef MEDIAPIPE_TASKS_CC_VISION_ORIENTED_OBJECT_DETECTOR_ORIENTED_OBJECT_DETECTOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_ORIENTED_OBJECT_DETECTOR_ORIENTED_OBJECT_DETECTOR_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/oriented_object_detector/proto/oriented_object_detector_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace oriented_object_detector {

// Alias the shared OrientedObjectDetectionResult struct as result type.
using OrientedObjectDetectorResult =
    ::mediapipe::tasks::components::containers::OrientedObjectDetectionResult;

// The options for configuring a mediapipe oriented object detector task.
struct OrientedObjectDetectorOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the TfLite
  // model file with metadata, accelerator options, op resolver, etc.
  tasks::core::BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // Oriented object detector has three running modes:
  // 1) The image mode for detecting objects on single image inputs.
  // 2) The video mode for detecting objects on the decoded frames of a video.
  // 3) The live stream mode for detecting objects on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the detection results asynchronously.
  core::RunningMode running_mode = core::RunningMode::IMAGE;

  // The maximum number of top-scored detection results to return. If < 0, all
  // available results will be returned. If 0, an invalid argument error is
  // returned. Note that models may intrinsically be limited to returning a
  // maximum number of results N: if the provided value here is above N, only N
  // results will be returned.
  int max_results = -1;

  // Score threshold to override the one provided in the model metadata (if
  // any). Detection results with a score below this value are rejected.
  float score_threshold = 0.25f;

  // IoU threshold for non-maximum suppression. Default 0.45.
  float iou_threshold = 0.45f;

  // If true, NMS is applied across all classes jointly.
  bool class_agnostic_nms = false;

  // Output tensor layout of the detect head.
  // NOTE: these values must stay numerically equal to the proto enum
  // OrientedObjectDetectorOptions.Layout (CHANNELS_FIRST=1, CHANNELS_LAST=2) —
  // the options→proto conversion static_casts between them. Keep in sync if the
  // proto is ever reordered.
  enum Layout {
    kChannelsFirst = 1,  // NCHW-style: [batch, (cx,cy,w,h,angle,scores...), anchors]
    kChannelsLast = 2,   // NHWC-style: [batch, anchors, (cx,cy,w,h,angle,scores...)]
  };
  Layout layout = kChannelsFirst;

  // Number of classes. If 0, derived from model metadata at graph build time.
  int num_classes = 0;

  // The locale to use for display names specified through the TFLite Model
  // Metadata, if any. Defaults to English.
  std::string display_names_locale = "en";

  // The allowlist of category names. If non-empty, detection results whose
  // category name is not in this set will be filtered out. Duplicate or unknown
  // category names are ignored. Mutually exclusive with category_denylist.
  std::vector<std::string> category_allowlist = {};

  // The denylist of category names. If non-empty, detection results whose
  // category name is in this set will be filtered out. Duplicate or unknown
  // category names are ignored. Mutually exclusive with category_allowlist.
  std::vector<std::string> category_denylist = {};

  // Static tiling configuration. Tiling is enabled when
  // tile_rows * tile_cols > 1 or explicit_tiles is non-empty.
  struct TilingOptions {
    int tile_rows = 1;
    int tile_cols = 1;
    float tile_overlap_fraction = 0.0f;
    // A frame-normalized tile given by its CENTER point and size (NOT
    // corner-based like RectF). Mutually exclusive with the grid params.
    struct TileRect {
      float x_center = 0.0f;
      float y_center = 0.0f;
      float width = 0.0f;
      float height = 0.0f;
    };
    std::vector<TileRect> explicit_tiles;
    // Per-tile (in-decoder) rotated NMS; <= 0 disables.
    float tile_local_nms_iou_threshold = 0.0f;
    // Per-tile cap after tile-local NMS; <= 0 disables.
    int max_detections_after_tile_nms = 0;
  };
  TilingOptions tiling;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM.
  std::function<void(absl::StatusOr<OrientedObjectDetectorResult>, const Image&,
                     int64_t)>
      result_callback = nullptr;
};

// Converts a public OrientedObjectDetectorOptions struct into the corresponding
// proto. Exposed here (outside the anonymous namespace in the .cc) so that unit
// tests can verify the options→proto mapping without constructing a live graph.
std::unique_ptr<proto::OrientedObjectDetectorOptions>
ConvertOrientedObjectDetectorOptionsToProto(
    OrientedObjectDetectorOptions* options);

// Performs oriented (rotated bounding box) object detection on single images,
// video frames, or live stream.
//
// The API expects a TFLite model with a single output tensor encoding
// oriented (rotated) bounding boxes with class scores.
//
// Input tensor:
//   (kTfLiteUInt8/kTfLiteFloat32)
//    - image input of size `[batch x height x width x channels]`.
//    - batch inference is not supported (`batch` is required to be 1).
//    - only RGB inputs are supported (`channels` is required to be 3).
//    - if type is kTfLiteFloat32, NormalizationOptions are required to be
//      attached to the metadata for input normalization.
class OrientedObjectDetector : public tasks::vision::core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates an OrientedObjectDetector from an OrientedObjectDetectorOptions to
  // process image data or streaming data. Oriented object detector can be
  // created with one of the following three running modes:
  // 1) Image mode for detecting objects on single image inputs.
  //    Users provide mediapipe::Image to the `Detect` method, and will
  //    receive the detection results as the return value.
  // 2) Video mode for detecting objects on the decoded frames of a video.
  // 3) Live stream mode for detecting objects on the live stream of the input
  //    data, such as from camera. Users call `DetectAsync` to push the image
  //    data into the OrientedObjectDetector, the detection results along with
  //    the input timestamp and the image that object detector runs on will be
  //    available in the result callback when the object detector finishes
  //    the work.
  static absl::StatusOr<std::unique_ptr<OrientedObjectDetector>> Create(
      std::unique_ptr<OrientedObjectDetectorOptions> options);

  // Performs object detection on the provided single image.
  // Only use this method when the OrientedObjectDetector is created with the
  // image running mode.
  //
  // The image can be of any size with format RGB or RGBA.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by
  // setting its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  //
  // For CPU images, the returned bounding boxes are expressed in the
  // unrotated input frame of reference coordinates system, i.e. in `[0,
  // image_width) x [0, image_height)`, which are the dimensions of the
  // underlying image data.
  absl::StatusOr<OrientedObjectDetectorResult> Detect(
      mediapipe::Image image,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Performs object detection on the provided video frame.
  // Only use this method when the OrientedObjectDetector is created with the
  // video running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide the video frame's timestamp (in milliseconds). The input timestamps
  // must be monotonically increasing.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by
  // setting its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  //
  // For CPU images, the returned bounding boxes are expressed in the
  // unrotated input frame of reference coordinates system, i.e. in `[0,
  // image_width) x [0, image_height)`, which are the dimensions of the
  // underlying image data.
  absl::StatusOr<OrientedObjectDetectorResult> DetectForVideo(
      mediapipe::Image image, int64_t timestamp_ms,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Sends live image data to perform object detection, and the results will be
  // available via the "result_callback" provided in the
  // OrientedObjectDetectorOptions. Only use this method when the
  // OrientedObjectDetector is created with the live stream running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide a timestamp (in milliseconds) to indicate when the input image is
  // sent to the object detector. The input timestamps must be monotonically
  // increasing.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by
  // setting its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  absl::Status DetectAsync(mediapipe::Image image, int64_t timestamp_ms,
                           std::optional<core::ImageProcessingOptions>
                               image_processing_options = std::nullopt);

  // Shuts down the OrientedObjectDetector when all works are done.
  absl::Status Close() { return runner_->Close(); }

 private:
  // Whether the task was created with tiling enabled (set once by Create).
  // In tiled mode the graph has no NORM_RECT input: Detect* variants must not
  // send a norm_rect packet, and region-of-interest/rotation requests are
  // rejected with kInvalidArgument.
  bool tiling_enabled_ = false;
};

}  // namespace oriented_object_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_ORIENTED_OBJECT_DETECTOR_ORIENTED_OBJECT_DETECTOR_H_
