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

#ifndef MEDIAPIPE_TASKS_CC_VISION_UTILS_TILED_DETECTION_UTILS_H_
#define MEDIAPIPE_TASKS_CC_VISION_UTILS_TILED_DETECTION_UTILS_H_

#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"

namespace mediapipe {
namespace tasks {
namespace vision {

// The [N,H,W,C] shape of the model input tensor consumed by the tiled front.
struct TiledModelInputDims {
  int batch = 0;
  int height = 0;
  int width = 0;
  int channels = 0;
};

// Returns true when the tiling options enable the tiled path (grid larger
// than one tile, or explicit tiles provided). Shared single source of truth
// for the detector wrappers and the task graph builders (both must agree on
// whether the graph declares a NORM_RECT input).
template <typename TilingProto>
bool TilingEnabled(const TilingProto& t) {
  return t.tile_rows() * t.tile_cols() > 1 || t.explicit_tiles_size() > 0;
}

// Validates the model input tensor for the tiled front and returns its
// [N,H,W,C] shape dims. The tiled front supports float32 BHWC only.
// Normalization criterion (spec §2): float32 + 4D are hard requirements; if
// TFLite Metadata NormalizationOptions exist and differ from (mean 0, std 255)
// -> InvalidArgument; if absent -> assume /255 (same implicit assumption as
// the single-image path for this model family).
// Shared by the OBB (oriented_object_detector) and axis-aligned YOLO
// (yolo_object_detector) tiled task graph builders.
absl::StatusOr<TiledModelInputDims> ValidateTiledModelInputAndGetDims(
    const tasks::core::ModelResources& model_resources);

// In tiled mode the task graph has no NORM_RECT input, so per-call
// ImageProcessingOptions cannot be honored: ROI is mutually exclusive with
// tiling, and rotation is unsupported (it would be silently ignored
// otherwise).
absl::Status CheckTiledImageProcessingOptions(
    const std::optional<core::ImageProcessingOptions>&
        image_processing_options);

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_UTILS_TILED_DETECTION_UTILS_H_
