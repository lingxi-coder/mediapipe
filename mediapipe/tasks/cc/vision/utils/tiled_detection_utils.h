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

#include <cstdint>
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
  // True when the model's batch axis is dynamic/unspecified (non-positive in
  // the model). The tiled front then emits a variable (valid tile count) batch.
  bool is_dynamic_batch = false;
};

// Returns true when the tiling options enable the tiled path (grid larger
// than one tile, or explicit tiles provided). Shared single source of truth
// for the detector wrappers and the task graph builders (both must agree on
// whether the graph declares a NORM_RECT input).
template <typename TilingProto>
bool TilingEnabled(const TilingProto& t) {
  return static_cast<int64_t>(t.tile_rows()) * t.tile_cols() > 1 ||
         t.explicit_tiles_size() > 0;
}

// Returns true when motion scheduling should be active: scheduling is opted in
// AND tiling is enabled. (Running-mode gating — VIDEO/LIVE_STREAM only — is
// enforced separately at the wrapper's Create(), where running_mode lives.)
template <typename TilingProto>
bool SchedulingEnabled(const TilingProto& t) {
  return t.enable_motion_scheduling() && TilingEnabled(t);
}

// Normalizes a raw TFLite input batch dimension for the tiled front. A
// non-positive value (a dynamic / unspecified batch axis) is treated as a
// dynamic batch: returns capacity 1 with is_dynamic=true, so the positive-
// capacity contract downstream holds and the front emits a variable
// (valid-count) batch rather than aborting. A positive value is a fixed batch
// and is returned unchanged.
struct NormalizedBatchDim {
  int batch_capacity = 1;
  bool is_dynamic = false;
};
NormalizedBatchDim NormalizeTiledBatchDim(int raw_batch);

// Validates the model input tensor for the tiled front and returns its
// [N,H,W,C] shape dims. Requires one float32 BHWC image input with positive
// spatial dimensions and 1, 3, or 4 channels. Single-channel inputs use
// grayscale metadata and scalar normalization; 3/4-channel inputs use RGB
// metadata. Fixed and dynamic batches are supported; shape_signature marks a
// dynamic batch even when shape contains
// a positive allocated batch size. If NormalizationOptions metadata is absent,
// tiled preprocessing assumes /255. Present normalization metadata must be
// well-formed, finite, and equal to mean 0 and std 255. Other malformed image
// metadata is also rejected, independently of the batch size.
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
