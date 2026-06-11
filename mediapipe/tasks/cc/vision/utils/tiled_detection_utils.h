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

#include "absl/status/statusor.h"
#include "flatbuffers/flatbuffers.h"
#include "mediapipe/tasks/cc/core/model_resources.h"

namespace mediapipe {
namespace tasks {
namespace vision {

// Validates the model input tensor for the tiled front and returns its
// [N,H,W,C] shape dims. The tiled front supports float32 BHWC only.
// Normalization criterion (spec §2): float32 + 4D are hard requirements; if
// TFLite Metadata NormalizationOptions exist and differ from (mean 0, std 255)
// -> InvalidArgument; if absent -> assume /255 (same implicit assumption as
// the single-image path for this model family).
// Shared by the OBB (oriented_object_detector) and axis-aligned YOLO
// (yolo_object_detector) tiled task graph builders.
absl::StatusOr<const flatbuffers::Vector<int32_t>*>
ValidateTiledModelInputAndGetDims(const core::ModelResources& model_resources);

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_UTILS_TILED_DETECTION_UTILS_H_
