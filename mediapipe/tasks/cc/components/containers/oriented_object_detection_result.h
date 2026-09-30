/* Copyright 2024 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_COMPONENTS_CONTAINERS_ORIENTED_OBJECT_DETECTION_RESULT_H_
#define MEDIAPIPE_TASKS_CC_COMPONENTS_CONTAINERS_ORIENTED_OBJECT_DETECTION_RESULT_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/tasks/cc/components/containers/category.h"

namespace mediapipe::tasks::components::containers {

// One oriented (rotated) bounding-box detection in PIXEL units.
struct OrientedObjectDetection {
  std::vector<Category> categories;
  float cx = 0.0f;        // box center x, pixels
  float cy = 0.0f;        // box center y, pixels
  float width = 0.0f;     // pixels
  float height = 0.0f;    // pixels
  float rotation = 0.0f;  // radians, counter-clockwise
  // Optional persistent track ID (set by BoTSORT tracking). std::nullopt when
  // the detection is not part of a track.
  std::optional<std::string> track_id = std::nullopt;
};

struct OrientedObjectDetectionResult {
  std::vector<OrientedObjectDetection> detections;
};

// Converts original-image-normalized OrientedDetection protos to the pixel-unit
// container. image_size is {width, height} in pixels. For non-square images,
// the normalized corners are mapped to pixel space and refit as the minimum-area
// enclosing oriented rectangle so width, height, and rotation remain coherent.
OrientedObjectDetectionResult ConvertToOrientedObjectDetectionResult(
    std::vector<mediapipe::OrientedDetection> detections_proto,
    std::pair<int, int> image_size);

}  // namespace mediapipe::tasks::components::containers
#endif  // MEDIAPIPE_TASKS_CC_COMPONENTS_CONTAINERS_ORIENTED_OBJECT_DETECTION_RESULT_H_
