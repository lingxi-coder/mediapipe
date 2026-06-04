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

#include "mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h"

#include <optional>
#include <utility>
#include <vector>

#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/tasks/cc/components/containers/category.h"

namespace mediapipe::tasks::components::containers {

constexpr int kDefaultCategoryIndex = -1;

OrientedObjectDetectionResult ConvertToOrientedObjectDetectionResult(
    std::vector<mediapipe::OrientedDetection> detections_proto,
    std::pair<int, int> image_size) {
  const float w = static_cast<float>(image_size.first);
  const float h = static_cast<float>(image_size.second);
  OrientedObjectDetectionResult result;
  result.detections.reserve(detections_proto.size());
  for (const auto& d : detections_proto) {
    OrientedObjectDetection od;
    for (int i = 0; i < d.score_size(); ++i) {
      od.categories.push_back(
          {/* index= */ d.label_id_size() > i ? d.label_id(i)
                                              : kDefaultCategoryIndex,
           /* score= */ d.score(i),
           /* category_name= */ d.label_size() > i
               ? std::make_optional(d.label(i)) : std::nullopt,
           /* display_name= */ d.display_name_size() > i
               ? std::make_optional(d.display_name(i)) : std::nullopt});
    }
    od.cx = d.cx() * w;
    od.cy = d.cy() * h;
    od.width = d.width() * w;
    od.height = d.height() * h;
    od.rotation = d.rotation();
    result.detections.push_back(std::move(od));
  }
  return result;
}

}  // namespace mediapipe::tasks::components::containers
