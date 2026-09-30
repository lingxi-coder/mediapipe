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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/tasks/cc/components/containers/category.h"

namespace mediapipe::tasks::components::containers {

namespace {

constexpr int kDefaultCategoryIndex = -1;

struct Point {
  float x;
  float y;
};

struct PixelOrientedBox {
  float width;
  float height;
  float rotation;
  float area;
};

std::optional<PixelOrientedBox> FitAlongEdge(
    const std::array<Point, 4>& corners, Point edge) {
  const float edge_length = std::hypot(edge.x, edge.y);
  if (edge_length <= std::numeric_limits<float>::epsilon()) {
    return std::nullopt;
  }

  const float axis_x = edge.x / edge_length;
  const float axis_y = edge.y / edge_length;
  const float normal_x = -axis_y;
  const float normal_y = axis_x;
  float min_axis = std::numeric_limits<float>::infinity();
  float max_axis = -std::numeric_limits<float>::infinity();
  float min_normal = std::numeric_limits<float>::infinity();
  float max_normal = -std::numeric_limits<float>::infinity();
  for (const Point& corner : corners) {
    const float axis_projection = corner.x * axis_x + corner.y * axis_y;
    const float normal_projection =
        corner.x * normal_x + corner.y * normal_y;
    min_axis = std::min(min_axis, axis_projection);
    max_axis = std::max(max_axis, axis_projection);
    min_normal = std::min(min_normal, normal_projection);
    max_normal = std::max(max_normal, normal_projection);
  }

  const float width = max_axis - min_axis;
  const float height = max_normal - min_normal;
  return PixelOrientedBox{
      .width = width,
      .height = height,
      .rotation = std::atan2(axis_y, axis_x),
      .area = width * height,
  };
}

PixelOrientedBox FitPixelOrientedBox(
    const mediapipe::OrientedDetection& detection, float image_width,
    float image_height) {
  const float cos_rotation = std::cos(detection.rotation());
  const float sin_rotation = std::sin(detection.rotation());
  const Point width_edge{
      detection.width() * cos_rotation * image_width,
      detection.width() * sin_rotation * image_height,
  };
  const Point height_edge{
      -detection.height() * sin_rotation * image_width,
      detection.height() * cos_rotation * image_height,
  };
  const Point center{detection.cx() * image_width,
                     detection.cy() * image_height};
  const std::array<Point, 4> corners = {{
      {center.x + (width_edge.x + height_edge.x) * 0.5f,
       center.y + (width_edge.y + height_edge.y) * 0.5f},
      {center.x + (width_edge.x - height_edge.x) * 0.5f,
       center.y + (width_edge.y - height_edge.y) * 0.5f},
      {center.x - (width_edge.x + height_edge.x) * 0.5f,
       center.y - (width_edge.y + height_edge.y) * 0.5f},
      {center.x - (width_edge.x - height_edge.x) * 0.5f,
       center.y - (width_edge.y - height_edge.y) * 0.5f},
  }};

  std::optional<PixelOrientedBox> best = FitAlongEdge(corners, width_edge);
  std::optional<PixelOrientedBox> height_aligned =
      FitAlongEdge(corners, height_edge);
  const float area_tolerance =
      best.has_value() ? std::max(1.0f, best->area) * 1e-6f : 0.0f;
  if (height_aligned.has_value() &&
      (!best.has_value() ||
       height_aligned->area < best->area - area_tolerance)) {
    best = height_aligned;
  }
  if (best.has_value()) {
    return *best;
  }

  return PixelOrientedBox{
      .width = 0.0f,
      .height = 0.0f,
      .rotation = std::atan2(image_height * sin_rotation,
                             image_width * cos_rotation),
      .area = 0.0f,
  };
}

}  // namespace

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
    const PixelOrientedBox pixel_box = FitPixelOrientedBox(d, w, h);
    od.cx = d.cx() * w;
    od.cy = d.cy() * h;
    od.width = pixel_box.width;
    od.height = pixel_box.height;
    od.rotation = pixel_box.rotation;
    if (d.has_track_id()) od.track_id = d.track_id();
    result.detections.push_back(std::move(od));
  }
  return result;
}

}  // namespace mediapipe::tasks::components::containers
