// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/calculators/tensor/detection_nms_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

namespace mediapipe {

float DetectionRelativeIoU(const Detection& a, const Detection& b) {
  const auto& ra = a.location_data().relative_bounding_box();
  const auto& rb = b.location_data().relative_bounding_box();

  const float area_a = ra.width() * ra.height();
  const float area_b = rb.width() * rb.height();
  if (area_a <= 0.0f || area_b <= 0.0f) return 0.0f;

  // Intersection
  const float inter_xmin = std::max(ra.xmin(), rb.xmin());
  const float inter_ymin = std::max(ra.ymin(), rb.ymin());
  const float inter_xmax = std::min(ra.xmin() + ra.width(),
                                    rb.xmin() + rb.width());
  const float inter_ymax = std::min(ra.ymin() + ra.height(),
                                    rb.ymin() + rb.height());

  const float inter_w = std::max(0.0f, inter_xmax - inter_xmin);
  const float inter_h = std::max(0.0f, inter_ymax - inter_ymin);
  const float inter_area = inter_w * inter_h;

  const float union_area = area_a + area_b - inter_area;
  if (union_area <= 0.0f) return 0.0f;

  return inter_area / union_area;
}

std::vector<Detection> GreedyDetectionNms(std::vector<Detection> dets,
                                          float iou_threshold,
                                          bool class_agnostic) {
  // Stable sort descending by score(0).
  std::stable_sort(dets.begin(), dets.end(),
                   [](const Detection& x, const Detection& y) {
                     return x.score(0) > y.score(0);
                   });

  const int n = static_cast<int>(dets.size());
  std::vector<bool> suppressed(n, false);
  std::vector<Detection> kept;

  for (int i = 0; i < n; ++i) {
    if (suppressed[i]) continue;
    kept.push_back(dets[i]);
    for (int j = i + 1; j < n; ++j) {
      if (suppressed[j]) continue;
      // Per-class guard: only suppress if same label_id(0) or class-agnostic.
      if (!class_agnostic &&
          dets[i].label_id(0) != dets[j].label_id(0)) {
        continue;
      }
      if (DetectionRelativeIoU(dets[i], dets[j]) > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return kept;
}

float OrientedDetectionIoU(const OrientedDetection& a,
                           const OrientedDetection& b) {
  // OpenCV expects the angle in degrees; OrientedDetection.rotation is radians.
  constexpr float kRadToDeg = 180.0f / static_cast<float>(M_PI);
  cv::RotatedRect ra(cv::Point2f(a.cx(), a.cy()),
                     cv::Size2f(a.width(), a.height()),
                     a.rotation() * kRadToDeg);
  cv::RotatedRect rb(cv::Point2f(b.cx(), b.cy()),
                     cv::Size2f(b.width(), b.height()),
                     b.rotation() * kRadToDeg);
  std::vector<cv::Point2f> inter;
  const int status = cv::rotatedRectangleIntersection(ra, rb, inter);
  if (status == cv::INTERSECT_NONE || inter.size() < 3) return 0.0f;
  // Order the intersection vertices before measuring area (matches the
  // defensive pattern in util/tracking box_util.cc).
  std::vector<cv::Point2f> hull;
  cv::convexHull(inter, hull);
  if (hull.size() < 3) return 0.0f;
  const double inter_area = cv::contourArea(hull);
  const double area_a = static_cast<double>(a.width()) * a.height();
  const double area_b = static_cast<double>(b.width()) * b.height();
  const double union_area = area_a + area_b - inter_area;
  if (union_area <= 0.0) return 0.0f;
  return static_cast<float>(inter_area / union_area);
}

std::vector<OrientedDetection> GreedyOrientedDetectionNms(
    std::vector<OrientedDetection> dets, float iou_threshold,
    bool class_agnostic) {
  auto top_score = [](const OrientedDetection& d) {
    return d.score_size() > 0 ? d.score(0) : 0.0f;
  };
  auto top_label = [](const OrientedDetection& d) {
    return d.label_id_size() > 0 ? d.label_id(0) : -1;
  };
  std::stable_sort(dets.begin(), dets.end(),
                   [&](const OrientedDetection& x, const OrientedDetection& y) {
                     return top_score(x) > top_score(y);
                   });

  const int n = static_cast<int>(dets.size());
  std::vector<bool> suppressed(n, false);
  std::vector<OrientedDetection> kept;
  for (int i = 0; i < n; ++i) {
    if (suppressed[i]) continue;
    kept.push_back(dets[i]);
    for (int j = i + 1; j < n; ++j) {
      if (suppressed[j]) continue;
      if (!class_agnostic && top_label(dets[i]) != top_label(dets[j])) continue;
      if (OrientedDetectionIoU(dets[i], dets[j]) >= iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return kept;
}

}  // namespace mediapipe
