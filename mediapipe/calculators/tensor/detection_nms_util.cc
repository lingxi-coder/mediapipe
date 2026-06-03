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
#include <vector>

#include "mediapipe/framework/formats/detection.pb.h"

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

}  // namespace mediapipe
