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
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/util/rotated_non_max_suppression_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

namespace mediapipe {
namespace api2 {

namespace {

// Rotated IoU of two oriented detections, computed in the frame-normalized
// coordinate space the OrientedDetection fields are defined in (see
// oriented_detection.proto). On a non-square frame that space is anisotropic
// w.r.t. pixels, so this is NOT pixel-space IoU; pixel-accurate suppression
// would need the frame aspect ratio plumbed in. OpenCV expects the angle in
// degrees; OrientedDetection.rotation is radians.
float RotatedIoU(const OrientedDetection& a, const OrientedDetection& b) {
  constexpr float kRadToDeg = 180.0f / static_cast<float>(M_PI);
  cv::RotatedRect ra(cv::Point2f(a.cx(), a.cy()),
                     cv::Size2f(a.width(), a.height()), a.rotation() * kRadToDeg);
  cv::RotatedRect rb(cv::Point2f(b.cx(), b.cy()),
                     cv::Size2f(b.width(), b.height()), b.rotation() * kRadToDeg);
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

}  // namespace

// Greedy rotated-IoU non-max suppression over a flat list of OrientedDetections.
class RotatedNonMaxSuppressionCalculator : public Node {
 public:
  static constexpr Input<std::vector<OrientedDetection>> kIn{
      "ORIENTED_DETECTIONS"};
  static constexpr Output<std::vector<OrientedDetection>> kOut{
      "ORIENTED_DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Open(CalculatorContext* cc) override {
    options_ =
        cc->Options<mediapipe::RotatedNonMaxSuppressionCalculatorOptions>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& dets = *kIn(cc);

    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j) {
      return TopScore(dets[i]) > TopScore(dets[j]);
    });

    std::vector<bool> removed(dets.size(), false);
    auto out = std::make_unique<std::vector<OrientedDetection>>();
    for (int oi = 0; oi < static_cast<int>(order.size()); ++oi) {
      const int i = order[oi];
      if (removed[i]) continue;
      out->push_back(dets[i]);
      if (options_.max_detections() >= 0 &&
          static_cast<int>(out->size()) >= options_.max_detections()) {
        break;
      }
      for (int oj = oi + 1; oj < static_cast<int>(order.size()); ++oj) {
        const int j = order[oj];
        if (removed[j]) continue;
        if (!options_.class_agnostic() && TopLabel(dets[i]) != TopLabel(dets[j]))
          continue;
        if (RotatedIoU(dets[i], dets[j]) >= options_.iou_threshold()) {
          removed[j] = true;
        }
      }
    }
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }

 private:
  static float TopScore(const OrientedDetection& d) {
    return d.score_size() > 0 ? d.score(0) : 0.0f;
  }
  static int TopLabel(const OrientedDetection& d) {
    return d.label_id_size() > 0 ? d.label_id(0) : -1;
  }

  mediapipe::RotatedNonMaxSuppressionCalculatorOptions options_;
};

MEDIAPIPE_REGISTER_NODE(RotatedNonMaxSuppressionCalculator);

}  // namespace api2
}  // namespace mediapipe
