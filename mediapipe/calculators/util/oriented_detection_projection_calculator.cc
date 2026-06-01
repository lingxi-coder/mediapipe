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

#include <array>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// Projects normalized OrientedDetections through a 4x4 affine (model-input ->
// original-image normalized coords) by transforming the four oriented corners
// and refitting a rotated rectangle. Output stays normalized.
class OrientedDetectionProjectionCalculator : public Node {
 public:
  static constexpr Input<std::vector<OrientedDetection>> kIn{
      "ORIENTED_DETECTIONS"};
  static constexpr Input<std::array<float, 16>> kMatrix{"PROJECTION_MATRIX"};
  static constexpr Output<std::vector<OrientedDetection>> kOut{
      "ORIENTED_DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kIn, kMatrix, kOut);

  absl::Status Process(CalculatorContext* cc) override {
    const auto& dets = *kIn(cc);
    const std::array<float, 16>& m = *kMatrix(cc);
    constexpr float kDegToRad = static_cast<float>(M_PI) / 180.0f;

    auto out = std::make_unique<std::vector<OrientedDetection>>();
    out->reserve(dets.size());
    for (const auto& d : dets) {
      const float c = std::cos(d.rotation()), s = std::sin(d.rotation());
      const float hw = d.width() / 2.0f, hh = d.height() / 2.0f;
      std::vector<cv::Point2f> pts;
      pts.reserve(4);
      for (const auto& off : {std::pair<float, float>(-hw, -hh),
                              std::pair<float, float>(hw, -hh),
                              std::pair<float, float>(hw, hh),
                              std::pair<float, float>(-hw, hh)}) {
        const float x = d.cx() + off.first * c - off.second * s;
        const float y = d.cy() + off.first * s + off.second * c;
        const float px = x * m[0] + y * m[1] + m[3];
        const float py = x * m[4] + y * m[5] + m[7];
        pts.emplace_back(px, py);
      }
      // NOTE: cv::minAreaRect's angle convention varies by OpenCV version and it
      // may report width/height transposed (with a 90° angle shift). The OBB is
      // equivalent regardless of which axis is labelled width vs height.
      cv::RotatedRect rr = cv::minAreaRect(pts);
      OrientedDetection projected = d;  // preserve scores/labels
      projected.set_cx(rr.center.x);
      projected.set_cy(rr.center.y);
      projected.set_width(rr.size.width);
      projected.set_height(rr.size.height);
      projected.set_rotation(rr.angle * kDegToRad);
      out->push_back(std::move(projected));
    }
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(OrientedDetectionProjectionCalculator);

}  // namespace api2
}  // namespace mediapipe
