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
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/detection_nms_util.h"
#include "mediapipe/calculators/util/rotated_non_max_suppression_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"

namespace mediapipe {
namespace api2 {

// Greedy rotated-IoU non-max suppression over a flat list of
// OrientedDetections, via the shared GreedyOrientedDetectionNms util. IoU is
// computed in the frame-normalized coordinate space the OrientedDetection
// fields are defined in (see oriented_detection.proto); on a non-square frame
// that space is anisotropic w.r.t. pixels, so this is NOT pixel-space IoU —
// pixel-accurate suppression would need the frame aspect ratio plumbed in.
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
    auto out = std::make_unique<std::vector<OrientedDetection>>(
        GreedyOrientedDetectionNms(*kIn(cc), options_.iou_threshold(),
                                   options_.class_agnostic()));
    // max_detections caps the kept list: explicit 0 keeps none, -1 (default)
    // = uncapped. The kept list is in descending-score order, so truncation
    // keeps the highest-scoring detections.
    if (options_.max_detections() >= 0 &&
        static_cast<int>(out->size()) > options_.max_detections()) {
      out->resize(options_.max_detections());
    }
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }

 private:
  mediapipe::RotatedNonMaxSuppressionCalculatorOptions options_;
};

MEDIAPIPE_REGISTER_NODE(RotatedNonMaxSuppressionCalculator);

}  // namespace api2
}  // namespace mediapipe
