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
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// Flattens batched oriented detections (one inner vector per batch row) to a
// single vector for the single-image Tasks API. Requires batch N <= 1.
class YoloObbBatchDetectionsToSingleCalculator : public Node {
 public:
  static constexpr Input<std::vector<std::vector<OrientedDetection>>> kIn{
      "ORIENTED_DETECTIONS"};
  static constexpr Output<std::vector<OrientedDetection>> kOut{
      "ORIENTED_DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Process(CalculatorContext* cc) override {
    const auto& batched = *kIn(cc);
    RET_CHECK_LE(batched.size(), 1u)
        << "single-image Task expects batch N<=1, got " << batched.size();
    std::vector<OrientedDetection> out;
    if (!batched.empty()) out = batched[0];
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(YoloObbBatchDetectionsToSingleCalculator);

}  // namespace api2
}  // namespace mediapipe
