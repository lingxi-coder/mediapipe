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

#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"

namespace mediapipe {
namespace api2 {

// Driven by the TICK stream (one packet per source frame). Emits the DATA
// vector at the tick timestamp if present, otherwise an empty vector. This
// converts the tracker's gappy DETECTIONS stream (the manager only emits on
// frames that have tracking boxes) into a dense one-packet-per-frame stream,
// so the downstream synchronized TRACKER_DETECTIONS input never stalls.
class DetectionsTickGateCalculator : public Node {
 public:
  static constexpr Input<std::vector<Detection>> kTick{"TICK"};
  static constexpr Input<std::vector<Detection>>::Optional kData{"DATA"};
  static constexpr Output<std::vector<Detection>> kOut{"DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kTick, kData, kOut);

  absl::Status Process(CalculatorContext* cc) override {
    std::vector<Detection> out;
    if (kData(cc).IsConnected() && !kData(cc).IsEmpty()) {
      out = *kData(cc);
    }
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(DetectionsTickGateCalculator);

}  // namespace api2
}  // namespace mediapipe
