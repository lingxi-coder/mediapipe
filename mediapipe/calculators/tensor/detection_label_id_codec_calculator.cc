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

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensor/detection_label_id_codec_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"

namespace mediapipe {
namespace api2 {

// Carries the integer label_id through the optical-flow tracker, which only
// preserves string labels (TrackedDetection::label_to_score_map). ENCODE
// stringifies each label_id into the parallel string `label`; DECODE parses it
// back and drops any detection it cannot classify (so no category=-1 box
// reaches the public API).
class DetectionLabelIdCodecCalculator : public Node {
 public:
  static constexpr Input<std::vector<Detection>> kIn{"DETECTIONS"};
  static constexpr Output<std::vector<Detection>> kOut{"DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<mediapipe::DetectionLabelIdCodecCalculatorOptions>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    std::vector<Detection> out;
    if (kIn(cc).IsEmpty()) {
      kOut(cc).Send(std::move(out));
      return absl::OkStatus();
    }
    const bool encode =
        options_.direction() ==
        mediapipe::DetectionLabelIdCodecCalculatorOptions::ENCODE;
    for (const Detection& d : *kIn(cc)) {
      Detection nd = d;
      if (encode) {
        // label_id is intentionally left in place; the tracker ignores it.
        nd.clear_label();
        for (int i = 0; i < nd.label_id_size(); ++i) {
          nd.add_label(absl::StrCat(nd.label_id(i)));
        }
        out.push_back(std::move(nd));
      } else {
        nd.clear_label_id();
        bool ok = nd.label_size() > 0;
        for (int i = 0; i < nd.label_size() && ok; ++i) {
          int id = 0;
          if (absl::SimpleAtoi(nd.label(i), &id)) {
            nd.add_label_id(id);
          } else {
            ok = false;
          }
        }
        if (!ok || nd.label_id_size() == 0) {
          ABSL_LOG_FIRST_N(WARNING, 1)
              << "DetectionLabelIdCodecCalculator: dropping a tracker "
                 "detection with no parseable label_id.";
          continue;
        }
        nd.clear_label();
        out.push_back(std::move(nd));
      }
    }
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }

 private:
  mediapipe::DetectionLabelIdCodecCalculatorOptions options_;
};

MEDIAPIPE_REGISTER_NODE(DetectionLabelIdCodecCalculator);

}  // namespace api2
}  // namespace mediapipe
