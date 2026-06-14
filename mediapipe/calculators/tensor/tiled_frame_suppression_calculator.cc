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

#include <cstddef>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/detection_nms_util.h"
#include "mediapipe/calculators/tensor/tiled_frame_suppression_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"

namespace mediapipe {
namespace api2 {

// Frame-level suppression: concatenate fresh tiled detections + tracker-updated
// detections and run ONE global NMS. Safe bypass: when bypass_single_tile and
// there is exactly one valid tile row (NUM_TILES <= 1) and no tracker
// detections, pass the fresh detections through unchanged (already tile-local
// NMSed; nothing cross-source to dedup).
class TiledFrameSuppressionCalculator : public Node {
 public:
  static constexpr Input<std::vector<Detection>> kInDets{"DETECTIONS"};
  static constexpr Input<std::vector<Detection>>::Optional kInTracker{
      "TRACKER_DETECTIONS"};
  static constexpr Input<int>::Optional kInNumTiles{"NUM_TILES"};
  static constexpr Output<std::vector<Detection>> kOut{"DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInDets, kInTracker, kInNumTiles, kOut);

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<mediapipe::TiledFrameSuppressionCalculatorOptions>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    std::vector<Detection> fresh =
        kInDets(cc).IsEmpty() ? std::vector<Detection>{} : *kInDets(cc);
    const bool tracker_present =
        kInTracker(cc).IsConnected() && !kInTracker(cc).IsEmpty() &&
        !kInTracker(cc)->empty();
    // Absent NUM_TILES => treat as multi-tile (no bypass).
    const int num_tiles =
        (kInNumTiles(cc).IsConnected() && !kInNumTiles(cc).IsEmpty())
            ? *kInNumTiles(cc)
            : 2;

    if (options_.bypass_single_tile() && !tracker_present && num_tiles <= 1) {
      kOut(cc).Send(std::move(fresh));  // bypass: nothing cross-source to dedup
      return absl::OkStatus();
    }
    std::vector<Detection> combined = std::move(fresh);
    if (tracker_present) {
      const auto& tr = *kInTracker(cc);
      if (options_.tracker_is_gap_fill_only()) {
        // Keep only tracker boxes that don't overlap any fresh box. The inner
        // loop is bounded by the original fresh count so appended tracker boxes
        // are never treated as "fresh".
        const size_t fresh_count = combined.size();
        for (const Detection& t : tr) {
          bool overlaps = false;
          for (size_t i = 0; i < fresh_count; ++i) {
            const Detection& f = combined[i];
            if (!options_.class_agnostic() && t.label_id_size() > 0 &&
                f.label_id_size() > 0 && t.label_id(0) != f.label_id(0)) {
              continue;
            }
            if (DetectionRelativeIoU(t, f) >= options_.iou_threshold()) {
              overlaps = true;
              break;
            }
          }
          if (!overlaps) combined.push_back(t);
        }
      } else {
        combined.insert(combined.end(), tr.begin(), tr.end());
      }
    }
    kOut(cc).Send(GreedyDetectionNms(std::move(combined),
                                     options_.iou_threshold(),
                                     options_.class_agnostic()));
    return absl::OkStatus();
  }

 private:
  mediapipe::TiledFrameSuppressionCalculatorOptions options_;
};

MEDIAPIPE_REGISTER_NODE(TiledFrameSuppressionCalculator);

}  // namespace api2
}  // namespace mediapipe
