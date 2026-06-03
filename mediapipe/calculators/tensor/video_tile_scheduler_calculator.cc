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
#include "mediapipe/calculators/tensor/video_tile_scheduler_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/tracking/flow_packager.pb.h"

namespace mediapipe {
namespace api2 {

// Per-frame video-mode tile scheduler. Skeleton: always DETECT (image-mode);
// tracking-driven DETECT/SKIP + prioritization arrive in later tasks. See
// docs/superpowers/specs/2026-06-03-video-tile-scheduler-design.md.
class VideoTileSchedulerCalculator : public Node {
 public:
  static constexpr Input<std::vector<NormalizedRect>> kInTiles{"TILES"};
  static constexpr Input<std::vector<Detection>> kInPriorDets{
      "PRIOR_DETECTIONS"};
  static constexpr Input<TrackingData>::Optional kInTracking{"TRACKING"};
  static constexpr Output<std::vector<NormalizedRect>> kOutTiles{"TILES"};
  static constexpr Output<bool> kOutRefresh{"REFRESH"};
  MEDIAPIPE_NODE_CONTRACT(kInTiles, kInPriorDets, kInTracking, kOutTiles,
                          kOutRefresh);

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<mediapipe::VideoTileSchedulerCalculatorOptions>();
    RET_CHECK_GE(options_.max_scheduled_tiles(), 0);
    RET_CHECK_GE(options_.min_global_features(), 0);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const std::vector<NormalizedRect>& base = *kInTiles(cc);
    // Image-mode skeleton: always DETECT, emit all base tiles, REFRESH=true.
    kOutTiles(cc).Send(std::vector<NormalizedRect>(base));
    kOutRefresh(cc).Send(true);
    return absl::OkStatus();
  }

 private:
  mediapipe::VideoTileSchedulerCalculatorOptions options_;
};

MEDIAPIPE_REGISTER_NODE(VideoTileSchedulerCalculator);

}  // namespace api2
}  // namespace mediapipe
