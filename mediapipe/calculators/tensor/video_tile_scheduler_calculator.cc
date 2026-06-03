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

#include <cmath>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/video_tile_scheduler_calculator.pb.h"
#include "mediapipe/calculators/tensor/video_tile_scheduler_util.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/tracking/flow_packager.pb.h"
#include "mediapipe/util/tracking/tracking.h"

namespace mediapipe {
namespace api2 {

// Per-frame video-mode tile scheduler. Tracking-driven DETECT/SKIP decision
// based on FlowPackager TrackingData signals. Tile prioritization arrives in
// later tasks. See
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
    const bool priors_empty =
        kInPriorDets(cc).IsEmpty() || kInPriorDets(cc)->empty();
    TrackingSignals signals;
    if (kInTracking(cc).IsConnected() && !kInTracking(cc).IsEmpty()) {
      signals = DecodeTrackingSignals(*kInTracking(cc));
    }
    if (ShouldRefreshFrame(options_, priors_empty, signals)) {
      kOutTiles(cc).Send(std::vector<NormalizedRect>(base));  // full list (Task 3 adds cap)
      kOutRefresh(cc).Send(true);
    } else {
      kOutTiles(cc).Send(std::vector<NormalizedRect>{});  // SKIP: no inference
      kOutRefresh(cc).Send(false);
    }
    return absl::OkStatus();
  }

 private:
  // Decodes TrackingData into plain TrackingSignals for the decision function.
  // Booleans (valid_background_model, is_duplicated, is_chunk_boundary) are
  // read via MotionVectorFrameFromTrackingData, which maps frame_flags — the
  // same source as direct flag reads. domain_width/domain_height/frame_aspect
  // must be set on the TrackingData for MotionVectorFrameFromTrackingData to
  // produce correct motion vectors (division by domain_*).
  static TrackingSignals DecodeTrackingSignals(const TrackingData& td) {
    MotionVectorFrame mvf;
    MotionVectorFrameFromTrackingData(td, &mvf);
    TrackingSignals s;
    s.present = true;
    s.valid_background_model = mvf.valid_background_model;
    s.is_duplicated = mvf.is_duplicated;
    s.is_chunk_boundary = mvf.is_chunk_boundary;
    s.feature_count = td.has_global_feature_count()
                          ? static_cast<int>(td.global_feature_count())
                          : static_cast<int>(mvf.motion_vectors.size());
    float sum = 0.0f;
    for (const auto& m : mvf.motion_vectors) {
      sum += std::hypot(m.object.x(), m.object.y());
    }
    s.mean_foreground_motion =
        mvf.motion_vectors.empty() ? 0.0f : sum / mvf.motion_vectors.size();
    return s;
  }

  mediapipe::VideoTileSchedulerCalculatorOptions options_;
};

MEDIAPIPE_REGISTER_NODE(VideoTileSchedulerCalculator);

}  // namespace api2
}  // namespace mediapipe
