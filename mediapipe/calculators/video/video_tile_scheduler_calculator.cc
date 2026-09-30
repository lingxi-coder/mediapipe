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
#include "mediapipe/calculators/video/video_tile_scheduler_calculator.pb.h"
#include "mediapipe/calculators/video/video_tile_scheduler_util.h"
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
// based on FlowPackager TrackingData signals. On DETECT, applies motion-based
// tile prioritization with max_scheduled_tiles cap. See
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
    // A missing TILES packet (bound advanced without a packet) leaves nothing
    // to schedule: emit the SKIP outputs (api2 Get() on an empty packet is
    // fatal).
    if (kInTiles(cc).IsEmpty()) {
      kOutTiles(cc).Send(std::vector<NormalizedRect>{});
      kOutRefresh(cc).Send(false);
      return absl::OkStatus();
    }
    const std::vector<NormalizedRect>& base = *kInTiles(cc);
    const bool priors_empty =
        kInPriorDets(cc).IsEmpty() || kInPriorDets(cc)->empty();
    const std::vector<Detection>& priors =
        priors_empty ? kEmptyDetections_ : *kInPriorDets(cc);

    // Decode MotionVectorFrame ONCE; derive both TrackingSignals and
    // FeaturePoints from the same decoded frame — no double decode.
    TrackingSignals signals;
    std::vector<FeaturePoint> features;
    float aspect = 1.0f;

    if (kInTracking(cc).IsConnected() && !kInTracking(cc).IsEmpty()) {
      MotionVectorFrame mvf;
      MotionVectorFrameFromTrackingData(*kInTracking(cc), &mvf);

      signals.present = true;
      signals.valid_background_model = mvf.valid_background_model;
      signals.is_duplicated = mvf.is_duplicated;
      signals.is_chunk_boundary = mvf.is_chunk_boundary;
      const TrackingData& td = *kInTracking(cc);
      signals.feature_count =
          td.has_global_feature_count()
              ? static_cast<int>(td.global_feature_count())
              : static_cast<int>(mvf.motion_vectors.size());
      float sum = 0.0f;
      for (const auto& m : mvf.motion_vectors) {
        sum += std::hypot(m.object.x(), m.object.y());
      }
      signals.mean_foreground_motion =
          mvf.motion_vectors.empty() ? 0.0f
                                     : sum / mvf.motion_vectors.size();

      aspect = mvf.aspect_ratio > 0.0f ? mvf.aspect_ratio : 1.0f;
      features.reserve(mvf.motion_vectors.size());
      for (const auto& m : mvf.motion_vectors) {
        FeaturePoint fp;
        fp.x = m.pos.x();
        fp.y = m.pos.y();
        fp.motion = std::hypot(m.object.x(), m.object.y());
        features.push_back(fp);
      }
    }

    if (ShouldRefreshFrame(options_, priors_empty, signals)) {
      kOutTiles(cc).Send(ScheduleTiles(base, options_.max_scheduled_tiles(),
                                       features, aspect, priors));
      kOutRefresh(cc).Send(true);
    } else {
      kOutTiles(cc).Send(std::vector<NormalizedRect>{});  // SKIP: no inference
      kOutRefresh(cc).Send(false);
    }
    return absl::OkStatus();
  }

 private:
  mediapipe::VideoTileSchedulerCalculatorOptions options_;
  // Empty detection list used when PRIOR_DETECTIONS is absent or empty,
  // so ScheduleTiles has a valid reference to iterate over.
  static const std::vector<Detection> kEmptyDetections_;
};

const std::vector<Detection> VideoTileSchedulerCalculator::kEmptyDetections_;

MEDIAPIPE_REGISTER_NODE(VideoTileSchedulerCalculator);

}  // namespace api2
}  // namespace mediapipe
