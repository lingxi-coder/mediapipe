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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_VIDEO_TILE_SCHEDULER_UTIL_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_VIDEO_TILE_SCHEDULER_UTIL_H_

#include "mediapipe/calculators/tensor/video_tile_scheduler_calculator.pb.h"

namespace mediapipe {

// Decoded per-frame tracking signals used by the scheduler decision.
struct TrackingSignals {
  bool present = false;
  bool valid_background_model = true;
  bool is_duplicated = false;
  bool is_chunk_boundary = false;
  int feature_count = 0;
  float mean_foreground_motion = 0.0f;
};

// Decides DETECT (true) vs SKIP (false). Pure function of options + signals +
// whether prior detections exist. Decision order: hard-refresh flags dominate,
// then skip flags, then uncertain-tracking default. No cadence/frame counter.
inline bool ShouldRefreshFrame(
    const VideoTileSchedulerCalculatorOptions& options, bool priors_empty,
    const TrackingSignals& s) {
  if (!s.present) return options.detect_without_tracking();
  if (priors_empty) return true;
  if (options.refresh_on_background_unstable() && !s.valid_background_model)
    return true;
  if (options.refresh_on_chunk_boundary() && s.is_chunk_boundary) return true;
  if (options.min_global_features() > 0 &&
      s.feature_count < options.min_global_features())
    return true;
  if (options.motion_refresh_threshold() > 0.0f &&
      s.mean_foreground_motion > options.motion_refresh_threshold())
    return true;
  if (options.skip_on_duplicated() && s.is_duplicated) return false;
  if (options.motion_skip_threshold() > 0.0f &&
      s.mean_foreground_motion < options.motion_skip_threshold())
    return false;
  return options.refresh_on_uncertain_tracking();
}

}  // namespace mediapipe
#endif  // MEDIAPIPE_CALCULATORS_TENSOR_VIDEO_TILE_SCHEDULER_UTIL_H_
