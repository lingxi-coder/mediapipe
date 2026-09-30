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

#ifndef MEDIAPIPE_CALCULATORS_VIDEO_VIDEO_TILE_SCHEDULER_UTIL_H_
#define MEDIAPIPE_CALCULATORS_VIDEO_VIDEO_TILE_SCHEDULER_UTIL_H_

#include <vector>

#include "mediapipe/calculators/video/video_tile_scheduler_calculator.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

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

// A decoded motion feature in the longest-side-normalized tracking domain
// (the domain MotionVectorFrame positions live in).
struct FeaturePoint {
  float x = 0.0f;       // longest-side-normalized position
  float y = 0.0f;
  float motion = 0.0f;  // foreground motion magnitude at this feature
};

// Maps a longest-side-normalized position to frame-normalized [0,1]^2.
// aspect = w/h; the longest dimension is normalized to 1 in the source domain.
inline void FeatureFramePos(float x, float y, float aspect, float* fx,
                            float* fy) {
  if (aspect >= 1.0f) {
    *fx = x;
    *fy = y * aspect;
  } else {
    *fx = x / aspect;
    *fy = y;
  }
}

// True if frame-normalized point (x,y) lies in the tile's axis-aligned rect.
inline bool PointInTile(float x, float y, const NormalizedRect& t) {
  const float x0 = t.x_center() - t.width() / 2.0f;
  const float y0 = t.y_center() - t.height() / 2.0f;
  return x >= x0 && x <= x0 + t.width() && y >= y0 && y <= y0 + t.height();
}

// Selects up to max_scheduled_tiles from `base`, prioritizing tiles by summed
// foreground motion of features inside them (features mapped via aspect). A
// tile with zero in-tile motion falls back to the count of prior-detection
// centers inside it. Returns selected tiles in ORIGINAL input order.
// max_scheduled_tiles <= 0, or base.size() <= max, returns all of `base`.
std::vector<NormalizedRect> ScheduleTiles(
    const std::vector<NormalizedRect>& base, int max_scheduled_tiles,
    const std::vector<FeaturePoint>& features, float aspect,
    const std::vector<Detection>& priors);

}  // namespace mediapipe
#endif  // MEDIAPIPE_CALCULATORS_VIDEO_VIDEO_TILE_SCHEDULER_UTIL_H_
