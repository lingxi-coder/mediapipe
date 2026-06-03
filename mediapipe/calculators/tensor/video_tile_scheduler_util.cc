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

#include "mediapipe/calculators/tensor/video_tile_scheduler_util.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace mediapipe {

std::vector<NormalizedRect> ScheduleTiles(
    const std::vector<NormalizedRect>& base, int max_scheduled_tiles,
    const std::vector<FeaturePoint>& features, float aspect,
    const std::vector<Detection>& priors) {
  const int n = static_cast<int>(base.size());
  if (max_scheduled_tiles <= 0 || n <= max_scheduled_tiles) return base;

  std::vector<float> score(n, 0.0f);
  for (const FeaturePoint& f : features) {
    float fx, fy;
    FeatureFramePos(f.x, f.y, aspect, &fx, &fy);
    for (int i = 0; i < n; ++i) {
      if (PointInTile(fx, fy, base[i])) score[i] += f.motion;
    }
  }
  // Per-tile fallback: a tile with no in-tile motion uses prior-center count.
  for (int i = 0; i < n; ++i) {
    if (score[i] != 0.0f) continue;
    int hits = 0;
    for (const Detection& d : priors) {
      if (!d.has_location_data() ||
          !d.location_data().has_relative_bounding_box()) {
        continue;
      }
      const auto& b = d.location_data().relative_bounding_box();
      if (PointInTile(b.xmin() + b.width() / 2.0f,
                      b.ymin() + b.height() / 2.0f, base[i])) {
        ++hits;
      }
    }
    score[i] = static_cast<float>(hits);
  }
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
    if (score[a] != score[b]) return score[a] > score[b];  // higher first
    return a < b;                                           // stable by index
  });
  std::vector<int> kept(idx.begin(), idx.begin() + max_scheduled_tiles);
  std::sort(kept.begin(), kept.end());  // restore original input order
  std::vector<NormalizedRect> out;
  out.reserve(kept.size());
  for (int i : kept) out.push_back(base[i]);
  return out;
}

}  // namespace mediapipe
