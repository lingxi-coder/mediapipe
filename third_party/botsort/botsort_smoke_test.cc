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

#include "BoTSORT.h"
#include "DataType.h"
#include "TrackerParams.h"
#include "gtest/gtest.h"
#include "opencv2/core.hpp"

namespace {

// Proves the vendored motion-only BoTSORT builds, links (no TensorRT/ONNX/
// INIReader), and tracks the SAME object across frames with GMC + ReID off.
//
// This asserts PERSISTENCE, not mere survival: it captures the output track's
// id (`Track::track_id`, a public int) on each frame and requires it to be
// STABLE across consecutive output frames. A tracker that spawned a fresh id
// every frame would fail this. We feed 3 deterministic frames of the same box
// drifting 2px/frame; BoT-SORT may keep a freshly-seen track tentative before
// confirming it, so we assert that the last two OUTPUT frames agree on the id.
TEST(BotsortSmokeTest, KeepsStableTrackIdAcrossFrames) {
  TrackerParams params;  // defaults: gmc_enabled=false, reid_enabled=false
  BoTSORT tracker(params);

  // Single static black frame, reused each step (motion comes from the box).
  cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));

  Detection d;
  d.bbox_tlwh = cv::Rect_<float>(50.f, 50.f, 20.f, 40.f);
  d.class_id = 1;
  d.confidence = 0.9f;

  // Deterministic drift: same object moves 2px to the right each frame.
  const float kStartX = 50.f;
  const float kDriftPerFrame = 2.f;
  const int kNumFrames = 3;

  std::vector<std::shared_ptr<Track>> tracks;
  std::vector<int> output_track_ids;  // id reported on each frame that output one
  for (int i = 0; i < kNumFrames; ++i) {
    Detection step = d;
    step.bbox_tlwh.x = kStartX + kDriftPerFrame * static_cast<float>(i);
    tracks = tracker.track({step}, frame);
    if (!tracks.empty()) {
      output_track_ids.push_back(tracks[0]->track_id);
    }
  }

  // Last frame must produce an output track for the (still-present) object.
  ASSERT_FALSE(tracks.empty());
  EXPECT_EQ(tracks[0]->get_class_id(), 1);

  // Persistence: at least two output frames, and the last two agree on the id
  // (the same object kept the same track_id rather than re-spawning).
  ASSERT_GE(output_track_ids.size(), 2u)
      << "Expected the track to appear in at least two output frames";
  EXPECT_EQ(output_track_ids[output_track_ids.size() - 1],
            output_track_ids[output_track_ids.size() - 2])
      << "Track id changed between consecutive output frames; the tracker is "
         "re-spawning ids instead of persisting the same track";
}

}  // namespace
