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
// INIReader), and tracks a moving box across two frames with GMC + ReID off.
TEST(BotsortSmokeTest, TracksAcrossTwoFrames) {
  TrackerParams params;  // defaults: gmc_enabled=false, reid_enabled=false
  BoTSORT tracker(params);

  cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));

  Detection d;
  d.bbox_tlwh = cv::Rect_<float>(50.f, 50.f, 20.f, 40.f);
  d.class_id = 1;
  d.confidence = 0.9f;
  tracker.track({d}, frame);

  Detection d2 = d;
  d2.bbox_tlwh = cv::Rect_<float>(52.f, 50.f, 20.f, 40.f);
  std::vector<std::shared_ptr<Track>> tracks = tracker.track({d2}, frame);

  ASSERT_FALSE(tracks.empty());
  EXPECT_EQ(tracks[0]->get_class_id(), 1);
}

}  // namespace
