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

#include <memory>
#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

NormalizedRect Rect(float xc, float yc, float w, float h) {
  NormalizedRect r;
  r.set_x_center(xc);
  r.set_y_center(yc);
  r.set_width(w);
  r.set_height(h);
  return r;
}

Detection Det(float score, float xmin, float ymin, float w, float h) {
  Detection d;
  d.add_score(score);
  d.mutable_location_data()->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* b = d.mutable_location_data()->mutable_relative_bounding_box();
  b->set_xmin(xmin);
  b->set_ymin(ymin);
  b->set_width(w);
  b->set_height(h);
  return d;
}

TEST(VideoTileSchedulerTest, DefaultsWithoutTrackingDetectEveryFrame) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "VideoTileSchedulerCalculator"
    input_stream: "TILES:tiles"
    input_stream: "PRIOR_DETECTIONS:priors"
    output_stream: "TILES:sched"
    output_stream: "REFRESH:refresh"
  )pb"));
  for (int t = 0; t < 2; ++t) {
    auto tiles = std::make_unique<std::vector<NormalizedRect>>();
    tiles->push_back(Rect(.25, .5, .5, 1.0));
    tiles->push_back(Rect(.75, .5, .5, 1.0));
    runner.MutableInputs()->Tag("TILES").packets.push_back(
        Adopt(tiles.release()).At(Timestamp(t)));
    auto priors = std::make_unique<std::vector<Detection>>();
    priors->push_back(Det(0.9, .4, .4, .2, .2));
    runner.MutableInputs()->Tag("PRIOR_DETECTIONS").packets.push_back(
        Adopt(priors.release()).At(Timestamp(t)));
  }
  MP_ASSERT_OK(runner.Run());
  const auto& sched = runner.Outputs().Tag("TILES").packets;
  const auto& refresh = runner.Outputs().Tag("REFRESH").packets;
  ASSERT_EQ(sched.size(), 2);
  ASSERT_EQ(refresh.size(), 2);
  for (int t = 0; t < 2; ++t) {
    EXPECT_TRUE(refresh[t].Get<bool>());
    EXPECT_EQ(sched[t].Get<std::vector<NormalizedRect>>().size(), 2);
  }
}

}  // namespace
}  // namespace mediapipe
