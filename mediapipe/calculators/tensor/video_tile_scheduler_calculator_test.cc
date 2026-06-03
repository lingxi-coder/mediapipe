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

#include "mediapipe/calculators/tensor/video_tile_scheduler_calculator.pb.h"
#include "mediapipe/calculators/tensor/video_tile_scheduler_util.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/tracking/flow_packager.pb.h"

namespace mediapipe {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

// Build a minimal valid TrackingData for through-calculator decode tests.
// domain_width/height/frame_aspect must be valid for MotionVectorFrameFromTrackingData.
TrackingData MakeTrackingData(int frame_flags) {
  TrackingData td;
  td.set_frame_flags(frame_flags);
  td.set_domain_width(100.0f);
  td.set_domain_height(100.0f);
  td.set_frame_aspect(1.0f);
  return td;
}

VideoTileSchedulerCalculatorOptions DefaultOptions() {
  return VideoTileSchedulerCalculatorOptions();
}

// ---------------------------------------------------------------------------
// Existing test (Task 1 baseline): no tracking, always DETECT.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// (a) Direct unit tests of ShouldRefreshFrame
// ---------------------------------------------------------------------------

TEST(ShouldRefreshFrameTest, NoTrackingDefaultsToDetect) {
  auto opts = DefaultOptions();
  TrackingSignals s;  // present=false
  // detect_without_tracking defaults to true
  EXPECT_TRUE(ShouldRefreshFrame(opts, /*priors_empty=*/false, s));
  // Explicitly set detect_without_tracking=false
  opts.set_detect_without_tracking(false);
  EXPECT_FALSE(ShouldRefreshFrame(opts, /*priors_empty=*/false, s));
}

TEST(ShouldRefreshFrameTest, EmptyPriorsForceDetect) {
  auto opts = DefaultOptions();
  TrackingSignals s;
  s.present = true;
  s.valid_background_model = true;
  EXPECT_TRUE(ShouldRefreshFrame(opts, /*priors_empty=*/true, s));
}

TEST(ShouldRefreshFrameTest, BackgroundUnstableForcesDetect) {
  auto opts = DefaultOptions();
  // refresh_on_background_unstable defaults to true
  TrackingSignals s;
  s.present = true;
  s.valid_background_model = false;
  EXPECT_TRUE(ShouldRefreshFrame(opts, /*priors_empty=*/false, s));
}

TEST(ShouldRefreshFrameTest, ChunkBoundaryForcesDetectEvenWhenDuplicated) {
  auto opts = DefaultOptions();
  // Both refresh_on_chunk_boundary and skip_on_duplicated default to true.
  // Chunk boundary takes priority (checked first).
  TrackingSignals s;
  s.present = true;
  s.is_chunk_boundary = true;
  s.is_duplicated = true;
  EXPECT_TRUE(ShouldRefreshFrame(opts, /*priors_empty=*/false, s));
}

TEST(ShouldRefreshFrameTest, DuplicatedSkipsWhenPriorsExist) {
  auto opts = DefaultOptions();
  // skip_on_duplicated defaults to true. No refresh triggers active.
  // Disable refresh_on_uncertain_tracking so uncertain-tracking default
  // doesn't override the skip.
  opts.set_refresh_on_uncertain_tracking(false);
  TrackingSignals s;
  s.present = true;
  s.is_duplicated = true;
  EXPECT_FALSE(ShouldRefreshFrame(opts, /*priors_empty=*/false, s));
}

TEST(ShouldRefreshFrameTest, LowFeatureCountForcesDetect) {
  auto opts = DefaultOptions();
  opts.set_min_global_features(10);
  TrackingSignals s;
  s.present = true;
  s.feature_count = 3;
  EXPECT_TRUE(ShouldRefreshFrame(opts, /*priors_empty=*/false, s));
}

TEST(ShouldRefreshFrameTest, HighForegroundMotionForcesDetect) {
  auto opts = DefaultOptions();
  opts.set_motion_refresh_threshold(0.1f);
  TrackingSignals s;
  s.present = true;
  s.mean_foreground_motion = 0.5f;
  EXPECT_TRUE(ShouldRefreshFrame(opts, /*priors_empty=*/false, s));
}

TEST(ShouldRefreshFrameTest, LowForegroundMotionSkipsWhenThresholdEnabled) {
  auto opts = DefaultOptions();
  opts.set_motion_skip_threshold(0.1f);
  // Disable refresh_on_uncertain_tracking so the skip-path is reached.
  opts.set_refresh_on_uncertain_tracking(false);
  TrackingSignals s;
  s.present = true;
  s.mean_foreground_motion = 0.01f;
  EXPECT_FALSE(ShouldRefreshFrame(opts, /*priors_empty=*/false, s));
}

TEST(ShouldRefreshFrameTest, UncertainTrackingDetectsByDefault) {
  auto opts = DefaultOptions();
  // No flag or motion trigger. refresh_on_uncertain_tracking defaults to true.
  TrackingSignals s;
  s.present = true;
  s.valid_background_model = true;
  s.is_duplicated = false;
  s.is_chunk_boundary = false;
  s.feature_count = 0;  // min_global_features=0 so no trigger
  s.mean_foreground_motion = 0.0f;
  EXPECT_TRUE(ShouldRefreshFrame(opts, /*priors_empty=*/false, s));
}

// ---------------------------------------------------------------------------
// (b) Through-calculator decode tests: validates DecodeTrackingSignals
// ---------------------------------------------------------------------------

// Feeds a TrackingData with FLAG_DUPLICATED; expects REFRESH=false and
// scheduled TILES to be empty (SKIP path).
TEST(VideoTileSchedulerTest, DuplicatedFrameSkipsThroughCalculator) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "VideoTileSchedulerCalculator"
    input_stream: "TILES:tiles"
    input_stream: "PRIOR_DETECTIONS:priors"
    input_stream: "TRACKING:tracking"
    output_stream: "TILES:sched"
    output_stream: "REFRESH:refresh"
    options {
      [mediapipe.VideoTileSchedulerCalculatorOptions.ext] {
        refresh_on_uncertain_tracking: false
      }
    }
  )pb"));

  auto tiles = std::make_unique<std::vector<NormalizedRect>>();
  tiles->push_back(Rect(.25, .5, .5, 1.0));
  tiles->push_back(Rect(.75, .5, .5, 1.0));
  runner.MutableInputs()->Tag("TILES").packets.push_back(
      Adopt(tiles.release()).At(Timestamp(0)));

  auto priors = std::make_unique<std::vector<Detection>>();
  priors->push_back(Det(0.9, .4, .4, .2, .2));
  runner.MutableInputs()->Tag("PRIOR_DETECTIONS").packets.push_back(
      Adopt(priors.release()).At(Timestamp(0)));

  auto td = std::make_unique<TrackingData>(
      MakeTrackingData(TrackingData::FLAG_DUPLICATED));
  runner.MutableInputs()->Tag("TRACKING").packets.push_back(
      Adopt(td.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& refresh = runner.Outputs().Tag("REFRESH").packets;
  const auto& sched = runner.Outputs().Tag("TILES").packets;
  ASSERT_EQ(refresh.size(), 1);
  ASSERT_EQ(sched.size(), 1);
  EXPECT_FALSE(refresh[0].Get<bool>());
  EXPECT_TRUE(sched[0].Get<std::vector<NormalizedRect>>().empty());
}

// Feeds a TrackingData with FLAG_BACKGROUND_UNSTABLE; expects REFRESH=true.
TEST(VideoTileSchedulerTest, BackgroundUnstableThroughCalculator) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "VideoTileSchedulerCalculator"
    input_stream: "TILES:tiles"
    input_stream: "PRIOR_DETECTIONS:priors"
    input_stream: "TRACKING:tracking"
    output_stream: "TILES:sched"
    output_stream: "REFRESH:refresh"
  )pb"));

  auto tiles = std::make_unique<std::vector<NormalizedRect>>();
  tiles->push_back(Rect(.25, .5, .5, 1.0));
  tiles->push_back(Rect(.75, .5, .5, 1.0));
  runner.MutableInputs()->Tag("TILES").packets.push_back(
      Adopt(tiles.release()).At(Timestamp(0)));

  auto priors = std::make_unique<std::vector<Detection>>();
  priors->push_back(Det(0.9, .4, .4, .2, .2));
  runner.MutableInputs()->Tag("PRIOR_DETECTIONS").packets.push_back(
      Adopt(priors.release()).At(Timestamp(0)));

  auto td = std::make_unique<TrackingData>(
      MakeTrackingData(TrackingData::FLAG_BACKGROUND_UNSTABLE));
  runner.MutableInputs()->Tag("TRACKING").packets.push_back(
      Adopt(td.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& refresh = runner.Outputs().Tag("REFRESH").packets;
  const auto& sched = runner.Outputs().Tag("TILES").packets;
  ASSERT_EQ(refresh.size(), 1);
  ASSERT_EQ(sched.size(), 1);
  EXPECT_TRUE(refresh[0].Get<bool>());
  EXPECT_EQ(sched[0].Get<std::vector<NormalizedRect>>().size(), 2);
}

}  // namespace
}  // namespace mediapipe
