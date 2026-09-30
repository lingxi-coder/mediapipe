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

#include "mediapipe/calculators/video/video_tile_scheduler_calculator.pb.h"
#include "mediapipe/calculators/video/video_tile_scheduler_util.h"
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

// ---------------------------------------------------------------------------
// (c) Direct unit tests of ScheduleTiles pure function
// ---------------------------------------------------------------------------

TEST(ScheduleTilesTest, NoCapEmitsAllTiles) {
  std::vector<NormalizedRect> base = {
      Rect(0.125f, 0.5f, 0.25f, 1.0f),
      Rect(0.375f, 0.5f, 0.25f, 1.0f),
      Rect(0.625f, 0.5f, 0.25f, 1.0f),
      Rect(0.875f, 0.5f, 0.25f, 1.0f),
  };
  std::vector<FeaturePoint> features;
  std::vector<Detection> priors;

  // max_scheduled_tiles=0 means no cap → all tiles returned.
  auto result = ScheduleTiles(base, /*max_scheduled_tiles=*/0, features,
                              /*aspect=*/1.0f, priors);
  ASSERT_EQ(result.size(), 4u);

  // base.size() <= max → all tiles returned.
  result = ScheduleTiles(base, /*max_scheduled_tiles=*/4, features,
                         /*aspect=*/1.0f, priors);
  ASSERT_EQ(result.size(), 4u);

  // base.size() <= max (larger cap) → all tiles returned.
  result = ScheduleTiles(base, /*max_scheduled_tiles=*/10, features,
                         /*aspect=*/1.0f, priors);
  ASSERT_EQ(result.size(), 4u);
}

TEST(ScheduleTilesTest, CapKeepsTopMotionTilesInStableOrder) {
  // 4 tiles along x axis, each width=0.25, centered at 0.125, 0.375, 0.625,
  // 0.875. Features with motion only inside tiles 0 and 2.
  std::vector<NormalizedRect> base = {
      Rect(0.125f, 0.5f, 0.25f, 1.0f),   // tile 0: x in [0, 0.25]
      Rect(0.375f, 0.5f, 0.25f, 1.0f),   // tile 1: x in [0.25, 0.5]
      Rect(0.625f, 0.5f, 0.25f, 1.0f),   // tile 2: x in [0.5, 0.75]
      Rect(0.875f, 0.5f, 0.25f, 1.0f),   // tile 3: x in [0.75, 1.0]
  };
  // aspect=1: FeatureFramePos is identity.
  std::vector<FeaturePoint> features = {
      {0.1f, 0.5f, 2.0f},   // inside tile 0, motion=2
      {0.6f, 0.5f, 3.0f},   // inside tile 2, motion=3
  };
  std::vector<Detection> priors;

  auto result = ScheduleTiles(base, /*max_scheduled_tiles=*/2, features,
                              /*aspect=*/1.0f, priors);
  // Top 2 by motion: tile 2 (score=3) and tile 0 (score=2); tiles 1,3 (0).
  // Returned in ORIGINAL input order → {tile 0, tile 2}.
  ASSERT_EQ(result.size(), 2u);
  EXPECT_NEAR(result[0].x_center(), 0.125f, 1e-5f);  // original tile 0
  EXPECT_NEAR(result[1].x_center(), 0.625f, 1e-5f);  // original tile 2
}

TEST(ScheduleTilesTest, PerTilePriorFallbackCanSelectZeroMotionTile) {
  // 3 tiles; motion only in tile 0; prior-detection center in tile 2.
  // max=2 → keep tiles 0 (motion) and 2 (prior fallback), NOT tile 1 (nothing).
  std::vector<NormalizedRect> base = {
      Rect(0.125f, 0.5f, 0.25f, 1.0f),   // tile 0: x in [0, 0.25]
      Rect(0.375f, 0.5f, 0.25f, 1.0f),   // tile 1: x in [0.25, 0.5]
      Rect(0.625f, 0.5f, 0.25f, 1.0f),   // tile 2: x in [0.5, 0.75]
  };
  std::vector<FeaturePoint> features = {
      {0.1f, 0.5f, 5.0f},   // inside tile 0, motion=5
  };
  // Prior center at (0.625, 0.5) → inside tile 2.
  // Det bbox: xmin=0.55, ymin=0.4, w=0.15, h=0.2 → center=(0.625, 0.5).
  std::vector<Detection> priors = {Det(0.9f, 0.55f, 0.4f, 0.15f, 0.2f)};

  auto result = ScheduleTiles(base, /*max_scheduled_tiles=*/2, features,
                              /*aspect=*/1.0f, priors);
  // tile 0: motion score=5, tile 1: score=0, tile 2: prior-fallback score=1.
  // Top 2: tile 0 and tile 2. Returned in original order.
  ASSERT_EQ(result.size(), 2u);
  EXPECT_NEAR(result[0].x_center(), 0.125f, 1e-5f);  // tile 0
  EXPECT_NEAR(result[1].x_center(), 0.625f, 1e-5f);  // tile 2 (prior fallback)
}

TEST(ScheduleTilesTest, AllZeroPriorityFallsBackToOriginalOrder) {
  // No features, no priors → all scores=0. Stable sort preserves index order.
  // max=2 → first 2 tiles by original order.
  std::vector<NormalizedRect> base = {
      Rect(0.125f, 0.5f, 0.25f, 1.0f),
      Rect(0.375f, 0.5f, 0.25f, 1.0f),
      Rect(0.625f, 0.5f, 0.25f, 1.0f),
      Rect(0.875f, 0.5f, 0.25f, 1.0f),
  };
  std::vector<FeaturePoint> features;
  std::vector<Detection> priors;

  auto result = ScheduleTiles(base, /*max_scheduled_tiles=*/2, features,
                              /*aspect=*/1.0f, priors);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_NEAR(result[0].x_center(), 0.125f, 1e-5f);  // tile 0
  EXPECT_NEAR(result[1].x_center(), 0.375f, 1e-5f);  // tile 1
}

TEST(ScheduleTilesTest, AspectMappingMatchesTrackingDomain) {
  // Landscape: aspect=16/9. Center of tracking domain is at (0.5, 0.5/aspect).
  // FeatureFramePos should map that back to frame-center (0.5, 0.5).
  {
    const float aspect = 16.0f / 9.0f;
    // In tracking domain, x∈[0,1], y∈[0,1/aspect]. Center=(0.5, 0.5/aspect).
    float fx, fy;
    FeatureFramePos(0.5f, 0.5f / aspect, aspect, &fx, &fy);
    EXPECT_NEAR(fx, 0.5f, 1e-4f);
    EXPECT_NEAR(fy, 0.5f, 1e-4f);
  }
  // Portrait: aspect=9/16. Center of tracking domain is at (0.5*aspect, 0.5).
  // FeatureFramePos should map that back to frame-center (0.5, 0.5).
  {
    const float aspect = 9.0f / 16.0f;
    // In tracking domain, x∈[0,aspect], y∈[0,1]. Center=(0.5*aspect, 0.5).
    float fx, fy;
    FeatureFramePos(0.5f * aspect, 0.5f, aspect, &fx, &fy);
    EXPECT_NEAR(fx, 0.5f, 1e-4f);
    EXPECT_NEAR(fy, 0.5f, 1e-4f);
  }
}

// ---------------------------------------------------------------------------
// (d) Through-calculator: DETECT with max_scheduled_tiles=1, no tracking.
// ---------------------------------------------------------------------------

TEST(VideoTileSchedulerTest, DetectWithMaxScheduledTilesCapsTileCount) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "VideoTileSchedulerCalculator"
    input_stream: "TILES:tiles"
    input_stream: "PRIOR_DETECTIONS:priors"
    output_stream: "TILES:sched"
    output_stream: "REFRESH:refresh"
    options {
      [mediapipe.VideoTileSchedulerCalculatorOptions.ext] {
        max_scheduled_tiles: 1
      }
    }
  )pb"));

  auto tiles = std::make_unique<std::vector<NormalizedRect>>();
  tiles->push_back(Rect(0.25f, 0.5f, 0.5f, 1.0f));
  tiles->push_back(Rect(0.75f, 0.5f, 0.5f, 1.0f));
  runner.MutableInputs()->Tag("TILES").packets.push_back(
      Adopt(tiles.release()).At(Timestamp(0)));

  // Priors exist so priors_empty=false; no tracking → detect_without_tracking
  // is true by default → DETECT path.
  auto priors = std::make_unique<std::vector<Detection>>();
  priors->push_back(Det(0.9f, 0.1f, 0.1f, 0.2f, 0.2f));
  runner.MutableInputs()->Tag("PRIOR_DETECTIONS").packets.push_back(
      Adopt(priors.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& sched = runner.Outputs().Tag("TILES").packets;
  const auto& refresh = runner.Outputs().Tag("REFRESH").packets;
  ASSERT_EQ(sched.size(), 1u);
  ASSERT_EQ(refresh.size(), 1u);
  EXPECT_TRUE(refresh[0].Get<bool>());
  // max_scheduled_tiles=1 caps from 2 tiles to 1.
  EXPECT_EQ(sched[0].Get<std::vector<NormalizedRect>>().size(), 1u);
}

// A frame whose TILES packet is missing (bound advanced without a packet)
// has nothing to schedule: emit the SKIP outputs (empty tiles, REFRESH=false)
// instead of dereferencing the missing packet (api2 Get() on empty is fatal).
TEST(VideoTileSchedulerTest, MissingTilesPacketEmitsSkip) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "VideoTileSchedulerCalculator"
    input_stream: "TILES:tiles"
    input_stream: "PRIOR_DETECTIONS:priors"
    output_stream: "TILES:sched"
    output_stream: "REFRESH:refresh"
  )pb"));
  auto priors = std::make_unique<std::vector<Detection>>();
  priors->push_back(Det(0.9, .4, .4, .2, .2));
  runner.MutableInputs()->Tag("PRIOR_DETECTIONS").packets.push_back(
      Adopt(priors.release()).At(Timestamp(0)));
  // No TILES packet at Timestamp(0).
  MP_ASSERT_OK(runner.Run());
  const auto& sched = runner.Outputs().Tag("TILES").packets;
  const auto& refresh = runner.Outputs().Tag("REFRESH").packets;
  ASSERT_EQ(sched.size(), 1u);
  ASSERT_EQ(refresh.size(), 1u);
  EXPECT_TRUE(sched[0].Get<std::vector<NormalizedRect>>().empty());
  EXPECT_FALSE(refresh[0].Get<bool>());
}

}  // namespace
}  // namespace mediapipe
