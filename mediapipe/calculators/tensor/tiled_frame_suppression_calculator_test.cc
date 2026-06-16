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
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

// Helper: create a Detection with one score, one label_id, and a
// RELATIVE_BOUNDING_BOX.
Detection Det(float score, int label_id,
              float xmin, float ymin, float w, float h) {
  Detection d;
  d.add_score(score);
  d.add_label_id(label_id);
  auto* ld = d.mutable_location_data();
  ld->set_format(mediapipe::LocationData::RELATIVE_BOUNDING_BOX);
  auto* rbb = ld->mutable_relative_bounding_box();
  rbb->set_xmin(xmin);
  rbb->set_ymin(ymin);
  rbb->set_width(w);
  rbb->set_height(h);
  return d;
}

// Push a vector<Detection> packet onto a named tag at timestamp 0.
void PushDets(CalculatorRunner* runner, const std::string& tag,
              std::vector<Detection> dets) {
  runner->MutableInputs()->Tag(tag).packets.push_back(
      MakePacket<std::vector<Detection>>(std::move(dets)).At(Timestamp(0)));
}

// Push an int packet onto a named tag at timestamp 0.
void PushInt(CalculatorRunner* runner, const std::string& tag, int val) {
  runner->MutableInputs()->Tag(tag).packets.push_back(
      MakePacket<int>(val).At(Timestamp(0)));
}

// Retrieve the output vector<Detection> at timestamp 0.
const std::vector<Detection>& GetOutput(const CalculatorRunner& runner) {
  return runner.Outputs()
      .Tag("DETECTIONS")
      .packets[0]
      .Get<std::vector<Detection>>();
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

// bypass_single_tile=true, NUM_TILES=1, no tracker → both overlapping boxes
// pass through unchanged (NMS is skipped).
TEST(TiledFrameSuppressionCalculatorTest, BypassSingleTileNoTrackerPassesThrough) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "NUM_TILES:num_tiles"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        bypass_single_tile: true
      }
    }
  )pb"));

  // Two heavily-overlapping boxes → without NMS both survive.
  std::vector<Detection> fresh = {
      Det(0.9f, 0, 0.0f, 0.0f, 0.5f, 0.5f),
      Det(0.8f, 0, 0.05f, 0.05f, 0.5f, 0.5f),
  };
  PushDets(&runner, "DETECTIONS", fresh);
  PushInt(&runner, "NUM_TILES", 1);

  MP_ASSERT_OK(runner.Run());
  const auto& out = GetOutput(runner);
  EXPECT_EQ(out.size(), 2u);  // bypass: NMS not run
}

// NUM_TILES=2, no tracker → NMS runs, two overlapping boxes → 1 kept.
TEST(TiledFrameSuppressionCalculatorTest, MultiTileRunsGlobalNms) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "NUM_TILES:num_tiles"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        bypass_single_tile: true
      }
    }
  )pb"));

  std::vector<Detection> fresh = {
      Det(0.9f, 0, 0.0f, 0.0f, 0.5f, 0.5f),
      Det(0.8f, 0, 0.05f, 0.05f, 0.5f, 0.5f),
  };
  PushDets(&runner, "DETECTIONS", fresh);
  PushInt(&runner, "NUM_TILES", 2);

  MP_ASSERT_OK(runner.Run());
  const auto& out = GetOutput(runner);
  EXPECT_EQ(out.size(), 1u);
}

// bypass_single_tile=true, NUM_TILES=1, tracker non-empty → bypass disabled,
// NMS runs over combined detections.
TEST(TiledFrameSuppressionCalculatorTest, TrackerPresentDisablesBypass) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "TRACKER_DETECTIONS:tracker"
    input_stream: "NUM_TILES:num_tiles"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        bypass_single_tile: true
      }
    }
  )pb"));

  // One fresh box + one heavily-overlapping tracker box → NMS should dedup.
  std::vector<Detection> fresh = {
      Det(0.9f, 0, 0.0f, 0.0f, 0.5f, 0.5f),
  };
  std::vector<Detection> tracker = {
      Det(0.8f, 0, 0.05f, 0.05f, 0.5f, 0.5f),
  };
  PushDets(&runner, "DETECTIONS", fresh);
  PushDets(&runner, "TRACKER_DETECTIONS", tracker);
  PushInt(&runner, "NUM_TILES", 1);

  MP_ASSERT_OK(runner.Run());
  const auto& out = GetOutput(runner);
  EXPECT_EQ(out.size(), 1u);
}

// bypass_single_tile=false (default), NUM_TILES=1, no tracker →
// NMS runs, two overlapping boxes → 1 kept.
TEST(TiledFrameSuppressionCalculatorTest, BypassDisabledRunsNms) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "NUM_TILES:num_tiles"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        bypass_single_tile: false
      }
    }
  )pb"));

  std::vector<Detection> fresh = {
      Det(0.9f, 0, 0.0f, 0.0f, 0.5f, 0.5f),
      Det(0.8f, 0, 0.05f, 0.05f, 0.5f, 0.5f),
  };
  PushDets(&runner, "DETECTIONS", fresh);
  PushInt(&runner, "NUM_TILES", 1);

  MP_ASSERT_OK(runner.Run());
  const auto& out = GetOutput(runner);
  EXPECT_EQ(out.size(), 1u);
}

// Fresh (2 disjoint) + tracker (2 disjoint from fresh and each other) →
// all 4 emitted after NMS.
TEST(TiledFrameSuppressionCalculatorTest, ConcatenatesFreshAndTracker) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "TRACKER_DETECTIONS:tracker"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
      }
    }
  )pb"));

  std::vector<Detection> fresh = {
      Det(0.9f, 0, 0.0f,  0.0f,  0.1f, 0.1f),
      Det(0.8f, 0, 0.2f,  0.0f,  0.1f, 0.1f),
  };
  std::vector<Detection> tracker = {
      Det(0.7f, 0, 0.5f,  0.0f,  0.1f, 0.1f),
      Det(0.6f, 0, 0.7f,  0.0f,  0.1f, 0.1f),
  };
  PushDets(&runner, "DETECTIONS", fresh);
  PushDets(&runner, "TRACKER_DETECTIONS", tracker);

  MP_ASSERT_OK(runner.Run());
  const auto& out = GetOutput(runner);
  EXPECT_EQ(out.size(), 4u);
}

// NUM_TILES not connected; bypass_single_tile=true, fresh=2 overlapping →
// absent NUM_TILES defaults to multi-tile (2), so NMS runs → 1 kept.
TEST(TiledFrameSuppressionCalculatorTest, AbsentNumTilesNoBypass) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        bypass_single_tile: true
      }
    }
  )pb"));

  std::vector<Detection> fresh = {
      Det(0.9f, 0, 0.0f, 0.0f, 0.5f, 0.5f),
      Det(0.8f, 0, 0.05f, 0.05f, 0.5f, 0.5f),
  };
  PushDets(&runner, "DETECTIONS", fresh);
  // NUM_TILES not connected → defaults to 2 → no bypass.

  MP_ASSERT_OK(runner.Run());
  const auto& out = GetOutput(runner);
  EXPECT_EQ(out.size(), 1u);
}

// gap-fill: a high-score tracker box overlapping a lower-score fresh box is
// dropped before NMS, so the FRESH box (its geometry/score) is what survives.
TEST(TiledFrameSuppressionCalculatorTest, GapFillDropsTrackerBoxOverlappingFresh) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "TRACKER_DETECTIONS:tracker"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        class_agnostic: true
        tracker_is_gap_fill_only: true
      }
    }
  )pb"));
  PushDets(&runner, "DETECTIONS", {Det(0.4f, 0, 0.10f, 0.10f, 0.40f, 0.40f)});
  PushDets(&runner, "TRACKER_DETECTIONS",
           {Det(0.95f, 0, 0.11f, 0.11f, 0.40f, 0.40f)});
  MP_ASSERT_OK(runner.Run());
  const auto& out = GetOutput(runner);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NEAR(out[0].score(0), 0.4f, 1e-5);
}

// gap-fill: a tracker box that does NOT overlap any fresh box is kept.
TEST(TiledFrameSuppressionCalculatorTest, GapFillKeepsNonOverlappingTrackerBox) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "TRACKER_DETECTIONS:tracker"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        class_agnostic: true
        tracker_is_gap_fill_only: true
      }
    }
  )pb"));
  PushDets(&runner, "DETECTIONS", {Det(0.4f, 0, 0.05f, 0.05f, 0.10f, 0.10f)});
  PushDets(&runner, "TRACKER_DETECTIONS",
           {Det(0.8f, 0, 0.70f, 0.70f, 0.10f, 0.10f)});
  MP_ASSERT_OK(runner.Run());
  EXPECT_EQ(GetOutput(runner).size(), 2u);
}

// Default (option absent) keeps today's blind-concatenate behavior: the HIGHER
// score wins among overlapping boxes.
TEST(TiledFrameSuppressionCalculatorTest, DefaultConcatenatesTrackerIntoNms) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "TRACKER_DETECTIONS:tracker"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        class_agnostic: true
      }
    }
  )pb"));
  PushDets(&runner, "DETECTIONS", {Det(0.4f, 0, 0.10f, 0.10f, 0.40f, 0.40f)});
  PushDets(&runner, "TRACKER_DETECTIONS",
           {Det(0.95f, 0, 0.11f, 0.11f, 0.40f, 0.40f)});
  MP_ASSERT_OK(runner.Run());
  const auto& out = GetOutput(runner);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NEAR(out[0].score(0), 0.95f, 1e-5);
}

// Approach A: a tracker detection overlapping a surviving fresh detection
// transfers its track_id onto the fresh one (geometry unchanged).
TEST(TiledFrameSuppressionCalculatorTest, TransfersTrackIdToOverlappingFresh) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "TRACKER_DETECTIONS:tracker"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        class_agnostic: true
        tracker_is_gap_fill_only: true
      }
    }
  )pb"));
  // fresh: one box (no track_id). tracker: SAME box, track_id "7".
  PushDets(&runner, "DETECTIONS", {Det(0.9f, 0, 0.10f, 0.10f, 0.40f, 0.40f)});
  Detection tracked = Det(0.8f, 0, 0.10f, 0.10f, 0.40f, 0.40f);
  tracked.set_track_id("7");
  PushDets(&runner, "TRACKER_DETECTIONS", {tracked});
  MP_ASSERT_OK(runner.Run());
  const auto& out = GetOutput(runner);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].track_id(), "7");
}

// A tracker detection overlapping no fresh box is appended as a gap-fill and
// keeps its own track_id.
TEST(TiledFrameSuppressionCalculatorTest, GapFillTrackerKeepsOwnTrackId) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "TRACKER_DETECTIONS:tracker"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        class_agnostic: true
        tracker_is_gap_fill_only: true
      }
    }
  )pb"));
  // fresh: box A (no id). tracker: box B far away, track_id "3".
  PushDets(&runner, "DETECTIONS", {Det(0.9f, 0, 0.05f, 0.05f, 0.10f, 0.10f)});
  Detection tracked = Det(0.8f, 0, 0.70f, 0.70f, 0.10f, 0.10f);
  tracked.set_track_id("3");
  PushDets(&runner, "TRACKER_DETECTIONS", {tracked});
  MP_ASSERT_OK(runner.Run());
  const auto& out = GetOutput(runner);
  ASSERT_EQ(out.size(), 2u);
  // The box-B detection (far away) carries track_id "3".
  bool found = false;
  for (const auto& d : out) {
    if (d.has_track_id() && d.track_id() == "3") found = true;
  }
  EXPECT_TRUE(found);
}

// No-op when the tracker carries no track_id (the BoxTracker path).
TEST(TiledFrameSuppressionCalculatorTest, NoTrackIdNoOp) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "TRACKER_DETECTIONS:tracker"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        class_agnostic: true
        tracker_is_gap_fill_only: true
      }
    }
  )pb"));
  // fresh: box A (no id). tracker: SAME box A, NO track_id.
  PushDets(&runner, "DETECTIONS", {Det(0.9f, 0, 0.10f, 0.10f, 0.40f, 0.40f)});
  PushDets(&runner, "TRACKER_DETECTIONS",
           {Det(0.8f, 0, 0.10f, 0.10f, 0.40f, 0.40f)});
  MP_ASSERT_OK(runner.Run());
  const auto& out = GetOutput(runner);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FALSE(out[0].has_track_id());
}

}  // namespace
}  // namespace mediapipe
