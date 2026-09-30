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
#include <set>
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

// A single-label oriented detection in normalized space.
OrientedDetection MakeOriented(float cx, float cy, float width, float height,
                               float rotation, int label_id, float score) {
  OrientedDetection d;
  d.set_cx(cx);
  d.set_cy(cy);
  d.set_width(width);
  d.set_height(height);
  d.set_rotation(rotation);
  d.add_label_id(label_id);
  d.add_score(score);
  return d;
}

// A solid gray frame; the motion-only tracker does not depend on content but
// needs a valid cv::Mat for the (default-off) GMC path.
Packet MakeImagePacket(int width, int height, int64_t ts) {
  auto frame = std::make_unique<ImageFrame>(ImageFormat::SRGB, width, height);
  frame->SetToZero();
  return Adopt(frame.release()).At(Timestamp(ts));
}

constexpr char kNodeConfig[] = R"pb(
  calculator: "OrientedBotsortTrackingCalculator"
  input_stream: "IMAGE:image"
  input_stream: "ORIENTED_DETECTIONS:dets"
  output_stream: "ORIENTED_DETECTIONS:tracked"
  node_options {
    [type.googleapis.com/mediapipe.BotsortTrackingCalculatorOptions] {
      track_high_threshold: 0.05
      track_low_threshold: 0.02
      new_track_threshold: 0.05
    }
  }
)pb";

// Several frames carrying the same rotated object drifting slightly. The
// tracker must assign a stable BoTSORT track id while leaving the fresh rotated
// geometry untouched.
TEST(OrientedBotsortTrackingCalculatorTest,
     SetsStableTrackIdAndPreservesRotation) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kNodeConfig));

  // Three frames; the object drifts in cx by 0.01 each step.
  for (int i = 0; i < 3; ++i) {
    runner.MutableInputs()->Tag("IMAGE").packets.push_back(
        MakeImagePacket(200, 200, i));
    runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
        MakePacket<std::vector<OrientedDetection>>(
            std::vector<OrientedDetection>{
                MakeOriented(0.40f + 0.01f * i, 0.40f, 0.10f, 0.10f,
                             /*rotation=*/0.5f, /*label_id=*/1,
                             /*score=*/0.9f)})
            .At(Timestamp(i)));
  }

  MP_ASSERT_OK(runner.Run());
  const auto& outs = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets;
  ASSERT_EQ(outs.size(), 3u);

  // The last frame carries a stable id and the fresh, untracked rotation.
  const auto& last = outs.back().Get<std::vector<OrientedDetection>>();
  ASSERT_FALSE(last.empty());
  EXPECT_TRUE(last[0].has_track_id());
  EXPECT_FALSE(last[0].track_id().empty());
  EXPECT_FLOAT_EQ(last[0].rotation(), 0.5f);

  // The same track id appears on the last two output frames.
  auto ids_of = [](const Packet& p) {
    std::set<std::string> ids;
    for (const auto& d : p.Get<std::vector<OrientedDetection>>()) {
      if (d.has_track_id()) ids.insert(d.track_id());
    }
    return ids;
  };
  const std::set<std::string> last_ids = ids_of(outs[2]);
  const std::set<std::string> prev_ids = ids_of(outs[1]);
  std::set<std::string> intersection;
  for (const std::string& id : last_ids) {
    if (prev_ids.count(id) > 0) intersection.insert(id);
  }
  EXPECT_FALSE(intersection.empty());
}

// An empty detections vector must still produce exactly one (empty) output
// packet so a downstream synchronized consumer never stalls.
TEST(OrientedBotsortTrackingCalculatorTest, EmptyDetectionsProducesEmptyPacket) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kNodeConfig));

  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      MakeImagePacket(200, 200, 0));
  runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      MakePacket<std::vector<OrientedDetection>>(
          std::vector<OrientedDetection>{})
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& outs = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets;
  ASSERT_EQ(outs.size(), 1u);
  EXPECT_TRUE(outs[0].Get<std::vector<OrientedDetection>>().empty());
}

}  // namespace
}  // namespace mediapipe
