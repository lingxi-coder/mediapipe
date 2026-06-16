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
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

// A normalized RELATIVE_BOUNDING_BOX detection with a single label/score.
Detection MakeDet(int label_id, float score, float xmin, float ymin, float w,
                  float h) {
  Detection d;
  d.add_score(score);
  d.add_label_id(label_id);
  auto* loc = d.mutable_location_data();
  loc->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* box = loc->mutable_relative_bounding_box();
  box->set_xmin(xmin);
  box->set_ymin(ymin);
  box->set_width(w);
  box->set_height(h);
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
  calculator: "BotsortTrackingCalculator"
  input_stream: "IMAGE:image"
  input_stream: "DETECTIONS:dets"
  output_stream: "DETECTIONS:out"
)pb";

// Two (then three) frames carrying the same labeled object, slightly moved. The
// tracker should emit a tracked detection that preserves the input label and a
// positive score, and that carries NO track_id/detection_id (parity contract).
TEST(BotsortTrackingCalculatorTest, PreservesLabelAndScoreAndCarriesNoTrackId) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kNodeConfig));

  // Frame 0.
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      MakeImagePacket(200, 200, 0));
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(
          std::vector<Detection>{MakeDet(7, 0.9f, 0.40f, 0.40f, 0.10f, 0.10f)})
          .At(Timestamp(0)));
  // Frame 1 (object moved slightly).
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      MakeImagePacket(200, 200, 1));
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(
          std::vector<Detection>{MakeDet(7, 0.9f, 0.42f, 0.41f, 0.10f, 0.10f)})
          .At(Timestamp(1)));
  // Frame 2 (object moved slightly again) - guarantees a confirmed track even
  // if frame 0 produced a tentative (empty) result.
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      MakeImagePacket(200, 200, 2));
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(
          std::vector<Detection>{MakeDet(7, 0.9f, 0.44f, 0.42f, 0.10f, 0.10f)})
          .At(Timestamp(2)));

  MP_ASSERT_OK(runner.Run());
  const auto& outs = runner.Outputs().Tag("DETECTIONS").packets;
  ASSERT_EQ(outs.size(), 3u);

  const auto& last = outs.back().Get<std::vector<Detection>>();
  ASSERT_FALSE(last.empty());
  EXPECT_EQ(last[0].label_id(0), 7);
  EXPECT_GT(last[0].score(0), 0.0f);
  EXPECT_FALSE(last[0].has_track_id());
  EXPECT_FALSE(last[0].has_detection_id());
}

// An empty detections vector must still produce exactly one (empty) output
// packet so a downstream synchronized consumer never stalls.
TEST(BotsortTrackingCalculatorTest, EmptyDetectionsProducesValidPacket) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kNodeConfig));

  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      MakeImagePacket(200, 200, 0));
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{})
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& outs = runner.Outputs().Tag("DETECTIONS").packets;
  ASSERT_EQ(outs.size(), 1u);
  EXPECT_TRUE(outs[0].Get<std::vector<Detection>>().empty());
}

}  // namespace
}  // namespace mediapipe
