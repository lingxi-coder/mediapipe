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

#include <string>
#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

Detection Det(float score, int label_id) {
  Detection d;
  d.add_score(score);
  d.add_label_id(label_id);
  auto* ld = d.mutable_location_data();
  ld->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* bb = ld->mutable_relative_bounding_box();
  bb->set_xmin(0.1f);
  bb->set_ymin(0.2f);
  bb->set_width(0.3f);
  bb->set_height(0.4f);
  return d;
}

CalculatorRunner MakeRunner(const std::string& direction) {
  return CalculatorRunner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      "calculator: \"DetectionLabelIdCodecCalculator\"\n"
      "input_stream: \"DETECTIONS:in\"\n"
      "output_stream: \"DETECTIONS:out\"\n"
      "options { [mediapipe.DetectionLabelIdCodecCalculatorOptions.ext] {"
      "  direction: " + direction + " } }"));
}

const std::vector<Detection>& Out(const CalculatorRunner& r) {
  return r.Outputs().Tag("DETECTIONS").packets[0].Get<std::vector<Detection>>();
}

TEST(DetectionLabelIdCodecCalculatorTest, EncodeWritesStringLabel) {
  CalculatorRunner runner = MakeRunner("ENCODE");
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{Det(0.7f, 8)})
          .At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = Out(runner);
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].label_size(), 1);
  EXPECT_EQ(out[0].label(0), "8");
  ASSERT_EQ(out[0].score_size(), 1);
  EXPECT_NEAR(out[0].score(0), 0.7f, 1e-6);
}

TEST(DetectionLabelIdCodecCalculatorTest, DecodeRoundTripsLabelId) {
  Detection tracked;
  tracked.add_label("8");
  tracked.add_score(0.5f);
  CalculatorRunner runner = MakeRunner("DECODE");
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{tracked})
          .At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = Out(runner);
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].label_id_size(), 1);
  EXPECT_EQ(out[0].label_id(0), 8);
  EXPECT_EQ(out[0].label_size(), 0);
}

TEST(DetectionLabelIdCodecCalculatorTest, DecodeDropsUnclassifiable) {
  Detection bad;
  bad.add_label("boat");
  bad.add_score(0.5f);
  Detection empty;
  empty.add_score(0.4f);
  CalculatorRunner runner = MakeRunner("DECODE");
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{bad, empty})
          .At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  EXPECT_TRUE(Out(runner).empty());
}

TEST(DetectionLabelIdCodecCalculatorTest, DecodeKeepsGoodDropsBad) {
  Detection good;
  good.add_label("8");
  good.add_score(0.9f);
  Detection bad;
  bad.add_label("boat");
  bad.add_score(0.5f);
  CalculatorRunner runner = MakeRunner("DECODE");
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{good, bad})
          .At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = Out(runner);
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].label_id_size(), 1);
  EXPECT_EQ(out[0].label_id(0), 8);
}

}  // namespace
}  // namespace mediapipe
