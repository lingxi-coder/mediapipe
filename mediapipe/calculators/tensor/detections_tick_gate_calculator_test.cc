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

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

Detection MakeDet() {
  Detection d;
  d.add_score(0.9f);
  d.add_label_id(0);
  return d;
}

// Two ticks; DATA only at the first. The gate must emit a packet at BOTH tick
// timestamps (the second one empty), so the downstream synchronized consumer
// never stalls on a gap.
TEST(DetectionsTickGateCalculatorTest, EmitsOnePacketPerTickEmptyOnGap) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsTickGateCalculator"
    input_stream: "TICK:tick"
    input_stream: "DATA:data"
    output_stream: "DETECTIONS:out"
  )pb"));

  runner.MutableInputs()->Tag("TICK").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{}).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TICK").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{}).At(Timestamp(1)));
  runner.MutableInputs()->Tag("DATA").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{MakeDet()})
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& outs = runner.Outputs().Tag("DETECTIONS").packets;
  ASSERT_EQ(outs.size(), 2u);
  EXPECT_EQ(outs[0].Timestamp(), Timestamp(0));
  EXPECT_EQ(outs[0].Get<std::vector<Detection>>().size(), 1u);
  EXPECT_EQ(outs[1].Timestamp(), Timestamp(1));
  EXPECT_TRUE(outs[1].Get<std::vector<Detection>>().empty());
}

}  // namespace
}  // namespace mediapipe
