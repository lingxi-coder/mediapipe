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
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

TEST(OrientedDetectionLabelIdToTextCalculatorTest, MapsLabelAndKeepsId) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "OrientedDetectionLabelIdToTextCalculator"
    input_stream: "in"
    output_stream: "out"
    options {
      [mediapipe.OrientedDetectionLabelIdToTextCalculatorOptions.ext] {
        label: "cat" label: "dog" keep_label_id: true
      }
    }
  )pb"));
  std::vector<OrientedDetection> in(1);
  in[0].add_label_id(1);
  in[0].add_score(0.9f);
  runner.MutableInputs()->Index(0).packets.push_back(
      MakePacket<std::vector<OrientedDetection>>(in).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs()
                        .Index(0)
                        .packets[0]
                        .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(out[0].label_size(), 1);
  EXPECT_EQ(out[0].label(0), "dog");
  ASSERT_EQ(out[0].label_id_size(), 1);
  EXPECT_EQ(out[0].label_id(0), 1);
}

TEST(OrientedDetectionLabelIdToTextCalculatorTest, ClearsIdWhenNotKept) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "OrientedDetectionLabelIdToTextCalculator"
    input_stream: "in"
    output_stream: "out"
    options {
      [mediapipe.OrientedDetectionLabelIdToTextCalculatorOptions.ext] {
        label: "cat" label: "dog"
      }
    }
  )pb"));
  std::vector<OrientedDetection> in(1);
  in[0].add_label_id(0);
  in[0].add_score(0.5f);
  runner.MutableInputs()->Index(0).packets.push_back(
      MakePacket<std::vector<OrientedDetection>>(in).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs()
                        .Index(0)
                        .packets[0]
                        .Get<std::vector<OrientedDetection>>();
  EXPECT_EQ(out[0].label(0), "cat");
  EXPECT_EQ(out[0].label_id_size(), 0);
}

}  // namespace
}  // namespace mediapipe
