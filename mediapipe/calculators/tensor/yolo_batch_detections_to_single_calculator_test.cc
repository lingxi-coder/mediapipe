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
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

Detection MakeDet(int label) {
  Detection d;
  d.add_label_id(label);
  d.add_score(0.9f);
  return d;
}

TEST(YoloBatchDetectionsToSingleCalculatorTest, FlattensSingleRow) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloBatchDetectionsToSingleCalculator"
    input_stream: "DETECTIONS:batched"
    output_stream: "DETECTIONS:flat"
  )pb"));

  auto in = std::make_unique<std::vector<std::vector<Detection>>>();
  in->push_back({MakeDet(0), MakeDet(1)});  // one batch row, two detections
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      Adopt(in.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Tag("DETECTIONS").packets[0]
                        .Get<std::vector<Detection>>();
  ASSERT_EQ(out.size(), 2);
  EXPECT_EQ(out[0].label_id(0), 0);
  EXPECT_EQ(out[1].label_id(0), 1);
}

TEST(YoloBatchDetectionsToSingleCalculatorTest, EmptyBatchEmitsEmpty) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloBatchDetectionsToSingleCalculator"
    input_stream: "DETECTIONS:batched"
    output_stream: "DETECTIONS:flat"
  )pb"));
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      Adopt(new std::vector<std::vector<Detection>>()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  EXPECT_TRUE(runner.Outputs().Tag("DETECTIONS").packets[0]
                  .Get<std::vector<Detection>>().empty());
}

TEST(YoloBatchDetectionsToSingleCalculatorTest, RejectsMultipleRows) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloBatchDetectionsToSingleCalculator"
    input_stream: "DETECTIONS:batched"
    output_stream: "DETECTIONS:flat"
  )pb"));
  auto in = std::make_unique<std::vector<std::vector<Detection>>>();
  in->push_back({MakeDet(0)});
  in->push_back({MakeDet(1)});  // two batch rows -> single-image Task expects N==1
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      Adopt(in.release()).At(Timestamp(0)));
  EXPECT_FALSE(runner.Run().ok());
}

}  // namespace
}  // namespace mediapipe
