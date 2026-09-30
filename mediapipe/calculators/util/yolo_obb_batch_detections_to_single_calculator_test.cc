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
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

OrientedDetection MakeObb(int label) {
  OrientedDetection d;
  d.set_cx(0.5f); d.set_cy(0.5f); d.set_width(0.2f); d.set_height(0.2f);
  d.set_rotation(0.0f); d.add_score(0.9f); d.add_label_id(label);
  return d;
}

TEST(YoloObbBatchDetectionsToSingleCalculatorTest, FlattensSingleRow) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloObbBatchDetectionsToSingleCalculator"
    input_stream: "ORIENTED_DETECTIONS:batched"
    output_stream: "ORIENTED_DETECTIONS:flat"
  )pb"));
  auto in = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  in->push_back({MakeObb(0), MakeObb(1)});
  runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(in.release()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                        .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(out.size(), 2);
  EXPECT_EQ(out[0].label_id(0), 0);
  EXPECT_EQ(out[1].label_id(0), 1);
}

TEST(YoloObbBatchDetectionsToSingleCalculatorTest, EmptyBatchEmitsEmpty) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloObbBatchDetectionsToSingleCalculator"
    input_stream: "ORIENTED_DETECTIONS:batched"
    output_stream: "ORIENTED_DETECTIONS:flat"
  )pb"));
  runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(new std::vector<std::vector<OrientedDetection>>()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  EXPECT_TRUE(runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                  .Get<std::vector<OrientedDetection>>().empty());
}

TEST(YoloObbBatchDetectionsToSingleCalculatorTest, RejectsMultipleRows) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloObbBatchDetectionsToSingleCalculator"
    input_stream: "ORIENTED_DETECTIONS:batched"
    output_stream: "ORIENTED_DETECTIONS:flat"
  )pb"));
  auto in = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  in->push_back({MakeObb(0)});
  in->push_back({MakeObb(1)});
  runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(in.release()).At(Timestamp(0)));
  EXPECT_FALSE(runner.Run().ok());
}

}  // namespace
}  // namespace mediapipe
