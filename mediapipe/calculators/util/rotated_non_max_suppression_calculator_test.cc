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

OrientedDetection MakeObb(float cx, float cy, float w, float h, float rot,
                          float score, int label) {
  OrientedDetection d;
  d.set_cx(cx); d.set_cy(cy); d.set_width(w); d.set_height(h);
  d.set_rotation(rot); d.add_score(score); d.add_label_id(label);
  return d;
}

TEST(RotatedNmsCalculatorTest, SuppressesHighOverlapKeepsHigherScore) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "RotatedNonMaxSuppressionCalculator"
    input_stream: "ORIENTED_DETECTIONS:in"
    output_stream: "ORIENTED_DETECTIONS:out"
    options {
      [mediapipe.RotatedNonMaxSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
      }
    }
  )pb"));

  auto in = std::make_unique<std::vector<OrientedDetection>>();
  in->push_back(MakeObb(0.5f, 0.5f, 0.4f, 0.4f, 0.0f, 0.95f, 0));
  in->push_back(MakeObb(0.5f, 0.5f, 0.4f, 0.4f, 0.0f, 0.80f, 0));
  in->push_back(MakeObb(0.9f, 0.9f, 0.1f, 0.1f, 0.0f, 0.70f, 0));
  runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(in.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                        .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(out.size(), 2);
  EXPECT_NEAR(out[0].score(0), 0.95f, 1e-5);
  EXPECT_NEAR(out[1].score(0), 0.70f, 1e-5);
}

TEST(RotatedNmsCalculatorTest, RotationSeparatesOtherwiseOverlappingBoxes) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "RotatedNonMaxSuppressionCalculator"
    input_stream: "ORIENTED_DETECTIONS:in"
    output_stream: "ORIENTED_DETECTIONS:out"
    options {
      [mediapipe.RotatedNonMaxSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
      }
    }
  )pb"));

  auto in = std::make_unique<std::vector<OrientedDetection>>();
  in->push_back(MakeObb(0.5f, 0.5f, 0.6f, 0.1f, 0.0f, 0.95f, 0));
  in->push_back(MakeObb(0.5f, 0.5f, 0.6f, 0.1f, 1.5708f, 0.80f, 0));  // ~90deg
  runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(in.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                        .Get<std::vector<OrientedDetection>>();
  EXPECT_EQ(out.size(), 2);  // low rotated-IoU -> both survive
}

TEST(RotatedNmsCalculatorTest, ClassAwareDoesNotSuppressAcrossClasses) {
  // Two overlapping boxes of DIFFERENT classes.
  auto build = [](bool agnostic) {
    CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
        agnostic
            ? R"pb(
                calculator: "RotatedNonMaxSuppressionCalculator"
                input_stream: "ORIENTED_DETECTIONS:in"
                output_stream: "ORIENTED_DETECTIONS:out"
                options {
                  [mediapipe.RotatedNonMaxSuppressionCalculatorOptions.ext] {
                    iou_threshold: 0.5 class_agnostic: true
                  }
                })pb"
            : R"pb(
                calculator: "RotatedNonMaxSuppressionCalculator"
                input_stream: "ORIENTED_DETECTIONS:in"
                output_stream: "ORIENTED_DETECTIONS:out"
                options {
                  [mediapipe.RotatedNonMaxSuppressionCalculatorOptions.ext] {
                    iou_threshold: 0.5 class_agnostic: false
                  }
                })pb"));
    auto in = std::make_unique<std::vector<OrientedDetection>>();
    in->push_back(MakeObb(0.5f, 0.5f, 0.4f, 0.4f, 0.0f, 0.95f, /*label=*/0));
    in->push_back(MakeObb(0.5f, 0.5f, 0.4f, 0.4f, 0.0f, 0.80f, /*label=*/1));
    runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
        Adopt(in.release()).At(Timestamp(0)));
    MP_EXPECT_OK(runner.Run());
    return runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
        .Get<std::vector<OrientedDetection>>()
        .size();
  };

  EXPECT_EQ(build(/*agnostic=*/false), 2u);  // different classes -> both kept
  EXPECT_EQ(build(/*agnostic=*/true), 1u);   // agnostic -> dup suppressed
}

}  // namespace
}  // namespace mediapipe
