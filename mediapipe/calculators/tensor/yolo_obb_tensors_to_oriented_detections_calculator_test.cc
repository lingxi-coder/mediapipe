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
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using BatchOrientedDetections = std::vector<std::vector<OrientedDetection>>;

std::unique_ptr<std::vector<Tensor>> MakeTensor(
    const Tensor::Shape& shape, const std::vector<float>& data) {
  auto tensor = Tensor(Tensor::ElementType::kFloat32, shape);
  auto write = tensor.GetCpuWriteView();
  std::copy(data.begin(), data.end(), write.buffer<float>());
  auto v = std::make_unique<std::vector<Tensor>>();
  v->push_back(std::move(tensor));
  return v;
}

TEST(YoloObbCalculatorTest, ChannelsFirstSingleClassWithAngle) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloObbTensorsToOrientedDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "ORIENTED_DETECTIONS:dets"
    options {
      [mediapipe.YoloObbTensorsToOrientedDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 1
        conf_threshold: 0.25
      }
    }
  )pb"));

  // Shape [N=1, C=6, A=2]; channels = cx,cy,w,h,score,angle ; index c*A + a.
  // Anchor0: cx .5 cy .5 w .2 h .4 score .9 angle 0.5  -> kept
  // Anchor1: score .1 -> dropped.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 6, 2},
                       {0.5f, 0.25f,   // cx
                        0.5f, 0.75f,   // cy
                        0.2f, 0.1f,    // w
                        0.4f, 0.1f,    // h
                        0.9f, 0.1f,    // score
                        0.5f, 0.0f})   // angle (radians)
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                          .Get<BatchOrientedDetections>();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 1);
  const OrientedDetection& d = batch[0][0];
  EXPECT_NEAR(d.cx(), 0.5f, 1e-5);
  EXPECT_NEAR(d.cy(), 0.5f, 1e-5);
  EXPECT_NEAR(d.width(), 0.2f, 1e-5);
  EXPECT_NEAR(d.height(), 0.4f, 1e-5);
  EXPECT_NEAR(d.rotation(), 0.5f, 1e-5);
  ASSERT_EQ(d.score_size(), 1);
  EXPECT_NEAR(d.score(0), 0.9f, 1e-5);
  EXPECT_EQ(d.label_id(0), 0);
}

TEST(YoloObbCalculatorTest, BatchNativeTwoRows) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloObbTensorsToOrientedDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "ORIENTED_DETECTIONS:dets"
    options {
      [mediapipe.YoloObbTensorsToOrientedDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 1
        conf_threshold: 0.25
      }
    }
  )pb"));

  // Shape [N=2, C=6, A=1]; index (n*C + c)*A + a, A=1 so = n*6 + c.
  // Row0 score .9 angle .3 (kept); Row1 score .1 (dropped) -> sizes {1,0}.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{2, 6, 1},
                       {0.5f, 0.5f, 0.2f, 0.4f, 0.9f, 0.3f,    // row 0
                        0.5f, 0.5f, 0.2f, 0.4f, 0.1f, 0.0f})   // row 1
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                          .Get<std::vector<std::vector<OrientedDetection>>>();
  ASSERT_EQ(batch.size(), 2);
  EXPECT_EQ(batch[0].size(), 1);
  EXPECT_EQ(batch[1].size(), 0);
  EXPECT_NEAR(batch[0][0].rotation(), 0.3f, 1e-5);
}

TEST(YoloObbCalculatorTest, MultiClassArgmaxAndAngleChannel) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloObbTensorsToOrientedDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "ORIENTED_DETECTIONS:dets"
    options {
      [mediapipe.YoloObbTensorsToOrientedDetectionsCalculatorOptions.ext] {
        num_classes: 3
        conf_threshold: 0.25
      }
    }
  )pb"));
  // CHANNELS_FIRST [1, 4+3+1=8, 1]: cx,cy,w,h, s0,s1,s2, angle (index c*A+a, A=1).
  // argmax over {0.1,0.8,0.3} -> class 1 @ 0.8; angle lives at channel 7 (4+3).
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 8, 1},
                       {0.5f, 0.5f, 0.2f, 0.4f, 0.1f, 0.8f, 0.3f, 0.7f})
                .release())
          .At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0].Get<BatchOrientedDetections>();
  ASSERT_EQ(batch.size(), 1u);
  ASSERT_EQ(batch[0].size(), 1u);
  const OrientedDetection& d = batch[0][0];
  EXPECT_NEAR(d.cx(), 0.5f, 1e-5);
  EXPECT_NEAR(d.cy(), 0.5f, 1e-5);
  EXPECT_NEAR(d.width(), 0.2f, 1e-5);
  EXPECT_NEAR(d.height(), 0.4f, 1e-5);
  EXPECT_EQ(d.label_id(0), 1);          // argmax picked class 1
  EXPECT_NEAR(d.score(0), 0.8f, 1e-5);
  EXPECT_NEAR(d.rotation(), 0.7f, 1e-5);  // angle read from channel 4+num_classes
}

TEST(YoloObbCalculatorTest, AllowClassesFiltersByIndex) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloObbTensorsToOrientedDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "ORIENTED_DETECTIONS:dets"
    options {
      [mediapipe.YoloObbTensorsToOrientedDetectionsCalculatorOptions.ext] {
        num_classes: 2 conf_threshold: 0.25 allow_classes: 1
      }
    }
  )pb"));
  // [1, 4+2+1=7, 2]: cx,cy,w,h, s0,s1, angle ; anchor0 class0(0.9), anchor1 class1(0.8).
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 7, 2},
                       {0.5f,0.5f, 0.5f,0.5f, 0.2f,0.2f, 0.2f,0.2f,
                        0.9f,0.1f, 0.1f,0.8f, 0.3f,0.4f}).release()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& batch = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                          .Get<BatchOrientedDetections>();
  ASSERT_EQ(batch[0].size(), 1u);
  EXPECT_EQ(batch[0][0].label_id(0), 1);
}
TEST(YoloObbCalculatorTest, IgnoreClassesDropsByIndex) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloObbTensorsToOrientedDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "ORIENTED_DETECTIONS:dets"
    options {
      [mediapipe.YoloObbTensorsToOrientedDetectionsCalculatorOptions.ext] {
        num_classes: 2 conf_threshold: 0.25 ignore_classes: 0
      }
    }
  )pb"));
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 7, 2},
                       {0.5f,0.5f, 0.5f,0.5f, 0.2f,0.2f, 0.2f,0.2f,
                        0.9f,0.1f, 0.1f,0.8f, 0.3f,0.4f}).release()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& batch = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                          .Get<BatchOrientedDetections>();
  ASSERT_EQ(batch[0].size(), 1u);
  EXPECT_EQ(batch[0][0].label_id(0), 1);
}

}  // namespace
}  // namespace mediapipe
