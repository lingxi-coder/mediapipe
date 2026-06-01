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
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using BatchDetections = std::vector<std::vector<Detection>>;

// Builds a single-tensor input packet from row-major float data.
std::unique_ptr<std::vector<Tensor>> MakeTensor(
    const Tensor::Shape& shape, const std::vector<float>& data) {
  auto tensor = Tensor(Tensor::ElementType::kFloat32, shape);
  auto write = tensor.GetCpuWriteView();
  std::copy(data.begin(), data.end(), write.buffer<float>());
  auto v = std::make_unique<std::vector<Tensor>>();
  v->push_back(std::move(tensor));
  return v;
}

TEST(YoloTensorsToDetectionsCalculatorTest, ChannelsFirstSingleClass) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 1
        conf_threshold: 0.25
      }
    }
  )pb"));

  // Shape [N=1, C=5, A=2]; channels = cx,cy,w,h,score; row-major c*A + a.
  // Anchor0: cx .5 cy .5 w .2 h .4 score .9  -> kept
  // Anchor1: cx .25 cy .75 w .1 h .1 score .1 -> dropped (< 0.25)
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 5, 2},
                       {0.5f, 0.25f, 0.5f, 0.75f, 0.2f, 0.1f, 0.4f, 0.1f,
                        0.9f, 0.1f})
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Tag("DETECTIONS").packets;
  ASSERT_EQ(out.size(), 1);
  const BatchDetections& batch = out[0].Get<BatchDetections>();
  ASSERT_EQ(batch.size(), 1);          // one batch row
  ASSERT_EQ(batch[0].size(), 1);       // one kept detection
  const Detection& d = batch[0][0];
  ASSERT_EQ(d.score_size(), 1);
  EXPECT_NEAR(d.score(0), 0.9f, 1e-5);
  ASSERT_EQ(d.label_id_size(), 1);
  EXPECT_EQ(d.label_id(0), 0);
  const auto& bb = d.location_data().relative_bounding_box();
  EXPECT_NEAR(bb.xmin(), 0.4f, 1e-5);   // cx - w/2
  EXPECT_NEAR(bb.ymin(), 0.3f, 1e-5);   // cy - h/2
  EXPECT_NEAR(bb.width(), 0.2f, 1e-5);
  EXPECT_NEAR(bb.height(), 0.4f, 1e-5);
}

TEST(YoloTensorsToDetectionsCalculatorTest, ChannelsLastSameResult) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_LAST
        num_classes: 1
        conf_threshold: 0.25
      }
    }
  )pb"));

  // Shape [N=1, A=2, C=5]; row-major a*C + c. Same two anchors as Task 2.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 2, 5},
                       {0.5f, 0.5f, 0.2f, 0.4f, 0.9f,
                        0.25f, 0.75f, 0.1f, 0.1f, 0.1f})
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<
          std::vector<std::vector<Detection>>>();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 1);
  const auto& bb = batch[0][0].location_data().relative_bounding_box();
  EXPECT_NEAR(bb.xmin(), 0.4f, 1e-5);
  EXPECT_NEAR(bb.ymin(), 0.3f, 1e-5);
  EXPECT_NEAR(bb.width(), 0.2f, 1e-5);
  EXPECT_NEAR(bb.height(), 0.4f, 1e-5);
}

TEST(YoloTensorsToDetectionsCalculatorTest, MultiClassArgmaxAndThreshold) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 3
        conf_threshold: 0.5
      }
    }
  )pb"));

  // Shape [N=1, C=7, A=2]; channels = cx,cy,w,h, s0,s1,s2 ; index c*A + a.
  // Anchor0: box(.5,.5,.2,.2) scores [.1,.8,.3] -> class 1, score .8 (kept)
  // Anchor1: box(.5,.5,.2,.2) scores [.4,.2,.1] -> max .4 < .5 (dropped)
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 7, 2},
                       {0.5f, 0.5f,   // cx
                        0.5f, 0.5f,   // cy
                        0.2f, 0.2f,   // w
                        0.2f, 0.2f,   // h
                        0.1f, 0.4f,   // s0
                        0.8f, 0.2f,   // s1
                        0.3f, 0.1f})  // s2
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<
          std::vector<std::vector<Detection>>>();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 1);
  EXPECT_EQ(batch[0][0].label_id(0), 1);
  EXPECT_NEAR(batch[0][0].score(0), 0.8f, 1e-5);
}

}  // namespace
}  // namespace mediapipe
