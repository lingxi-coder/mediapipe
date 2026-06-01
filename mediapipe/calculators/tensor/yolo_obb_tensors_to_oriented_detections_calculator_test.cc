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

}  // namespace
}  // namespace mediapipe
