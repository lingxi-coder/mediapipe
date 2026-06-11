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

#include <cstring>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

std::unique_ptr<ImageFrame> WhiteFrame(int w, int h) {
  auto f = std::make_unique<ImageFrame>(ImageFormat::SRGB, w, h);
  std::memset(f->MutablePixelData(), 255, f->Height() * f->WidthStep());
  return f;
}

TEST(TiledDetectionFrontGraphTest, EmitsBatchesWithInfo) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    output_stream: "tensors"
    output_stream: "info"
    node {
      calculator: "mediapipe.tiled_detection.TiledDetectionFrontGraph"
      input_stream: "IMAGE:image"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:info"
      options {
        [mediapipe.TiledDetectionFrontGraphOptions.ext] {
          tile_grid { cols: 2 }
          batch_capacity: 2
          input_height: 8
          input_width: 16
          input_channels: 3
        }
      }
    }
  )pb");
  std::vector<Packet> tensors, info;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("tensors", [&](const Packet& p) {
    tensors.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.ObserveOutputStream("info", [&](const Packet& p) {
    info.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image", Adopt(WhiteFrame(64, 48).release()).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  ASSERT_EQ(tensors.size(), 1u);  // 2 tiles / cap 2 -> 1 batch
  const auto& tensor_vec = tensors[0].Get<std::vector<Tensor>>();
  ASSERT_FALSE(tensor_vec.empty());
  const Tensor& t = tensor_vec[0];
  ASSERT_EQ(t.shape().dims.size(), 4u);
  EXPECT_EQ(t.shape().dims[0], 2);   // batch
  EXPECT_EQ(t.shape().dims[1], 8);   // height
  EXPECT_EQ(t.shape().dims[2], 16);  // width
  EXPECT_EQ(t.shape().dims[3], 3);   // channels
  ASSERT_EQ(info.size(), 1u);
  EXPECT_EQ(info[0].Get<TensorBatchInfo>().valid_count, 2);
}

}  // namespace
}  // namespace mediapipe
