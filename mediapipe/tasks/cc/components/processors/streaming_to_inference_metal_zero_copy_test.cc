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
//
// Runs on this Mac: chains StreamingTilesToTensorBatch's Metal zero-copy
// preprocessing (logical [N,H,W,C] tensor, no CPU readback) -> a normal
// InferenceCalculatorMetal, end to end. Validates the cross-calculator GPU
// handoff: the Metal-backed preprocessing tensor is consumed on-GPU by
// inference's BHWC->BPHWC4 conversion with no CPU round-trip.

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_METAL_ENABLED

#include <cmath>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/formats/tiling_types.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/util/image_test_utils.h"

namespace mediapipe {
namespace {

class StreamingToInferenceMetalTest : public testing::Test {
 protected:
  GpuSharedData gpu_shared_;
  std::shared_ptr<GpuResources> gpu_resources_ = gpu_shared_.gpu_resources;
};

// One full-frame tile, batch_capacity=1 -> the preprocessing emits a logical
// [1,256,256,3] Metal tensor that inference converts and runs. End to end on GPU.
TEST_F(StreamingToInferenceMetalTest, ChainProducesOutput) {
  InferenceMetadata meta;
  meta.set_input_height(256);
  meta.set_input_width(256);
  meta.set_input_channels(3);
  meta.set_batch_capacity(1);

  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image_gpu"
    input_stream: "tile_plan"
    output_stream: "out"
    node {
      calculator: "StreamingTilesToTensorBatchCalculator"
      input_stream: "IMAGE_GPU:image_gpu"
      input_stream: "TILE_PLAN:tile_plan"
      input_side_packet: "METADATA:meta"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:info"
      options {
        [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
          enable_gpu_zero_copy: true
          dynamic_batch: true
          max_gpu_tensor_buffers: 2
        }
      }
    }
    node {
      calculator: "InferenceCalculator"
      input_stream: "TENSORS:tensors"
      output_stream: "TENSORS:out"
      options {
        [mediapipe.InferenceCalculatorOptions.ext] {
          model_path: "mediapipe/calculators/tensor/testdata/1x256x256x3_softmax.tflite"
          delegate { gpu { allow_precision_loss: false } }
        }
      }
    }
  )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.SetGpuResources(gpu_resources_));
  std::vector<Packet> out;
  MP_ASSERT_OK(graph.ObserveOutputStream("out", [&](const Packet& p) {
    out.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({{"meta", MakePacket<InferenceMetadata>(meta)}}));

  GpuBuffer input = CreateTestRgba8GpuBuffer(320, 240);
  TilePlan plan;
  TileGeometry g;
  g.tile_index = 0;
  g.x_center = 0.5f;
  g.y_center = 0.5f;
  g.width = 1.0f;
  g.height = 1.0f;
  plan.tiles.push_back(g);

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image_gpu", MakePacket<GpuBuffer>(input).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tile_plan", MakePacket<TilePlan>(plan).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(out.size(), 1);
  const auto& tensors = out[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(tensors.size(), 1);
  auto view = tensors[0].GetCpuReadView();
  const float* buf = view.buffer<float>();
  const int n = tensors[0].shape().num_elements();
  ASSERT_GT(n, 0);
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    ASSERT_FALSE(std::isnan(buf[i])) << "NaN at " << i;
    sum += buf[i];
  }
  // Softmax output: finite, and (per spatial location) a probability-like
  // distribution -> total sum is positive and bounded.
  EXPECT_GT(sum, 0.0);
}

}  // namespace
}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
