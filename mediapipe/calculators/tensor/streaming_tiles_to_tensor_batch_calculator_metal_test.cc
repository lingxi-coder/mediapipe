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
// Runs on this Mac (Metal). Drives StreamingTilesToTensorBatchCalculator's Metal
// zero-copy path through a real CalculatorGraph with GpuResources: GPU buffer +
// tile plan -> logical [N,H,W,C] batch tensor, asserting shape,
// no-CPU-readback, and multi-batch semantics. On non-Metal configs the whole TU
// compiles to nothing.

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_METAL_ENABLED

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.pb.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/util/image_test_utils.h"

namespace mediapipe {
namespace {

TileGeometry NormTile(int index, float x0, float y0, float w, float h) {
  TileGeometry g;
  g.tile_index = index;
  g.width = w;
  g.height = h;
  g.x_center = x0 + w / 2.0f;
  g.y_center = y0 + h / 2.0f;
  return g;
}

struct RunResult {
  std::vector<Packet> tensors;
  std::vector<Packet> info;
};

absl::Status RunZeroCopy(std::shared_ptr<GpuResources> gpu_resources,
                         const std::string& options_body, int out_w, int out_h,
                         int channels, int batch_capacity, int in_w, int in_h,
                         const std::vector<TileGeometry>& tiles,
                         RunResult* out) {
  InferenceMetadata meta;
  meta.set_input_height(out_h);
  meta.set_input_width(out_w);
  meta.set_input_channels(channels);
  meta.set_batch_capacity(batch_capacity);

  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image_gpu"
    input_stream: "tile_plan"
    output_stream: "tensors"
    output_stream: "info"
    node {
      calculator: "StreamingTilesToTensorBatchCalculator"
      input_stream: "IMAGE_GPU:image_gpu"
      input_stream: "TILE_PLAN:tile_plan"
      input_side_packet: "METADATA:meta"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:info"
      options { [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {} }
    }
  )pb");
  config.mutable_node(0)
      ->mutable_options()
      ->MutableExtension(StreamingTilesToTensorBatchCalculatorOptions::ext)
      ->MergeFrom(
          ParseTextProtoOrDie<StreamingTilesToTensorBatchCalculatorOptions>(
              options_body));

  CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  MP_RETURN_IF_ERROR(graph.ObserveOutputStream("tensors", [out](const Packet& p) {
    out->tensors.push_back(p);
    return absl::OkStatus();
  }));
  MP_RETURN_IF_ERROR(graph.ObserveOutputStream("info", [out](const Packet& p) {
    out->info.push_back(p);
    return absl::OkStatus();
  }));
  MP_RETURN_IF_ERROR(
      graph.StartRun({{"meta", MakePacket<InferenceMetadata>(meta)}}));
  GpuBuffer input = CreateTestRgba8GpuBuffer(in_w, in_h);
  TilePlan plan;
  plan.tiles = tiles;
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "image_gpu", MakePacket<GpuBuffer>(input).At(Timestamp(0))));
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "tile_plan", MakePacket<TilePlan>(plan).At(Timestamp(0))));
  MP_RETURN_IF_ERROR(graph.WaitUntilIdle());
  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  return graph.WaitUntilDone();
}

// Minimal GPU test fixture (replicates gpu_test_base's GpuResources setup,
// which is a package-private target in //mediapipe/gpu).
class StreamingTilesMetalTest : public testing::Test {
 protected:
  GpuSharedData gpu_shared_;
  std::shared_ptr<GpuResources> gpu_resources_ = gpu_shared_.gpu_resources;
};

TEST_F(StreamingTilesMetalTest, ZeroCopyEmitsLogicalBatchNoReadback) {
  RunResult r;
  MP_ASSERT_OK(RunZeroCopy(
      gpu_resources_,
      "enable_gpu_zero_copy: true max_gpu_tensor_buffers: 2 dynamic_batch: true",
      /*out_w=*/8, /*out_h=*/8, /*channels=*/3, /*batch_capacity=*/4,
      /*in_w=*/64, /*in_h=*/48,
      {NormTile(0, 0.0f, 0.0f, 0.5f, 1.0f), NormTile(1, 0.5f, 0.0f, 0.5f, 1.0f)},
      &r));
  ASSERT_EQ(r.tensors.size(), 1);
  const auto& t = r.tensors[0].Get<std::vector<Tensor>>()[0];
  const Tensor::Shape& shape = t.shape();
  ASSERT_EQ(shape.dims.size(), 4);
  EXPECT_EQ(shape.dims[0], 2);  // dynamic batch: N == valid tiles
  EXPECT_EQ(shape.dims[1], 8);
  EXPECT_EQ(shape.dims[2], 8);
  EXPECT_EQ(shape.dims[3], 3);  // logical BHWC (model input channels)

  // No-readback: GPU-resident, not CPU-materialized, before any explicit read.
  EXPECT_TRUE(t.ready_on_gpu());
  EXPECT_FALSE(t.ready_on_cpu());

  // Explicit test-side readback: normalized [0,1].
  auto view = t.GetCpuReadView();
  const float* buf = view.buffer<float>();
  const int n = shape.num_elements();
  for (int i = 0; i < n; ++i) {
    EXPECT_GE(buf[i], 0.0f) << "elem " << i;
    EXPECT_LE(buf[i], 1.0001f) << "elem " << i;
  }
}

TEST_F(StreamingTilesMetalTest, MultiBatchEmitsCeil) {
  RunResult r;
  MP_ASSERT_OK(RunZeroCopy(
      gpu_resources_,
      "enable_gpu_zero_copy: true max_gpu_tensor_buffers: 4 dynamic_batch: true",
      /*out_w=*/8, /*out_h=*/8, /*channels=*/3, /*batch_capacity=*/2,
      /*in_w=*/64, /*in_h=*/48,
      {NormTile(0, 0.0f, 0.0f, 0.33f, 1.0f), NormTile(1, 0.33f, 0.0f, 0.33f, 1.0f),
       NormTile(2, 0.66f, 0.0f, 0.34f, 1.0f)},
      &r));
  EXPECT_EQ(r.tensors.size(), 2);  // ceil(3/2)
  ASSERT_GE(r.info.size(), 2u);
  EXPECT_EQ(r.info[0].Get<TensorBatchInfo>().valid_count, 2);
  EXPECT_EQ(r.info[1].Get<TensorBatchInfo>().valid_count, 1);
}

}  // namespace
}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
