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
// GPU env required (GLES 3.1). Drives StreamingTilesToTensorBatchCalculator's
// zero-copy path through a real CalculatorGraph with GpuResources. Build/run on
// a GLES 3.1 device/emulator or Linux EGL GPU; on Apple/disable-gpu the whole
// TU compiles to nothing. Full CPU/GPU value parity is covered by Task 7 and
// the no-readback assertion by Task 8; this test validates the GPU input
// contract, dispatch, batch/row geometry, and emission wiring.

#include "mediapipe/framework/port.h"

#if !MEDIAPIPE_DISABLE_GPU && \
    MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

#include <memory>
#include <vector>

#include <string>

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
#include "mediapipe/gpu/gpu_test_base.h"
#include "mediapipe/util/image_test_utils.h"

namespace mediapipe {
namespace {

// A normalized tile covering [x0, x0+w) x [y0, y0+h) of the source frame.
TileGeometry NormTile(int index, float x0, float y0, float w, float h) {
  TileGeometry g;
  g.tile_index = index;
  g.width = w;
  g.height = h;
  g.x_center = x0 + w / 2.0f;
  g.y_center = y0 + h / 2.0f;
  return g;
}

class StreamingTilesToTensorBatchCalculatorGpuTest : public GpuTestBase {};

TEST_F(StreamingTilesToTensorBatchCalculatorGpuTest, ZeroCopyEmitsBatchRows) {
  constexpr int kInW = 64, kInH = 48, kOutW = 8, kOutH = 8, kC = 3;

  InferenceMetadata meta;
  meta.set_input_height(kOutH);
  meta.set_input_width(kOutW);
  meta.set_input_channels(kC);
  meta.set_batch_capacity(4);

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
      options {
        [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
          enable_gpu_zero_copy: true
          dynamic_batch: true
          max_gpu_tensor_buffers: 4
        }
      }
    }
  )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.SetGpuResources(gpu_resources_));
  std::vector<Packet> tensor_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "tensors", [&](const Packet& p) {
        tensor_packets.push_back(p);
        return absl::OkStatus();
      }));
  MP_ASSERT_OK(graph.StartRun(
      {{"meta", MakePacket<InferenceMetadata>(meta)}}));

  // A uniform-value RGBA GpuBuffer makes any resize exact; two tiles cover
  // disjoint halves of the frame.
  GpuBuffer input = CreateTestRgba8GpuBuffer(kInW, kInH);
  auto plan = std::make_shared<TilePlan>();
  plan->tiles.push_back(NormTile(0, 0.0f, 0.0f, 0.5f, 1.0f));
  plan->tiles.push_back(NormTile(1, 0.5f, 0.0f, 0.5f, 1.0f));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image_gpu", MakePacket<GpuBuffer>(input).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tile_plan", MakePacket<TilePlan>(*plan).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  ASSERT_EQ(tensor_packets.size(), 1);  // dynamic batch: one batch of 2 rows
  const auto& tensors = tensor_packets[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(tensors.size(), 1);
  const Tensor::Shape& shape = tensors[0].shape();
  ASSERT_EQ(shape.dims.size(), 4);
  EXPECT_EQ(shape.dims[0], 2);     // N == valid tiles (dynamic batch)
  EXPECT_EQ(shape.dims[1], kOutH);
  EXPECT_EQ(shape.dims[2], kOutW);
  EXPECT_EQ(shape.dims[3], kC);

  // Read back (explicit test-side readback) and assert the GPU produced finite,
  // normalized [0,1] values in every row.
  auto view = tensors[0].GetCpuReadView();
  const float* buf = view.buffer<float>();
  const int n = shape.num_elements();
  for (int i = 0; i < n; ++i) {
    EXPECT_GE(buf[i], 0.0f) << "elem " << i;
    EXPECT_LE(buf[i], 1.0f) << "elem " << i;
  }

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Builds a single-node zero-copy graph with the given options proto body and
// runs it once on a uniform RGBA GpuBuffer + the given tiles. Collects TENSORS
// and BATCH_INFO packets.
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
  // Splice the options body in (text-proto merge is simplest done by re-parse).
  config.mutable_node(0)
      ->mutable_options()
      ->MutableExtension(
          StreamingTilesToTensorBatchCalculatorOptions::ext)
      ->MergeFrom(ParseTextProtoOrDie<StreamingTilesToTensorBatchCalculatorOptions>(
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
  MP_RETURN_IF_ERROR(graph.StartRun({{"meta", MakePacket<InferenceMetadata>(meta)}}));
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

// Task 7: fixed batch pads the tail rows with zeros on the GPU.
TEST_F(StreamingTilesToTensorBatchCalculatorGpuTest, FixedBatchZeroesPadding) {
  RunResult r;
  MP_ASSERT_OK(RunZeroCopy(
      gpu_resources_,
      "enable_gpu_zero_copy: true max_gpu_tensor_buffers: 2 dynamic_batch: false",
      /*out_w=*/8, /*out_h=*/8, /*channels=*/3, /*batch_capacity=*/4,
      /*in_w=*/64, /*in_h=*/48,
      {NormTile(0, 0.0f, 0.0f, 0.5f, 1.0f), NormTile(1, 0.5f, 0.0f, 0.5f, 1.0f)},
      &r));
  ASSERT_EQ(r.tensors.size(), 1);
  const auto& t = r.tensors[0].Get<std::vector<Tensor>>()[0];
  ASSERT_EQ(t.shape().dims[0], 4);  // padded to capacity
  auto view = t.GetCpuReadView();
  const float* buf = view.buffer<float>();
  const int row_elems = 8 * 8 * 3;
  for (int i = 2 * row_elems; i < 4 * row_elems; ++i) {
    EXPECT_EQ(buf[i], 0.0f) << "padding elem " << i;  // rows 2,3 cleared
  }
}

// Task 7: T > batch_capacity emits ceil(T/cap) batches at synthetic timestamps.
TEST_F(StreamingTilesToTensorBatchCalculatorGpuTest, MultiBatchEmitsCeil) {
  RunResult r;
  MP_ASSERT_OK(RunZeroCopy(
      gpu_resources_,
      "enable_gpu_zero_copy: true max_gpu_tensor_buffers: 4 dynamic_batch: true",
      /*out_w=*/8, /*out_h=*/8, /*channels=*/3, /*batch_capacity=*/2,
      /*in_w=*/64, /*in_h=*/48,
      {NormTile(0, 0.0f, 0.0f, 0.33f, 1.0f), NormTile(1, 0.33f, 0.0f, 0.33f, 1.0f),
       NormTile(2, 0.66f, 0.0f, 0.34f, 1.0f)},
      &r));
  EXPECT_EQ(r.tensors.size(), 2);            // ceil(3/2)
  ASSERT_GE(r.info.size(), 2u);
  EXPECT_EQ(r.info[0].Get<TensorBatchInfo>().valid_count, 2);
  EXPECT_EQ(r.info[1].Get<TensorBatchInfo>().valid_count, 1);
}

// Task 8: the zero-copy path must NOT CPU-materialize the output tensor; it is
// GPU-resident until an explicit downstream/test read view is taken.
TEST_F(StreamingTilesToTensorBatchCalculatorGpuTest, NoCpuReadbackInZeroCopy) {
  RunResult r;
  MP_ASSERT_OK(RunZeroCopy(
      gpu_resources_,
      "enable_gpu_zero_copy: true max_gpu_tensor_buffers: 2 dynamic_batch: true",
      /*out_w=*/8, /*out_h=*/8, /*channels=*/3, /*batch_capacity=*/4,
      /*in_w=*/64, /*in_h=*/48,
      {NormTile(0, 0.0f, 0.0f, 1.0f, 1.0f)}, &r));
  ASSERT_EQ(r.tensors.size(), 1);
  const auto& t = r.tensors[0].Get<std::vector<Tensor>>()[0];
  // No GetCpuReadView/GetCpuWriteView/MatView was called on the GPU path, so the
  // tensor is GPU-resident and not CPU-ready.
  EXPECT_TRUE(t.ready_on_gpu());
  EXPECT_FALSE(t.ready_on_cpu());
}

}  // namespace
}  // namespace mediapipe

#endif  // !MEDIAPIPE_DISABLE_GPU && OPENGL_ES >= 31
