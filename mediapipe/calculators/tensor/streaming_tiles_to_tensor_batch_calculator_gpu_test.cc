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

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
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

}  // namespace
}  // namespace mediapipe

#endif  // !MEDIAPIPE_DISABLE_GPU && OPENGL_ES >= 31
