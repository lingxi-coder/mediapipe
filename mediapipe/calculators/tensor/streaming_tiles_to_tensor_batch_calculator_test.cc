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

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "mediapipe/calculators/tensor/tiling_matrix_utils.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

InferenceMetadata Meta(int batch_capacity, int h, int w, int c, bool dynamic) {
  InferenceMetadata md;
  md.set_batch_capacity(batch_capacity);
  md.set_input_height(h);
  md.set_input_width(w);
  md.set_input_channels(c);
  md.set_is_dynamic_batch(dynamic);
  md.set_tensor_layout("BHWC");
  return md;
}

std::unique_ptr<ImageFrame> WhiteFrame(int w, int h) {
  auto f = std::make_unique<ImageFrame>(ImageFormat::SRGB, w, h);
  std::memset(f->MutablePixelData(), 255, f->Height() * f->WidthStep());
  return f;
}

TilePlan TwoTiles() {
  TilePlan p;
  TileGeometry a;
  a.tile_index = 0;
  a.x_center = .25;
  a.y_center = .5;
  a.width = .5;
  a.height = 1.0;
  p.tiles.push_back(a);
  TileGeometry b;
  b.tile_index = 1;
  b.x_center = .75;
  b.y_center = .5;
  b.width = .5;
  b.height = 1.0;
  p.tiles.push_back(b);
  return p;
}

TEST(StreamingTilesTest, FixedBatchPadsToCapacity) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    input_side_packet: "METADATA:meta"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
  )pb"));
  runner.MutableSidePackets()->Tag("METADATA") =
      MakePacket<InferenceMetadata>(Meta(4, 8, 8, 3, /*dynamic=*/false));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      Adopt(WhiteFrame(16, 16).release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(TwoTiles()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& tpk = runner.Outputs().Tag("TENSORS").packets;
  const auto& ipk = runner.Outputs().Tag("BATCH_INFO").packets;
  ASSERT_EQ(tpk.size(), 1);
  ASSERT_EQ(ipk.size(), 1);
  const auto& tensors = tpk[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(tensors.size(), 1);
  EXPECT_EQ(tensors[0].shape().dims[0], 4);
  EXPECT_EQ(tensors[0].shape().dims[1], 8);
  EXPECT_EQ(tensors[0].shape().dims[2], 8);
  EXPECT_EQ(tensors[0].shape().dims[3], 3);
  const auto& info = ipk[0].Get<TensorBatchInfo>();
  EXPECT_EQ(info.batch_capacity, 4);
  EXPECT_EQ(info.valid_count, 2);
  EXPECT_EQ(info.total_batches, 1);
  ASSERT_EQ(info.tile_indices.size(), 2);
  EXPECT_EQ(info.tile_indices[0], 0);
}

TEST(StreamingTilesTest, DynamicBatchNoPadding) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    input_side_packet: "METADATA:meta"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
  )pb"));
  runner.MutableSidePackets()->Tag("METADATA") =
      MakePacket<InferenceMetadata>(Meta(4, 8, 8, 3, /*dynamic=*/true));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      Adopt(WhiteFrame(16, 16).release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(TwoTiles()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& tensors =
      runner.Outputs().Tag("TENSORS").packets[0].Get<std::vector<Tensor>>();
  EXPECT_EQ(tensors[0].shape().dims[0], 2);
  const auto& info =
      runner.Outputs().Tag("BATCH_INFO").packets[0].Get<TensorBatchInfo>();
  EXPECT_EQ(info.valid_count, 2);
}

// Checks that geometry is populated and that applying tile_to_image_matrices[0]
// to the tile-center (0.5, 0.5) in tile-normalized space yields the tile's
// frame center within 2e-2.  We use a 16x16 frame; integer pixel rounding on a
// small frame loosens the tolerance slightly compared to a large frame.
TEST(StreamingTilesTest, GeometryPopulatedAndMatricesRoundTrip) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    input_side_packet: "METADATA:meta"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
  )pb"));
  runner.MutableSidePackets()->Tag("METADATA") =
      MakePacket<InferenceMetadata>(Meta(4, 8, 8, 3, /*dynamic=*/false));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      Adopt(WhiteFrame(16, 16).release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(TwoTiles()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());

  const auto& ipk = runner.Outputs().Tag("BATCH_INFO").packets;
  ASSERT_EQ(ipk.size(), 1);
  const auto& info = ipk[0].Get<TensorBatchInfo>();

  // geometry must be non-null and correctly sized
  ASSERT_NE(info.geometry, nullptr);
  ASSERT_EQ(static_cast<int>(info.geometry->tile_to_image_matrices.size()),
            info.valid_count);
  ASSERT_EQ(static_cast<int>(info.geometry->tile_geometries.size()),
            info.valid_count);

  // Apply tile_to_image_matrices[0] to the tile-center (0.5, 0.5) in
  // tile-normalized space and check it lands at the tile's frame center.
  // Tolerance is 2e-2 due to integer pixel rounding on a 16x16 frame.
  const TileGeometry& tg = info.geometry->tile_geometries[0];
  float out_x = 0.0f, out_y = 0.0f;
  ApplyMatrix(info.geometry->tile_to_image_matrices[0], 0.5f, 0.5f,
              &out_x, &out_y);
  EXPECT_NEAR(out_x, tg.x_center, 2e-2f);
  EXPECT_NEAR(out_y, tg.y_center, 2e-2f);
}

// Verifies that enabling max_cached_tile_matrices produces identical geometry
// matrices and pixel buffers compared to running with the cache disabled.
// Feed the same (image, plan) twice through each graph and check that:
//   (a) tile_to_image_matrices are element-wise equal between cached/uncached
//   (b) output tensor float buffers are byte-identical between cached/uncached
TEST(StreamingTilesTest, CacheOnVsCacheOffIdenticalResults) {
  auto run_graph = [](bool enable_cache) {
    CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
        enable_cache
            ? R"pb(
                calculator: "StreamingTilesToTensorBatchCalculator"
                input_stream: "IMAGE:image"
                input_stream: "TILE_PLAN:plan"
                input_side_packet: "METADATA:meta"
                output_stream: "TENSORS:tensors"
                output_stream: "BATCH_INFO:info"
                options {
                  [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
                    max_cached_tile_matrices: 4
                  }
                }
              )pb"
            : R"pb(
                calculator: "StreamingTilesToTensorBatchCalculator"
                input_stream: "IMAGE:image"
                input_stream: "TILE_PLAN:plan"
                input_side_packet: "METADATA:meta"
                output_stream: "TENSORS:tensors"
                output_stream: "BATCH_INFO:info"
              )pb"));
    runner.MutableSidePackets()->Tag("METADATA") =
        MakePacket<InferenceMetadata>(Meta(4, 8, 8, 3, /*dynamic=*/false));
    // Send the same frame + plan twice so a cache hit can occur on the 2nd call.
    for (int t = 0; t < 2; ++t) {
      runner.MutableInputs()->Tag("IMAGE").packets.push_back(
          Adopt(WhiteFrame(16, 16).release()).At(Timestamp(t)));
      runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
          MakePacket<TilePlan>(TwoTiles()).At(Timestamp(t)));
    }
    MP_EXPECT_OK(runner.Run());
    return std::make_pair(
        runner.Outputs().Tag("TENSORS").packets,
        runner.Outputs().Tag("BATCH_INFO").packets);
  };

  auto [cached_tensors, cached_infos] = run_graph(/*enable_cache=*/true);
  auto [uncached_tensors, uncached_infos] = run_graph(/*enable_cache=*/false);

  ASSERT_EQ(cached_tensors.size(), 2u);
  ASSERT_EQ(uncached_tensors.size(), 2u);

  for (int t = 0; t < 2; ++t) {
    const auto& ct = cached_tensors[t].Get<std::vector<Tensor>>();
    const auto& ut = uncached_tensors[t].Get<std::vector<Tensor>>();
    ASSERT_EQ(ct.size(), 1u);
    ASSERT_EQ(ut.size(), 1u);

    // Compare geometry matrices element-wise.
    const auto& ci = cached_infos[t].Get<TensorBatchInfo>();
    const auto& ui = uncached_infos[t].Get<TensorBatchInfo>();
    ASSERT_NE(ci.geometry, nullptr);
    ASSERT_NE(ui.geometry, nullptr);
    ASSERT_EQ(ci.geometry->tile_to_image_matrices.size(),
              ui.geometry->tile_to_image_matrices.size());
    for (size_t m = 0; m < ci.geometry->tile_to_image_matrices.size(); ++m) {
      for (int e = 0; e < 16; ++e) {
        EXPECT_FLOAT_EQ(ci.geometry->tile_to_image_matrices[m][e],
                        ui.geometry->tile_to_image_matrices[m][e])
            << "mismatch at frame=" << t << " matrix=" << m << " elem=" << e;
      }
    }

    // Compare output tensor float buffers byte-for-byte.
    const Tensor& cached_t = ct[0];
    const Tensor& uncached_t = ut[0];
    ASSERT_EQ(cached_t.shape().dims, uncached_t.shape().dims);
    auto cr = cached_t.GetCpuReadView();
    auto ur = uncached_t.GetCpuReadView();
    const float* cbuf = cr.buffer<float>();
    const float* ubuf = ur.buffer<float>();
    const int elems = cached_t.shape().num_elements();
    for (int e = 0; e < elems; ++e) {
      EXPECT_EQ(cbuf[e], ubuf[e])
          << "tensor mismatch at frame=" << t << " elem=" << e;
    }
  }
}

}  // namespace
}  // namespace mediapipe
