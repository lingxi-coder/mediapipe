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

#include "mediapipe/framework/formats/tiling_cache_stats.h"
#include "mediapipe/util/tiling_matrix_utils.h"
#include "mediapipe/framework/formats/tiling_types.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::testing::HasSubstr;

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

TilePlan ThreeTiles() {
  // Three vertical thirds of the frame (tile_index 0, 1, 2).
  TilePlan p;
  for (int i = 0; i < 3; ++i) {
    TileGeometry g;
    g.tile_index = i;
    g.x_center = (i + 0.5f) / 3.0f;
    g.y_center = 0.5f;
    g.width = 1.0f / 3.0f;
    g.height = 1.0f;
    p.tiles.push_back(g);
  }
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

TEST(StreamingTilesTest, RgbaInputDropsAlphaForRgbModel) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    input_side_packet: "METADATA:meta"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
  )pb"));
  runner.MutableSidePackets()->Tag("METADATA") =
      MakePacket<InferenceMetadata>(Meta(1, 2, 2, 3, /*dynamic=*/true));

  auto frame = std::make_unique<ImageFrame>(ImageFormat::SRGBA, 2, 2);
  const unsigned char pixels[2][2][4] = {
      {{10, 20, 30, 40}, {50, 60, 70, 80}},
      {{90, 100, 110, 120}, {130, 140, 150, 160}},
  };
  for (int y = 0; y < 2; ++y) {
    std::memcpy(frame->MutablePixelData() + y * frame->WidthStep(), pixels[y],
                sizeof(pixels[y]));
  }

  TilePlan plan;
  TileGeometry tile;
  tile.tile_index = 0;
  tile.x_center = 0.5f;
  tile.y_center = 0.5f;
  tile.width = 1.0f;
  tile.height = 1.0f;
  plan.tiles.push_back(tile);

  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      Adopt(frame.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(std::move(plan)).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());

  const auto& tensors =
      runner.Outputs().Tag("TENSORS").packets[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(tensors.size(), 1u);
  EXPECT_EQ(tensors[0].shape().dims, (std::vector<int>{1, 2, 2, 3}));
  const auto read = tensors[0].GetCpuReadView();
  const float* output = read.buffer<float>();
  const unsigned char expected[] = {10,  20,  30,  50,  60,  70,
                                    90,  100, 110, 130, 140, 150};
  for (int i = 0; i < 12; ++i) {
    EXPECT_NEAR(output[i], expected[i] / 255.0f, 1e-6f) << "index " << i;
  }
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

// Verifies multi-batch emission when tile count T exceeds batch_capacity.
// cap=2, T=3 → 2 batches: batch[0] has tiles {0,1} (valid_count=2),
// batch[1] has tile {2} (valid_count=1). Each batch is emitted at a distinct
// synthetic timestamp (batch_ts_) so the two output Timestamps differ.
TEST(StreamingTilesTest, MultiBatchEmissionWhenTExceedsCap) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    input_side_packet: "METADATA:meta"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
  )pb"));
  // cap=2, dynamic so N matches valid rows, T=3 → 2 batches
  runner.MutableSidePackets()->Tag("METADATA") =
      MakePacket<InferenceMetadata>(Meta(2, 8, 8, 3, /*dynamic=*/true));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      Adopt(WhiteFrame(8, 8).release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(ThreeTiles()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());

  const auto& ipk = runner.Outputs().Tag("BATCH_INFO").packets;
  const auto& tpk = runner.Outputs().Tag("TENSORS").packets;

  // Exactly 2 batches emitted.
  ASSERT_EQ(ipk.size(), 2u);
  ASSERT_EQ(tpk.size(), 2u);

  const auto& info0 = ipk[0].Get<TensorBatchInfo>();
  const auto& info1 = ipk[1].Get<TensorBatchInfo>();

  // total_batches == 2 in both.
  EXPECT_EQ(info0.total_batches, 2);
  EXPECT_EQ(info1.total_batches, 2);

  // batch_index: 0 then 1.
  EXPECT_EQ(info0.batch_index, 0);
  EXPECT_EQ(info1.batch_index, 1);

  // valid_count: 2 tiles in first batch, 1 in second.
  EXPECT_EQ(info0.valid_count, 2);
  EXPECT_EQ(info1.valid_count, 1);

  // tile_indices cover all three tiles {0, 1, 2} across both batches.
  ASSERT_EQ(info0.tile_indices.size(), 2u);
  ASSERT_EQ(info1.tile_indices.size(), 1u);
  EXPECT_EQ(info0.tile_indices[0], 0);
  EXPECT_EQ(info0.tile_indices[1], 1);
  EXPECT_EQ(info1.tile_indices[0], 2);

  // The two output packets must have distinct timestamps.
  EXPECT_NE(ipk[0].Timestamp(), ipk[1].Timestamp());
  // Timestamps are Timestamp(0) and Timestamp(1) (batch_ts_ starts at 0).
  EXPECT_EQ(ipk[0].Timestamp(), Timestamp(0));
  EXPECT_EQ(ipk[1].Timestamp(), Timestamp(1));
}

// Verifies that enabling max_cpu_tensor_workspaces=2 produces identical tensor
// float buffers and geometry matrices compared to running with the pool disabled
// (max_cpu_tensor_workspaces=0, the default) over three consecutive frames.
// This proves the pooled path is correctness-neutral and doesn't crash across
// multiple frames.
TEST(StreamingTilesTest, CpuPoolOnVsOffIdenticalResults) {
  auto run_graph = [](int max_workspaces) {
    std::string node_text = R"pb(
      calculator: "StreamingTilesToTensorBatchCalculator"
      input_stream: "IMAGE:image"
      input_stream: "TILE_PLAN:plan"
      input_side_packet: "METADATA:meta"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:info"
    )pb";
    std::string pooled_node_text = R"pb(
      calculator: "StreamingTilesToTensorBatchCalculator"
      input_stream: "IMAGE:image"
      input_stream: "TILE_PLAN:plan"
      input_side_packet: "METADATA:meta"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:info"
      options {
        [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
          max_cpu_tensor_workspaces: 2
        }
      }
    )pb";
    CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
        max_workspaces > 0 ? pooled_node_text : node_text));
    runner.MutableSidePackets()->Tag("METADATA") =
        MakePacket<InferenceMetadata>(Meta(4, 8, 8, 3, /*dynamic=*/false));
    for (int t = 0; t < 3; ++t) {
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

  auto [pooled_tensors, pooled_infos] = run_graph(/*max_workspaces=*/2);
  auto [default_tensors, default_infos] = run_graph(/*max_workspaces=*/0);

  ASSERT_EQ(pooled_tensors.size(), 3u);
  ASSERT_EQ(default_tensors.size(), 3u);

  for (int t = 0; t < 3; ++t) {
    const auto& pt = pooled_tensors[t].Get<std::vector<Tensor>>();
    const auto& dt = default_tensors[t].Get<std::vector<Tensor>>();
    ASSERT_EQ(pt.size(), 1u);
    ASSERT_EQ(dt.size(), 1u);

    // (a) Compare output tensor float buffers byte-for-byte.
    const Tensor& pooled_t = pt[0];
    const Tensor& default_t = dt[0];
    ASSERT_EQ(pooled_t.shape().dims, default_t.shape().dims);
    auto pr = pooled_t.GetCpuReadView();
    auto dr = default_t.GetCpuReadView();
    const float* pbuf = pr.buffer<float>();
    const float* dbuf = dr.buffer<float>();
    const int elems = pooled_t.shape().num_elements();
    for (int e = 0; e < elems; ++e) {
      EXPECT_EQ(pbuf[e], dbuf[e])
          << "tensor mismatch at frame=" << t << " elem=" << e;
    }

    // (b) Compare geometry tile_to_image_matrices element-wise.
    const auto& pi = pooled_infos[t].Get<TensorBatchInfo>();
    const auto& di = default_infos[t].Get<TensorBatchInfo>();
    ASSERT_NE(pi.geometry, nullptr);
    ASSERT_NE(di.geometry, nullptr);
    ASSERT_EQ(pi.geometry->tile_to_image_matrices.size(),
              di.geometry->tile_to_image_matrices.size());
    for (size_t m = 0; m < pi.geometry->tile_to_image_matrices.size(); ++m) {
      for (int elem = 0; elem < 16; ++elem) {
        EXPECT_FLOAT_EQ(pi.geometry->tile_to_image_matrices[m][elem],
                        di.geometry->tile_to_image_matrices[m][elem])
            << "matrix mismatch at frame=" << t << " matrix=" << m
            << " elem=" << elem;
      }
    }
  }
}

// Verifies that the optional CACHE_STATS output reports matrix cache hits.
// Feed the same (image, plan) twice with max_cached_tile_matrices=4 and
// emit_cache_stats=true. After two frames:
//   - tile_matrix.misses >= 1 (first frame is a miss),
//   - tile_matrix.hits >= 1 (second frame hits the cached geometry).
TEST(StreamingTilesTest, CacheStatsReportsMatrixHits) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    input_side_packet: "METADATA:meta"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
    output_stream: "CACHE_STATS:stats"
    options {
      [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
        max_cached_tile_matrices: 4
        emit_cache_stats: true
      }
    }
  )pb"));
  runner.MutableSidePackets()->Tag("METADATA") =
      MakePacket<InferenceMetadata>(Meta(4, 8, 8, 3, /*dynamic=*/false));
  // Send the same frame + plan twice so a cache hit occurs on the 2nd call.
  for (int t = 0; t < 2; ++t) {
    runner.MutableInputs()->Tag("IMAGE").packets.push_back(
        Adopt(WhiteFrame(16, 16).release()).At(Timestamp(t)));
    runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
        MakePacket<TilePlan>(TwoTiles()).At(Timestamp(t)));
  }
  MP_ASSERT_OK(runner.Run());

  const auto& stats_packets = runner.Outputs().Tag("CACHE_STATS").packets;
  ASSERT_EQ(stats_packets.size(), 2u);

  // Both packets are at the source timestamps (0 and 1).
  EXPECT_EQ(stats_packets[0].Timestamp(), Timestamp(0));
  EXPECT_EQ(stats_packets[1].Timestamp(), Timestamp(1));

  // After the second frame, cumulative stats must show at least one miss
  // (first frame) and at least one hit (second frame, same geometry key).
  const TilingCacheStats& last = stats_packets[1].Get<TilingCacheStats>();
  EXPECT_GE(last.tile_matrix.misses, 1)
      << "expected at least one cache miss (first frame)";
  EXPECT_GE(last.tile_matrix.hits, 1)
      << "expected at least one cache hit (second frame, same geometry)";
}

// A frame whose TILE_PLAN packet is missing (the bound advanced past the
// image's timestamp without a packet) must behave like an empty plan: emit
// the empty-frame BATCH_INFO so the merge accumulator still sees the frame,
// and produce no tensors — instead of dereferencing the missing packet
// (api2 Get() on an empty packet is fatal).
TEST(StreamingTilesTest, MissingTilePlanPacketEmitsEmptyFrameInfo) {
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
      Adopt(WhiteFrame(16, 16).release()).At(Timestamp(7)));
  // No TILE_PLAN packet at Timestamp(7).
  MP_ASSERT_OK(runner.Run());
  EXPECT_TRUE(runner.Outputs().Tag("TENSORS").packets.empty());
  const auto& ipk = runner.Outputs().Tag("BATCH_INFO").packets;
  ASSERT_EQ(ipk.size(), 1);
  const auto& info = ipk[0].Get<TensorBatchInfo>();
  EXPECT_EQ(info.source_frame_timestamp, 7);
  EXPECT_EQ(info.total_batches, 0);
  EXPECT_EQ(info.valid_count, 0);
}

// Metadata coming from options (METADATA side packet not connected) behaves
// identically to the side-packet path.
TEST(StreamingTilesTest, OptionsBorneMetadataWorksWithoutSidePacket) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
    options { [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
      metadata_batch_capacity: 4 metadata_input_height: 8
      metadata_input_width: 8 metadata_input_channels: 3 } }
  )pb"));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      Adopt(WhiteFrame(16, 16).release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(TwoTiles()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  ASSERT_EQ(runner.Outputs().Tag("TENSORS").packets.size(), 1u);
  const auto& tensors =
      runner.Outputs().Tag("TENSORS").packets[0].Get<std::vector<Tensor>>();
  EXPECT_EQ(tensors[0].shape().dims[0], 4);  // fixed batch = cap, padded
  EXPECT_EQ(tensors[0].shape().dims[1], 8);
  const auto& info =
      runner.Outputs().Tag("BATCH_INFO").packets[0].Get<TensorBatchInfo>();
  EXPECT_EQ(info.valid_count, 2);
}

// Both sources missing -> Open fails (via the existing RET_CHECKs).
TEST(StreamingTilesTest, MissingMetadataEverywhereFailsOpen) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image" input_stream: "TILE_PLAN:plan"
    output_stream: "TENSORS:tensors" output_stream: "BATCH_INFO:info"
  )pb"));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      Adopt(WhiteFrame(16, 16).release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(TwoTiles()).At(Timestamp(0)));
  absl::Status st = runner.Run();
  EXPECT_FALSE(st.ok());
  EXPECT_THAT(st.message(), HasSubstr("input_height"));
}

// When BOTH the METADATA side packet and options-borne metadata are provided,
// the side packet wins and the options metadata is ignored (pins existing
// behavior; a warning is logged).
TEST(StreamingTilesTest, SidePacketWinsOverOptionsMetadata) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    input_side_packet: "METADATA:meta"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
    options { [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
      metadata_batch_capacity: 2 metadata_input_height: 99
      metadata_input_width: 99 metadata_input_channels: 3 } }
  )pb"));
  runner.MutableSidePackets()->Tag("METADATA") =
      MakePacket<InferenceMetadata>(Meta(4, 8, 8, 3, /*dynamic=*/false));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      Adopt(WhiteFrame(16, 16).release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(TwoTiles()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& tpk = runner.Outputs().Tag("TENSORS").packets;
  ASSERT_EQ(tpk.size(), 1);
  const auto& tensors = tpk[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(tensors.size(), 1);
  EXPECT_EQ(tensors[0].shape().dims[0], 4);  // side packet wins, not 2
  EXPECT_EQ(tensors[0].shape().dims[1], 8);  // side packet wins, not 99
}

}  // namespace
}  // namespace mediapipe
