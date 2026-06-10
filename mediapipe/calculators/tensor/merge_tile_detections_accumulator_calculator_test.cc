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

#include "mediapipe/calculators/tensor/tiling_matrix_utils.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

OrientedDetection Obb(float cx, float cy, float w, float h) {
  OrientedDetection d;
  d.set_cx(cx);
  d.set_cy(cy);
  d.set_width(w);
  d.set_height(h);
  d.set_rotation(0.0f);
  d.add_score(0.9f);
  d.add_label_id(0);
  return d;
}

// Builds a TileBatchGeometry for the given tiles using pixel ROIs computed from
// tile normalized coordinates over a frame_w x frame_h frame.
std::shared_ptr<TileBatchGeometry> MakeGeom(
    const std::vector<TileGeometry>& tiles, int fw, int fh) {
  auto geom = std::make_shared<TileBatchGeometry>();
  for (const TileGeometry& g : tiles) {
    TilePixelRoi roi;
    roi.x = static_cast<int>(g.x0() * fw + 0.5f);
    roi.y = static_cast<int>(g.y0() * fh + 0.5f);
    roi.width = static_cast<int>(g.width * fw + 0.5f);
    roi.height = static_cast<int>(g.height * fh + 0.5f);
    geom->tile_indices.push_back(g.tile_index);
    geom->tile_geometries.push_back(g);
    geom->effective_pixel_rois.push_back(roi);
    geom->tile_to_image_matrices.push_back(TileToImageMatrix(roi, fw, fh));
  }
  return geom;
}

TEST(MergeTileAccumulatorTest, ProjectsTileLocalToFrame) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MergeTileDetectionsAccumulatorCalculator"
        input_stream: "ORIENTED_DETECTIONS:dets"
        input_stream: "BATCH_INFO:info"
        output_stream: "ORIENTED_DETECTIONS:merged"
      )pb"));

  TileGeometry l;
  l.tile_index = 0;
  l.x_center = .25f;
  l.y_center = .5f;
  l.width = .5f;
  l.height = 1.0f;
  TileGeometry r;
  r.tile_index = 1;
  r.x_center = .75f;
  r.y_center = .5f;
  r.width = .5f;
  r.height = 1.0f;

  auto geom = MakeGeom({l, r}, 100, 100);

  auto batch = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch->push_back({Obb(0.5f, 0.5f, 0.4f, 0.4f)});
  batch->push_back({Obb(0.5f, 0.5f, 0.4f, 0.4f)});

  TensorBatchInfo info;
  info.source_frame_timestamp = 0;
  info.batch_index = 0;
  info.total_batches = 1;
  info.batch_capacity = 2;
  info.valid_count = 2;
  info.tile_indices = {0, 1};
  info.geometry = geom;

  runner.MutableInputs()
      ->Tag("ORIENTED_DETECTIONS")
      .packets.push_back(Adopt(batch.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& merged = runner.Outputs()
                           .Tag("ORIENTED_DETECTIONS")
                           .packets[0]
                           .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(merged.size(), 2);
  EXPECT_NEAR(merged[0].cx(), 0.25f, 1e-5);    // 0 + 0.5*0.5
  EXPECT_NEAR(merged[0].cy(), 0.5f, 1e-5);
  EXPECT_NEAR(merged[0].width(), 0.20f, 1e-5);  // 0.4*0.5
  EXPECT_NEAR(merged[0].height(), 0.40f, 1e-5); // 0.4*1.0
  EXPECT_NEAR(merged[1].cx(), 0.75f, 1e-5);    // 0.5 + 0.5*0.5
}

TEST(MergeTileAccumulatorTest, DropsPaddedRows) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MergeTileDetectionsAccumulatorCalculator"
        input_stream: "ORIENTED_DETECTIONS:dets"
        input_stream: "BATCH_INFO:info"
        output_stream: "ORIENTED_DETECTIONS:merged"
      )pb"));

  TileGeometry t;
  t.tile_index = 0;
  t.x_center = .5f;
  t.y_center = .5f;
  t.width = 1.0f;
  t.height = 1.0f;

  // Geometry has only 1 row (valid_count=1); second batch row is padded.
  auto geom = MakeGeom({t}, 100, 100);

  auto batch = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch->push_back({Obb(0.5f, 0.5f, 0.2f, 0.2f)});
  batch->push_back({Obb(0.5f, 0.5f, 0.9f, 0.9f)});  // padded row -> dropped
  TensorBatchInfo info;
  info.source_frame_timestamp = 0;
  info.total_batches = 1;
  info.batch_capacity = 2;
  info.valid_count = 1;
  info.tile_indices = {0};
  info.geometry = geom;

  runner.MutableInputs()
      ->Tag("ORIENTED_DETECTIONS")
      .packets.push_back(Adopt(batch.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& merged = runner.Outputs()
                           .Tag("ORIENTED_DETECTIONS")
                           .packets[0]
                           .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(merged.size(), 1);  // padded row dropped
}

// A boundary tile whose requested normalized geometry (width 0.6) was clamped
// at sample time to a narrower effective pixel ROI (50px of a 100px frame =>
// 0.5). Both the projected center AND size must follow the effective ROI, so a
// tile-local box of width 0.4 maps to 0.4 * 0.5 = 0.20 — NOT 0.4 * 0.6 = 0.24
// (which scaling by the requested tile width would wrongly produce).
TEST(MergeTileAccumulatorTest, BoundaryTileScalesSizeByEffectiveRoiNotRequested) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MergeTileDetectionsAccumulatorCalculator"
        input_stream: "ORIENTED_DETECTIONS:dets"
        input_stream: "BATCH_INFO:info"
        output_stream: "ORIENTED_DETECTIONS:merged"
      )pb"));

  TileGeometry g;
  g.tile_index = 0;
  g.x_center = 0.3f;  // requested x0=0, width 0.6 -> [0, 0.6]
  g.y_center = 0.5f;
  g.width = 0.6f;
  g.height = 1.0f;

  // Effective ROI clamped to the left half (50px) — diverges from requested.
  auto geom = std::make_shared<TileBatchGeometry>();
  TilePixelRoi roi{/*x=*/0, /*y=*/0, /*width=*/50, /*height=*/100};
  geom->tile_indices.push_back(0);
  geom->tile_geometries.push_back(g);
  geom->effective_pixel_rois.push_back(roi);
  geom->tile_to_image_matrices.push_back(TileToImageMatrix(roi, 100, 100));

  auto batch = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch->push_back({Obb(0.5f, 0.5f, 0.4f, 0.4f)});  // tile-local center + size

  TensorBatchInfo info;
  info.source_frame_timestamp = 0;
  info.total_batches = 1;
  info.batch_capacity = 1;
  info.valid_count = 1;
  info.tile_indices = {0};
  info.geometry = geom;

  runner.MutableInputs()
      ->Tag("ORIENTED_DETECTIONS")
      .packets.push_back(Adopt(batch.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& merged = runner.Outputs()
                           .Tag("ORIENTED_DETECTIONS")
                           .packets[0]
                           .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(merged.size(), 1);
  EXPECT_NEAR(merged[0].cx(), 0.25f, 1e-4);     // center of [0, 0.5]
  EXPECT_NEAR(merged[0].width(), 0.20f, 1e-4);  // 0.4 * 0.5 (effective), not 0.24
  EXPECT_NEAR(merged[0].height(), 0.40f, 1e-4); // 0.4 * 1.0
}

// An anisotropically clamped tile (effective ROI 50x100 of a 100x100 frame:
// sx=0.5, sy=1.0) must transform a ROTATED box's axes through the scale —
// not scale w/h per-axis while keeping the angle. For rotation=pi/2 the
// box's w-axis lies along tile-local Y (so it scales by sy) and its h-axis
// along X (scales by sx); naive per-axis scaling swaps the two factors.
TEST(MergeTileAccumulatorTest, AnisotropicTileTransformsRotatedBoxAxes) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MergeTileDetectionsAccumulatorCalculator"
        input_stream: "ORIENTED_DETECTIONS:dets"
        input_stream: "BATCH_INFO:info"
        output_stream: "ORIENTED_DETECTIONS:merged"
      )pb"));

  constexpr float kHalfPi = 1.57079632679f;
  TileGeometry g;
  g.tile_index = 0;
  g.x_center = 0.25f;
  g.y_center = 0.5f;
  g.width = 0.5f;
  g.height = 1.0f;

  auto geom = std::make_shared<TileBatchGeometry>();
  TilePixelRoi roi{/*x=*/0, /*y=*/0, /*width=*/50, /*height=*/100};
  geom->tile_indices.push_back(0);
  geom->tile_geometries.push_back(g);
  geom->effective_pixel_rois.push_back(roi);
  geom->tile_to_image_matrices.push_back(TileToImageMatrix(roi, 100, 100));

  OrientedDetection d = Obb(0.5f, 0.5f, 0.4f, 0.2f);
  d.set_rotation(kHalfPi);
  auto batch = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch->push_back({d});

  TensorBatchInfo info;
  info.source_frame_timestamp = 0;
  info.total_batches = 1;
  info.batch_capacity = 1;
  info.valid_count = 1;
  info.tile_indices = {0};
  info.geometry = geom;

  runner.MutableInputs()
      ->Tag("ORIENTED_DETECTIONS")
      .packets.push_back(Adopt(batch.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& merged = runner.Outputs()
                           .Tag("ORIENTED_DETECTIONS")
                           .packets[0]
                           .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(merged.size(), 1);
  EXPECT_NEAR(merged[0].cx(), 0.25f, 1e-4);
  EXPECT_NEAR(merged[0].cy(), 0.5f, 1e-4);
  EXPECT_NEAR(merged[0].width(), 0.4f, 1e-4);    // w-axis along Y -> sy=1.0
  EXPECT_NEAR(merged[0].height(), 0.1f, 1e-4);   // h-axis along X -> sx=0.5
  EXPECT_NEAR(merged[0].rotation(), kHalfPi, 1e-4);
}

// Isotropic tile (sx == sy): per-axis scaling is exact and the rotation must
// pass through unchanged — pins that the verified square-tile behavior is
// untouched by the anisotropic-transform branch.
TEST(MergeTileAccumulatorTest, IsotropicTilePreservesRotationAndScales) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MergeTileDetectionsAccumulatorCalculator"
        input_stream: "ORIENTED_DETECTIONS:dets"
        input_stream: "BATCH_INFO:info"
        output_stream: "ORIENTED_DETECTIONS:merged"
      )pb"));

  TileGeometry g;
  g.tile_index = 0;
  g.x_center = 0.25f;
  g.y_center = 0.25f;
  g.width = 0.5f;
  g.height = 0.5f;

  auto geom = std::make_shared<TileBatchGeometry>();
  TilePixelRoi roi{/*x=*/0, /*y=*/0, /*width=*/50, /*height=*/50};
  geom->tile_indices.push_back(0);
  geom->tile_geometries.push_back(g);
  geom->effective_pixel_rois.push_back(roi);
  geom->tile_to_image_matrices.push_back(TileToImageMatrix(roi, 100, 100));

  OrientedDetection d = Obb(0.5f, 0.5f, 0.4f, 0.2f);
  d.set_rotation(0.7f);
  auto batch = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch->push_back({d});

  TensorBatchInfo info;
  info.source_frame_timestamp = 0;
  info.total_batches = 1;
  info.batch_capacity = 1;
  info.valid_count = 1;
  info.tile_indices = {0};
  info.geometry = geom;

  runner.MutableInputs()
      ->Tag("ORIENTED_DETECTIONS")
      .packets.push_back(Adopt(batch.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& merged = runner.Outputs()
                           .Tag("ORIENTED_DETECTIONS")
                           .packets[0]
                           .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(merged.size(), 1);
  EXPECT_NEAR(merged[0].cx(), 0.25f, 1e-4);
  EXPECT_NEAR(merged[0].cy(), 0.25f, 1e-4);
  EXPECT_NEAR(merged[0].width(), 0.2f, 1e-4);
  EXPECT_NEAR(merged[0].height(), 0.1f, 1e-4);
  EXPECT_NEAR(merged[0].rotation(), 0.7f, 1e-4);
}

// The producer's empty-frame protocol (T==0, e.g. scheduler SKIP) sends a
// BATCH_INFO packet and only advances the detections-side timestamp bound —
// no ORIENTED_DETECTIONS packet ever exists at that timestamp. The merge must
// emit the empty source-frame result instead of dereferencing the missing
// packet (api2 Get() on an empty packet is fatal).
TEST(MergeTileAccumulatorTest, EmptyFrameWithoutDetectionsPacketEmitsEmptyResult) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MergeTileDetectionsAccumulatorCalculator"
        input_stream: "ORIENTED_DETECTIONS:dets"
        input_stream: "BATCH_INFO:info"
        output_stream: "ORIENTED_DETECTIONS:merged"
      )pb"));

  TensorBatchInfo info;
  info.source_frame_timestamp = 1234;
  info.total_batches = 0;
  info.valid_count = 0;  // geometry stays nullptr, like the real producer

  // Only BATCH_INFO carries a packet; the detections stream stays empty (its
  // bound advances past Timestamp(0) when the runner closes the sources).
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& packets = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets;
  ASSERT_EQ(packets.size(), 1);
  EXPECT_EQ(packets[0].Timestamp(), Timestamp(1234));
  EXPECT_TRUE(packets[0].Get<std::vector<OrientedDetection>>().empty());
}

// A batch whose decoded detections packet is missing (bound-only) must still
// count toward total_batches so the frame completes with the rows that did
// arrive, instead of crashing or waiting forever.
TEST(MergeTileAccumulatorTest, MissingDetectionsPacketStillCountsBatch) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MergeTileDetectionsAccumulatorCalculator"
        input_stream: "ORIENTED_DETECTIONS:dets"
        input_stream: "BATCH_INFO:info"
        output_stream: "ORIENTED_DETECTIONS:merged"
      )pb"));

  TileGeometry t;
  t.tile_index = 0;
  t.x_center = .5f;
  t.y_center = .5f;
  t.width = 1.0f;
  t.height = 1.0f;
  auto geom = MakeGeom({t}, 100, 100);

  // Batch 0 of 2: one detection present.
  auto batch = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch->push_back({Obb(0.5f, 0.5f, 0.2f, 0.2f)});
  TensorBatchInfo info0;
  info0.source_frame_timestamp = 50;
  info0.batch_index = 0;
  info0.total_batches = 2;
  info0.valid_count = 1;
  info0.tile_indices = {0};
  info0.geometry = geom;
  runner.MutableInputs()
      ->Tag("ORIENTED_DETECTIONS")
      .packets.push_back(Adopt(batch.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info0).At(Timestamp(0)));

  // Batch 1 of 2: BATCH_INFO only — the detections packet was never produced.
  TensorBatchInfo info1 = info0;
  info1.batch_index = 1;
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info1).At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run());
  const auto& packets = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets;
  ASSERT_EQ(packets.size(), 1);
  EXPECT_EQ(packets[0].Timestamp(), Timestamp(50));
  EXPECT_EQ(packets[0].Get<std::vector<OrientedDetection>>().size(), 1);
}

}  // namespace
}  // namespace mediapipe
