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
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

constexpr char kNodeConfig[] = R"pb(
  calculator: "MergeTileBoxDetectionsAccumulatorCalculator"
  input_stream: "DETECTIONS:dets"
  input_stream: "BATCH_INFO:info"
  output_stream: "DETECTIONS:merged"
)pb";

Detection Box(float xmin, float ymin, float w, float h) {
  Detection d;
  d.add_score(0.9f);
  d.add_label_id(0);
  auto* ld = d.mutable_location_data();
  ld->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* bb = ld->mutable_relative_bounding_box();
  bb->set_xmin(xmin);
  bb->set_ymin(ymin);
  bb->set_width(w);
  bb->set_height(h);
  return d;
}

// Builds a TileBatchGeometry for the given tiles using pixel ROIs computed
// from tile normalized coordinates over a fw x fh frame.
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

TEST(MergeTileBoxAccumulatorTest, ProjectsTileLocalToFrame) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kNodeConfig));

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

  auto batch = std::make_unique<std::vector<std::vector<Detection>>>();
  batch->push_back({Box(0.3f, 0.3f, 0.4f, 0.4f)});
  batch->push_back({Box(0.3f, 0.3f, 0.4f, 0.4f)});

  TensorBatchInfo info;
  info.source_frame_timestamp = 0;
  info.batch_index = 0;
  info.total_batches = 1;
  info.batch_capacity = 2;
  info.valid_count = 2;
  info.tile_indices = {0, 1};
  info.geometry = geom;

  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      Adopt(batch.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& merged = runner.Outputs()
                           .Tag("DETECTIONS")
                           .packets[0]
                           .Get<std::vector<Detection>>();
  ASSERT_EQ(merged.size(), 2);
  const auto& bb0 = merged[0].location_data().relative_bounding_box();
  EXPECT_NEAR(bb0.xmin(), 0.15f, 1e-5);    // 0 + 0.3*0.5
  EXPECT_NEAR(bb0.ymin(), 0.3f, 1e-5);     // 0.3*1.0
  EXPECT_NEAR(bb0.width(), 0.20f, 1e-5);   // 0.4*0.5
  EXPECT_NEAR(bb0.height(), 0.40f, 1e-5);  // 0.4*1.0
  const auto& bb1 = merged[1].location_data().relative_bounding_box();
  EXPECT_NEAR(bb1.xmin(), 0.65f, 1e-5);    // 0.5 + 0.3*0.5
}

TEST(MergeTileBoxAccumulatorTest, DropsPaddedRowsAndWaitsForAllBatches) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kNodeConfig));

  TileGeometry t;
  t.tile_index = 0;
  t.x_center = .5f;
  t.y_center = .5f;
  t.width = 1.0f;
  t.height = 1.0f;
  auto geom = MakeGeom({t}, 100, 100);

  // Two batches for one source frame; each has 1 valid row + 1 padded row.
  for (int b = 0; b < 2; ++b) {
    auto batch = std::make_unique<std::vector<std::vector<Detection>>>();
    batch->push_back({Box(0.4f, 0.4f, 0.2f, 0.2f)});
    batch->push_back({Box(0.0f, 0.0f, 0.9f, 0.9f)});  // padded row -> dropped
    TensorBatchInfo info;
    info.source_frame_timestamp = 77;
    info.batch_index = b;
    info.total_batches = 2;
    info.batch_capacity = 2;
    info.valid_count = 1;
    info.tile_indices = {0};
    info.geometry = geom;
    runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
        Adopt(batch.release()).At(Timestamp(b)));
    runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
        MakePacket<TensorBatchInfo>(info).At(Timestamp(b)));
  }

  MP_ASSERT_OK(runner.Run());
  const auto& packets = runner.Outputs().Tag("DETECTIONS").packets;
  ASSERT_EQ(packets.size(), 1);  // ONE merged result for the source frame
  EXPECT_EQ(packets[0].Timestamp(), Timestamp(77));
  EXPECT_EQ(packets[0].Get<std::vector<Detection>>().size(), 2);
}

// Mirrors the empty-frame protocol guards of the OrientedDetection merge:
// BATCH_INFO with no paired DETECTIONS packet (T==0 / scheduler SKIP) must
// emit an empty result at the source timestamp, not crash.
TEST(MergeTileBoxAccumulatorTest, EmptyFrameWithoutDetectionsPacketEmitsEmptyResult) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kNodeConfig));

  TensorBatchInfo info;
  info.source_frame_timestamp = 1234;
  info.total_batches = 0;
  info.valid_count = 0;  // geometry stays nullptr, like the real producer

  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& packets = runner.Outputs().Tag("DETECTIONS").packets;
  ASSERT_EQ(packets.size(), 1);
  EXPECT_EQ(packets[0].Timestamp(), Timestamp(1234));
  EXPECT_TRUE(packets[0].Get<std::vector<Detection>>().empty());
}

// A batch whose detections packet is missing (bound-only) still counts toward
// total_batches so the frame completes with the rows that did arrive.
TEST(MergeTileBoxAccumulatorTest, MissingDetectionsPacketStillCountsBatch) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kNodeConfig));

  TileGeometry t;
  t.tile_index = 0;
  t.x_center = .5f;
  t.y_center = .5f;
  t.width = 1.0f;
  t.height = 1.0f;
  auto geom = MakeGeom({t}, 100, 100);

  auto batch = std::make_unique<std::vector<std::vector<Detection>>>();
  batch->push_back({Box(0.4f, 0.4f, 0.2f, 0.2f)});
  TensorBatchInfo info0;
  info0.source_frame_timestamp = 50;
  info0.batch_index = 0;
  info0.total_batches = 2;
  info0.valid_count = 1;
  info0.tile_indices = {0};
  info0.geometry = geom;
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      Adopt(batch.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info0).At(Timestamp(0)));

  TensorBatchInfo info1 = info0;
  info1.batch_index = 1;
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info1).At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run());
  const auto& packets = runner.Outputs().Tag("DETECTIONS").packets;
  ASSERT_EQ(packets.size(), 1);
  EXPECT_EQ(packets[0].Timestamp(), Timestamp(50));
  EXPECT_EQ(packets[0].Get<std::vector<Detection>>().size(), 1);
}

}  // namespace
}  // namespace mediapipe
