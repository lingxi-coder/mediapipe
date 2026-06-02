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

}  // namespace
}  // namespace mediapipe
