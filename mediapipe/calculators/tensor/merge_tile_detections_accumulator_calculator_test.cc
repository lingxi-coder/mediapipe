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
  d.set_cx(cx); d.set_cy(cy); d.set_width(w); d.set_height(h);
  d.set_rotation(0.0f); d.add_score(0.9f); d.add_label_id(0);
  return d;
}

TEST(MergeTileAccumulatorTest, ProjectsTileLocalToFrame) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "MergeTileDetectionsAccumulatorCalculator"
    input_stream: "ORIENTED_DETECTIONS:dets"
    input_stream: "BATCH_INFO:info"
    input_stream: "TILE_PLAN:plan"
    output_stream: "ORIENTED_DETECTIONS:merged"
  )pb"));

  TilePlan plan;
  TileGeometry l; l.tile_index=0; l.x_center=.25; l.y_center=.5; l.width=.5; l.height=1.0;
  TileGeometry r; r.tile_index=1; r.x_center=.75; r.y_center=.5; r.width=.5; r.height=1.0;
  plan.tiles = {l, r};

  auto batch = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch->push_back({Obb(0.5f, 0.5f, 0.4f, 0.4f)});
  batch->push_back({Obb(0.5f, 0.5f, 0.4f, 0.4f)});

  TensorBatchInfo info;
  info.source_frame_timestamp = 0; info.batch_index = 0; info.total_batches = 1;
  info.batch_capacity = 2; info.valid_count = 2; info.tile_indices = {0, 1};

  runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(batch.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(plan).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& merged = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                           .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(merged.size(), 2);
  EXPECT_NEAR(merged[0].cx(), 0.25f, 1e-5);   // 0 + 0.5*0.5
  EXPECT_NEAR(merged[0].cy(), 0.5f, 1e-5);
  EXPECT_NEAR(merged[0].width(), 0.20f, 1e-5); // 0.4*0.5
  EXPECT_NEAR(merged[0].height(), 0.40f, 1e-5);// 0.4*1.0
  EXPECT_NEAR(merged[1].cx(), 0.75f, 1e-5);   // 0.5 + 0.5*0.5
}

TEST(MergeTileAccumulatorTest, DropsPaddedRows) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "MergeTileDetectionsAccumulatorCalculator"
    input_stream: "ORIENTED_DETECTIONS:dets"
    input_stream: "BATCH_INFO:info"
    input_stream: "TILE_PLAN:plan"
    output_stream: "ORIENTED_DETECTIONS:merged"
  )pb"));

  TilePlan plan;
  TileGeometry t; t.tile_index=0; t.x_center=.5; t.y_center=.5; t.width=1.0; t.height=1.0;
  plan.tiles = {t};

  auto batch = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch->push_back({Obb(0.5f, 0.5f, 0.2f, 0.2f)});
  batch->push_back({Obb(0.5f, 0.5f, 0.9f, 0.9f)});  // padded row -> dropped
  TensorBatchInfo info;
  info.source_frame_timestamp = 0; info.total_batches = 1; info.batch_capacity = 2;
  info.valid_count = 1; info.tile_indices = {0};

  runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(batch.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(plan).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& merged = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                           .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(merged.size(), 1);  // padded row dropped
}

}  // namespace
}  // namespace mediapipe
