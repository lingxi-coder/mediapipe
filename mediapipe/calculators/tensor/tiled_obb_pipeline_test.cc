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
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

// merge -> global rotated NMS. Two overlapping tiles each detect the SAME
// real-world object in their overlap; after projection the two boxes coincide
// and global NMS keeps one. Proves the single frame-global NMS dedups across
// tiles.
TEST(TiledObbPipelineTest, MergeThenGlobalNmsDedupsAcrossTiles) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    input_stream: "plan"
    output_stream: "out"
    node {
      calculator: "MergeTileDetectionsAccumulatorCalculator"
      input_stream: "ORIENTED_DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
      input_stream: "TILE_PLAN:plan"
      output_stream: "ORIENTED_DETECTIONS:merged"
    }
    node {
      calculator: "RotatedNonMaxSuppressionCalculator"
      input_stream: "ORIENTED_DETECTIONS:merged"
      output_stream: "ORIENTED_DETECTIONS:out"
      options {
        [mediapipe.RotatedNonMaxSuppressionCalculatorOptions.ext] {
          iou_threshold: 0.5
        }
      }
    }
  )pb");

  // Overlapping tiles: left covers x[0,0.6], right covers x[0.4,1.0].
  TilePlan plan;
  TileGeometry l; l.tile_index=0; l.x_center=.3; l.y_center=.5; l.width=.6; l.height=1.0;
  TileGeometry r; r.tile_index=1; r.x_center=.7; r.y_center=.5; r.width=.6; r.height=1.0;
  plan.tiles = {l, r};

  // Object truly at frame (0.5,0.5,0.2,0.2). Tile-local coords:
  //   left:  cx=(0.5-0.0)/0.6=0.83333, w=0.2/0.6=0.33333
  //   right: cx=(0.5-0.4)/0.6=0.16667, w=0.33333
  auto obb = [](float cx, float cy, float w, float h, float s) {
    OrientedDetection d; d.set_cx(cx); d.set_cy(cy); d.set_width(w);
    d.set_height(h); d.set_rotation(0); d.add_score(s); d.add_label_id(0);
    return d;
  };
  auto batch = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch->push_back({obb(0.83333f, 0.5f, 0.33333f, 0.2f, 0.9f)});   // left
  batch->push_back({obb(0.16667f, 0.5f, 0.33333f, 0.2f, 0.8f)});   // right
  TensorBatchInfo info;
  info.source_frame_timestamp=0; info.total_batches=1; info.batch_capacity=2;
  info.valid_count=2; info.tile_indices={0,1};

  std::vector<Packet> out_packets;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("out", [&](const Packet& p) {
    out_packets.push_back(p); return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "plan", MakePacket<TilePlan>(plan).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info", MakePacket<TensorBatchInfo>(info).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets", Adopt(batch.release()).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(out_packets.size(), 1);
  const auto& out = out_packets[0].Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(out.size(), 1);  // both tiles -> same frame box -> NMS keeps one
  EXPECT_NEAR(out[0].cx(), 0.5f, 1e-3);
  EXPECT_NEAR(out[0].width(), 0.2f, 1e-3);
  EXPECT_NEAR(out[0].score(0), 0.9f, 1e-5);  // higher-scoring kept
}

}  // namespace
}  // namespace mediapipe
