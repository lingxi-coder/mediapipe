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

#include <cstring>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/tiling_matrix_utils.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

std::unique_ptr<ImageFrame> WhiteFrame(int w, int h) {
  auto f = std::make_unique<ImageFrame>(ImageFormat::SRGB, w, h);
  std::memset(f->MutablePixelData(), 255, f->Height() * f->WidthStep());
  return f;
}

TEST(TiledDetectionFrontGraphTest, EmitsBatchesWithInfo) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    output_stream: "tensors"
    output_stream: "info"
    node {
      calculator: "mediapipe.tiled_detection.TiledDetectionFrontGraph"
      input_stream: "IMAGE:image"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:info"
      options {
        [mediapipe.TiledDetectionFrontGraphOptions.ext] {
          tile_grid { cols: 2 }
          batch_capacity: 2
          input_height: 8
          input_width: 16
          input_channels: 3
        }
      }
    }
  )pb");
  std::vector<Packet> tensors, info;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("tensors", [&](const Packet& p) {
    tensors.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.ObserveOutputStream("info", [&](const Packet& p) {
    info.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image", Adopt(WhiteFrame(64, 48).release()).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  ASSERT_EQ(tensors.size(), 1u);  // 2 tiles / cap 2 -> 1 batch
  const auto& tensor_vec = tensors[0].Get<std::vector<Tensor>>();
  ASSERT_FALSE(tensor_vec.empty());
  const Tensor& t = tensor_vec[0];
  ASSERT_EQ(t.shape().dims.size(), 4u);
  EXPECT_EQ(t.shape().dims[0], 2);   // batch
  EXPECT_EQ(t.shape().dims[1], 8);   // height
  EXPECT_EQ(t.shape().dims[2], 16);  // width
  EXPECT_EQ(t.shape().dims[3], 3);   // channels
  ASSERT_EQ(info.size(), 1u);
  EXPECT_EQ(info[0].Get<TensorBatchInfo>().valid_count, 2);
}

OrientedDetection Obb(float cx, float cy, float w, float h, float score) {
  OrientedDetection d;
  d.set_cx(cx);
  d.set_cy(cy);
  d.set_width(w);
  d.set_height(h);
  d.set_rotation(0.0f);
  d.add_score(score);
  d.add_label_id(0);
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

CalculatorGraphConfig ObbMergeGraphConfig() {
  return ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    output_stream: "merged"
    node {
      calculator: "mediapipe.tiled_detection.TiledObbMergeGraph"
      input_stream: "ORIENTED_DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
      output_stream: "ORIENTED_DETECTIONS:merged"
      options {
        [mediapipe.TiledObbMergeGraphOptions.ext] { iou_threshold: 0.5 }
      }
    }
  )pb");
}

TEST(TiledObbMergeGraphTest, MergesBatchesAndRunsGlobalNms) {
  auto config = ObbMergeGraphConfig();

  TileGeometry left;
  left.tile_index = 0;
  left.x_center = .25f;
  left.y_center = .5f;
  left.width = .5f;
  left.height = 1.0f;
  TileGeometry right;
  right.tile_index = 1;
  right.x_center = .75f;
  right.y_center = .5f;
  right.width = .5f;
  right.height = 1.0f;

  // Two batches for ONE source frame (ts=77). Both detections project to the
  // SAME frame-space box centered at (0.5, 0.5), size (0.2, 0.4):
  //   batch 0 (left tile):  tile-local cx=1.0 -> frame 0 + 1.0*0.5 = 0.5
  //   batch 1 (right tile): tile-local cx=0.0 -> frame 0.5 + 0.0*0.5 = 0.5
  // IoU = 1 >= 0.5, so the global NMS must keep ONE (the 0.9-score box).
  std::vector<std::vector<OrientedDetection>> batch0 = {
      {Obb(1.0f, 0.5f, 0.4f, 0.4f, 0.9f)}};
  std::vector<std::vector<OrientedDetection>> batch1 = {
      {Obb(0.0f, 0.5f, 0.4f, 0.4f, 0.8f)}};

  TensorBatchInfo info0;
  info0.source_frame_timestamp = 77;
  info0.batch_index = 0;
  info0.total_batches = 2;
  info0.valid_count = 1;
  info0.tile_indices = {0};
  info0.geometry = MakeGeom({left}, 100, 100);
  TensorBatchInfo info1 = info0;
  info1.batch_index = 1;
  info1.tile_indices = {1};
  info1.geometry = MakeGeom({right}, 100, 100);

  std::vector<Packet> merged_packets;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("merged", [&](const Packet& p) {
    merged_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets", MakePacket<std::vector<std::vector<OrientedDetection>>>(batch0)
                  .At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info", MakePacket<TensorBatchInfo>(info0).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets", MakePacket<std::vector<std::vector<OrientedDetection>>>(batch1)
                  .At(Timestamp(1))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info", MakePacket<TensorBatchInfo>(info1).At(Timestamp(1))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(merged_packets.size(), 1u);
  EXPECT_EQ(merged_packets[0].Timestamp(), Timestamp(77));
  const auto& merged =
      merged_packets[0].Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(merged.size(), 1u);
  ASSERT_EQ(merged[0].score_size(), 1);
  EXPECT_NEAR(merged[0].score(0), 0.9f, 1e-5);
  EXPECT_NEAR(merged[0].cx(), 0.5f, 1e-4);
  EXPECT_NEAR(merged[0].width(), 0.2f, 1e-4);
}

// Pins the no-bypass design decision: even for a single-tile frame the global
// rotated NMS still runs (tile-local NMS defaults off upstream, so bypassing
// would hand un-deduped raw detections to the caller).
TEST(TiledObbMergeGraphTest, SingleTileStillRunsGlobalNms) {
  auto config = ObbMergeGraphConfig();

  TileGeometry full;
  full.tile_index = 0;
  full.x_center = .5f;
  full.y_center = .5f;
  full.width = 1.0f;
  full.height = 1.0f;

  // ONE batch, ONE tile row holding TWO overlapping same-class OBBs.
  std::vector<std::vector<OrientedDetection>> batch = {
      {Obb(0.5f, 0.5f, 0.4f, 0.4f, 0.9f),
       Obb(0.5f, 0.5f, 0.4f, 0.4f, 0.8f)}};

  TensorBatchInfo info;
  info.source_frame_timestamp = 5;
  info.batch_index = 0;
  info.total_batches = 1;
  info.valid_count = 1;
  info.tile_indices = {0};
  info.geometry = MakeGeom({full}, 100, 100);

  std::vector<Packet> merged_packets;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("merged", [&](const Packet& p) {
    merged_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets", MakePacket<std::vector<std::vector<OrientedDetection>>>(batch)
                  .At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info", MakePacket<TensorBatchInfo>(info).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(merged_packets.size(), 1u);
  EXPECT_EQ(merged_packets[0].Timestamp(), Timestamp(5));
  const auto& merged =
      merged_packets[0].Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(merged.size(), 1u);  // global NMS ran despite a single tile
  ASSERT_EQ(merged[0].score_size(), 1);
  EXPECT_NEAR(merged[0].score(0), 0.9f, 1e-5);
}

// Discriminates the options forwarding in TiledObbMergeGraph: the subgraph's
// defaults match the NMS calculator's defaults, so only a NON-default option
// value can prove the `no.set_*` forwarding lines exist. Two DISJOINT boxes
// (IoU = 0) survive any iou_threshold; only a forwarded max_detections: 1 can
// drop one of them.
TEST(TiledObbMergeGraphTest, ForwardsMaxDetectionsOption) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    output_stream: "merged"
    node {
      calculator: "mediapipe.tiled_detection.TiledObbMergeGraph"
      input_stream: "ORIENTED_DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
      output_stream: "ORIENTED_DETECTIONS:merged"
      options {
        [mediapipe.TiledObbMergeGraphOptions.ext] {
          iou_threshold: 0.5
          max_detections: 1
        }
      }
    }
  )pb");

  TileGeometry full;
  full.tile_index = 0;
  full.x_center = .5f;
  full.y_center = .5f;
  full.width = 1.0f;
  full.height = 1.0f;

  // ONE batch, ONE tile row holding TWO DISJOINT same-class OBBs.
  std::vector<std::vector<OrientedDetection>> batch = {
      {Obb(0.2f, 0.2f, 0.1f, 0.1f, 0.9f),
       Obb(0.7f, 0.7f, 0.1f, 0.1f, 0.8f)}};

  TensorBatchInfo info;
  info.source_frame_timestamp = 9;
  info.batch_index = 0;
  info.total_batches = 1;
  info.valid_count = 1;
  info.tile_indices = {0};
  info.geometry = MakeGeom({full}, 100, 100);

  std::vector<Packet> merged_packets;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("merged", [&](const Packet& p) {
    merged_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets", MakePacket<std::vector<std::vector<OrientedDetection>>>(batch)
                  .At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info", MakePacket<TensorBatchInfo>(info).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(merged_packets.size(), 1u);
  EXPECT_EQ(merged_packets[0].Timestamp(), Timestamp(9));
  const auto& merged =
      merged_packets[0].Get<std::vector<OrientedDetection>>();
  // Default max_detections (-1) would keep BOTH disjoint boxes; the
  // forwarded max_detections: 1 keeps only the top-scoring one.
  ASSERT_EQ(merged.size(), 1u);
  ASSERT_EQ(merged[0].score_size(), 1);
  EXPECT_NEAR(merged[0].score(0), 0.9f, 1e-5);
}

}  // namespace
}  // namespace mediapipe
