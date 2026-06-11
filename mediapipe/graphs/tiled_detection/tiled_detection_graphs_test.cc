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

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/tiling_matrix_utils.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/location_data.pb.h"
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

Detection Box(float score, float xmin, float ymin, float w, float h) {
  Detection d;
  d.add_score(score);
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

TileGeometry MakeTile(int tile_index, float x_center, float y_center,
                      float width, float height) {
  TileGeometry g;
  g.tile_index = tile_index;
  g.x_center = x_center;
  g.y_center = y_center;
  g.width = width;
  g.height = height;
  return g;
}

// A single tile covering the whole source frame.
TileGeometry FullFrameTile() { return MakeTile(0, .5f, .5f, 1.0f, 1.0f); }

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

// Builds a TensorBatchInfo for one batch of the frame at source timestamp
// `source_ts`; valid_count and tile_indices are derived from `geom`'s rows.
TensorBatchInfo MakeBatchInfo(int64_t source_ts, int batch_index,
                              int total_batches,
                              std::shared_ptr<TileBatchGeometry> geom) {
  TensorBatchInfo info;
  info.source_frame_timestamp = source_ts;
  info.batch_index = batch_index;
  info.total_batches = total_batches;
  info.valid_count = static_cast<int>(geom->tile_indices.size());
  info.tile_indices = geom->tile_indices;
  info.geometry = std::move(geom);
  return info;
}

Packet ObbBatch(std::vector<std::vector<OrientedDetection>> rows) {
  return MakePacket<std::vector<std::vector<OrientedDetection>>>(
      std::move(rows));
}

Packet BoxBatch(std::vector<std::vector<Detection>> rows) {
  return MakePacket<std::vector<std::vector<Detection>>>(std::move(rows));
}

// Runs a merge graph whose graph-level streams are "dets" (per-batch
// detections), "info" (TensorBatchInfo), and "merged" (output): initializes
// `config`, feeds each (detections, batch info) pair at consecutive synthetic
// timestamps, and returns the packets observed on "merged".
absl::StatusOr<std::vector<Packet>> RunMergeGraph(
    const CalculatorGraphConfig& config,
    std::vector<std::pair<Packet, TensorBatchInfo>> batches) {
  std::vector<Packet> merged_packets;
  CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  MP_RETURN_IF_ERROR(graph.ObserveOutputStream("merged", [&](const Packet& p) {
    merged_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  for (int i = 0; i < static_cast<int>(batches.size()); ++i) {
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "dets", batches[i].first.At(Timestamp(i))));
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "info", MakePacket<TensorBatchInfo>(std::move(batches[i].second))
                    .At(Timestamp(i))));
  }
  MP_RETURN_IF_ERROR(graph.CloseAllPacketSources());
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return merged_packets;
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

CalculatorGraphConfig BoxMergeGraphConfig() {
  return ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    output_stream: "merged"
    node {
      calculator: "mediapipe.tiled_detection.TiledBoxMergeGraph"
      input_stream: "DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
      output_stream: "DETECTIONS:merged"
      options {
        [mediapipe.TiledBoxMergeGraphOptions.ext] { iou_threshold: 0.5 }
      }
    }
  )pb");
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

TEST(TiledObbMergeGraphTest, MergesBatchesAndRunsGlobalNms) {
  // Two batches for ONE source frame (ts=77), one per half-frame tile. Both
  // detections project to the SAME frame-space box centered at (0.5, 0.5),
  // size (0.2, 0.4):
  //   batch 0 (left tile):  tile-local cx=1.0 -> frame 0 + 1.0*0.5 = 0.5
  //   batch 1 (right tile): tile-local cx=0.0 -> frame 0.5 + 0.0*0.5 = 0.5
  // IoU = 1 >= 0.5, so the global NMS must keep ONE (the 0.9-score box).
  TileGeometry left = MakeTile(0, .25f, .5f, .5f, 1.0f);
  TileGeometry right = MakeTile(1, .75f, .5f, .5f, 1.0f);

  MP_ASSERT_OK_AND_ASSIGN(
      std::vector<Packet> merged_packets,
      RunMergeGraph(ObbMergeGraphConfig(),
                    {{ObbBatch({{Obb(1.0f, 0.5f, 0.4f, 0.4f, 0.9f)}}),
                      MakeBatchInfo(77, 0, 2, MakeGeom({left}, 100, 100))},
                     {ObbBatch({{Obb(0.0f, 0.5f, 0.4f, 0.4f, 0.8f)}}),
                      MakeBatchInfo(77, 1, 2, MakeGeom({right}, 100, 100))}}));

  ASSERT_EQ(merged_packets.size(), 1u);
  EXPECT_EQ(merged_packets[0].Timestamp(), Timestamp(77));
  const auto& merged = merged_packets[0].Get<std::vector<OrientedDetection>>();
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
  // ONE batch, ONE tile row holding TWO overlapping same-class OBBs.
  MP_ASSERT_OK_AND_ASSIGN(
      std::vector<Packet> merged_packets,
      RunMergeGraph(ObbMergeGraphConfig(),
                    {{ObbBatch({{Obb(0.5f, 0.5f, 0.4f, 0.4f, 0.9f),
                                 Obb(0.5f, 0.5f, 0.4f, 0.4f, 0.8f)}}),
                      MakeBatchInfo(5, 0, 1,
                                    MakeGeom({FullFrameTile()}, 100, 100))}}));

  ASSERT_EQ(merged_packets.size(), 1u);
  EXPECT_EQ(merged_packets[0].Timestamp(), Timestamp(5));
  const auto& merged = merged_packets[0].Get<std::vector<OrientedDetection>>();
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

  // ONE batch, ONE tile row holding TWO DISJOINT same-class OBBs.
  MP_ASSERT_OK_AND_ASSIGN(
      std::vector<Packet> merged_packets,
      RunMergeGraph(config,
                    {{ObbBatch({{Obb(0.2f, 0.2f, 0.1f, 0.1f, 0.9f),
                                 Obb(0.7f, 0.7f, 0.1f, 0.1f, 0.8f)}}),
                      MakeBatchInfo(9, 0, 1,
                                    MakeGeom({FullFrameTile()}, 100, 100))}}));

  ASSERT_EQ(merged_packets.size(), 1u);
  EXPECT_EQ(merged_packets[0].Timestamp(), Timestamp(9));
  const auto& merged = merged_packets[0].Get<std::vector<OrientedDetection>>();
  // Default max_detections (-1) would keep BOTH disjoint boxes; the
  // forwarded max_detections: 1 keeps only the top-scoring one.
  ASSERT_EQ(merged.size(), 1u);
  ASSERT_EQ(merged[0].score_size(), 1);
  EXPECT_NEAR(merged[0].score(0), 0.9f, 1e-5);
}

TEST(TiledBoxMergeGraphTest, MergesBatchesAndRunsGlobalNms) {
  // Two batches for ONE source frame (ts=77). Both tiles are full-frame, so
  // identical tile-local boxes project to the SAME frame-space box; IoU = 1
  // >= 0.5, so the global NMS must keep ONE (the 0.9-score box).
  auto geom = MakeGeom({FullFrameTile()}, 100, 100);

  MP_ASSERT_OK_AND_ASSIGN(
      std::vector<Packet> merged_packets,
      RunMergeGraph(BoxMergeGraphConfig(),
                    {{BoxBatch({{Box(0.9f, 0.3f, 0.3f, 0.2f, 0.4f)}}),
                      MakeBatchInfo(77, 0, 2, geom)},
                     {BoxBatch({{Box(0.8f, 0.3f, 0.3f, 0.2f, 0.4f)}}),
                      MakeBatchInfo(77, 1, 2, geom)}}));

  ASSERT_EQ(merged_packets.size(), 1u);
  EXPECT_EQ(merged_packets[0].Timestamp(), Timestamp(77));
  const auto& merged = merged_packets[0].Get<std::vector<Detection>>();
  ASSERT_EQ(merged.size(), 1u);
  ASSERT_EQ(merged[0].score_size(), 1);
  EXPECT_NEAR(merged[0].score(0), 0.9f, 1e-5);
  const auto& bb = merged[0].location_data().relative_bounding_box();
  EXPECT_NEAR(bb.xmin(), 0.3f, 1e-4);
  EXPECT_NEAR(bb.width(), 0.2f, 1e-4);
}

// Pins the no-bypass design decision: even for a single-tile frame the global
// NMS still runs (the suppression calculator's bypass_single_tile is left
// default false and NUM_TILES is not connected, so bypassing is impossible).
TEST(TiledBoxMergeGraphTest, SingleTileStillRunsGlobalNms) {
  // ONE batch, ONE tile row holding TWO overlapping same-class boxes.
  MP_ASSERT_OK_AND_ASSIGN(
      std::vector<Packet> merged_packets,
      RunMergeGraph(BoxMergeGraphConfig(),
                    {{BoxBatch({{Box(0.9f, 0.3f, 0.3f, 0.2f, 0.2f),
                                 Box(0.8f, 0.31f, 0.31f, 0.2f, 0.2f)}}),
                      MakeBatchInfo(5, 0, 1,
                                    MakeGeom({FullFrameTile()}, 100, 100))}}));

  ASSERT_EQ(merged_packets.size(), 1u);
  EXPECT_EQ(merged_packets[0].Timestamp(), Timestamp(5));
  const auto& merged = merged_packets[0].Get<std::vector<Detection>>();
  ASSERT_EQ(merged.size(), 1u);  // global NMS ran despite a single tile
  ASSERT_EQ(merged[0].score_size(), 1);
  EXPECT_NEAR(merged[0].score(0), 0.9f, 1e-5);
}

// Discriminates the max_detections forwarding in TiledBoxMergeGraph: three
// DISJOINT boxes (IoU = 0) survive any iou_threshold; with the default
// max_detections (-1) all three would be kept, so only a forwarded
// max_detections: 2 can cap the output. NMS output is descending-score, so
// the cap keeps the highest-scoring boxes.
TEST(TiledBoxMergeGraphTest, MaxDetectionsCaps) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    output_stream: "merged"
    node {
      calculator: "mediapipe.tiled_detection.TiledBoxMergeGraph"
      input_stream: "DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
      output_stream: "DETECTIONS:merged"
      options {
        [mediapipe.TiledBoxMergeGraphOptions.ext] {
          iou_threshold: 0.5
          max_detections: 2
        }
      }
    }
  )pb");

  // ONE batch, ONE tile row holding THREE DISJOINT same-class boxes.
  MP_ASSERT_OK_AND_ASSIGN(
      std::vector<Packet> merged_packets,
      RunMergeGraph(config,
                    {{BoxBatch({{Box(0.9f, 0.1f, 0.1f, 0.1f, 0.1f),
                                 Box(0.8f, 0.4f, 0.4f, 0.1f, 0.1f),
                                 Box(0.7f, 0.7f, 0.7f, 0.1f, 0.1f)}}),
                      MakeBatchInfo(9, 0, 1,
                                    MakeGeom({FullFrameTile()}, 100, 100))}}));

  ASSERT_EQ(merged_packets.size(), 1u);
  EXPECT_EQ(merged_packets[0].Timestamp(), Timestamp(9));
  const auto& merged = merged_packets[0].Get<std::vector<Detection>>();
  // Default max_detections (-1) would keep all THREE disjoint boxes; the
  // forwarded max_detections: 2 keeps only the two top-scoring ones.
  ASSERT_EQ(merged.size(), 2u);
  ASSERT_EQ(merged[0].score_size(), 1);
  EXPECT_NEAR(merged[0].score(0), 0.9f, 1e-5);
  ASSERT_EQ(merged[1].score_size(), 1);
  EXPECT_NEAR(merged[1].score(0), 0.8f, 1e-5);
}

// max_detections: 0 is ambiguous (cap-to-zero vs. uncapped) and the internal
// ClipVectorSizeCalculator cannot represent it, so the subgraph must reject it
// at graph-init time with a message that names the offending option.
TEST(TiledBoxMergeGraphTest, MaxDetectionsZeroRejectedAtInit) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    output_stream: "merged"
    node {
      calculator: "mediapipe.tiled_detection.TiledBoxMergeGraph"
      input_stream: "DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
      output_stream: "DETECTIONS:merged"
      options {
        [mediapipe.TiledBoxMergeGraphOptions.ext] { max_detections: 0 }
      }
    }
  )pb");
  CalculatorGraph graph;
  absl::Status status = graph.Initialize(config);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(std::string(status.message()),
              testing::HasSubstr("TiledBoxMergeGraphOptions.max_detections"));
}

}  // namespace
}  // namespace mediapipe
