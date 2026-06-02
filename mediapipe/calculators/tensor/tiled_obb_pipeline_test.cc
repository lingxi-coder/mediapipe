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
#include <array>
#include <memory>
#include <vector>

#include "mediapipe/calculators/tensor/tiling_matrix_utils.h"
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
    output_stream: "out"
    node {
      calculator: "MergeTileDetectionsAccumulatorCalculator"
      input_stream: "ORIENTED_DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
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
  // On a 100x100 frame:
  //   left ROI:  {x=0,  y=0, width=60, height=100}
  //   right ROI: {x=40, y=0, width=60, height=100}
  TileGeometry l;
  l.tile_index = 0;
  l.x_center = .3f;
  l.y_center = .5f;
  l.width = .6f;
  l.height = 1.0f;
  TileGeometry r;
  r.tile_index = 1;
  r.x_center = .7f;
  r.y_center = .5f;
  r.width = .6f;
  r.height = 1.0f;

  TilePixelRoi left_roi{0, 0, 60, 100};
  TilePixelRoi right_roi{40, 0, 60, 100};

  auto geom = std::make_shared<TileBatchGeometry>();
  geom->tile_indices = {0, 1};
  geom->tile_geometries = {l, r};
  geom->effective_pixel_rois = {left_roi, right_roi};
  geom->tile_to_image_matrices = {TileToImageMatrix(left_roi, 100, 100),
                                   TileToImageMatrix(right_roi, 100, 100)};

  // Object truly at frame (0.5,0.5,0.2,0.2). Tile-local coords:
  //   left:  cx=(0.5-0.0)/0.6=0.83333, w=0.2/0.6=0.33333
  //   right: cx=(0.5-0.4)/0.6=0.16667, w=0.33333
  auto obb = [](float cx, float cy, float w, float h, float s) {
    OrientedDetection d;
    d.set_cx(cx);
    d.set_cy(cy);
    d.set_width(w);
    d.set_height(h);
    d.set_rotation(0);
    d.add_score(s);
    d.add_label_id(0);
    return d;
  };
  auto batch = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch->push_back({obb(0.83333f, 0.5f, 0.33333f, 0.2f, 0.9f)});  // left
  batch->push_back({obb(0.16667f, 0.5f, 0.33333f, 0.2f, 0.8f)});  // right
  TensorBatchInfo info;
  info.source_frame_timestamp = 0;
  info.total_batches = 1;
  info.batch_capacity = 2;
  info.valid_count = 2;
  info.tile_indices = {0, 1};
  info.geometry = geom;

  std::vector<Packet> out_packets;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("out", [&](const Packet& p) {
    out_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
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

// ---------------------------------------------------------------------------
// Shared helper: build a TileBatchGeometry from parallel tile/ROI vectors.
// ---------------------------------------------------------------------------
std::shared_ptr<TileBatchGeometry> MakeGeom(
    const std::vector<TileGeometry>& tiles,
    const std::vector<TilePixelRoi>& rois, int fw, int fh) {
  auto g = std::make_shared<TileBatchGeometry>();
  for (size_t i = 0; i < tiles.size(); ++i) {
    g->tile_indices.push_back(tiles[i].tile_index);
    g->tile_geometries.push_back(tiles[i]);
    g->effective_pixel_rois.push_back(rois[i]);
    const auto m = TileToImageMatrix(rois[i], fw, fh);
    g->tile_to_image_matrices.push_back(m);
    g->image_to_tile_matrices.push_back(InvertAffine2d(m));
  }
  return g;
}

// ---------------------------------------------------------------------------
// Helper: build a simple OrientedDetection.
// ---------------------------------------------------------------------------
OrientedDetection MakeObb(float cx, float cy, float w, float h, float s) {
  OrientedDetection d;
  d.set_cx(cx);
  d.set_cy(cy);
  d.set_width(w);
  d.set_height(h);
  d.set_rotation(0);
  d.add_score(s);
  d.add_label_id(0);
  return d;
}

// ---------------------------------------------------------------------------
// Test: two batches for ONE source frame accumulate and emit ONCE at the
// source timestamp Timestamp(0), even though they arrived at Timestamp(0)
// and Timestamp(1) respectively (the synthetic batch timestamps).
//
// This proves the EndLoop / multi-batch merge path:
//   Batch 0 (synthetic ts=0): left tile  x[0, 0.5], detection projects to
//       frame cx≈0.25.
//   Batch 1 (synthetic ts=1): right tile x[0.5, 1.0], detection projects to
//       frame cx≈0.75.
// The two projected boxes are far apart so NMS keeps both.
// The accumulator emits at Timestamp(source_frame_timestamp=0), which is
// BEFORE the consuming timestamp of Timestamp(1) — legal under
// TimestampChange::Arbitrary().
// ---------------------------------------------------------------------------
TEST(TiledObbPipelineTest, MultiBatchAccumulatesToSingleSourceTimestamp) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    output_stream: "out"
    node {
      calculator: "MergeTileDetectionsAccumulatorCalculator"
      input_stream: "ORIENTED_DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
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

  // Batch 0: left tile covers x[0, 0.5] on a 100x100 frame.
  TileGeometry left_tile;
  left_tile.tile_index = 0;
  left_tile.x_center = 0.25f;
  left_tile.y_center = 0.5f;
  left_tile.width = 0.5f;
  left_tile.height = 1.0f;
  TilePixelRoi left_roi{0, 0, 50, 100};
  auto geom0 = MakeGeom({left_tile}, {left_roi}, 100, 100);

  // Batch 1: right tile covers x[0.5, 1.0] on a 100x100 frame.
  TileGeometry right_tile;
  right_tile.tile_index = 1;
  right_tile.x_center = 0.75f;
  right_tile.y_center = 0.5f;
  right_tile.width = 0.5f;
  right_tile.height = 1.0f;
  TilePixelRoi right_roi{50, 0, 50, 100};
  auto geom1 = MakeGeom({right_tile}, {right_roi}, 100, 100);

  // Tile-local detection at (0.5, 0.5, 0.4, 0.4) projects to:
  //   left tile:  frame cx = 0 + 0.5*0.5 = 0.25, width = 0.4*0.5 = 0.2
  //   right tile: frame cx = 0.5 + 0.5*0.5 = 0.75, width = 0.4*0.5 = 0.2
  auto batch0 = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch0->push_back({MakeObb(0.5f, 0.5f, 0.4f, 0.4f, 0.9f)});

  auto batch1 = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch1->push_back({MakeObb(0.5f, 0.5f, 0.4f, 0.4f, 0.8f)});

  // Both batches share the same source_frame_timestamp=0, total_batches=2.
  TensorBatchInfo info0;
  info0.source_frame_timestamp = 0;
  info0.batch_timestamp = 0;
  info0.batch_index = 0;
  info0.total_batches = 2;
  info0.batch_capacity = 1;
  info0.batch_size = 1;
  info0.valid_count = 1;
  info0.tile_indices = {0};
  info0.geometry = geom0;

  TensorBatchInfo info1;
  info1.source_frame_timestamp = 0;
  info1.batch_timestamp = 1;
  info1.batch_index = 1;
  info1.total_batches = 2;
  info1.batch_capacity = 1;
  info1.batch_size = 1;
  info1.valid_count = 1;
  info1.tile_indices = {1};
  info1.geometry = geom1;

  std::vector<Packet> out_packets;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("out", [&](const Packet& p) {
    out_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));

  // Feed batch 0 at synthetic Timestamp(0).
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info", MakePacket<TensorBatchInfo>(info0).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets", Adopt(batch0.release()).At(Timestamp(0))));

  // Feed batch 1 at synthetic Timestamp(1).
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info", MakePacket<TensorBatchInfo>(info1).At(Timestamp(1))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets", Adopt(batch1.release()).At(Timestamp(1))));

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  // Must emit EXACTLY ONE packet at the source frame timestamp Timestamp(0),
  // not one per synthetic batch timestamp.
  ASSERT_EQ(out_packets.size(), 1);
  EXPECT_EQ(out_packets[0].Timestamp(), Timestamp(0));

  const auto& out = out_packets[0].Get<std::vector<OrientedDetection>>();
  // Both boxes project to different frame locations so NMS keeps both.
  ASSERT_EQ(out.size(), 2);
  // Collect cx values and check the expected pair regardless of ordering.
  std::vector<float> cxs;
  for (const auto& d : out) cxs.push_back(d.cx());
  std::sort(cxs.begin(), cxs.end());
  EXPECT_NEAR(cxs[0], 0.25f, 1e-3);
  EXPECT_NEAR(cxs[1], 0.75f, 1e-3);
}

// ---------------------------------------------------------------------------
// Test: a single BATCH_INFO with total_batches=0 / valid_count=0 (empty
// frame) paired with an empty detections vector emits exactly one packet at
// Timestamp(0) containing zero detections, and the NMS output is also one
// empty packet at Timestamp(0).
//
// NOTE: In a fully-wired graph an empty frame produces a BATCH_INFO with no
// paired TENSORS packet; whether the synchronized downstream is reached
// depends on graph-level timestamp-bound propagation from the inference
// branch. This unit test feeds the synchronized empty detections packet
// directly to verify merge's own logic. The end-to-end graph-sync behavior
// for empty frames must be validated when the full inference graph is
// assembled (future work).
// ---------------------------------------------------------------------------
TEST(TiledObbPipelineTest, EmptyFrameEmitsEmptyResultAtSourceTimestamp) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    output_stream: "out"
    node {
      calculator: "MergeTileDetectionsAccumulatorCalculator"
      input_stream: "ORIENTED_DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
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

  // Empty frame: no tiles, no detections.
  TensorBatchInfo info;
  info.source_frame_timestamp = 0;
  info.batch_timestamp = 0;
  info.batch_index = 0;
  info.total_batches = 0;
  info.batch_capacity = 0;
  info.batch_size = 0;
  info.valid_count = 0;
  info.geometry = nullptr;

  auto empty_batch =
      std::make_unique<std::vector<std::vector<OrientedDetection>>>();

  std::vector<Packet> out_packets;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("out", [&](const Packet& p) {
    out_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info", MakePacket<TensorBatchInfo>(info).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets", Adopt(empty_batch.release()).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  // Merge emits once at the source timestamp; NMS passes it through.
  ASSERT_EQ(out_packets.size(), 1);
  EXPECT_EQ(out_packets[0].Timestamp(), Timestamp(0));
  const auto& out = out_packets[0].Get<std::vector<OrientedDetection>>();
  EXPECT_EQ(out.size(), 0);
}

// ---------------------------------------------------------------------------
// Test: TWO distinct source frames flow through merge back-to-back and the
// merged output stream stays strictly monotonic. Frame A (source ts 0) emits
// 2 batches at synthetic ts 0 and 1; frame B (source ts 100) emits 1 batch at
// synthetic ts 2. Merge must emit A's result at Timestamp(0) THEN B's at
// Timestamp(100), in order. This locks in the cross-frame ordering that the
// single-frame multi-batch test cannot exercise: the producer serializes all
// of A's batches before any of B's, so A completes (and emits at its earlier
// source ts) before B does.
// ---------------------------------------------------------------------------
TEST(TiledObbPipelineTest, MultipleFramesEmitInSourceTimestampOrder) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    output_stream: "out"
    node {
      calculator: "MergeTileDetectionsAccumulatorCalculator"
      input_stream: "ORIENTED_DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
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

  // One full-frame tile on a 100x100 frame, reused for every batch.
  TileGeometry tile;
  tile.tile_index = 0;
  tile.x_center = 0.5f;
  tile.y_center = 0.5f;
  tile.width = 1.0f;
  tile.height = 1.0f;
  TilePixelRoi roi{0, 0, 100, 100};

  // Builds a single-row BATCH_INFO for (source frame ts, batch index, total).
  auto make_info = [&](int64_t src_ts, int batch_index, int total_batches) {
    TensorBatchInfo info;
    info.source_frame_timestamp = src_ts;
    info.batch_index = batch_index;
    info.total_batches = total_batches;
    info.batch_capacity = 1;
    info.batch_size = 1;
    info.valid_count = 1;
    info.tile_indices = {0};
    info.geometry = MakeGeom({tile}, {roi}, 100, 100);
    return info;
  };
  auto make_batch = [](float cx) {
    auto b = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
    b->push_back({MakeObb(cx, 0.5f, 0.2f, 0.2f, 0.9f)});
    return b;
  };

  std::vector<Packet> out_packets;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("out", [&](const Packet& p) {
    out_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));

  // Frame A (source ts 0): two batches at synthetic ts 0 and 1.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info", MakePacket<TensorBatchInfo>(make_info(0, 0, 2)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets", Adopt(make_batch(0.3f).release()).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info", MakePacket<TensorBatchInfo>(make_info(0, 1, 2)).At(Timestamp(1))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets", Adopt(make_batch(0.7f).release()).At(Timestamp(1))));
  // Frame B (source ts 100): one batch at synthetic ts 2.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info",
      MakePacket<TensorBatchInfo>(make_info(100, 0, 1)).At(Timestamp(2))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets", Adopt(make_batch(0.5f).release()).At(Timestamp(2))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  // Two frames -> two merged outputs, strictly increasing source timestamps.
  ASSERT_EQ(out_packets.size(), 2);
  EXPECT_EQ(out_packets[0].Timestamp(), Timestamp(0));
  EXPECT_EQ(out_packets[1].Timestamp(), Timestamp(100));
  // Frame A accumulated both batches (cx 0.3 and 0.7, far apart -> NMS keeps 2).
  EXPECT_EQ(out_packets[0].Get<std::vector<OrientedDetection>>().size(), 2);
  EXPECT_EQ(out_packets[1].Get<std::vector<OrientedDetection>>().size(), 1);
}

}  // namespace
}  // namespace mediapipe
