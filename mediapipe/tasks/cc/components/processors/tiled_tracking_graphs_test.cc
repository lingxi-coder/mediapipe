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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/video/botsort_tracking_calculator.pb.h"
#include "mediapipe/util/tiling_matrix_utils.h"
#include "mediapipe/framework/formats/tiling_types.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/components/processors/proto/tiled_detection_graph_options.pb.h"

namespace mediapipe {
namespace {

std::unique_ptr<ImageFrame> WhiteFrame(int w, int h) {
  auto f = std::make_unique<ImageFrame>(ImageFormat::SRGB, w, h);
  std::memset(f->MutablePixelData(), 255, f->Height() * f->WidthStep());
  return f;
}

Detection Box(float score, int label_id, float xmin, float ymin, float w,
              float h) {
  Detection d;
  d.add_score(score);
  d.add_label_id(label_id);
  auto* ld = d.mutable_location_data();
  ld->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* bb = ld->mutable_relative_bounding_box();
  bb->set_xmin(xmin);
  bb->set_ymin(ymin);
  bb->set_width(w);
  bb->set_height(h);
  return d;
}

OrientedDetection Obb(float cx, float cy, float w, float h, float score,
                      int label_id) {
  OrientedDetection d;
  d.set_cx(cx);
  d.set_cy(cy);
  d.set_width(w);
  d.set_height(h);
  d.set_rotation(0.0f);
  d.add_score(score);
  d.add_label_id(label_id);
  return d;
}

TileGeometry FullFrameTile2() {
  TileGeometry g;
  g.tile_index = 0;
  g.x_center = 0.5f;
  g.y_center = 0.5f;
  g.width = 1.0f;
  g.height = 1.0f;
  return g;
}

std::shared_ptr<TileBatchGeometry> Geom1(int fw, int fh) {
  auto geom = std::make_shared<TileBatchGeometry>();
  TileGeometry g = FullFrameTile2();
  TilePixelRoi roi;
  roi.x = 0;
  roi.y = 0;
  roi.width = fw;
  roi.height = fh;
  geom->tile_indices.push_back(g.tile_index);
  geom->tile_geometries.push_back(g);
  geom->effective_pixel_rois.push_back(roi);
  geom->tile_to_image_matrices.push_back(TileToImageMatrix(roi, fw, fh));
  return geom;
}

std::shared_ptr<TileBatchGeometry> GeomRoi(int x, int y, int width,
                                           int height, int fw, int fh,
                                           int tile_index) {
  auto geom = std::make_shared<TileBatchGeometry>();
  TileGeometry g;
  g.tile_index = tile_index;
  g.x_center = (x + width / 2.0f) / fw;
  g.y_center = (y + height / 2.0f) / fh;
  g.width = static_cast<float>(width) / fw;
  g.height = static_cast<float>(height) / fh;
  TilePixelRoi roi{x, y, width, height};
  geom->tile_indices.push_back(tile_index);
  geom->tile_geometries.push_back(g);
  geom->effective_pixel_rois.push_back(roi);
  geom->tile_to_image_matrices.push_back(TileToImageMatrix(roi, fw, fh));
  return geom;
}

TensorBatchInfo Info1(int64_t source_ts, std::shared_ptr<TileBatchGeometry> g) {
  TensorBatchInfo info;
  info.source_frame_timestamp = source_ts;
  info.batch_index = 0;
  info.total_batches = 1;
  info.valid_count = 1;
  info.tile_indices = g->tile_indices;
  info.geometry = std::move(g);
  return info;
}

// Test-only stand-in for inference + YOLO decode. It preserves the real
// front-end batch cadence and metadata while emitting one tile-local detection
// per valid row, so front-to-merge scheduler wiring can run without a model.
class CannedTileDetectionsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("BATCH_INFO").Set<TensorBatchInfo>();
    cc->Outputs()
        .Tag("DETECTIONS")
        .Set<std::vector<std::vector<Detection>>>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const TensorBatchInfo& info =
        cc->Inputs().Tag("BATCH_INFO").Get<TensorBatchInfo>();
    if (info.total_batches == 0) {
      return absl::OkStatus();
    }
    std::vector<std::vector<Detection>> rows(info.valid_count);
    for (auto& row : rows) {
      row.push_back(Box(0.9f, 7, 0.25f, 0.25f, 0.5f, 0.5f));
    }
    cc->Outputs()
        .Tag("DETECTIONS")
        .AddPacket(MakePacket<std::vector<std::vector<Detection>>>(
                       std::move(rows))
                       .At(cc->InputTimestamp()));
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(CannedTileDetectionsCalculator);

TEST(TiledBoxTrackMergeGraphTest,
     SchedulerRefreshFeedsBotsortMergeWithoutModelFixture) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    input_stream: "priors"
    output_stream: "merged"
    output_stream: "refresh"
    node {
      calculator: "mediapipe.tiled_detection.TiledDetectionStreamFrontGraph"
      input_stream: "IMAGE:image"
      input_stream: "PRIOR_DETECTIONS:priors"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:batch_info"
      output_stream: "REFRESH:refresh"
      options {
        [mediapipe.TiledDetectionFrontGraphOptions.ext] {
          tile_grid { cols: 2 }
          batch_capacity: 2
          input_height: 64
          input_width: 64
          input_channels: 3
        }
      }
    }
    node {
      calculator: "CannedTileDetectionsCalculator"
      input_stream: "BATCH_INFO:batch_info"
      output_stream: "DETECTIONS:tile_detections"
    }
    node {
      calculator: "mediapipe.tiled_detection.TiledBoxTrackMergeGraph"
      input_stream: "DETECTIONS:tile_detections"
      input_stream: "BATCH_INFO:batch_info"
      input_stream: "IMAGE:image"
      input_stream: "REFRESH:refresh"
      output_stream: "DETECTIONS:merged"
      node_options {
        [type.googleapis.com/mediapipe.TiledBoxMergeGraphOptions] {
          iou_threshold: 0.5
          class_agnostic: true
          tracking {
            tracker_type: BOTSORT
            track_high_threshold: 0.05
            track_low_threshold: 0.02
            new_track_threshold: 0.05
          }
        }
      }
    }
  )pb");

  std::vector<Packet> merged;
  std::vector<Packet> refresh;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("merged", [&](const Packet& p) {
    merged.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.ObserveOutputStream("refresh", [&](const Packet& p) {
    refresh.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < 3; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image", Adopt(WhiteFrame(64, 64).release()).At(Timestamp(i))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "priors", MakePacket<std::vector<Detection>>(
                      std::vector<Detection>{
                          Box(0.9f, 7, 0.3f, 0.3f, 0.2f, 0.2f)})
                      .At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(refresh.size(), 3u);
  ASSERT_EQ(merged.size(), 3u);
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(refresh[i].Get<bool>());
    EXPECT_EQ(merged[i].Timestamp(), Timestamp(i));
    EXPECT_FALSE(merged[i].Get<std::vector<Detection>>().empty());
  }
}

TEST(TiledBoxTrackMergeGraphTest,
     PacketPresenceFeedsBotsortMergeWithoutModelFixture) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    output_stream: "merged"
    node {
      calculator: "mediapipe.tiled_detection.TiledDetectionFrontGraph"
      input_stream: "IMAGE:image"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:batch_info"
      options {
        [mediapipe.TiledDetectionFrontGraphOptions.ext] {
          tile_grid { cols: 2 }
          batch_capacity: 2
          input_height: 64
          input_width: 64
          input_channels: 3
        }
      }
    }
    node {
      calculator: "CannedTileDetectionsCalculator"
      input_stream: "BATCH_INFO:batch_info"
      output_stream: "DETECTIONS:tile_detections"
    }
    node {
      calculator: "PacketPresenceCalculator"
      input_stream: "PACKET:image"
      output_stream: "PRESENCE:refresh"
    }
    node {
      calculator: "mediapipe.tiled_detection.TiledBoxTrackMergeGraph"
      input_stream: "DETECTIONS:tile_detections"
      input_stream: "BATCH_INFO:batch_info"
      input_stream: "IMAGE:image"
      input_stream: "REFRESH:refresh"
      output_stream: "DETECTIONS:merged"
      node_options {
        [type.googleapis.com/mediapipe.TiledBoxMergeGraphOptions] {
          iou_threshold: 0.5
          class_agnostic: true
          tracking {
            tracker_type: BOTSORT
            track_high_threshold: 0.05
            track_low_threshold: 0.02
            new_track_threshold: 0.05
          }
        }
      }
    }
  )pb");

  std::vector<Packet> merged;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("merged", [&](const Packet& p) {
    merged.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < 3; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image", Adopt(WhiteFrame(64, 64).release()).At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(merged.size(), 3u);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(merged[i].Timestamp(), Timestamp(i));
    EXPECT_FALSE(merged[i].Get<std::vector<Detection>>().empty());
  }
}

// TiledTrackingGraph: IMAGE + DETECTIONS -> TRACKER_DETECTIONS. Feeds 3 frames
// of fresh detections through the real optical-flow tracker and asserts the
// tick gate materializes EXACTLY one TRACKER_DETECTIONS packet per source
// frame (no stall — including the first frame, before any flow exists).
TEST(TiledTrackingGraphTest, EmitsOnePacketPerFrameNoStall) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    input_stream: "dets"
    output_stream: "tracker"
    node {
      calculator: "mediapipe.tiled_detection.TiledTrackingGraph"
      input_stream: "IMAGE:image"
      input_stream: "DETECTIONS:dets"
      output_stream: "TRACKER_DETECTIONS:tracker"
    }
  )pb");

  std::vector<Packet> out;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("tracker", [&](const Packet& p) {
    out.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < 3; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image", Adopt(WhiteFrame(64, 64).release()).At(Timestamp(i))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "dets",
        MakePacket<std::vector<Detection>>(
            std::vector<Detection>{Box(0.9f, 8, 0.3f, 0.3f, 0.2f, 0.2f)})
            .At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(out.size(), 3u);  // one TRACKER_DETECTIONS packet per source frame
  for (const Packet& p : out) {
    for (const Detection& d : p.Get<std::vector<Detection>>()) {
      EXPECT_GT(d.label_id_size(), 0);
      EXPECT_EQ(d.label_size(), 0);
    }
  }
}

// TiledTrackingGraph with tracker_type: BOTSORT. Routes the fresh detections
// through BotsortTrackingCalculator + a tick gate (no label-id codec). Feeds 3
// frames (same object drifting a couple normalized units so BoTSORT can confirm
// the track) and asserts EXACTLY one TRACKER_DETECTIONS packet per source frame,
// then validates the "no codec needed for BOTSORT" claim on the last packet: the
// tracked detection still carries the original label_id and a score natively.
TEST(TiledTrackingGraphTest, BotsortEmitsOnePacketPerFrame) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    input_stream: "dets"
    input_stream: "refresh"
    input_stream: "rois"
    output_stream: "tracker"
    node {
      calculator: "mediapipe.tiled_detection.TiledTrackingGraph"
      input_stream: "IMAGE:image"
      input_stream: "DETECTIONS:dets"
      input_stream: "REFRESH:refresh"
      input_stream: "OBSERVED_ROIS:rois"
      output_stream: "TRACKER_DETECTIONS:tracker"
      node_options {
        [type.googleapis.com/mediapipe.TiledTrackingGraphOptions] {
          tracker_type: BOTSORT
          track_low_threshold: 0.02
          track_high_threshold: 0.05
        }
      }
    }
  )pb");

  std::vector<Packet> out;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("tracker", [&](const Packet& p) {
    out.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  constexpr int kLabelId = 5;  // distinct, to prove it survives the tracker
  for (int i = 0; i < 3; ++i) {
    const float drift = 0.01f * i;  // same object drifting a couple norm units
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image", Adopt(WhiteFrame(200, 200).release()).At(Timestamp(i))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "dets",
        MakePacket<std::vector<Detection>>(std::vector<Detection>{
            Box(0.9f, kLabelId, 0.3f + drift, 0.3f + drift, 0.2f, 0.2f)})
            .At(Timestamp(i))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "refresh", MakePacket<bool>(true).At(Timestamp(i))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "rois", MakePacket<std::vector<TilePixelRoi>>(
                    std::vector<TilePixelRoi>{{0, 0, 200, 200}})
                    .At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(out.size(), 3u);  // one TRACKER_DETECTIONS packet per source frame
  // BoTSORT may need a frame to confirm a track; assert on the last output. The
  // tracked detection must carry label_id + score natively (no codec path), and
  // the label_id must equal the one fed in.
  const auto& last = out.back().Get<std::vector<Detection>>();
  ASSERT_FALSE(last.empty());
  EXPECT_GT(last[0].label_id_size(), 0);
  EXPECT_GT(last[0].score_size(), 0);
  EXPECT_EQ(last[0].label_id(0), kLabelId);
}

// TiledBoxTrackMergeGraph: per-batch DETECTIONS + BATCH_INFO + IMAGE -> merged
// DETECTIONS. On a white frame the tracker emits nothing, so the merged output
// is exactly the fresh detection (tracker gap-fill drops nothing), at the
// source timestamp — proving the merge+suppression+IMAGE wiring runs without
// stalling.
TEST(TiledBoxTrackMergeGraphTest, FreshDetectionFlowsThroughWithTracker) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    input_stream: "image"
    output_stream: "merged"
    node {
      calculator: "mediapipe.tiled_detection.TiledBoxTrackMergeGraph"
      input_stream: "DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
      input_stream: "IMAGE:image"
      output_stream: "DETECTIONS:merged"
      options {
        [mediapipe.TiledBoxMergeGraphOptions.ext] {
          iou_threshold: 0.5
          class_agnostic: true
        }
      }
    }
  )pb");

  std::vector<Packet> merged;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("merged", [&](const Packet& p) {
    merged.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));

  auto geom = Geom1(100, 100);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets",
      MakePacket<std::vector<std::vector<Detection>>>(
          std::vector<std::vector<Detection>>{
              {Box(0.9f, 8, 0.3f, 0.3f, 0.2f, 0.2f)}})
          .At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info", MakePacket<TensorBatchInfo>(Info1(50, geom)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image", Adopt(WhiteFrame(100, 100).release()).At(Timestamp(50))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(merged.size(), 1u);
  EXPECT_EQ(merged[0].Timestamp(), Timestamp(50));
  const auto& dets = merged[0].Get<std::vector<Detection>>();
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_NEAR(dets[0].score(0), 0.9f, 1e-5);
}

// A five-tile frame with a fixed inference capacity of two arrives as three
// synthetic-time batches. The merge must wait for all three, emit one
// source-time packet, and give BoTSORT the union of the exact effective ROIs.
TEST(TiledBoxTrackMergeGraphTest, BotsortWaitsForAllBatchesAndUsesObservedRois) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    input_stream: "image"
    input_stream: "refresh"
    output_stream: "merged"
    node {
      calculator: "mediapipe.tiled_detection.TiledBoxTrackMergeGraph"
      input_stream: "DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
      input_stream: "IMAGE:image"
      input_stream: "REFRESH:refresh"
      output_stream: "DETECTIONS:merged"
      node_options {
        [type.googleapis.com/mediapipe.TiledBoxMergeGraphOptions] {
          iou_threshold: 0.5
          class_agnostic: true
          tracking {
            tracker_type: BOTSORT
            track_high_threshold: 0.05
            track_low_threshold: 0.02
            new_track_threshold: 0.05
          }
        }
      }
    }
  )pb");

  std::vector<Packet> merged;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("merged", [&](const Packet& p) {
    merged.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));

  constexpr int kWidth = 100;
  constexpr int kHeight = 100;
  constexpr int64_t kSourceTimestamp = 100;
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image", Adopt(WhiteFrame(kWidth, kHeight).release())
                    .At(Timestamp(kSourceTimestamp))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "refresh", MakePacket<bool>(true).At(Timestamp(kSourceTimestamp))));

  // T=5, B=2: rows are {0,1}, {2,3}, {4}. The first two batches cannot
  // complete the source frame and therefore cannot update the tracker.
  for (int batch_index = 0; batch_index < 3; ++batch_index) {
    const int valid_rows = batch_index == 2 ? 1 : 2;
    std::vector<std::vector<Detection>> rows;
    auto geometry = std::make_shared<TileBatchGeometry>();
    for (int row = 0; row < valid_rows; ++row) {
      const int tile_index = batch_index * 2 + row;
      const int x = tile_index * 20;
      const auto one = GeomRoi(x, 0, 20, 100, kWidth, kHeight, tile_index);
      geometry->tile_indices.push_back(tile_index);
      geometry->tile_geometries.push_back(one->tile_geometries[0]);
      geometry->effective_pixel_rois.push_back(one->effective_pixel_rois[0]);
      geometry->tile_to_image_matrices.push_back(
          one->tile_to_image_matrices[0]);
      rows.push_back({Box(0.9f, 4, 0.25f, 0.25f, 0.5f, 0.5f)});
    }
    TensorBatchInfo info;
    info.source_frame_timestamp = kSourceTimestamp;
    info.batch_index = batch_index;
    info.total_batches = 3;
    info.batch_capacity = 2;
    info.batch_size = 2;
    info.valid_count = valid_rows;
    info.tile_indices = geometry->tile_indices;
    info.geometry = std::move(geometry);

    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "dets", MakePacket<std::vector<std::vector<Detection>>>(
                          std::move(rows))
                    .At(Timestamp(batch_index))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "info", MakePacket<TensorBatchInfo>(std::move(info))
                    .At(Timestamp(batch_index))));
    if (batch_index < 2) {
      EXPECT_TRUE(merged.empty());
    }
  }

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  ASSERT_EQ(merged.size(), 1u);
  EXPECT_EQ(merged[0].Timestamp(), Timestamp(kSourceTimestamp));
  EXPECT_EQ(merged[0].Get<std::vector<Detection>>().size(), 5u);
}

TEST(TiledObbTrackMergeGraphTest, ForwardsNominalFrameRateToTracker) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    input_stream: "image"
    output_stream: "merged"
    node {
      calculator: "mediapipe.tiled_detection.TiledObbTrackMergeGraph"
      input_stream: "ORIENTED_DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
      input_stream: "IMAGE:image"
      output_stream: "ORIENTED_DETECTIONS:merged"
      options {
        [mediapipe.TiledObbMergeGraphOptions.ext] {
          tracking {
            tracker_type: BOTSORT
            nominal_frame_rate: 60
          }
        }
      }
    }
  )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  int tracker_nodes = 0;
  for (const auto& node : graph.Config().node()) {
    if (node.calculator() != "OrientedBotsortTrackingCalculator") continue;
    ++tracker_nodes;
    EXPECT_EQ(node.options()
                  .GetExtension(BotsortTrackingCalculatorOptions::ext)
                  .nominal_frame_rate(),
              60);
  }
  EXPECT_EQ(tracker_nodes, 1);
}

// TiledObbTrackMergeGraph: per-batch ORIENTED_DETECTIONS + BATCH_INFO + IMAGE ->
// merged ORIENTED_DETECTIONS with BoTSORT track ids. Feeds 3 frames of a single
// full-frame tile carrying one oriented detection (drifting a couple normalized
// units so BoTSORT can confirm the track) and asserts EXACTLY one merged packet
// per source frame at the source timestamp, and that the last frame's output
// carries a track_id (BoTSORT only emits ids for confirmed tracks).
TEST(TiledObbTrackMergeGraphTest, EmitsTrackIdAcrossFrames) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    input_stream: "image"
    output_stream: "merged"
    node {
      calculator: "mediapipe.tiled_detection.TiledObbTrackMergeGraph"
      input_stream: "ORIENTED_DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
      input_stream: "IMAGE:image"
      output_stream: "ORIENTED_DETECTIONS:merged"
      node_options {
        [type.googleapis.com/mediapipe.TiledObbMergeGraphOptions] {
          iou_threshold: 0.5
          class_agnostic: true
          tracking {
            tracker_type: BOTSORT
            track_high_threshold: 0.05
            track_low_threshold: 0.02
            new_track_threshold: 0.05
          }
        }
      }
    }
  )pb");

  std::vector<Packet> merged;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("merged", [&](const Packet& p) {
    merged.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));

  constexpr int kLabelId = 3;
  for (int i = 0; i < 3; ++i) {
    const float drift = 0.01f * i;  // same object drifting a couple norm units
    auto geom = Geom1(200, 200);
    // ONE batch, ONE full-frame tile row holding one oriented detection.
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "dets",
        MakePacket<std::vector<std::vector<OrientedDetection>>>(
            std::vector<std::vector<OrientedDetection>>{
                {Obb(0.5f + drift, 0.5f + drift, 0.2f, 0.2f, 0.9f, kLabelId)}})
            .At(Timestamp(i))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "info", MakePacket<TensorBatchInfo>(Info1(i, geom)).At(Timestamp(i))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image", Adopt(WhiteFrame(200, 200).release()).At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(merged.size(), 3u);  // one merged packet per source frame
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(merged[i].Timestamp(), Timestamp(i));
    // Geometry is unchanged by the tracker (ID-only): one box per frame.
    ASSERT_EQ(merged[i].Get<std::vector<OrientedDetection>>().size(), 1u);
  }
  // BoTSORT confirms a track after a couple frames; assert on the last output.
  const auto& last = merged.back().Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(last.size(), 1u);
  EXPECT_TRUE(last[0].has_track_id());
  EXPECT_FALSE(last[0].track_id().empty());
  // The fresh detection geometry/label is preserved (ID-only association).
  ASSERT_EQ(last[0].label_id_size(), 1);
  EXPECT_EQ(last[0].label_id(0), kLabelId);
  EXPECT_NEAR(last[0].width(), 0.2f, 1e-4);

  // Track-id STABILITY: the same object must keep its id across consecutive
  // frames. Collect the ids on the last two output frames and assert at least
  // one id appears on BOTH (matches the calculator unit test's stability
  // property). With a 3-frame feed BoTSORT has confirmed the track by frame 2,
  // so its id is present on both frames 1 and 2.
  ASSERT_GE(merged.size(), 2u);
  auto track_ids = [](const Packet& p) {
    std::vector<std::string> ids;
    for (const OrientedDetection& d :
         p.Get<std::vector<OrientedDetection>>()) {
      if (d.has_track_id() && !d.track_id().empty()) {
        ids.push_back(d.track_id());
      }
    }
    return ids;
  };
  const std::vector<std::string> prev_ids =
      track_ids(merged[merged.size() - 2]);
  const std::vector<std::string> last_ids = track_ids(merged.back());
  ASSERT_FALSE(prev_ids.empty());
  ASSERT_FALSE(last_ids.empty());
  bool shared_id = false;
  for (const std::string& id : last_ids) {
    if (std::find(prev_ids.begin(), prev_ids.end(), id) != prev_ids.end()) {
      shared_id = true;
      break;
    }
  }
  EXPECT_TRUE(shared_id)
      << "expected a track id to persist across the last two frames";
}

}  // namespace
}  // namespace mediapipe
