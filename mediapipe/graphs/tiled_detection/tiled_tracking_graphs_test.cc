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
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/graphs/tiled_detection/tiled_detection_graphs.pb.h"

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
    output_stream: "tracker"
    node {
      calculator: "mediapipe.tiled_detection.TiledTrackingGraph"
      input_stream: "IMAGE:image"
      input_stream: "DETECTIONS:dets"
      output_stream: "TRACKER_DETECTIONS:tracker"
      node_options {
        [type.googleapis.com/mediapipe.TiledTrackingGraphOptions] {
          tracker_type: BOTSORT
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
}

}  // namespace
}  // namespace mediapipe
