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

}  // namespace
}  // namespace mediapipe
