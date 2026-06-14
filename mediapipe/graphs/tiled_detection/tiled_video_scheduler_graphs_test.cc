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
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/tracking/tracking.h"

namespace mediapipe {
namespace {

// A frame with a high-contrast moving square on a textured background, so the
// optical-flow tracker finds features to track (a flat/white frame yields no
// flow). `shift` translates the square to create real motion between frames.
std::unique_ptr<ImageFrame> TexturedFrame(int w, int h, int shift) {
  auto f = std::make_unique<ImageFrame>(ImageFormat::SRGB, w, h);
  uint8_t* p = f->MutablePixelData();
  const int stride = f->WidthStep();
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const bool square = (x > 10 + shift && x < 40 + shift && y > 10 && y < 40);
      const uint8_t v = square ? 0 : (((x / 4 + y / 4) % 2) ? 220 : 60);
      uint8_t* px = p + y * stride + x * 3;
      px[0] = px[1] = px[2] = v;
    }
  }
  return f;
}

TEST(OpticalFlowTrackingGraphTest, ProducesTrackingDataAndTerminates) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    output_stream: "tracking"
    node {
      calculator: "OpticalFlowTrackingGraph"
      input_stream: "IMAGE:image"
      output_stream: "TRACKING:tracking"
    }
  )pb");

  std::vector<Packet> tracking_packets;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("tracking", [&](const Packet& p) {
    tracking_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < 4; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image", Adopt(TexturedFrame(64, 64, /*shift=*/2 * i).release())
                     .At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_GE(tracking_packets.size(), 1u);
  for (const Packet& p : tracking_packets) {
    EXPECT_NO_THROW((void)p.Get<TrackingData>());
  }
}

// One full-frame detection (frame-normalized) used as a non-empty prior so the
// scheduler does not auto-DETECT via the priors-empty rule.
std::vector<Detection> OnePrior() {
  Detection d;
  d.add_score(0.9f);
  d.add_label_id(0);
  d.mutable_location_data()->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* bb = d.mutable_location_data()->mutable_relative_bounding_box();
  bb->set_xmin(0.3f);
  bb->set_ymin(0.3f);
  bb->set_width(0.2f);
  bb->set_height(0.2f);
  return {d};
}

// The stream front drives the scheduler through its OWN real optical-flow pass.
// Asserts the graph terminates (no stall on real buffered MotionAnalysis flow —
// the [P1] no-stall arbiter) and DETECTs on moving frames (>=1 frame with
// inference batches). Deterministic SKIP-mechanics (empty TILES ->
// total_batches==0 -> empty merged) are already proven by
// video_tile_scheduler_pipeline_test.SkipFramesEmitEmptyMergedResults.
TEST(TiledDetectionStreamFrontGraphTest, DetectsOnMotionAndTerminates) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    input_stream: "priors"
    output_stream: "batch_info"
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
  )pb");

  std::vector<int> batches_per_frame;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("batch_info", [&](const Packet& p) {
    batches_per_frame.push_back(p.Get<TensorBatchInfo>().total_batches);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < 4; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image", Adopt(TexturedFrame(64, 64, /*shift=*/3 * i).release())
                     .At(Timestamp(i))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "priors",
        Adopt(new std::vector<Detection>(OnePrior())).At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_GE(batches_per_frame.size(), 1u);
  int detect_frames = 0;
  for (int b : batches_per_frame) {
    if (b > 0) ++detect_frames;
  }
  EXPECT_GE(detect_frames, 1)
      << "moving frames with non-empty priors should DETECT at least once";
}

}  // namespace
}  // namespace mediapipe
