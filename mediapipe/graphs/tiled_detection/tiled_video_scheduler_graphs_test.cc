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
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
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

}  // namespace
}  // namespace mediapipe
