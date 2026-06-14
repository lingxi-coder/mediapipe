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
#include "absl/status/statusor.h"
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

}  // namespace
}  // namespace mediapipe
