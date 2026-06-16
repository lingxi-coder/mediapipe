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

#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/botsort_tracking_calculator.pb.h"
#include "mediapipe/calculators/tensor/detection_label_id_codec_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/graphs/tiled_detection/tiled_detection_graphs.pb.h"

namespace mediapipe {
namespace tiled_detection {

// A "mediapipe.tiled_detection.TiledTrackingGraph" produces class-carrying
// tracker-propagated detections from video frames + per-frame fresh
// detections, one packet per source frame:
// DETECTIONS -[ENCODE label_id->string]-> ObjectTrackingSubgraphCpu(VIDEO,
// DETECTIONS) -[DECODE string->label_id, drop unclassifiable]->
// DetectionsTickGate(TICK=DETECTIONS) -> TRACKER_DETECTIONS.
//
// Inputs:
//   IMAGE       - ImageFrame (the source video frame; fed to optical flow).
//   DETECTIONS  - std::vector<Detection> (merged-fresh, frame-normalized,
//                 carries label_id + score), one packet per source frame.
// Outputs:
//   TRACKER_DETECTIONS - std::vector<Detection> (carries label_id), EXACTLY
//                 one packet per source frame (empty when nothing tracked).
class TiledTrackingGraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    api2::builder::Graph graph;
    auto image = graph.In("IMAGE").Cast<ImageFrame>();
    auto fresh = graph.In("DETECTIONS").Cast<std::vector<Detection>>();
    const auto& opts = sc->Options<TiledTrackingGraphOptions>();

    if (opts.tracker_type() == TiledTrackingGraphOptions::BOTSORT) {
      // BoTSORT carries label_id/score natively, so no label-id codec is
      // needed: route the fresh detections straight through the tracker and a
      // tick gate keyed on the fresh stream (one packet per source frame).
      auto& bot = graph.AddNode("BotsortTrackingCalculator");
      auto& bo = bot.GetOptions<BotsortTrackingCalculatorOptions>();
      bo.set_track_high_threshold(opts.track_high_threshold());
      bo.set_track_low_threshold(opts.track_low_threshold());
      bo.set_new_track_threshold(opts.new_track_threshold());
      bo.set_track_buffer(opts.track_buffer());
      bo.set_match_threshold(opts.match_threshold());
      bo.set_enable_gmc(opts.enable_gmc());
      image >> bot.In("IMAGE");
      fresh >> bot.In("DETECTIONS");

      auto& gate = graph.AddNode("DetectionsTickGateCalculator");
      fresh >> gate.In("TICK");
      bot.Out("DETECTIONS") >> gate.In("DATA");
      gate.Out("DETECTIONS") >> graph.Out("TRACKER_DETECTIONS");
      return graph.GetConfig();
    }

    auto& encode = graph.AddNode("DetectionLabelIdCodecCalculator");
    encode.GetOptions<DetectionLabelIdCodecCalculatorOptions>().set_direction(
        DetectionLabelIdCodecCalculatorOptions::ENCODE);
    fresh >> encode.In("DETECTIONS");

    auto& tracker = graph.AddNode("ObjectTrackingSubgraphCpu");
    image >> tracker.In("VIDEO");
    encode.Out("DETECTIONS") >> tracker.In("DETECTIONS");

    auto& decode = graph.AddNode("DetectionLabelIdCodecCalculator");
    decode.GetOptions<DetectionLabelIdCodecCalculatorOptions>().set_direction(
        DetectionLabelIdCodecCalculatorOptions::DECODE);
    tracker.Out("DETECTIONS") >> decode.In("DETECTIONS");

    auto& gate = graph.AddNode("DetectionsTickGateCalculator");
    fresh >> gate.In("TICK");
    decode.Out("DETECTIONS") >> gate.In("DATA");

    gate.Out("DETECTIONS") >> graph.Out("TRACKER_DETECTIONS");
    return graph.GetConfig();
  }
};

// NOTE: keep the fully-qualified type name on a single line. The
// REGISTER_MEDIAPIPE_GRAPH macro stringifies its argument with `#name`, so a
// line break here would inject a stray space into the registered name and the
// graph would never be found by lookup.
// clang-format off
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tiled_detection::TiledTrackingGraph);  // NOLINT(whitespace/line_length)
// clang-format on

}  // namespace tiled_detection
}  // namespace mediapipe
