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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/core/clip_vector_size_calculator.pb.h"
#include "mediapipe/calculators/util/tiled_frame_suppression_calculator.pb.h"
#include "mediapipe/framework/formats/tiling_types.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/tasks/cc/components/processors/proto/tiled_detection_graph_options.pb.h"

namespace mediapipe {
namespace tiled_detection {

// A "mediapipe.tiled_detection.TiledBoxTrackMergeGraph" is the stream-mode
// sibling of TiledBoxMergeGraph: it merges per-batch tile-local axis-aligned
// detections back into source-frame space, then FUSES them with optical-flow
// tracker-propagated detections via a single global NMS (fresh-wins):
//   DETECTIONS + BATCH_INFO -> MergeTileBoxDetectionsAccumulator -> merged_fresh
//   merged_fresh + IMAGE -> TiledTrackingGraph -> tracker_dets
//   TiledFrameSuppression(DETECTIONS=merged_fresh,
//                         TRACKER_DETECTIONS=tracker_dets,
//                         tracker_is_gap_fill_only=true)
//   [-> ClipDetectionVectorSize if max_detections >= 1].
//
// Inputs:
//   DETECTIONS - std::vector<std::vector<Detection>> (per-batch tile-local).
//   BATCH_INFO - TensorBatchInfo.
//   IMAGE      - ImageFrame (source video frame; drives the tracker's flow).
// Outputs:
//   DETECTIONS - std::vector<Detection> (merged + tracker-fused, one packet
//                per source frame, frame-normalized).
class TiledBoxTrackMergeGraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    const auto& options = sc->Options<TiledBoxMergeGraphOptions>();
    if (options.max_detections() == 0) {
      return absl::InvalidArgumentError(
          "TiledBoxMergeGraphOptions.max_detections must be -1 (uncapped) or "
          ">= 1; got 0.");
    }
    api2::builder::Graph graph;

    auto& merge = graph.AddNode("MergeTileBoxDetectionsAccumulatorCalculator");
    graph.In("DETECTIONS") >> merge.In("DETECTIONS");
    graph.In("BATCH_INFO") >> merge.In("BATCH_INFO");
    auto merged_fresh = merge.Out("DETECTIONS").Cast<std::vector<Detection>>();
    auto observed_rois =
        merge.Out("OBSERVED_ROIS").Cast<std::vector<TilePixelRoi>>();

    auto& track = graph.AddNode("mediapipe.tiled_detection.TiledTrackingGraph");
    if (options.has_tracking()) {
      track.GetOptions<TiledTrackingGraphOptions>().CopyFrom(options.tracking());
    }
    graph.In("IMAGE") >> track.In("IMAGE");
    merged_fresh >> track.In("DETECTIONS");
    if (options.tracking().tracker_type() ==
        TiledTrackingGraphOptions::BOTSORT) {
      graph.In("REFRESH") >> track.In("REFRESH");
      observed_rois >> track.In("OBSERVED_ROIS");
    }

    auto& nms = graph.AddNode("TiledFrameSuppressionCalculator");
    auto& no = nms.GetOptions<TiledFrameSuppressionCalculatorOptions>();
    no.set_iou_threshold(options.iou_threshold());
    no.set_class_agnostic(options.class_agnostic());
    no.set_tracker_is_gap_fill_only(true);
    merged_fresh >> nms.In("DETECTIONS");
    track.Out("TRACKER_DETECTIONS") >> nms.In("TRACKER_DETECTIONS");

    if (options.max_detections() >= 1) {
      auto& clip = graph.AddNode("ClipDetectionVectorSizeCalculator");
      clip.GetOptions<ClipVectorSizeCalculatorOptions>().set_max_vec_size(
          options.max_detections());
      nms.Out("DETECTIONS") >> clip.In("");
      clip.Out("") >> graph.Out("DETECTIONS");
    } else {
      nms.Out("DETECTIONS") >> graph.Out("DETECTIONS");
    }
    return graph.GetConfig();
  }
};

// NOTE: keep the fully-qualified type name on a single line. The
// REGISTER_MEDIAPIPE_GRAPH macro stringifies its argument with `#name`, so a
// line break here would inject a stray space into the registered name and the
// graph would never be found by lookup.
// clang-format off
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tiled_detection::TiledBoxTrackMergeGraph);  // NOLINT(whitespace/line_length)
// clang-format on

}  // namespace tiled_detection
}  // namespace mediapipe
