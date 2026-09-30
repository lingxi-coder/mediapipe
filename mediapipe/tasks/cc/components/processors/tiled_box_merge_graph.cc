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
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/tasks/cc/components/processors/proto/tiled_detection_graph_options.pb.h"

namespace mediapipe {
namespace tiled_detection {

// A "mediapipe.tiled_detection.TiledBoxMergeGraph" merges per-batch tile-local
// axis-aligned box detections back into source-frame space and runs a global
// NMS over the merged set:
// DETECTIONS + BATCH_INFO -> MergeTileBoxDetectionsAccumulator ->
// TiledFrameSuppression [-> ClipDetectionVectorSize if max_detections >= 1].
//
// The global NMS ALWAYS runs — there is no single-tile bypass. The internal
// TiledFrameSuppressionCalculator is used WITHOUT its optional NUM_TILES /
// TRACKER_DETECTIONS inputs and with bypass_single_tile left default false;
// those optional inputs are the seam for future tracker integration. Tile-
// local NMS defaults off upstream, so bypassing would return un-deduped raw
// detections for single-tile configs; re-running greedy NMS on an already-
// deduped set is an idempotent no-op.
//
// Inputs:
//   DETECTIONS - std::vector<std::vector<Detection>>
//     Per-batch tile-local detections (one inner vector per batch row), ONE
//     PACKET PER BATCH at synthetic timestamps paired with BATCH_INFO.
//   BATCH_INFO - TensorBatchInfo
//     Per-batch metadata (source frame timestamp, batch index/size,
//     valid_count, tile geometry) used to project rows to frame space.
//
// Outputs:
//   DETECTIONS - std::vector<Detection>
//     Merged, globally NMS-ed frame-space detections, ONE PACKET PER SOURCE
//     FRAME emitted at the source frame timestamp. NMS output is descending
//     by score, so the optional max_detections cap keeps the highest scores.
//
// Example:
// node {
//   calculator: "mediapipe.tiled_detection.TiledBoxMergeGraph"
//   input_stream: "DETECTIONS:tile_detections"
//   input_stream: "BATCH_INFO:batch_info"
//   output_stream: "DETECTIONS:detections"
//   options {
//     [mediapipe.TiledBoxMergeGraphOptions.ext] {
//       iou_threshold: 0.5
//     }
//   }
// }
class TiledBoxMergeGraph : public Subgraph {
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

    auto& nms = graph.AddNode("TiledFrameSuppressionCalculator");
    auto& no = nms.GetOptions<TiledFrameSuppressionCalculatorOptions>();
    no.set_iou_threshold(options.iou_threshold());
    no.set_class_agnostic(options.class_agnostic());
    merge.Out("DETECTIONS") >> nms.In("DETECTIONS");

    if (options.max_detections() >= 1) {
      auto& clip = graph.AddNode("ClipDetectionVectorSizeCalculator");
      auto& co = clip.GetOptions<ClipVectorSizeCalculatorOptions>();
      co.set_max_vec_size(options.max_detections());
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
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tiled_detection::TiledBoxMergeGraph);  // NOLINT(whitespace/line_length)
// clang-format on

}  // namespace tiled_detection
}  // namespace mediapipe
