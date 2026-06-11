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
#include "mediapipe/calculators/util/rotated_non_max_suppression_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/graphs/tiled_detection/tiled_detection_graphs.pb.h"

namespace mediapipe {
namespace tiled_detection {

// A "mediapipe.tiled_detection.TiledObbMergeGraph" merges per-batch tile-local
// oriented detections back into source-frame space and runs a global rotated
// NMS over the merged set:
// ORIENTED_DETECTIONS + BATCH_INFO -> MergeTileDetectionsAccumulator ->
// RotatedNonMaxSuppression.
//
// The global NMS ALWAYS runs — there is no single-tile bypass. Tile-local NMS
// defaults off upstream, so bypassing would return un-deduped raw detections
// for single-tile configs; re-running greedy NMS on an already-deduped set is
// an idempotent no-op.
//
// Inputs:
//   ORIENTED_DETECTIONS - std::vector<std::vector<OrientedDetection>>
//     Per-batch tile-local detections (one inner vector per batch row), ONE
//     PACKET PER BATCH at synthetic timestamps paired with BATCH_INFO.
//   BATCH_INFO - TensorBatchInfo
//     Per-batch metadata (source frame timestamp, batch index/size,
//     valid_count, tile geometry) used to project rows to frame space.
//
// Outputs:
//   ORIENTED_DETECTIONS - std::vector<OrientedDetection>
//     Merged, globally NMS-ed frame-space detections, ONE PACKET PER SOURCE
//     FRAME emitted at the source frame timestamp.
//
// Example:
// node {
//   calculator: "mediapipe.tiled_detection.TiledObbMergeGraph"
//   input_stream: "ORIENTED_DETECTIONS:tile_detections"
//   input_stream: "BATCH_INFO:batch_info"
//   output_stream: "ORIENTED_DETECTIONS:detections"
//   options {
//     [mediapipe.TiledObbMergeGraphOptions.ext] {
//       iou_threshold: 0.5
//     }
//   }
// }
class TiledObbMergeGraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    const auto& options = sc->Options<TiledObbMergeGraphOptions>();
    api2::builder::Graph graph;

    auto& merge = graph.AddNode("MergeTileDetectionsAccumulatorCalculator");
    graph.In("ORIENTED_DETECTIONS") >> merge.In("ORIENTED_DETECTIONS");
    graph.In("BATCH_INFO") >> merge.In("BATCH_INFO");

    auto& nms = graph.AddNode("RotatedNonMaxSuppressionCalculator");
    auto& no = nms.GetOptions<RotatedNonMaxSuppressionCalculatorOptions>();
    no.set_iou_threshold(options.iou_threshold());
    no.set_max_detections(options.max_detections());
    no.set_class_agnostic(options.class_agnostic());
    merge.Out("ORIENTED_DETECTIONS") >> nms.In("ORIENTED_DETECTIONS");

    nms.Out("ORIENTED_DETECTIONS") >> graph.Out("ORIENTED_DETECTIONS");
    return graph.GetConfig();
  }
};

// NOTE: keep the fully-qualified type name on a single line. The
// REGISTER_MEDIAPIPE_GRAPH macro stringifies its argument with `#name`, so a
// line break here would inject a stray space into the registered name and the
// graph would never be found by lookup.
// clang-format off
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tiled_detection::TiledObbMergeGraph);  // NOLINT(whitespace/line_length)
// clang-format on

}  // namespace tiled_detection
}  // namespace mediapipe
