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
#include "mediapipe/calculators/video/botsort_tracking_calculator.pb.h"
#include "mediapipe/calculators/util/rotated_non_max_suppression_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/tasks/cc/components/processors/proto/tiled_detection_graph_options.pb.h"

namespace mediapipe {
namespace tiled_detection {

// Stream-mode sibling of TiledObbMergeGraph: merges per-batch tile-local oriented
// detections into frame space + global rotated NMS, then assigns BoTSORT track
// ids via OrientedBotsortTrackingCalculator (ID-only; geometry unchanged). One
// packet per source frame.
//
// Unlike the axis-aligned TiledBoxTrackMergeGraph -- which fuses fresh +
// tracker detections via TiledFrameSuppression with gap-fill -- this OBB
// version is a simpler linear chain with NO gap-fill, because BoTSORT runs
// ID-only on AABBs and rotation is not tracked (so tracker-only oriented boxes
// can't be emitted).
//
// Inputs:  ORIENTED_DETECTIONS (vector<vector<OrientedDetection>>),
//          BATCH_INFO (TensorBatchInfo), IMAGE (ImageFrame).
// Outputs: ORIENTED_DETECTIONS (vector<OrientedDetection>, track_id set).
class TiledObbTrackMergeGraph : public Subgraph {
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

    auto& track = graph.AddNode("OrientedBotsortTrackingCalculator");
    auto& to = track.GetOptions<BotsortTrackingCalculatorOptions>();
    const auto& tr = options.tracking();
    to.set_track_high_threshold(tr.track_high_threshold());
    to.set_track_low_threshold(tr.track_low_threshold());
    to.set_new_track_threshold(tr.new_track_threshold());
    to.set_track_buffer(tr.track_buffer());
    to.set_match_threshold(tr.match_threshold());
    to.set_enable_gmc(tr.enable_gmc());
    to.set_nominal_frame_rate(tr.nominal_frame_rate());
    graph.In("IMAGE") >> track.In("IMAGE");
    nms.Out("ORIENTED_DETECTIONS") >> track.In("ORIENTED_DETECTIONS");

    track.Out("ORIENTED_DETECTIONS") >> graph.Out("ORIENTED_DETECTIONS");
    return graph.GetConfig();
  }
};

// NOTE: keep the fully-qualified type name on a single line. The
// REGISTER_MEDIAPIPE_GRAPH macro stringifies its argument with `#name`, so a
// line break here would inject a stray space into the registered name and the
// graph would never be found by lookup.
// clang-format off
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tiled_detection::TiledObbTrackMergeGraph);  // NOLINT(whitespace/line_length)
// clang-format on

}  // namespace tiled_detection
}  // namespace mediapipe
