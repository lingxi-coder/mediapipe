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
#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.pb.h"
#include "mediapipe/calculators/tensor/tile_grid_calculator.pb.h"
#include "mediapipe/calculators/tensor/video_tile_scheduler_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/graphs/tiled_detection/tiled_detection_graphs.pb.h"

namespace mediapipe {
namespace tiled_detection {

// Stream-mode sibling of TiledDetectionFrontGraph: runs its own optical-flow
// pass and a VideoTileSchedulerCalculator between TileGrid and
// TileSpecToTilePlan, so low-motion frames SKIP inference (empty TILES ->
// empty TilePlan -> BATCH_INFO-only frame). The scheduler's REFRESH bool
// (DETECT=true / SKIP=false) is exposed as a subgraph output; callers may
// consume it for telemetry/re-detection gating or ignore it. SKIP itself does
// not depend on REFRESH — it rides the empty-TILES path (empty TILES -> empty
// TilePlan -> BATCH_INFO-only frame).
//
// Inputs:
//   IMAGE - ImageFrame.
//   PRIOR_DETECTIONS - std::vector<Detection> (previous frame's merged
//     detections; the caller closes a PreviousLoopback back-edge).
// Outputs:
//   TENSORS - std::vector<Tensor> (one packet per batch; none on a SKIP frame).
//   BATCH_INFO - TensorBatchInfo (total_batches==0 on a SKIP frame).
//   REFRESH - bool (true=DETECT this frame, false=SKIP); optional for callers.
class TiledDetectionStreamFrontGraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    const auto& options = sc->Options<TiledDetectionFrontGraphOptions>();
    api2::builder::Graph graph;
    auto image = graph.In("IMAGE").Cast<ImageFrame>();
    auto priors = graph.In("PRIOR_DETECTIONS").Cast<std::vector<Detection>>();

    // Own optical-flow pass (sub-project C's tracker keeps its own flow).
    auto& flow = graph.AddNode("OpticalFlowTrackingGraph");
    image >> flow.In("IMAGE");

    auto& grid = graph.AddNode("TileGridCalculator");
    grid.GetOptions<TileGridCalculatorOptions>() = options.tile_grid();
    image >> grid.In("TICK");

    auto& scheduler = graph.AddNode("VideoTileSchedulerCalculator");
    scheduler.GetOptions<VideoTileSchedulerCalculatorOptions>()
        .set_max_scheduled_tiles(options.max_scheduled_tiles());
    grid.Out("TILES") >> scheduler.In("TILES");
    priors >> scheduler.In("PRIOR_DETECTIONS");
    flow.Out("TRACKING") >> scheduler.In("TRACKING");
    // REFRESH is exposed as a subgraph output so callers may optionally use it
    // to gate re-detection; leaving it unobserved is safe.
    scheduler.Out("REFRESH") >> graph.Out("REFRESH");

    auto& plan = graph.AddNode("TileSpecToTilePlanCalculator");
    scheduler.Out("TILES") >> plan.In("TILES");

    auto& batcher = graph.AddNode("StreamingTilesToTensorBatchCalculator");
    auto& bo =
        batcher.GetOptions<StreamingTilesToTensorBatchCalculatorOptions>();
    bo.set_metadata_batch_capacity(options.batch_capacity());
    bo.set_metadata_input_height(options.input_height());
    bo.set_metadata_input_width(options.input_width());
    bo.set_metadata_input_channels(options.input_channels());
    bo.set_metadata_is_dynamic_batch(options.is_dynamic_batch());
    image >> batcher.In("IMAGE");
    plan.Out("TILE_PLAN") >> batcher.In("TILE_PLAN");

    batcher.Out("TENSORS") >> graph.Out("TENSORS");
    batcher.Out("BATCH_INFO") >> graph.Out("BATCH_INFO");
    return graph.GetConfig();
  }
};

// NOTE: keep the fully-qualified type name on a single line. The
// REGISTER_MEDIAPIPE_GRAPH macro stringifies its argument with `#name`, so a
// line break here would inject a stray space into the registered name and the
// graph would never be found by lookup.
// clang-format off
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tiled_detection::TiledDetectionStreamFrontGraph);  // NOLINT(whitespace/line_length)
// clang-format on

}  // namespace tiled_detection
}  // namespace mediapipe
