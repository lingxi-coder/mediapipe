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
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/graphs/tiled_detection/tiled_detection_graphs.pb.h"

namespace mediapipe {
namespace tiled_detection {

// A "mediapipe.tiled_detection.TiledDetectionFrontGraph" splits an input
// image into a grid of tiles and packs them into batched input tensors:
// IMAGE -> TileGrid -> TileSpecToTilePlan -> StreamingTilesToTensorBatch.
// Inference is the caller's responsibility.
//
// Inputs:
//   IMAGE - ImageFrame
//     Image to tile and batch.
//
// Outputs:
//   TENSORS - std::vector<Tensor>
//     Batched NHWC float tensors, ONE PACKET PER BATCH, emitted at synthetic
//     timestamps. Callers must not assume alignment with input timestamps;
//     pair each packet with the corresponding BATCH_INFO packet instead.
//   BATCH_INFO - TensorBatchInfo
//     Per-batch metadata (source frame timestamp, batch index/size,
//     valid_count, tile geometry) for mapping batch rows back to the
//     original image.
//
// Example:
// node {
//   calculator: "mediapipe.tiled_detection.TiledDetectionFrontGraph"
//   input_stream: "IMAGE:image"
//   output_stream: "TENSORS:tensors"
//   output_stream: "BATCH_INFO:batch_info"
//   options {
//     [mediapipe.TiledDetectionFrontGraphOptions.ext] {
//       tile_grid { cols: 2 }
//       batch_capacity: 2
//       input_height: 320
//       input_width: 320
//       input_channels: 3
//     }
//   }
// }
class TiledDetectionFrontGraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    const auto& options = sc->Options<TiledDetectionFrontGraphOptions>();
    api2::builder::Graph graph;
    auto image = graph.In("IMAGE").Cast<ImageFrame>();

    auto& grid = graph.AddNode("TileGridCalculator");
    grid.GetOptions<TileGridCalculatorOptions>() = options.tile_grid();
    image >> grid.In("TICK");

    auto& plan = graph.AddNode("TileSpecToTilePlanCalculator");
    grid.Out("TILES") >> plan.In("TILES");

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
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tiled_detection::TiledDetectionFrontGraph);  // NOLINT(whitespace/line_length)
// clang-format on

}  // namespace tiled_detection
}  // namespace mediapipe
