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

#include <cmath>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.pb.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// Validates externally-supplied tiles and emits a TilePlan. Axis-aligned only.
class TileSpecToTilePlanCalculator : public Node {
 public:
  static constexpr Input<std::vector<NormalizedRect>> kInTiles{"TILES"};
  static constexpr Output<TilePlan> kOutPlan{"TILE_PLAN"};
  MEDIAPIPE_NODE_CONTRACT(kInTiles, kOutPlan);

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<mediapipe::TileSpecToTilePlanCalculatorOptions>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& tiles = *kInTiles(cc);
    if (options_.max_tiles_per_frame() > 0) {
      RET_CHECK_LE(static_cast<int>(tiles.size()), options_.max_tiles_per_frame())
          << "tile count exceeds max_tiles_per_frame";
    }
    auto plan = std::make_unique<TilePlan>();
    plan->tiles.reserve(tiles.size());
    for (int i = 0; i < static_cast<int>(tiles.size()); ++i) {
      const NormalizedRect& r = tiles[i];
      RET_CHECK(std::isfinite(r.x_center()) && std::isfinite(r.y_center()) &&
                std::isfinite(r.width()) && std::isfinite(r.height()))
          << "tile " << i << " has non-finite values";
      RET_CHECK_GT(r.width(), 0.0f) << "tile " << i << " width must be > 0";
      RET_CHECK_GT(r.height(), 0.0f) << "tile " << i << " height must be > 0";
      RET_CHECK(!r.has_rotation() || r.rotation() == 0.0f)
          << "rotated tiles are not supported yet (tile " << i << ")";
      TileGeometry g;
      g.tile_index = i;
      g.x_center = r.x_center();
      g.y_center = r.y_center();
      g.width = r.width();
      g.height = r.height();
      RET_CHECK(g.x0() < 1.0f && g.y0() < 1.0f &&
                g.x0() + g.width > 0.0f && g.y0() + g.height > 0.0f)
          << "tile " << i << " does not intersect the frame";
      plan->tiles.push_back(g);
    }
    kOutPlan(cc).Send(std::move(plan));
    return absl::OkStatus();
  }

 private:
  mediapipe::TileSpecToTilePlanCalculatorOptions options_;
};

MEDIAPIPE_REGISTER_NODE(TileSpecToTilePlanCalculator);

}  // namespace api2
}  // namespace mediapipe
