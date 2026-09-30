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

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/util/tile_grid_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// Emits a static tile list on every TICK (computed once in Open). Grid
// formula: tile_w = 1 / (cols - (cols-1)*overlap); stride = tile_w *
// (1 - overlap).
class TileGridCalculator : public Node {
 public:
  static constexpr Input<AnyType> kTick{"TICK"};
  static constexpr Output<std::vector<NormalizedRect>> kTiles{"TILES"};
  MEDIAPIPE_NODE_CONTRACT(kTick, kTiles);

  absl::Status Open(CalculatorContext* cc) override {
    const auto& o = cc->Options<mediapipe::TileGridCalculatorOptions>();
    RET_CHECK_GE(o.rows(), 1);
    RET_CHECK_GE(o.cols(), 1);
    RET_CHECK_GE(o.overlap_fraction(), 0.0f);
    RET_CHECK_LT(o.overlap_fraction(), 1.0f);
    if (!o.explicit_tiles().empty()) {
      RET_CHECK(o.rows() == 1 && o.cols() == 1)
          << "explicit_tiles is mutually exclusive with a rows/cols grid";
      RET_CHECK(!o.has_overlap_fraction())
          << "overlap_fraction is ignored with explicit_tiles; do not set "
             "both";
      for (const auto& t : o.explicit_tiles()) {
        RET_CHECK(t.width() > 0 && t.height() > 0);
        NormalizedRect r;
        r.set_x_center(t.x_center());
        r.set_y_center(t.y_center());
        r.set_width(t.width());
        r.set_height(t.height());
        tiles_.push_back(std::move(r));
      }
      return absl::OkStatus();
    }
    const float of = o.overlap_fraction();
    const float tw = 1.0f / (o.cols() - (o.cols() - 1) * of);
    const float th = 1.0f / (o.rows() - (o.rows() - 1) * of);
    const float sx = tw * (1.0f - of), sy = th * (1.0f - of);
    for (int r = 0; r < o.rows(); ++r) {
      for (int c = 0; c < o.cols(); ++c) {
        NormalizedRect t;
        t.set_x_center(c * sx + tw / 2);
        t.set_y_center(r * sy + th / 2);
        t.set_width(tw);
        t.set_height(th);
        tiles_.push_back(std::move(t));
      }
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    kTiles(cc).Send(tiles_);
    return absl::OkStatus();
  }

 private:
  std::vector<NormalizedRect> tiles_;
};

MEDIAPIPE_REGISTER_NODE(TileGridCalculator);

}  // namespace api2
}  // namespace mediapipe
