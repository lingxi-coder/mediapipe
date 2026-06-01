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

#include <memory>
#include <vector>

#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

NormalizedRect Rect(float xc, float yc, float w, float h) {
  NormalizedRect r;
  r.set_x_center(xc); r.set_y_center(yc); r.set_width(w); r.set_height(h);
  return r;
}

TEST(TileSpecToTilePlanCalculatorTest, ValidatesAndIndexesTiles) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TileSpecToTilePlanCalculator"
    input_stream: "TILES:tiles"
    output_stream: "TILE_PLAN:plan"
  )pb"));
  auto tiles = std::make_unique<std::vector<NormalizedRect>>();
  tiles->push_back(Rect(0.25f, 0.25f, 0.5f, 0.5f));
  tiles->push_back(Rect(0.75f, 0.75f, 0.5f, 0.5f));
  runner.MutableInputs()->Tag("TILES").packets.push_back(
      Adopt(tiles.release()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& plan = runner.Outputs().Tag("TILE_PLAN").packets[0].Get<TilePlan>();
  ASSERT_EQ(plan.tiles.size(), 2);
  EXPECT_EQ(plan.tiles[0].tile_index, 0);
  EXPECT_EQ(plan.tiles[1].tile_index, 1);
  EXPECT_NEAR(plan.tiles[0].x0(), 0.0f, 1e-5);
  EXPECT_NEAR(plan.tiles[0].width, 0.5f, 1e-5);
}

TEST(TileSpecToTilePlanCalculatorTest, RejectsNonPositiveSize) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TileSpecToTilePlanCalculator"
    input_stream: "TILES:tiles"
    output_stream: "TILE_PLAN:plan"
  )pb"));
  auto tiles = std::make_unique<std::vector<NormalizedRect>>();
  tiles->push_back(Rect(0.5f, 0.5f, 0.0f, 0.5f));  // zero width -> invalid
  runner.MutableInputs()->Tag("TILES").packets.push_back(
      Adopt(tiles.release()).At(Timestamp(0)));
  EXPECT_FALSE(runner.Run().ok());
}

TEST(TileSpecToTilePlanCalculatorTest, EmptyTilesEmitsEmptyPlan) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TileSpecToTilePlanCalculator"
    input_stream: "TILES:tiles"
    output_stream: "TILE_PLAN:plan"
  )pb"));
  runner.MutableInputs()->Tag("TILES").packets.push_back(
      Adopt(new std::vector<NormalizedRect>()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  EXPECT_TRUE(runner.Outputs().Tag("TILE_PLAN").packets[0].Get<TilePlan>().tiles.empty());
}

}  // namespace
}  // namespace mediapipe
