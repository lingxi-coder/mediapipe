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

#include "absl/strings/str_format.h"
#include "mediapipe/framework/formats/tiling_cache_stats.h"
#include "mediapipe/framework/formats/tiling_types.h"
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

// Runs the calculator over `frames` tile-lists (one per timestamp) with the
// given cache capacity; returns the emitted TilePlans in order.
std::vector<TilePlan> RunPlan(
    int max_cached_tile_plans,
    const std::vector<std::vector<NormalizedRect>>& frames) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      absl::StrFormat(R"pb(
        calculator: "TileSpecToTilePlanCalculator"
        input_stream: "TILES:tiles"
        output_stream: "TILE_PLAN:plan"
        options {
          [mediapipe.TileSpecToTilePlanCalculatorOptions.ext] {
            max_cached_tile_plans: %d
          }
        }
      )pb",
                      max_cached_tile_plans)));
  for (int t = 0; t < static_cast<int>(frames.size()); ++t) {
    auto tiles = std::make_unique<std::vector<NormalizedRect>>(frames[t]);
    runner.MutableInputs()->Tag("TILES").packets.push_back(
        Adopt(tiles.release()).At(Timestamp(t)));
  }
  MP_EXPECT_OK(runner.Run());
  std::vector<TilePlan> out;
  for (const Packet& p : runner.Outputs().Tag("TILE_PLAN").packets) {
    out.push_back(p.Get<TilePlan>());
  }
  return out;
}

// Caching on must produce TilePlans identical to caching off, across repeated
// identical frames (cache hits) and a changed frame (cache miss / recompute).
TEST(TileSpecToTilePlanCalculatorTest, CacheEnabledMatchesDisabled) {
  const std::vector<NormalizedRect> a = {Rect(0.25f, 0.25f, 0.5f, 0.5f),
                                         Rect(0.75f, 0.75f, 0.5f, 0.5f)};
  const std::vector<NormalizedRect> b = {Rect(0.5f, 0.5f, 0.4f, 0.4f)};
  const std::vector<std::vector<NormalizedRect>> frames = {a, a, b, a};

  const std::vector<TilePlan> off = RunPlan(/*max_cached_tile_plans=*/0, frames);
  const std::vector<TilePlan> on = RunPlan(/*max_cached_tile_plans=*/4, frames);

  ASSERT_EQ(off.size(), frames.size());
  ASSERT_EQ(on.size(), off.size());
  for (size_t f = 0; f < off.size(); ++f) {
    ASSERT_EQ(on[f].tiles.size(), off[f].tiles.size());
    for (size_t i = 0; i < off[f].tiles.size(); ++i) {
      EXPECT_EQ(on[f].tiles[i].tile_index, off[f].tiles[i].tile_index);
      EXPECT_FLOAT_EQ(on[f].tiles[i].x_center, off[f].tiles[i].x_center);
      EXPECT_FLOAT_EQ(on[f].tiles[i].y_center, off[f].tiles[i].y_center);
      EXPECT_FLOAT_EQ(on[f].tiles[i].width, off[f].tiles[i].width);
      EXPECT_FLOAT_EQ(on[f].tiles[i].height, off[f].tiles[i].height);
    }
  }
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

TEST(TileSpecToTilePlanCalculatorTest, CacheStatsReportsTilePlanHits) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TileSpecToTilePlanCalculator"
    input_stream: "TILES:tiles"
    output_stream: "TILE_PLAN:plan"
    output_stream: "CACHE_STATS:stats"
    options {
      [mediapipe.TileSpecToTilePlanCalculatorOptions.ext] {
        max_cached_tile_plans: 4
        emit_cache_stats: true
      }
    }
  )pb"));

  // Feed the same tile list twice (timestamps 0 and 1).
  // Frame 0 is a cache miss; frame 1 is a cache hit.
  const std::vector<NormalizedRect> tiles = {Rect(0.25f, 0.25f, 0.5f, 0.5f)};
  for (int t = 0; t < 2; ++t) {
    auto input = std::make_unique<std::vector<NormalizedRect>>(tiles);
    runner.MutableInputs()->Tag("TILES").packets.push_back(
        Adopt(input.release()).At(Timestamp(t)));
  }

  MP_ASSERT_OK(runner.Run());

  const auto& stats_packets = runner.Outputs().Tag("CACHE_STATS").packets;
  ASSERT_EQ(stats_packets.size(), 2);

  const TilingCacheStats& last = stats_packets[1].Get<TilingCacheStats>();
  EXPECT_GE(last.tile_plan.hits, 1);    // second frame hit
  EXPECT_GE(last.tile_plan.misses, 1);  // first frame miss
}

}  // namespace
}  // namespace mediapipe
