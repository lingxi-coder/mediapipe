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
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

constexpr float kTol = 1e-5f;

// Builds a runner for TileGridCalculator with `options_body` merged into the
// options extension and one int tick packet per timestamp 0..ticks-1.
std::unique_ptr<CalculatorRunner> MakeRunner(const std::string& options_body,
                                             int ticks) {
  auto runner = std::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
          absl::StrFormat(R"pb(
                            calculator: "TileGridCalculator"
                            input_stream: "TICK:tick"
                            output_stream: "TILES:tiles"
                            options {
                              [mediapipe.TileGridCalculatorOptions.ext] { %s }
                            }
                          )pb",
                          options_body)));
  for (int t = 0; t < ticks; ++t) {
    runner->MutableInputs()->Tag("TICK").packets.push_back(
        MakePacket<int>(0).At(Timestamp(t)));
  }
  return runner;
}

// Runs the calculator over `ticks` ticks and returns the contents of the last
// TILES packet (expects one output packet per tick).
std::vector<NormalizedRect> RunTiles(const std::string& options_body,
                                     int ticks) {
  auto runner = MakeRunner(options_body, ticks);
  MP_EXPECT_OK(runner->Run());
  const auto& packets = runner->Outputs().Tag("TILES").packets;
  EXPECT_EQ(static_cast<int>(packets.size()), ticks);
  if (packets.empty()) {
    ADD_FAILURE() << "no TILES packets emitted";
    return {};
  }
  return packets.back().Get<std::vector<NormalizedRect>>();
}

TEST(TileGridCalculatorTest, DefaultIsSingleFullFrameTile) {
  const std::vector<NormalizedRect> tiles = RunTiles("", /*ticks=*/1);
  ASSERT_EQ(tiles.size(), 1);
  EXPECT_NEAR(tiles[0].x_center(), 0.5f, kTol);
  EXPECT_NEAR(tiles[0].y_center(), 0.5f, kTol);
  EXPECT_NEAR(tiles[0].width(), 1.0f, kTol);
  EXPECT_NEAR(tiles[0].height(), 1.0f, kTol);
}

TEST(TileGridCalculatorTest, TwoColsWithOverlap) {
  const std::vector<NormalizedRect> tiles =
      RunTiles("cols: 2 overlap_fraction: 0.2", /*ticks=*/1);
  ASSERT_EQ(tiles.size(), 2);
  const float w = 1.0f / 1.8f;  // 1 / (cols - (cols-1)*overlap)
  EXPECT_NEAR(tiles[0].width(), w, kTol);
  EXPECT_NEAR(tiles[0].x_center(), w / 2, kTol);
  EXPECT_NEAR(tiles[1].x_center(), w * 0.8f + w / 2, kTol);
  // The last tile's right edge reaches the frame border.
  EXPECT_NEAR(tiles[1].x_center() + tiles[1].width() / 2, 1.0f, kTol);
  EXPECT_NEAR(tiles[0].height(), 1.0f, kTol);
}

TEST(TileGridCalculatorTest, RowsColsGrid) {
  const std::vector<NormalizedRect> tiles =
      RunTiles("rows: 2 cols: 3", /*ticks=*/1);
  ASSERT_EQ(tiles.size(), 6);  // row-major
  EXPECT_NEAR(tiles[0].width(), 1.0f / 3, kTol);
  EXPECT_NEAR(tiles[0].height(), 0.5f, kTol);
  // Row-major ordering: index 1 = row 0, col 1.
  EXPECT_NEAR(tiles[1].x_center(), 0.5f, 1e-5);
  EXPECT_NEAR(tiles[1].y_center(), 0.25f, 1e-5);
  EXPECT_NEAR(tiles[5].x_center(), 1.0f - 1.0f / 6, kTol);
  EXPECT_NEAR(tiles[5].y_center(), 0.75f, kTol);
}

TEST(TileGridCalculatorTest, ExplicitTilesPassThroughEveryTick) {
  const std::vector<NormalizedRect> tiles = RunTiles(
      "explicit_tiles { x_center: 0.3 y_center: 0.4 width: 0.2 height: 0.6 }",
      /*ticks=*/2);
  ASSERT_EQ(tiles.size(), 1);
  EXPECT_NEAR(tiles[0].x_center(), 0.3f, kTol);
  EXPECT_NEAR(tiles[0].y_center(), 0.4f, kTol);
  EXPECT_NEAR(tiles[0].width(), 0.2f, kTol);
  EXPECT_NEAR(tiles[0].height(), 0.6f, kTol);
}

TEST(TileGridCalculatorTest, GridAndExplicitMutuallyExclusive) {
  auto runner = MakeRunner(
      "cols: 2 "
      "explicit_tiles { x_center: 0.5 y_center: 0.5 width: 0.5 height: 0.5 }",
      /*ticks=*/1);
  EXPECT_FALSE(runner->Run().ok());
}

TEST(TileGridCalculatorTest, InvalidParamsFailOpen) {
  const std::vector<std::string> bad_options = {
      "cols: 0",
      "overlap_fraction: 1.0",
      "explicit_tiles { width: 0 height: 1 }",
      "explicit_tiles { x_center: .5 y_center: .5 width: 1 height: 1 } "
      "overlap_fraction: 0.5",
  };
  for (const std::string& options_body : bad_options) {
    auto runner = MakeRunner(options_body, /*ticks=*/1);
    EXPECT_FALSE(runner->Run().ok()) << "options: " << options_body;
  }
}

}  // namespace
}  // namespace mediapipe
