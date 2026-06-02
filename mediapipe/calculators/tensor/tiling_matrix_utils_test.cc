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
#include "mediapipe/calculators/tensor/tiling_matrix_utils.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

// Left tile covering x[0,0.6] of a 100x100 frame -> roi {0,0,60,100}.
TEST(TilingMatrixUtilsTest, ProjectsTileLocalToFrameNormalized) {
  TilePixelRoi roi{/*x=*/0, /*y=*/0, /*width=*/60, /*height=*/100};
  const auto m = TileToImageMatrix(roi, /*frame_w=*/100, /*frame_h=*/100);
  // Object at frame (0.5,0.5) is tile-local cx = (0.5-0.0)/0.6 = 0.83333.
  float fx, fy;
  ApplyMatrix(m, 0.83333f, 0.5f, &fx, &fy);
  EXPECT_NEAR(fx, 0.5f, 1e-3);
  EXPECT_NEAR(fy, 0.5f, 1e-3);
  // Width scales by tile fraction: a tile-local width of 0.33333 -> 0.2 frame.
  float x0, y0, x1, y1;
  ApplyMatrix(m, 0.83333f - 0.33333f / 2, 0.5f, &x0, &y0);
  ApplyMatrix(m, 0.83333f + 0.33333f / 2, 0.5f, &x1, &y1);
  EXPECT_NEAR(x1 - x0, 0.2f, 1e-3);
}

TEST(TilingMatrixUtilsTest, InverseRoundTrips) {
  TilePixelRoi roi{10, 20, 60, 50};
  const auto m = TileToImageMatrix(roi, 100, 100);
  const auto inv = InvertAffine2d(m);
  float fx, fy, tx, ty;
  ApplyMatrix(m, 0.3f, 0.7f, &fx, &fy);
  ApplyMatrix(inv, fx, fy, &tx, &ty);
  EXPECT_NEAR(tx, 0.3f, 1e-4);
  EXPECT_NEAR(ty, 0.7f, 1e-4);
}

}  // namespace
}  // namespace mediapipe
