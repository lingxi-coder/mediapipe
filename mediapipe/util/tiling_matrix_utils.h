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
#ifndef MEDIAPIPE_UTIL_TILING_MATRIX_UTILS_H_
#define MEDIAPIPE_UTIL_TILING_MATRIX_UTILS_H_

#include <array>

#include "mediapipe/framework/formats/tiling_types.h"

namespace mediapipe {

// Builds the row-major 4x4 that maps a point in the tile's normalized [0,1]
// space to the source frame's normalized [0,1] space, for an AXIS-ALIGNED tile
// whose effective sampled region is `roi` pixels within a `frame_w` x `frame_h`
// frame. Matches the ImageToTensor transform with zero rotation and no flip,
// including scaling Z by the tile width relative to the frame width.
std::array<float, 16> TileToImageMatrix(const TilePixelRoi& roi, int frame_w,
                                      int frame_h);

// Applies a row-major 4x4 to the homogeneous 2D point (x, y, 0, 1); returns the
// transformed (x', y') (w assumed 1 for affine tile transforms).
inline void ApplyMatrix(const std::array<float, 16>& m, float x, float y,
                        float* out_x, float* out_y) {
  *out_x = m[0] * x + m[1] * y + m[3];
  *out_y = m[4] * x + m[5] * y + m[7];
}

// Inverts the affine 2D part of a row-major 4x4 (rotation+scale+translation)
// into the reverse map (image-norm -> tile-norm). Tile transforms are affine,
// so this is exact.
std::array<float, 16> InvertAffine2d(const std::array<float, 16>& m);

}  // namespace mediapipe
#endif  // MEDIAPIPE_UTIL_TILING_MATRIX_UTILS_H_
