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
#include "mediapipe/util/tiling_matrix_utils.h"

namespace mediapipe {

std::array<float, 16> TileToImageMatrix(const TilePixelRoi& roi, int frame_w,
                                      int frame_h) {
  const float a = static_cast<float>(roi.width);
  const float b = static_cast<float>(roi.height);
  const float flip = 1.0f;
  const float c = 1.0f;  // cos(0), since tiles are axis-aligned.
  const float d = 0.0f;  // sin(0).
  const float e = roi.x + roi.width / 2.0f;
  const float f = roi.y + roi.height / 2.0f;
  const float g = 1.0f / frame_w;
  const float h = 1.0f / frame_h;

  // Keep the ImageToTensor formula's floating-point operation order. Computing
  // translations directly as roi.x / frame_w or roi.y / frame_h can round
  // differently, as can replacing multiplication by a reciprocal with division.
  return {
      a * c * flip * g, -b * d * g, 0.0f,
      (-0.5f * a * c * flip + 0.5f * b * d + e) * g,
      a * d * flip * h, b * c * h, 0.0f,
      (-0.5f * b * c - 0.5f * a * d * flip + f) * h,
      0.0f, 0.0f, a * g, 0.0f,
      0.0f, 0.0f, 0.0f, 1.0f,
  };
}

std::array<float, 16> InvertAffine2d(const std::array<float, 16>& m) {
  const float a = m[0], b = m[1], c = m[4], d = m[5];
  const float tx = m[3], ty = m[7];
  const float det = a * d - b * c;
  const float inv_det = det != 0.0f ? 1.0f / det : 0.0f;
  const float ia = d * inv_det, ib = -b * inv_det;
  const float ic = -c * inv_det, id = a * inv_det;
  std::array<float, 16> r = {0};
  r[0] = ia; r[1] = ib; r[3] = -(ia * tx + ib * ty);
  r[4] = ic; r[5] = id; r[7] = -(ic * tx + id * ty);
  r[10] = 1.0f; r[15] = 1.0f;
  return r;
}

}  // namespace mediapipe
