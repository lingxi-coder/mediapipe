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

namespace mediapipe {

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
