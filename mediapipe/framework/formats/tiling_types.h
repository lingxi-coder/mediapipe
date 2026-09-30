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
#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_TILING_TYPES_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_TILING_TYPES_H_

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace mediapipe {

// One validated tile: an axis-aligned crop over one source frame, in
// normalized [0,1] coordinates. (Rotated tiles are not supported yet.)
struct TileGeometry {
  int tile_index = 0;
  float x_center = 0.0f;  // normalized
  float y_center = 0.0f;
  float width = 0.0f;     // normalized, > 0
  float height = 0.0f;    // normalized, > 0

  // Top-left of the tile in normalized frame coords (derived).
  float x0() const { return x_center - width / 2.0f; }
  float y0() const { return y_center - height / 2.0f; }
};

// Per-frame validated tile list.
struct TilePlan {
  std::vector<TileGeometry> tiles;
};

// Effective integer pixel ROI a tile actually sampled from the source frame,
// after rounding + clamping to the frame. Matrices describe THIS ROI, not the
// requested normalized rect, so boundary tiles project precisely.
struct TilePixelRoi {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

// Immutable per-batch geometry, shared (shared_ptr) so it travels on the
// BATCH_INFO packet without copying matrix/geometry vectors. All vectors are
// indexed by valid row [0, valid_count) in row order.
struct TileBatchGeometry {
  std::vector<int> tile_indices;
  std::vector<TileGeometry> tile_geometries;
  std::vector<TilePixelRoi> effective_pixel_rois;
  // Row-major 4x4. image_to_tensor maps tile-normalized [0,1] -> image-norm
  // [0,1]; tensor_to_image is its inverse (image-norm -> tile-norm). Stored so
  // merge projects detections (tile space) to frame space without re-deriving.
  std::vector<std::array<float, 16>> tile_to_image_matrices;
  std::vector<std::array<float, 16>> image_to_tile_matrices;
};

// Travels with each emitted inference batch so the merge step can regroup and
// drop padding. Timestamps are in microseconds.
struct TensorBatchInfo {
  int64_t source_frame_timestamp = 0;
  int64_t batch_timestamp = 0;  // loop-internal monotonic ts this batch emitted at
  int batch_index = 0;     // 0-based index of this batch within the frame
  int total_batches = 1;   // total batches the frame will emit
  int batch_capacity = 1;  // N dimension of the emitted tensor
  int batch_size = 0;      // emitted N dimension of the tensor
  int valid_count = 0;     // valid rows [0, valid_count); rest is padding
  // Tile index for each valid row, in row order.
  std::vector<int> tile_indices;  // retained for back-compat; mirrors geometry
  // Non-null when valid_count > 0; all its vectors have size == valid_count.
  std::shared_ptr<const TileBatchGeometry> geometry;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_TILING_TYPES_H_
