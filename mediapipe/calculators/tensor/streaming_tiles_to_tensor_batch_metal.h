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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_STREAMING_TILES_TO_TENSOR_BATCH_METAL_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_STREAMING_TILES_TO_TENSOR_BATCH_METAL_H_

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_METAL_ENABLED

#import <Metal/Metal.h>

#include <memory>

#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"  // BorderMode
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"      // RotatedRect
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {

// Metal render-pipeline writer that crops/resizes/normalizes one tile of an
// input texture directly into row `tile_row` of a destination MTLBuffer laid out
// as the TFLite Metal delegate's PHWC4 input: physical [N, out_h, out_w, 4]
// float32, contiguous. Adapted from ImageToTensorMetalConverter
// (image_to_tensor_converter_metal.cc): the one structural change is that the
// render target MTLTexture is aliased over the destination buffer at this tile's
// row BYTE OFFSET (= tile_row * out_h * out_w * 4 * sizeof(float)) instead of
// offset 0, so each tile fills its own batch row. The caller renders every valid
// tile into the SAME buffer (one buffer per batch) before reading it.
//
// Logical model input is RGB (channels == 3); physical storage is RGBA32Float
// (C4 == 4) with the padded 4th channel written 0, matching PHWC4 for C <= 4.
class TiledBatchMetalWriter {
 public:
  // out_w/out_h: per-tile (= per-row) width/height. channels is the LOGICAL
  // channel count (must be 3); the physical row is always RGBA (4) F32.
  static absl::StatusOr<std::unique_ptr<TiledBatchMetalWriter>> Create(
      id<MTLDevice> device, int out_w, int out_h, int channels,
      BorderMode border_mode);

  // sub_rect: the tile's RotatedRect over the source texture (same convention as
  //   the CPU path / image_to_tensor).
  // tile_row: destination batch row in [0, N).
  // alpha/beta: value-range normalization (alpha=1, beta=0 for [0,1]; Metal
  //   samples uint8 textures as [0,1]).
  // command_buffer: the command buffer to encode into (caller commits).
  // dest: the whole-batch physical [N,out_h,out_w,4] F32 MTLBuffer (e.g. the
  //   output Tensor's MtlBufferView buffer).
  absl::Status RenderTileRow(id<MTLTexture> input_texture,
                             const RotatedRect& sub_rect, int tile_row,
                             float alpha, float beta,
                             id<MTLCommandBuffer> command_buffer,
                             id<MTLBuffer> dest);

  int out_w() const { return out_w_; }
  int out_h() const { return out_h_; }
  // Bytes per physical batch row = out_h * out_w * 4 * sizeof(float).
  size_t row_bytes() const { return row_bytes_; }

 private:
  TiledBatchMetalWriter(id<MTLDevice> device,
                        id<MTLRenderPipelineState> pipeline_state, int out_w,
                        int out_h, size_t texture_offset_alignment);

  id<MTLDevice> device_;
  id<MTLRenderPipelineState> pipeline_state_;
  id<MTLBuffer> positions_buffer_;
  id<MTLBuffer> tex_coords_buffer_;
  int out_w_ = 0;
  int out_h_ = 0;
  size_t bytes_per_pixel_row_ = 0;  // out_w * 4 * sizeof(float)
  size_t row_bytes_ = 0;            // out_h * bytes_per_pixel_row_
  size_t texture_offset_alignment_ = 0;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // MEDIAPIPE_CALCULATORS_TENSOR_STREAMING_TILES_TO_TENSOR_BATCH_METAL_H_
