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

// Metal COMPUTE writer that crops/resizes/normalizes one tile of an input
// texture directly into row `tile_row` of a destination MTLBuffer laid out as a
// logical BHWC tensor: contiguous float32 [N, out_h, out_w, 3], writing 3 floats
// per pixel at 3*(row_base + y*out_w + x) where row_base = tile_row*out_h*out_w.
//
// This is the Metal analogue of the GLES 3.1 SSBO writer
// (streaming_tiles_to_tensor_batch_gl): a compute shader samples the input
// texture through the GetRotatedSubRectToRectTransformMatrix sub-rect transform
// and writes packed RGB. The emitted tensor is the LOGICAL model input
// [N,H,W,C]; downstream InferenceCalculatorMetal performs its normal
// BHWC->BPHWC4 conversion (which handles batch correctly for any N). This avoids
// the batch-1-only restriction of writing the delegate's physical PHWC4/SHWBC4
// layout directly. RGB (channels == 3) only for v1.
//
// When physical_phwc4=true, writes 4 floats/pixel (RGB + 0.0 pad) into a
// physical PHWC4 [1,H,W,4] buffer at index 4*(tile_row*out_h*out_w + y*out_w +
// x), matching the TFLite Metal delegate's N=1 input layout for direct binding.
class TiledBatchMetalWriter {
 public:
  // out_w/out_h: per-tile (= per-row) width/height. channels must be 3.
  // physical_phwc4: when true, writes 4 floats/pixel (RGB + 0.0) instead of 3.
  static absl::StatusOr<std::unique_ptr<TiledBatchMetalWriter>> Create(
      id<MTLDevice> device, int out_w, int out_h, int channels,
      BorderMode border_mode, bool physical_phwc4 = false);

  // sub_rect: the tile's RotatedRect over the source texture (same convention as
  //   the CPU path / image_to_tensor).
  // tile_row: destination batch row in [0, N).
  // alpha/beta: value-range normalization (alpha=1, beta=0 for [0,1]; Metal
  //   samples uint8 textures as [0,1]).
  // command_buffer: the command buffer to encode into (caller commits).
  // dest: the whole-batch logical [N,out_h,out_w,3] float32 MTLBuffer (e.g. the
  //   output Tensor's MtlBufferView buffer).
  absl::Status WriteTileRow(id<MTLTexture> input_texture,
                            const RotatedRect& sub_rect, int tile_row,
                            float alpha, float beta,
                            id<MTLCommandBuffer> command_buffer,
                            id<MTLBuffer> dest);

  int out_w() const { return out_w_; }
  int out_h() const { return out_h_; }

 private:
  TiledBatchMetalWriter(id<MTLDevice> device,
                        id<MTLComputePipelineState> pipeline, int out_w,
                        int out_h, bool physical_phwc4)
      : device_(device),
        pipeline_(pipeline),
        out_w_(out_w),
        out_h_(out_h),
        physical_phwc4_(physical_phwc4) {}

  id<MTLDevice> device_;
  id<MTLComputePipelineState> pipeline_;
  int out_w_ = 0;
  int out_h_ = 0;
  bool physical_phwc4_ = false;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // MEDIAPIPE_CALCULATORS_TENSOR_STREAMING_TILES_TO_TENSOR_BATCH_METAL_H_
