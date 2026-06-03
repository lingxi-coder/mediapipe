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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_STREAMING_TILES_TO_TENSOR_BATCH_GL_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_STREAMING_TILES_TO_TENSOR_BATCH_GL_H_

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

#include <memory>

#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"  // BorderMode
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"      // RotatedRect
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gl_context.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/command_queue.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"

namespace mediapipe {

// GLES 3.1 compute writer that crops/resizes/normalizes one tile of an input
// texture directly into row `tile_row` of a single [N,H,W,C] float SSBO bound
// at binding 0. Adapted from ImageToTensorGlBufferConverter
// (image_to_tensor_converter_gl_buffer.cc): the one structural change is a
// `row_base` shader uniform so each tile writes into its own batch row instead
// of the start of the buffer. The caller dispatches every valid tile into the
// SAME destination buffer (one buffer per batch) before reading it back, so
// WriteTileRow never clears the destination.
//
// RGB (channels == 3) only for v1, matching the reference converter; the shader
// writes 3 floats per pixel.
class TiledBatchGlWriter {
 public:
  // out_w/out_h: per-tile (= per-row) tensor width/height. channels must be 3.
  // border_mode/input_starts_at_bottom mirror the reference converter's options
  // and select the same shader variants.
  static absl::StatusOr<std::unique_ptr<TiledBatchGlWriter>> Create(
      const mediapipe::GlContext& gl_context, int out_w, int out_h, int channels,
      BorderMode border_mode, bool input_starts_at_bottom);

  // sub_rect: the tile's RotatedRect over the source texture (built from the
  //   effective pixel ROI; same convention as the CPU path / image_to_tensor).
  // texture_size: source texture (H, W) in pixels.
  // tile_row: destination batch row in [0, N).
  // alpha/beta: value-range normalization (e.g. 1/255, 0 for [0,1]).
  // command_queue: GLES 3.1 command queue (owned by the caller).
  // dest: the whole-batch [N,H,W,C] SSBO (obtained from
  //   Tensor::GetOpenGlBufferWriteView()); offset 0, covers all rows.
  absl::Status WriteTileRow(const tflite::gpu::gl::GlTexture& texture,
                            const tflite::gpu::HW& texture_size,
                            const RotatedRect& sub_rect, int tile_row,
                            float alpha, float beta,
                            tflite::gpu::gl::CommandQueue* command_queue,
                            tflite::gpu::gl::GlBuffer* dest);

  int out_w() const { return out_w_; }
  int out_h() const { return out_h_; }

 private:
  TiledBatchGlWriter(tflite::gpu::gl::GlProgram program,
                     tflite::gpu::uint3 workgroup_size, int out_w, int out_h,
                     bool use_custom_zero_border, BorderMode border_mode)
      : program_(std::move(program)),
        workgroup_size_(workgroup_size),
        out_w_(out_w),
        out_h_(out_h),
        use_custom_zero_border_(use_custom_zero_border),
        border_mode_(border_mode) {}

  tflite::gpu::gl::GlProgram program_;
  tflite::gpu::uint3 workgroup_size_;
  int out_w_ = 0;
  int out_h_ = 0;
  bool use_custom_zero_border_ = false;
  BorderMode border_mode_ = BorderMode::kReplicate;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#endif  // MEDIAPIPE_CALCULATORS_TENSOR_STREAMING_TILES_TO_TENSOR_BATCH_GL_H_
