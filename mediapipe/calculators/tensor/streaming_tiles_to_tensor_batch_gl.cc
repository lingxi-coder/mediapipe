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

#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_gl.h"

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

#include <array>
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter_gl_utils.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_context.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/command_queue.h"
#include "tensorflow/lite/delegates/gpu/gl/converters/util.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"

namespace mediapipe {
namespace {

// Identical to the reference converter's shader EXCEPT for the `row_base`
// uniform: the linear write index is offset by row_base so this dispatch fills
// row `tile_row` of an [N,H,W,3] batch SSBO instead of the start of the buffer.
constexpr char kShaderCode[] = R"(
layout(std430) buffer;

precision highp float;

layout(binding = 0) writeonly buffer B0 {
  float elements[];
} output_data;

uniform ivec2 out_size;
uniform float alpha;
uniform float beta;
uniform int row_base;        // tile_row * out_height * out_width
uniform mat4 transform_matrix;
uniform mediump sampler2D input_data;

void main() {
    int out_width = out_size.x;
    int out_height = out_size.y;

    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= out_width || gid.y >= out_height) {
        return;
    }

    // transform from output range to [0, 1]
    float normal_x = (float(gid.x) + 0.5f) / float(out_width);
    float normal_y = (float(gid.y) + 0.5f) / float(out_height);
    vec4 tc = vec4(normal_x, normal_y, 0.0, 1.0);

    // Apply transformation from roi coordinates to original image coordinates.
    tc = transform_matrix * tc;
#ifdef INPUT_STARTS_AT_BOTTOM
    // Opengl texture sampler has origin in lower left corner,
    // so we invert y coordinate.
    tc.y = 1.0f - tc.y;
#endif  // INPUT_STARTS_AT_BOTTOM
    vec4 src_value = alpha * texture(input_data, tc.xy) + beta;

#ifdef CUSTOM_ZERO_BORDER_MODE
    float out_of_bounds =
      float(tc.x < 0.0 || tc.x > 1.0 || tc.y < 0.0 || tc.y > 1.0);
    src_value = mix(src_value, vec4(0.0, 0.0, 0.0, 0.0), out_of_bounds);
#endif

    // Write into this tile's batch row: row_base offsets the linear pixel index
    // by tile_row * out_height * out_width.
    int linear_index = row_base + gid.y * out_width + gid.x;

    // output_data.elements is populated as though it contains vec3 elements.
    int first_component_index = 3 * linear_index;
    output_data.elements[first_component_index] = src_value.r;
    output_data.elements[first_component_index + 1] = src_value.g;
    output_data.elements[first_component_index + 2] = src_value.b;
}
)";

absl::Status SetMat4x4(const tflite::gpu::gl::GlProgram& program,
                       const std::string& name, float* data) {
  GLint uniform_id;
  MP_RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glGetUniformLocation, &uniform_id,
                                        program.id(), name.c_str()));
  return TFLITE_GPU_CALL_GL(glProgramUniformMatrix4fv, program.id(), uniform_id,
                            1, GL_TRUE, data);
}

}  // namespace

absl::StatusOr<std::unique_ptr<TiledBatchGlWriter>> TiledBatchGlWriter::Create(
    const mediapipe::GlContext& gl_context, int out_w, int out_h, int channels,
    BorderMode border_mode, bool input_starts_at_bottom) {
  RET_CHECK_GT(out_w, 0);
  RET_CHECK_GT(out_h, 0);
  RET_CHECK_EQ(channels, 3)
      << "TiledBatchGlWriter v1 supports RGB (channels == 3) only; got "
      << channels;

  const bool use_custom_zero_border =
      border_mode == BorderMode::kZero && !IsGlClampToBorderSupported(gl_context);

  const tflite::gpu::uint3 workgroup_size = {8, 8, 1};
  std::string starts_at_bottom_def;
  if (input_starts_at_bottom) {
    starts_at_bottom_def = R"(
      #define INPUT_STARTS_AT_BOTTOM;
    )";
  }
  std::string custom_zero_border_mode_def;
  if (use_custom_zero_border) {
    custom_zero_border_mode_def = R"(
      #define CUSTOM_ZERO_BORDER_MODE
    )";
  }
  const std::string full_shader_source = absl::StrCat(
      tflite::gpu::gl::GetShaderHeader(workgroup_size), starts_at_bottom_def,
      custom_zero_border_mode_def, kShaderCode);

  tflite::gpu::gl::GlShader shader;
  MP_RETURN_IF_ERROR(tflite::gpu::gl::GlShader::CompileShader(
      GL_COMPUTE_SHADER, full_shader_source, &shader));
  tflite::gpu::gl::GlProgram program;
  MP_RETURN_IF_ERROR(
      tflite::gpu::gl::GlProgram::CreateWithShader(shader, &program));

  return absl::WrapUnique(new TiledBatchGlWriter(
      std::move(program), workgroup_size, out_w, out_h, use_custom_zero_border,
      border_mode));
}

absl::Status TiledBatchGlWriter::WriteTileRow(
    const tflite::gpu::gl::GlTexture& texture,
    const tflite::gpu::HW& texture_size, const RotatedRect& sub_rect,
    int tile_row, float alpha, float beta,
    tflite::gpu::gl::CommandQueue* command_queue,
    tflite::gpu::gl::GlBuffer* dest) {
  RET_CHECK_GE(tile_row, 0);
  std::array<float, 16> transform_mat;
  GetRotatedSubRectToRectTransformMatrix(sub_rect, texture_size.w,
                                         texture_size.h,
                                         /*flip_horizontally=*/false,
                                         &transform_mat);
  MP_RETURN_IF_ERROR(texture.BindAsSampler2D(0));

  // a) Filtering.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // b) Clamping.
  switch (border_mode_) {
    case BorderMode::kReplicate: {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      break;
    }
    case BorderMode::kZero: {
      if (!use_custom_zero_border_) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR,
                         std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}.data());
      }
      break;
    }
  }

  MP_RETURN_IF_ERROR(dest->BindToIndex(0));
  MP_RETURN_IF_ERROR(program_.SetParameter({"input_data", 0}));
  MP_RETURN_IF_ERROR(
      SetMat4x4(program_, "transform_matrix", transform_mat.data()));
  MP_RETURN_IF_ERROR(program_.SetParameter(
      {"out_size", tflite::gpu::int2(out_w_, out_h_)}));
  MP_RETURN_IF_ERROR(program_.SetParameter({"alpha", alpha}));
  MP_RETURN_IF_ERROR(program_.SetParameter({"beta", beta}));
  // The one tiling-specific uniform: offset the linear write index to this
  // tile's batch row. row_base = tile_row * out_height * out_width.
  MP_RETURN_IF_ERROR(program_.SetParameter(
      {"row_base", tile_row * out_h_ * out_w_}));

  tflite::gpu::uint3 num_workgroups = tflite::gpu::DivideRoundUp(
      tflite::gpu::uint3{static_cast<uint32_t>(out_w_),
                         static_cast<uint32_t>(out_h_), 1},
      workgroup_size_);
  MP_RETURN_IF_ERROR(command_queue->Dispatch(program_, num_workgroups));

  // Resetting to MediaPipe texture param defaults.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  return absl::OkStatus();
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
