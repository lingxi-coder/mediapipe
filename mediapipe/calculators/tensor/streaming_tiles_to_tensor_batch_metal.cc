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

#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal.h"

#if MEDIAPIPE_METAL_ENABLED

#import <Metal/Metal.h>

#include <array>
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace {

// clang-format off
// A square formed by 2 triangles (same as image_to_tensor_converter_metal).
const float kBasicSquareVertices[] = {
    -1, 1,  0, 1,
    1,  1,  0, 1,
    1,  -1, 0, 1,
    -1, 1,  0, 1,
    1,  -1, 0, 1,
    -1, -1, 0, 1,
};
const float kBasicTextureVertices[] = {
    0, 0, 0, 1,
    1, 0, 0, 1,
    1, 1, 0, 1,
    0, 0, 0, 1,
    1, 1, 0, 1,
    0, 1, 0, 1,
};
// clang-format on

constexpr char kShaderLibHeader[] = R"(
  #include <metal_stdlib>
  using namespace metal;
  struct TextureVertex {
    float4 position [[position]];
    float2 uv;
  };
)";

constexpr char kVertexShader[] = R"(
  vertex TextureVertex vertexShader(
      constant float4 *position [[buffer(0)]],
      device float4* tex_coords [[buffer(1)]],
      constant float4x4& transform_matrix [[buffer(2)]],
      uint vid [[vertex_id]]) {
    TextureVertex vert;
    vert.position = position[vid];
    vert.uv = (tex_coords[vid] * transform_matrix).xy;
    return vert;
  }
)";

// Physical output is always RGBA32Float (PHWC4, C4=4); the padded 4th channel
// is written 0.
constexpr char kFragmentShader[] = R"(
  fragment float4 fragmentShader(TextureVertex vertex_output [[stage_in]],
                                 texture2d<float> texture [[texture(0)]],
                                 constant float* parameters [[buffer(1)]]) {
    const float alpha = parameters[0];
    const float beta = parameters[1];
    #ifdef CLAMP_TO_ZERO
    constexpr sampler linear_sampler(address::clamp_to_zero, min_filter::linear,
      mag_filter::linear);
    #endif
    #ifdef CLAMP_TO_EDGE
    constexpr sampler linear_sampler(address::clamp_to_edge, min_filter::linear,
      mag_filter::linear);
    #endif
    float4 texture_pixel = texture.sample(linear_sampler, vertex_output.uv);
    return float4(alpha * texture_pixel.rgb + beta, 0);
  }
)";

constexpr int kNumPhysicalChannels = 4;

absl::Status MakePipelineState(id<MTLDevice> device, BorderMode border_mode,
                               id<MTLRenderPipelineState>* pipeline_state) {
  std::string clamp_def;
  switch (border_mode) {
    case BorderMode::kReplicate:
      clamp_def = "\n#define CLAMP_TO_EDGE\n";
      break;
    case BorderMode::kZero:
      clamp_def = "\n#define CLAMP_TO_ZERO\n";
      break;
  }
  const std::string shader_lib =
      absl::StrCat(kShaderLibHeader, clamp_def, kVertexShader, kFragmentShader);
  NSError* error = nil;
  id<MTLLibrary> library = [device
      newLibraryWithSource:[NSString stringWithUTF8String:shader_lib.c_str()]
                   options:nil
                     error:&error];
  RET_CHECK(library != nil) << "Couldn't create shader library: "
                            << [[error localizedDescription] UTF8String];
  id<MTLFunction> vertex_function = [library newFunctionWithName:@"vertexShader"];
  RET_CHECK(vertex_function != nil) << "no vertexShader";
  id<MTLFunction> fragment_function =
      [library newFunctionWithName:@"fragmentShader"];
  RET_CHECK(fragment_function != nil) << "no fragmentShader";

  MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
  desc.vertexFunction = vertex_function;
  desc.fragmentFunction = fragment_function;
  desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
  *pipeline_state = [device newRenderPipelineStateWithDescriptor:desc
                                                           error:&error];
  RET_CHECK(error == nil) << "Couldn't create pipeline state: "
                          << [[error localizedDescription] UTF8String];
  return absl::OkStatus();
}

}  // namespace

TiledBatchMetalWriter::TiledBatchMetalWriter(
    id<MTLDevice> device, id<MTLRenderPipelineState> pipeline_state, int out_w,
    int out_h, size_t texture_offset_alignment)
    : device_(device),
      pipeline_state_(pipeline_state),
      out_w_(out_w),
      out_h_(out_h),
      bytes_per_pixel_row_(static_cast<size_t>(out_w) * kNumPhysicalChannels *
                           sizeof(float)),
      row_bytes_(static_cast<size_t>(out_h) * out_w * kNumPhysicalChannels *
                 sizeof(float)),
      texture_offset_alignment_(texture_offset_alignment) {
  positions_buffer_ =
      [device_ newBufferWithBytes:kBasicSquareVertices
                           length:sizeof(kBasicSquareVertices)
                          options:MTLResourceOptionCPUCacheModeDefault];
  tex_coords_buffer_ =
      [device_ newBufferWithBytes:kBasicTextureVertices
                           length:sizeof(kBasicTextureVertices)
                          options:MTLResourceOptionCPUCacheModeDefault];
}

absl::StatusOr<std::unique_ptr<TiledBatchMetalWriter>>
TiledBatchMetalWriter::Create(id<MTLDevice> device, int out_w, int out_h,
                              int channels, BorderMode border_mode) {
  RET_CHECK(device != nil);
  RET_CHECK_GT(out_w, 0);
  RET_CHECK_GT(out_h, 0);
  RET_CHECK_EQ(channels, 3)
      << "TiledBatchMetalWriter v1 supports logical RGB (channels == 3) only; "
         "got "
      << channels;
  id<MTLRenderPipelineState> pipeline_state = nil;
  MP_RETURN_IF_ERROR(MakePipelineState(device, border_mode, &pipeline_state));
  const size_t alignment = [device
      minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatRGBA32Float];
  return absl::WrapUnique(new TiledBatchMetalWriter(
      device, pipeline_state, out_w, out_h, alignment == 0 ? 1 : alignment));
}

absl::Status TiledBatchMetalWriter::RenderTileRow(
    id<MTLTexture> input_texture, const RotatedRect& sub_rect, int tile_row,
    float alpha, float beta, id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> dest) {
  RET_CHECK(command_buffer != nil);
  RET_CHECK(dest != nil);
  RET_CHECK_GE(tile_row, 0);

  const size_t offset = static_cast<size_t>(tile_row) * row_bytes_;
  // newTextureWithDescriptor:offset: requires the offset to be a multiple of the
  // device's minimum linear texture alignment for the pixel format. With
  // contiguous PHWC4 rows this holds for typical H*W; assert so a bad config
  // fails loudly instead of returning a nil texture.
  RET_CHECK_EQ(offset % texture_offset_alignment_, 0u)
      << "row byte offset " << offset << " is not a multiple of the Metal "
      << "linear-texture alignment " << texture_offset_alignment_
      << " for out_w=" << out_w_ << " out_h=" << out_h_
      << "; pad rows to alignment (deferred) for these dimensions";

  MTLTextureDescriptor* texture_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                   width:out_w_
                                  height:out_h_
                               mipmapped:NO];
  texture_desc.usage = MTLTextureUsageRenderTarget;
  id<MTLTexture> output_texture =
      [dest newTextureWithDescriptor:texture_desc
                              offset:offset
                         bytesPerRow:bytes_per_pixel_row_];
  RET_CHECK(output_texture != nil)
      << "failed aliasing output texture over dest buffer at offset " << offset;

  std::array<float, 16> transform_mat;
  GetRotatedSubRectToRectTransformMatrix(sub_rect, input_texture.width,
                                         input_texture.height,
                                         /*flip_horizontally=*/false,
                                         &transform_mat);
  id<MTLBuffer> transform_mat_buffer =
      [device_ newBufferWithBytes:&transform_mat
                           length:sizeof(transform_mat)
                          options:MTLResourceOptionCPUCacheModeDefault];
  float parameters[] = {alpha, beta};

  MTLRenderPassDescriptor* render_pass_desc =
      [MTLRenderPassDescriptor renderPassDescriptor];
  render_pass_desc.colorAttachments[0].texture = output_texture;
  render_pass_desc.colorAttachments[0].storeAction = MTLStoreActionStore;
  render_pass_desc.colorAttachments[0].loadAction = MTLLoadActionClear;

  id<MTLRenderCommandEncoder> command_encoder =
      [command_buffer renderCommandEncoderWithDescriptor:render_pass_desc];
  [command_encoder setRenderPipelineState:pipeline_state_];
  [command_encoder setVertexBuffer:positions_buffer_ offset:0 atIndex:0];
  [command_encoder setVertexBuffer:tex_coords_buffer_ offset:0 atIndex:1];
  [command_encoder setVertexBuffer:transform_mat_buffer offset:0 atIndex:2];
  [command_encoder setFragmentTexture:input_texture atIndex:0];
  [command_encoder setFragmentBytes:&parameters
                             length:sizeof(parameters)
                            atIndex:1];
  [command_encoder drawPrimitives:MTLPrimitiveTypeTriangle
                      vertexStart:0
                      vertexCount:6];
  [command_encoder endEncoding];
  return absl::OkStatus();
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
