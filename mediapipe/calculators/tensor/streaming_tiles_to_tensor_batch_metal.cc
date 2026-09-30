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

#include <algorithm>
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

// Shared preamble for both compute shaders: uniforms struct + sampler selection
// via CLAMP_TO_ZERO define. Both layouts clamp interpolation to the source
// pixel centers of axis-aligned tiles, matching CPU crop/resize.
constexpr char kShaderPreamble[] = R"(
  #include <metal_stdlib>
  using namespace metal;

  struct Uniforms {
    int out_w;
    int out_h;
    int row_base;   // tile_row * out_h * out_w
    float alpha;
    float beta;
    float roi_inset_x;
    float roi_inset_y;
  };
)";

// Logical [N,H,W,3] writer — 3 floats/pixel, tight packed RGB. Default mode.
constexpr char kComputeShader[] = R"(
  kernel void tileWriter(
      texture2d<float, access::sample> input_texture [[texture(0)]],
      device float* output_data [[buffer(0)]],
      constant Uniforms& u [[buffer(1)]],
      constant float4x4& transform_matrix [[buffer(2)]],
      uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= u.out_w || int(gid.y) >= u.out_h) {
      return;
    }
    float nx = (float(gid.x) + 0.5) / float(u.out_w);
    float ny = (float(gid.y) + 0.5) / float(u.out_h);
    float2 inset = float2(u.roi_inset_x, u.roi_inset_y);
    float2 roi_coord = clamp(float2(nx, ny), inset, float2(1.0) - inset);
    float4 tc = float4(roi_coord, 0.0, 1.0) * transform_matrix;
    #ifdef CLAMP_TO_ZERO
    constexpr sampler linear_sampler(address::clamp_to_zero, min_filter::linear,
                                     mag_filter::linear);
    #else
    constexpr sampler linear_sampler(address::clamp_to_edge, min_filter::linear,
                                     mag_filter::linear);
    #endif
    float4 px = input_texture.sample(linear_sampler, tc.xy) * u.alpha + u.beta;
    int linear_index = u.row_base + int(gid.y) * u.out_w + int(gid.x);
    output_data[3 * linear_index + 0] = px.r;
    output_data[3 * linear_index + 1] = px.g;
    output_data[3 * linear_index + 2] = px.b;
  }
)";

// Physical PHWC4 [1,H,W,4] writer — 4 floats/pixel: RGB + 0.0 pad.
// Matches the TFLite Metal delegate's N=1 input layout for direct binding.
constexpr char kComputeShaderPhwc4[] = R"(
  kernel void tileWriter(
      texture2d<float, access::sample> input_texture [[texture(0)]],
      device float* output_data [[buffer(0)]],
      constant Uniforms& u [[buffer(1)]],
      constant float4x4& transform_matrix [[buffer(2)]],
      uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= u.out_w || int(gid.y) >= u.out_h) {
      return;
    }
    float nx = (float(gid.x) + 0.5) / float(u.out_w);
    float ny = (float(gid.y) + 0.5) / float(u.out_h);
    float2 inset = float2(u.roi_inset_x, u.roi_inset_y);
    float2 roi_coord = clamp(float2(nx, ny), inset, float2(1.0) - inset);
    float4 tc = float4(roi_coord, 0.0, 1.0) * transform_matrix;
    #ifdef CLAMP_TO_ZERO
    constexpr sampler linear_sampler(address::clamp_to_zero, min_filter::linear,
                                     mag_filter::linear);
    #else
    constexpr sampler linear_sampler(address::clamp_to_edge, min_filter::linear,
                                     mag_filter::linear);
    #endif
    float4 px = input_texture.sample(linear_sampler, tc.xy) * u.alpha + u.beta;
    int linear_index = u.row_base + int(gid.y) * u.out_w + int(gid.x);
    output_data[4 * linear_index + 0] = px.r;
    output_data[4 * linear_index + 1] = px.g;
    output_data[4 * linear_index + 2] = px.b;
    output_data[4 * linear_index + 3] = 0.0;
  }
)";

struct Uniforms {
  int out_w;
  int out_h;
  int row_base;
  float alpha;
  float beta;
  float roi_inset_x;
  float roi_inset_y;
};

}  // namespace

absl::StatusOr<std::unique_ptr<TiledBatchMetalWriter>>
TiledBatchMetalWriter::Create(id<MTLDevice> device, int out_w, int out_h,
                              int channels, BorderMode border_mode,
                              bool physical_phwc4) {
  RET_CHECK(device != nil);
  RET_CHECK_GT(out_w, 0);
  RET_CHECK_GT(out_h, 0);
  RET_CHECK_EQ(channels, 3)
      << "TiledBatchMetalWriter v1 supports RGB (channels == 3) only; got "
      << channels;

  std::string clamp_def;
  if (border_mode == BorderMode::kZero) {
    clamp_def = "\n#define CLAMP_TO_ZERO\n";
  }
  // Select shader body based on output layout mode.
  const char* shader_body =
      physical_phwc4 ? kComputeShaderPhwc4 : kComputeShader;
  const std::string source =
      absl::StrCat(clamp_def, kShaderPreamble, shader_body);

  NSError* error = nil;
  id<MTLLibrary> library =
      [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                           options:nil
                             error:&error];
  RET_CHECK(library != nil) << "Couldn't create compute library: "
                            << [[error localizedDescription] UTF8String];
  id<MTLFunction> function = [library newFunctionWithName:@"tileWriter"];
  RET_CHECK(function != nil) << "no tileWriter function";
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  RET_CHECK(pipeline != nil) << "Couldn't create compute pipeline: "
                             << [[error localizedDescription] UTF8String];

  return absl::WrapUnique(
      new TiledBatchMetalWriter(device, pipeline, out_w, out_h));
}

absl::Status TiledBatchMetalWriter::WriteTileRow(
    id<MTLTexture> input_texture, const RotatedRect& sub_rect, int tile_row,
    float alpha, float beta, id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> dest) {
  RET_CHECK(command_buffer != nil);
  RET_CHECK(dest != nil);
  RET_CHECK(input_texture != nil);
  RET_CHECK_GE(tile_row, 0);
  RET_CHECK_GT(sub_rect.width, 0.0f);
  RET_CHECK_GT(sub_rect.height, 0.0f);

  std::array<float, 16> transform_mat;
  GetRotatedSubRectToRectTransformMatrix(sub_rect, input_texture.width,
                                         input_texture.height,
                                         /*flip_horizontally=*/false,
                                         &transform_mat);
  // Preserve the reference sampling convention for rotated rects. The
  // calculator supplies axis-aligned integer pixel ROIs.
  const float inset_x = sub_rect.rotation == 0.0f
                            ? std::min(0.5f, 0.5f / sub_rect.width)
                            : 0.0f;
  const float inset_y = sub_rect.rotation == 0.0f
                            ? std::min(0.5f, 0.5f / sub_rect.height)
                            : 0.0f;
  Uniforms u{out_w_, out_h_, tile_row * out_h_ * out_w_,
             alpha, beta, inset_x, inset_y};

  id<MTLComputeCommandEncoder> encoder =
      [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:pipeline_];
  [encoder setTexture:input_texture atIndex:0];
  [encoder setBuffer:dest offset:0 atIndex:0];
  [encoder setBytes:&u length:sizeof(u) atIndex:1];
  [encoder setBytes:transform_mat.data()
             length:sizeof(transform_mat)
            atIndex:2];
  const NSUInteger tg = 8;
  MTLSize threadgroup = MTLSizeMake(tg, tg, 1);
  MTLSize grid = MTLSizeMake((out_w_ + tg - 1) / tg, (out_h_ + tg - 1) / tg, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
  [encoder endEncoding];
  return absl::OkStatus();
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
