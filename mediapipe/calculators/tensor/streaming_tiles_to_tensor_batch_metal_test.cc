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
//
// Runs on this Mac (Metal). Unit-tests TiledBatchMetalWriter: render two tiles
// of a uniform-color texture into rows 0 and 1 of a physical [2,H,W,4] tensor
// and verify each row.

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_METAL_ENABLED

#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/formats/tensor_mtl_buffer_view.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

// A uniform-color RGBA8Unorm texture; uniform color makes any resize exact, so
// each output pixel equals color/255 (Metal samples Unorm textures as [0,1]).
id<MTLTexture> MakeUniformTexture(id<MTLDevice> device, int w, int h, uint8_t r,
                                  uint8_t g, uint8_t b) {
  MTLTextureDescriptor* desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:w
                                  height:h
                               mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 4);
  for (size_t i = 0; i < pixels.size(); i += 4) {
    pixels[i] = r;
    pixels[i + 1] = g;
    pixels[i + 2] = b;
    pixels[i + 3] = 255;
  }
  [texture replaceRegion:MTLRegionMake2D(0, 0, w, h)
             mipmapLevel:0
               withBytes:pixels.data()
             bytesPerRow:w * 4];
  return texture;
}

RotatedRect PixelRoiRect(int x, int y, int rw, int rh) {
  RotatedRect rect;
  rect.center_x = x + rw / 2.0f;
  rect.center_y = y + rh / 2.0f;
  rect.width = rw;
  rect.height = rh;
  rect.rotation = 0.0f;
  return rect;
}

TEST(StreamingTilesToTensorBatchMetalTest, WritesEachTileIntoItsBatchRow) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ASSERT_NE(device, nil) << "no Metal device";
  id<MTLCommandQueue> queue = [device newCommandQueue];

  constexpr int kSrcW = 16, kSrcH = 16;
  constexpr int kOutW = 8, kOutH = 8;  // 8*8*4*4 = 1024 B/row (256-aligned)
  constexpr int kBatch = 2;
  constexpr int kC4 = 4;
  constexpr uint8_t kR = 30, kG = 120, kB = 210;

  std::vector<float> cpu;
  @autoreleasepool {
    id<MTLTexture> input = MakeUniformTexture(device, kSrcW, kSrcH, kR, kG, kB);
    auto writer_or = TiledBatchMetalWriter::Create(device, kOutW, kOutH,
                                                   /*channels=*/3,
                                                   BorderMode::kReplicate);
    MP_ASSERT_OK(writer_or);
    auto writer = std::move(writer_or).value();

    // Physical PHWC4 batch tensor: [N, H, W, 4].
    Tensor tensor(Tensor::ElementType::kFloat32,
                  Tensor::Shape{kBatch, kOutH, kOutW, kC4});
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    {
      auto wv = MtlBufferView::GetWriteView(tensor, cb);
      MP_ASSERT_OK(writer->RenderTileRow(
          input, PixelRoiRect(0, 0, kSrcW, kSrcH), /*tile_row=*/0,
          /*alpha=*/1.0f, /*beta=*/0.0f, cb, wv.buffer()));
      MP_ASSERT_OK(writer->RenderTileRow(
          input, PixelRoiRect(4, 2, 8, 8), /*tile_row=*/1,
          /*alpha=*/1.0f, /*beta=*/0.0f, cb, wv.buffer()));
    }
    [cb commit];
    [cb waitUntilCompleted];

    auto view = tensor.GetCpuReadView();
    const float* buf = view.buffer<float>();
    cpu.assign(buf, buf + tensor.shape().num_elements());
  }

  ASSERT_EQ(cpu.size(), static_cast<size_t>(kBatch * kOutH * kOutW * kC4));
  const float er = kR / 255.0f, eg = kG / 255.0f, eb = kB / 255.0f;
  for (int row = 0; row < kBatch; ++row) {
    for (int p = 0; p < kOutH * kOutW; ++p) {
      const int base = (row * kOutH * kOutW + p) * kC4;
      EXPECT_NEAR(cpu[base + 0], er, 2e-3) << "row " << row << " px " << p;
      EXPECT_NEAR(cpu[base + 1], eg, 2e-3) << "row " << row << " px " << p;
      EXPECT_NEAR(cpu[base + 2], eb, 2e-3) << "row " << row << " px " << p;
      EXPECT_NEAR(cpu[base + 3], 0.0f, 2e-3) << "padded chan row " << row;
    }
  }
}

// The Metal sampling matrix must match the CPU converter convention: a
// full-frame sub-rect maps normalized output coords to themselves.
TEST(StreamingTilesToTensorBatchMetalTest, FullFrameMatrixIsIdentity) {
  constexpr int kW = 16, kH = 16;
  std::array<float, 16> mat;
  GetRotatedSubRectToRectTransformMatrix(PixelRoiRect(0, 0, kW, kH), kW, kH,
                                         /*flip_horizontally=*/false, &mat);
  EXPECT_NEAR(mat[0], 1.0f, 1e-5);
  EXPECT_NEAR(mat[5], 1.0f, 1e-5);
  EXPECT_NEAR(mat[3], 0.0f, 1e-5);
  EXPECT_NEAR(mat[7], 0.0f, 1e-5);
}

}  // namespace
}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
