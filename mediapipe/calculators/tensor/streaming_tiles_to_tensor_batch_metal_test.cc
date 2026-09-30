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
// Exercises Metal tile sampling and logical BHWC / physical PHWC4 layouts.

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

id<MTLTexture> MakeRgbaTexture(id<MTLDevice> device, int w, int h,
                               const std::vector<uint8_t>& pixels) {
  MTLTextureDescriptor* desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:w
                                  height:h
                               mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  [texture replaceRegion:MTLRegionMake2D(0, 0, w, h)
             mipmapLevel:0
               withBytes:pixels.data()
             bytesPerRow:w * 4];
  return texture;
}

// Uniform color makes any resize exact: each output pixel is color/255.
id<MTLTexture> MakeUniformTexture(id<MTLDevice> device, int w, int h, uint8_t r,
                                  uint8_t g, uint8_t b) {
  std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 4);
  for (size_t i = 0; i < pixels.size(); i += 4) {
    pixels[i] = r;
    pixels[i + 1] = g;
    pixels[i + 2] = b;
    pixels[i + 3] = 255;
  }
  return MakeRgbaTexture(device, w, h, pixels);
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
  constexpr int kOutW = 8, kOutH = 8;
  constexpr int kBatch = 2;
  constexpr int kC = 3;  // logical BHWC, tight RGB
  constexpr uint8_t kR = 30, kG = 120, kB = 210;

  std::vector<float> cpu;
  @autoreleasepool {
    id<MTLTexture> input = MakeUniformTexture(device, kSrcW, kSrcH, kR, kG, kB);
    auto writer_or = TiledBatchMetalWriter::Create(device, kOutW, kOutH,
                                                   /*channels=*/kC,
                                                   BorderMode::kReplicate);
    MP_ASSERT_OK(writer_or);
    auto writer = std::move(writer_or).value();

    // Logical BHWC batch tensor: [N, H, W, 3] (tight, batch-outermost).
    Tensor tensor(Tensor::ElementType::kFloat32,
                  Tensor::Shape{kBatch, kOutH, kOutW, kC});
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    {
      auto wv = MtlBufferView::GetWriteView(tensor, cb);
      MP_ASSERT_OK(writer->WriteTileRow(
          input, PixelRoiRect(0, 0, kSrcW, kSrcH), /*tile_row=*/0,
          /*alpha=*/1.0f, /*beta=*/0.0f, cb, wv.buffer()));
      MP_ASSERT_OK(writer->WriteTileRow(
          input, PixelRoiRect(4, 2, 8, 8), /*tile_row=*/1,
          /*alpha=*/1.0f, /*beta=*/0.0f, cb, wv.buffer()));
    }
    [cb commit];
    [cb waitUntilCompleted];

    auto view = tensor.GetCpuReadView();
    const float* buf = view.buffer<float>();
    cpu.assign(buf, buf + tensor.shape().num_elements());
  }

  ASSERT_EQ(cpu.size(), static_cast<size_t>(kBatch * kOutH * kOutW * kC));
  const float er = kR / 255.0f, eg = kG / 255.0f, eb = kB / 255.0f;
  for (int row = 0; row < kBatch; ++row) {
    for (int p = 0; p < kOutH * kOutW; ++p) {
      const int base = (row * kOutH * kOutW + p) * kC;
      EXPECT_NEAR(cpu[base + 0], er, 2e-3) << "row " << row << " px " << p;
      EXPECT_NEAR(cpu[base + 1], eg, 2e-3) << "row " << row << " px " << p;
      EXPECT_NEAR(cpu[base + 2], eb, 2e-3) << "row " << row << " px " << p;
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

// Physical PHWC4 mode: 4 floats/pixel, RGB + 0.0 pad.  Verifies that the
// padded 4th channel is exactly 0.0 and the RGB channels match the expected
// normalized color values (within GPU bilinear tolerance).
TEST(StreamingTilesToTensorBatchMetalTest, PhysicalPhwc4WriterWritesPaddedRgba) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ASSERT_NE(device, nil) << "no Metal device";
  id<MTLCommandQueue> queue = [device newCommandQueue];

  constexpr int kSrcW = 16, kSrcH = 16;
  constexpr int kOutW = 4, kOutH = 4;
  constexpr int kPhysC = 4;  // physical PHWC4 stride
  constexpr uint8_t kR = 80, kG = 160, kB = 200;

  std::vector<float> cpu;
  @autoreleasepool {
    id<MTLTexture> input = MakeUniformTexture(device, kSrcW, kSrcH, kR, kG, kB);

    // Create writer in physical PHWC4 mode; channels is still the logical 3.
    auto writer_or = TiledBatchMetalWriter::Create(device, kOutW, kOutH,
                                                   /*channels=*/3,
                                                   BorderMode::kReplicate,
                                                   /*physical_phwc4=*/true);
    MP_ASSERT_OK(writer_or);
    auto writer = std::move(writer_or).value();

    // Physical [1, kOutH, kOutW, 4] tensor (batch==1, 4 channels).
    Tensor tensor(Tensor::ElementType::kFloat32,
                  Tensor::Shape{1, kOutH, kOutW, kPhysC});

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    {
      auto wv = MtlBufferView::GetWriteView(tensor, cb);
      // Full-frame crop: sub_rect covers the whole source image.
      MP_ASSERT_OK(writer->WriteTileRow(
          input, PixelRoiRect(0, 0, kSrcW, kSrcH), /*tile_row=*/0,
          /*alpha=*/1.0f, /*beta=*/0.0f, cb, wv.buffer()));
    }
    [cb commit];
    [cb waitUntilCompleted];

    auto view = tensor.GetCpuReadView();
    const float* buf = view.buffer<float>();
    cpu.assign(buf, buf + tensor.shape().num_elements());
  }

  ASSERT_EQ(cpu.size(),
            static_cast<size_t>(1 * kOutH * kOutW * kPhysC));

  const float er = kR / 255.0f, eg = kG / 255.0f, eb = kB / 255.0f;
  for (int p = 0; p < kOutH * kOutW; ++p) {
    const int base = p * kPhysC;
    EXPECT_NEAR(cpu[base + 0], er, 2e-2) << "px " << p << " R";
    EXPECT_NEAR(cpu[base + 1], eg, 2e-2) << "px " << p << " G";
    EXPECT_NEAR(cpu[base + 2], eb, 2e-2) << "px " << p << " B";
    EXPECT_EQ(cpu[base + 3], 0.0f) << "px " << p << " pad channel must be 0";
  }
}

class StreamingTilesMetalCropTest : public testing::TestWithParam<bool> {};

TEST_P(StreamingTilesMetalCropTest, UpsamplingReplicatesCropEdges) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ASSERT_NE(device, nil);
  id<MTLCommandQueue> queue = [device newCommandQueue];
  constexpr int kSrcW = 6, kSrcH = 5, kOutW = 4, kOutH = 4;
  const bool physical_phwc4 = GetParam();
  const int stride = physical_phwc4 ? 4 : 3;

  // A nonuniform 2x2 tile surrounded by unrelated magenta pixels. Its four
  // corners are black, red, green and white. Resizing that crop alone gives
  // R=tx, G=ty, B=tx*ty, with tx/ty = {0, .25, .75, 1} for a 4x4 output.
  std::vector<uint8_t> pixels(kSrcW * kSrcH * 4);
  for (int y = 0; y < kSrcH; ++y) {
    for (int x = 0; x < kSrcW; ++x) {
      const int offset = (y * kSrcW + x) * 4;
      const bool inside = x >= 2 && x < 4 && y >= 1 && y < 3;
      pixels[offset] = !inside || x == 3 ? 255 : 0;
      pixels[offset + 1] = inside && y == 2 ? 255 : 0;
      pixels[offset + 2] = !inside || (x == 3 && y == 2) ? 255 : 0;
      pixels[offset + 3] = 255;
    }
  }
  id<MTLTexture> input = MakeRgbaTexture(device, kSrcW, kSrcH, pixels);
  auto writer_or = TiledBatchMetalWriter::Create(
      device, kOutW, kOutH, /*channels=*/3, BorderMode::kReplicate,
      physical_phwc4);
  MP_ASSERT_OK(writer_or);
  auto writer = std::move(writer_or).value();
  const std::array<RotatedRect, 3> rois = {
      PixelRoiRect(2, 1, 2, 2), PixelRoiRect(3, 2, 1, 1),
      PixelRoiRect(0, 0, kSrcW, kSrcH)};
  constexpr std::array<float, 4> kWeights = {0.0f, 0.25f, 0.75f, 1.0f};
  for (int sample = 0; sample < rois.size(); ++sample) {
    SCOPED_TRACE(sample);
    // Physical delegate tensors have N=1. Also test single-pixel crops and
    // the zero-fill dispatch used for padding with the same writer instance.
    Tensor tensor(Tensor::ElementType::kFloat32,
                  Tensor::Shape{1, kOutH, kOutW, stride});
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    {
      auto write = MtlBufferView::GetWriteView(tensor, cb);
      MP_ASSERT_OK(writer->WriteTileRow(
          input, rois[sample], /*tile_row=*/0,
          /*alpha=*/sample == 2 ? 0.0f : 1.0f, /*beta=*/0.0f, cb,
          write.buffer()));
    }
    [cb commit];
    [cb waitUntilCompleted];
    ASSERT_EQ(cb.error, nil);
    auto read = tensor.GetCpuReadView();
    const float* values = read.buffer<float>();
    for (int y = 0; y < kOutH; ++y) {
      for (int x = 0; x < kOutW; ++x) {
        const int offset = (y * kOutW + x) * stride;
        const std::array<float, 3> expected =
            sample == 0
                ? std::array<float, 3>{kWeights[x], kWeights[y],
                                        kWeights[x] * kWeights[y]}
                : std::array<float, 3>{sample == 1 ? 1.0f : 0.0f,
                                        sample == 1 ? 1.0f : 0.0f,
                                        sample == 1 ? 1.0f : 0.0f};
        for (int c = 0; c < 3; ++c) {
          // CPU resize rounds to uint8 before normalization; allow one LSB.
          EXPECT_NEAR(values[offset + c], expected[c], 1.0f / 255.0f)
              << "x=" << x << " y=" << y << " channel=" << c;
        }
        if (physical_phwc4) EXPECT_EQ(values[offset + 3], 0.0f);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(LogicalAndPhysical, StreamingTilesMetalCropTest,
                         testing::Values(false, true));

}  // namespace
}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
