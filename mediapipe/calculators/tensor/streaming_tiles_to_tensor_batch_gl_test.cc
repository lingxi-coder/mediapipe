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
// GPU env required: GLES 3.1. This test exercises the live compute shader and
// must be built/run on a GLES 3.1 device/emulator or Linux EGL GPU (NOT under
// --define MEDIAPIPE_DISABLE_GPU=1, and not on Apple where the GLES path is
// compiled out). On unsupported configs the whole TU compiles to nothing.

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_gl.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_context.h"
#include "tflite/delegates/gpu/common/types.h"
#include "tflite/delegates/gpu/gl/command_queue.h"
#include "tflite/delegates/gpu/gl/gl_buffer.h"
#include "tflite/delegates/gpu/gl/gl_texture.h"
#include "tflite/delegates/gpu/gl/request_gpu_info.h"

namespace mediapipe {
namespace {

GLuint MakeRgbaTexture(int w, int h, const std::vector<uint8_t>& pixels) {
  GLuint texture = 0;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               pixels.data());
  glBindTexture(GL_TEXTURE_2D, 0);
  return texture;
}

// Uniform color makes bilinear resampling exact regardless of the sub-rect.
GLuint MakeUniformRgbaTexture(int w, int h, uint8_t r, uint8_t g, uint8_t b) {
  std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 4);
  for (size_t i = 0; i < pixels.size(); i += 4) {
    pixels[i] = r;
    pixels[i + 1] = g;
    pixels[i + 2] = b;
    pixels[i + 3] = 255;
  }
  return MakeRgbaTexture(w, h, pixels);
}

// A RotatedRect covering an axis-aligned region [x, x+rw) x [y, y+rh) of the
// source texture, matching the calculator's effective-pixel-ROI convention.
RotatedRect PixelRoiRect(int x, int y, int rw, int rh) {
  RotatedRect rect;
  rect.center_x = x + rw / 2.0f;
  rect.center_y = y + rh / 2.0f;
  rect.width = rw;
  rect.height = rh;
  rect.rotation = 0.0f;
  return rect;
}

TEST(StreamingTilesToTensorBatchGlTest, WritesEachTileIntoItsBatchRow) {
  auto status_or_context = mediapipe::GlContext::Create(nullptr, false);
  MP_ASSERT_OK(status_or_context);
  auto context = status_or_context.value();

  constexpr int kSrcW = 8, kSrcH = 6;
  constexpr int kOutW = 4, kOutH = 4, kChannels = 3;
  constexpr int kBatch = 2;
  constexpr uint8_t kR = 30, kG = 120, kB = 210;

  std::vector<float> cpu;  // read back here
  context->Run([&]() {
    tflite::gpu::GpuInfo gpu_info;
    MP_ASSERT_OK(tflite::gpu::gl::RequestGpuInfo(&gpu_info));
    ASSERT_TRUE(gpu_info.IsApiOpenGl31OrAbove());
    auto command_queue = tflite::gpu::gl::NewCommandQueue(gpu_info);

    GLuint tex_name = MakeUniformRgbaTexture(kSrcW, kSrcH, kR, kG, kB);
    tflite::gpu::gl::GlTexture input_texture(
        GL_TEXTURE_2D, tex_name, GL_RGBA,
        kSrcW * kSrcH * 4 * sizeof(uint8_t), /*layer=*/0, /*owned=*/true);

    auto writer_or = TiledBatchGlWriter::Create(
        *context, kOutW, kOutH, kChannels, BorderMode::kReplicate,
        /*input_starts_at_bottom=*/false);
    MP_ASSERT_OK(writer_or);
    auto writer = std::move(writer_or).value();

    Tensor tensor(Tensor::ElementType::kFloat32,
                  Tensor::Shape{kBatch, kOutH, kOutW, kChannels});
    {
      auto wv = tensor.GetOpenGlBufferWriteView();
      tflite::gpu::gl::GlBuffer dest(GL_SHADER_STORAGE_BUFFER, wv.name(),
                                     tensor.bytes(), /*offset=*/0,
                                     /*has_ownership=*/false);
      // Row 0: full-frame sub-rect. Row 1: a sub-region. Both sample the same
      // uniform color. GL samples an unsigned-normalized RGBA8 texture as
      // [0,1] already, so alpha=1 (NOT 1/255) yields color/255 — matching the
      // CPU path's uint8 -> *(1/255) -> [0,1].
      MP_ASSERT_OK(writer->WriteTileRow(
          input_texture, tflite::gpu::HW(kSrcH, kSrcW),
          PixelRoiRect(0, 0, kSrcW, kSrcH), /*tile_row=*/0,
          /*alpha=*/1.0f, /*beta=*/0.0f, command_queue.get(), &dest));
      MP_ASSERT_OK(writer->WriteTileRow(
          input_texture, tflite::gpu::HW(kSrcH, kSrcW),
          PixelRoiRect(2, 1, 4, 4), /*tile_row=*/1,
          /*alpha=*/1.0f, /*beta=*/0.0f, command_queue.get(), &dest));
      MP_ASSERT_OK(command_queue->WaitForCompletion());
    }  // write view destroyed -> GL fence created

    auto rv = tensor.GetCpuReadView();
    const float* buf = rv.buffer<float>();
    cpu.assign(buf, buf + tensor.shape().num_elements());
  });

  ASSERT_EQ(cpu.size(), kBatch * kOutH * kOutW * kChannels);
  const float er = kR / 255.0f, eg = kG / 255.0f, eb = kB / 255.0f;
  for (int row = 0; row < kBatch; ++row) {
    for (int p = 0; p < kOutH * kOutW; ++p) {
      const int base = (row * kOutH * kOutW + p) * kChannels;
      EXPECT_NEAR(cpu[base + 0], er, 1e-4) << "row " << row << " px " << p;
      EXPECT_NEAR(cpu[base + 1], eg, 1e-4) << "row " << row << " px " << p;
      EXPECT_NEAR(cpu[base + 2], eb, 1e-4) << "row " << row << " px " << p;
    }
  }
}

// The GPU sampling matrix must be exactly the one the CPU converter convention
// produces, so CPU and GPU projections agree. For a full-frame sub-rect the
// output->input sampling map is the identity on [0,1]^2.
TEST(StreamingTilesToTensorBatchGlTest, FullFrameMatrixMatchesConverter) {
  constexpr int kW = 8, kH = 6;
  std::array<float, 16> mat;
  GetRotatedSubRectToRectTransformMatrix(PixelRoiRect(0, 0, kW, kH), kW, kH,
                                         /*flip_horizontally=*/false, &mat);
  // Row-major mat4: x' = mat[0]*x + mat[1]*y + ... For a full-frame sub-rect,
  // normalized output coords map to themselves: scale 1, no translation.
  EXPECT_NEAR(mat[0], 1.0f, 1e-5);   // x scale
  EXPECT_NEAR(mat[5], 1.0f, 1e-5);   // y scale
  EXPECT_NEAR(mat[3], 0.0f, 1e-5);   // x translate
  EXPECT_NEAR(mat[7], 0.0f, 1e-5);   // y translate
}

class StreamingTilesGlCropTest : public testing::TestWithParam<bool> {};

TEST_P(StreamingTilesGlCropTest, UpsamplingReplicatesCropEdges) {
  auto context_or = GlContext::Create(nullptr, false);
  MP_ASSERT_OK(context_or);
  auto context = context_or.value();
  constexpr int kSrcW = 6, kSrcH = 5, kOutW = 4, kOutH = 4, kChannels = 3;
  constexpr int kBatch = 3;
  const bool starts_at_bottom = GetParam();
  std::vector<uint8_t> pixels(kSrcW * kSrcH * 4);
  for (int y = 0; y < kSrcH; ++y) {
    for (int x = 0; x < kSrcW; ++x) {
      const int texture_y = starts_at_bottom ? kSrcH - 1 - y : y;
      const int offset = (texture_y * kSrcW + x) * 4;
      // The 2x2 ROI corners are black, red, green and white. Magenta pixels
      // outside it must not bleed into the upsampled tile at any edge.
      const bool inside = x >= 2 && x < 4 && y >= 1 && y < 3;
      pixels[offset] = !inside || x == 3 ? 255 : 0;
      pixels[offset + 1] = inside && y == 2 ? 255 : 0;
      pixels[offset + 2] = !inside || (x == 3 && y == 2) ? 255 : 0;
      pixels[offset + 3] = 255;
    }
  }
  std::vector<float> cpu;
  context->Run([&]() {
    tflite::gpu::GpuInfo gpu_info;
    MP_ASSERT_OK(tflite::gpu::gl::RequestGpuInfo(&gpu_info));
    ASSERT_TRUE(gpu_info.IsApiOpenGl31OrAbove());
    auto queue = tflite::gpu::gl::NewCommandQueue(gpu_info);
    tflite::gpu::gl::GlTexture input(
        GL_TEXTURE_2D, MakeRgbaTexture(kSrcW, kSrcH, pixels), GL_RGBA,
        pixels.size(), /*layer=*/0, /*owned=*/true);
    auto writer_or = TiledBatchGlWriter::Create(
        *context, kOutW, kOutH, kChannels, BorderMode::kReplicate,
        starts_at_bottom);
    MP_ASSERT_OK(writer_or);
    auto writer = std::move(writer_or).value();
    Tensor tensor(Tensor::ElementType::kFloat32,
                  Tensor::Shape{kBatch, kOutH, kOutW, kChannels});
    {
      auto write = tensor.GetOpenGlBufferWriteView();
      tflite::gpu::gl::GlBuffer dest(GL_SHADER_STORAGE_BUFFER, write.name(),
                                     tensor.bytes(), /*offset=*/0,
                                     /*has_ownership=*/false);
      const tflite::gpu::HW size(kSrcH, kSrcW);
      MP_ASSERT_OK(writer->WriteTileRow(input, size, PixelRoiRect(2, 1, 2, 2),
                                        /*tile_row=*/0, /*alpha=*/1.0f,
                                        /*beta=*/0.0f, queue.get(), &dest));
      MP_ASSERT_OK(writer->WriteTileRow(input, size, PixelRoiRect(3, 2, 1, 1),
                                        /*tile_row=*/1, /*alpha=*/1.0f,
                                        /*beta=*/0.0f, queue.get(), &dest));
      MP_ASSERT_OK(writer->WriteTileRow(
          input, size, PixelRoiRect(0, 0, kSrcW, kSrcH), /*tile_row=*/2,
          /*alpha=*/0.0f, /*beta=*/0.0f, queue.get(), &dest));
      MP_ASSERT_OK(queue->WaitForCompletion());
    }
    auto read = tensor.GetCpuReadView();
    const float* values = read.buffer<float>();
    cpu.assign(values, values + tensor.shape().num_elements());
  });
  ASSERT_EQ(cpu.size(), kBatch * kOutW * kOutH * kChannels);
  constexpr std::array<float, 4> kWeights = {0.0f, 0.25f, 0.75f, 1.0f};
  for (int y = 0; y < kOutH; ++y) {
    for (int x = 0; x < kOutW; ++x) {
      const int offset = (y * kOutW + x) * kChannels;
      const std::array<float, 3> expected = {
          kWeights[x], kWeights[y], kWeights[x] * kWeights[y]};
      for (int c = 0; c < kChannels; ++c) {
        // CPU resize rounds to uint8 before normalization; allow one LSB.
        EXPECT_NEAR(cpu[offset + c], expected[c], 1.0f / 255.0f)
            << "x=" << x << " y=" << y << " channel=" << c;
        EXPECT_NEAR(cpu[kOutW * kOutH * kChannels + offset + c], 1.0f,
                    1.0f / 255.0f);
        EXPECT_EQ(cpu[2 * kOutW * kOutH * kChannels + offset + c], 0.0f);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(TextureOrigins, StreamingTilesGlCropTest,
                         testing::Values(false, true));

}  // namespace
}  // namespace mediapipe

#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
