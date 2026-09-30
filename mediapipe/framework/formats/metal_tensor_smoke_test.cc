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
// Phase 5 (Metal) Task 1 — feasibility GATE. Proves the macOS Metal runtime +
// MtlBufferView actually EXECUTE on this machine (the gpu_buffer_format config
// fix only proved they compile). If this fails to build or run, the Metal
// zero-copy plan is blocked.

#import <Metal/Metal.h>

#include <vector>

#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/formats/tensor_mtl_buffer_view.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

// Writes known floats into a Tensor's MTLBuffer on the GPU (blit from a source
// buffer) and reads them back, proving: a Tensor can be Metal-backed, a Metal
// command runs here, and the result is observable.
TEST(MetalTensorSmokeTest, WriteAndReadBackMetalBackedTensor) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ASSERT_NE(device, nil) << "no Metal device";
  id<MTLCommandQueue> queue = [device newCommandQueue];

  constexpr int kN = 8;
  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{kN});

  std::vector<float> src(kN);
  for (int i = 0; i < kN; ++i) src[i] = i + 0.5f;
  id<MTLBuffer> src_buf =
      [device newBufferWithBytes:src.data()
                          length:sizeof(float) * kN
                         options:MTLResourceStorageModeShared];

  id<MTLCommandBuffer> cb = [queue commandBuffer];
  {
    // GetWriteView allocates/links the Tensor's MTLBuffer.
    auto wv = MtlBufferView::GetWriteView(tensor, cb);
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src_buf
            sourceOffset:0
                toBuffer:wv.buffer()
       destinationOffset:0
                    size:sizeof(float) * kN];
    [blit endEncoding];
  }
  [cb commit];
  [cb waitUntilCompleted];

  auto cpu = tensor.GetCpuReadView();
  const float* out = cpu.buffer<float>();
  for (int i = 0; i < kN; ++i) EXPECT_FLOAT_EQ(out[i], i + 0.5f);
}

}  // namespace
}  // namespace mediapipe
