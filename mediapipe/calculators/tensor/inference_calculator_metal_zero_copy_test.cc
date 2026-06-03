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
// Runs on this Mac (Metal). Verifies InferenceCalculatorMetal's
// metal_external_input_zero_copy mode: binding a physical PHWC4 packet buffer
// directly to the delegate input (skipping BHWC->BPHWC4 conversion) produces
// the SAME inference output as the normal path on the equivalent logical input.

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_METAL_ENABLED

#include <cstring>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_replace.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

namespace mediapipe {
namespace {

constexpr int kH = 256, kW = 256, kC = 3, kC4 = 4;

// Deterministic [0,1] value for logical channel c at (y,x).
float Pattern(int y, int x, int c) {
  return ((y * 7 + x * 13 + c * 29) % 251) / 251.0f;
}

class InferenceMetalZeroCopyTest : public testing::Test {
 protected:
  GpuSharedData gpu_shared_;
  std::shared_ptr<GpuResources> gpu_resources_ = gpu_shared_.gpu_resources;

  // Runs the softmax model once through InferenceCalculatorMetal and returns
  // the flattened first output tensor. `channels` is the input tensor's last
  // dim (3 = logical normal path, 4 = physical PHWC4 zero-copy path).
  std::vector<float> RunOnce(bool zero_copy, int channels) {
    const std::string zc =
        zero_copy ? "metal_external_input_zero_copy: true" : "";
    const std::string proto = absl::StrReplaceAll(
        R"pb(
          input_stream: "tensors"
          output_stream: "out"
          node {
            calculator: "InferenceCalculator"
            input_stream: "TENSORS:tensors"
            output_stream: "TENSORS:out"
            options {
              [mediapipe.InferenceCalculatorOptions.ext] {
                model_path: "mediapipe/calculators/tensor/testdata/1x256x256x3_softmax.tflite"
                delegate { gpu { allow_precision_loss: false $ZC } }
              }
            }
          }
        )pb",
        {{"$ZC", zc}});
    auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(proto);

    CalculatorGraph graph;
    EXPECT_TRUE(graph.Initialize(config).ok());
    EXPECT_TRUE(graph.SetGpuResources(gpu_resources_).ok());
    std::vector<Packet> out_packets;
    EXPECT_TRUE(graph
                    .ObserveOutputStream("out",
                                         [&](const Packet& p) {
                                           out_packets.push_back(p);
                                           return absl::OkStatus();
                                         })
                    .ok());
    EXPECT_TRUE(graph.StartRun({}).ok());

    // Build the input tensor with `channels` last dim; fill logical channels
    // 0..kC-1 with the pattern, padded channel (if any) with 0.
    std::vector<Tensor> input;
    input.emplace_back(Tensor::ElementType::kFloat32,
                       Tensor::Shape{1, kH, kW, channels});
    {
      auto w = input[0].GetCpuWriteView();
      float* buf = w.buffer<float>();
      std::memset(buf, 0, sizeof(float) * kH * kW * channels);
      for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
          float* px = buf + (y * kW + x) * channels;
          for (int c = 0; c < kC; ++c) px[c] = Pattern(y, x, c);
        }
      }
    }
    EXPECT_TRUE(graph
                    .AddPacketToInputStream(
                        "tensors", MakePacket<std::vector<Tensor>>(
                                       std::move(input))
                                       .At(Timestamp(0)))
                    .ok());
    EXPECT_TRUE(graph.WaitUntilIdle().ok());
    EXPECT_TRUE(graph.CloseAllInputStreams().ok());
    EXPECT_TRUE(graph.WaitUntilDone().ok());

    std::vector<float> result;
    if (out_packets.size() == 1) {
      const auto& tensors = out_packets[0].Get<std::vector<Tensor>>();
      if (!tensors.empty()) {
        auto v = tensors[0].GetCpuReadView();
        const float* b = v.buffer<float>();
        result.assign(b, b + tensors[0].shape().num_elements());
      }
    }
    return result;
  }
};

TEST_F(InferenceMetalZeroCopyTest, DirectBindMatchesNormalPath) {
  std::vector<float> normal = RunOnce(/*zero_copy=*/false, /*channels=*/kC);
  std::vector<float> direct = RunOnce(/*zero_copy=*/true, /*channels=*/kC4);
  ASSERT_FALSE(normal.empty());
  ASSERT_EQ(normal.size(), direct.size());
  for (size_t i = 0; i < normal.size(); ++i) {
    EXPECT_NEAR(direct[i], normal[i], 1e-4) << "output elem " << i;
  }
}

}  // namespace
}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
