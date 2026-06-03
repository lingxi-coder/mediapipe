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

// Deterministic [0,1] value for logical channel c at (y,x) on a given frame.
float Pattern(int y, int x, int c, int frame = 0) {
  return ((y * 7 + x * 13 + c * 29 + frame * 101) % 251) / 251.0f;
}

class InferenceMetalZeroCopyTest : public testing::Test {
 protected:
  GpuSharedData gpu_shared_;
  std::shared_ptr<GpuResources> gpu_resources_ = gpu_shared_.gpu_resources;

  // Runs the softmax model for `num_frames` consecutive frames through one
  // InferenceCalculatorMetal instance and returns each frame's flattened first
  // output tensor. `channels` is the input tensor's last dim (3 = logical normal
  // path, 4 = physical PHWC4 zero-copy path). Running >1 frame exercises the
  // per-Process delegate rebinding in zero-copy mode.
  std::vector<std::vector<float>> RunFrames(bool zero_copy, int channels,
                                            int num_frames) {
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

    for (int frame = 0; frame < num_frames; ++frame) {
      // Build the input tensor with `channels` last dim; fill logical channels
      // 0..kC-1 with this frame's pattern, padded channel (if any) with 0.
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
            for (int c = 0; c < kC; ++c) px[c] = Pattern(y, x, c, frame);
          }
        }
      }
      EXPECT_TRUE(graph
                      .AddPacketToInputStream(
                          "tensors", MakePacket<std::vector<Tensor>>(
                                         std::move(input))
                                         .At(Timestamp(frame)))
                      .ok());
    }
    EXPECT_TRUE(graph.WaitUntilIdle().ok());
    EXPECT_TRUE(graph.CloseAllInputStreams().ok());
    EXPECT_TRUE(graph.WaitUntilDone().ok());

    std::vector<std::vector<float>> results;
    for (const Packet& p : out_packets) {
      const auto& tensors = p.Get<std::vector<Tensor>>();
      if (tensors.empty()) {
        results.emplace_back();
        continue;
      }
      auto v = tensors[0].GetCpuReadView();
      const float* b = v.buffer<float>();
      results.emplace_back(b, b + tensors[0].shape().num_elements());
    }
    return results;
  }
};

TEST_F(InferenceMetalZeroCopyTest, DirectBindMatchesNormalPath) {
  auto normal = RunFrames(/*zero_copy=*/false, /*channels=*/kC, 1);
  auto direct = RunFrames(/*zero_copy=*/true, /*channels=*/kC4, 1);
  ASSERT_EQ(normal.size(), 1u);
  ASSERT_EQ(direct.size(), 1u);
  ASSERT_FALSE(normal[0].empty());
  ASSERT_EQ(normal[0].size(), direct[0].size());
  for (size_t i = 0; i < normal[0].size(); ++i) {
    EXPECT_NEAR(direct[0][i], normal[0][i], 1e-4) << "output elem " << i;
  }
}

// Per-frame TFLGpuDelegateBindMetalBufferToTensor must re-point the delegate
// input each Process(); each frame's output must match the normal path's.
TEST_F(InferenceMetalZeroCopyTest, MultiFrameRebindMatchesNormalPath) {
  constexpr int kFrames = 3;
  auto normal = RunFrames(/*zero_copy=*/false, /*channels=*/kC, kFrames);
  auto direct = RunFrames(/*zero_copy=*/true, /*channels=*/kC4, kFrames);
  ASSERT_EQ(normal.size(), static_cast<size_t>(kFrames));
  ASSERT_EQ(direct.size(), static_cast<size_t>(kFrames));
  // Frames carry distinct patterns, so distinct outputs prove rebinding took
  // effect (not a stale buffer reused across frames).
  for (int f = 0; f < kFrames; ++f) {
    ASSERT_EQ(normal[f].size(), direct[f].size()) << "frame " << f;
    for (size_t i = 0; i < normal[f].size(); ++i) {
      EXPECT_NEAR(direct[f][i], normal[f][i], 1e-4) << "frame " << f << " elem " << i;
    }
  }
  EXPECT_NE(direct[0], direct[1]);  // different inputs -> different outputs
}

}  // namespace
}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
