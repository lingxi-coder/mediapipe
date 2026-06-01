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

// Graph-level test: verifies that InferenceCalculatorCpu emits a populated
// METADATA side packet when the optional output side packet is wired.

#include <string>
#include <vector>

#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

TEST(InferenceCalculatorMetadataTest, MetadataSidePacketIsPopulated) {
  // Build a graph config that:
  //   1. Uses InferenceCalculatorCpu directly (CPU backend guaranteed).
  //   2. Wires the optional METADATA output side packet.
  //   3. Provides the model via model_path in options (same as the main
  //      inference_calculator_test smoketests).
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "tensor_in"
    output_side_packet: "metadata"

    node {
      calculator: "InferenceCalculatorCpu"
      input_stream: "TENSORS:tensor_in"
      output_stream: "TENSORS:tensor_out"
      output_side_packet: "METADATA:metadata"
      options {
        [mediapipe.InferenceCalculatorOptions.ext] {
          model_path: "mediapipe/calculators/tensor/testdata/add.bin"
          delegate { tflite {} }
        }
      }
    }
  )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  // Push one dummy tensor so the graph can process (model requires input).
  Tensor input_tensor(Tensor::ElementType::kFloat32,
                      Tensor::Shape{1, 8, 8, 3});
  {
    auto view = input_tensor.GetCpuWriteView();
    float* buf = view.buffer<float>();
    const int n = input_tensor.shape().num_elements();
    for (int i = 0; i < n; ++i) buf[i] = 1.0f;
  }
  std::vector<Tensor> inputs;
  inputs.push_back(std::move(input_tensor));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tensor_in",
      MakePacket<std::vector<Tensor>>(std::move(inputs)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseInputStream("tensor_in"));
  MP_ASSERT_OK(graph.WaitUntilDone());

  // Retrieve the METADATA side packet.
  auto status_or_packet = graph.GetOutputSidePacket("metadata");
  MP_ASSERT_OK(status_or_packet);
  const InferenceMetadata& md =
      status_or_packet.value().Get<InferenceMetadata>();

  EXPECT_FALSE(md.input().empty());
  EXPECT_FALSE(md.output().empty());
  EXPECT_EQ(md.backend(), "cpu");

  // add.bin has a 4-D input [1, 8, 8, 3], so spatial and batch fields must be
  // set.
  EXPECT_GT(md.batch_capacity(), 0);
  EXPECT_GT(md.input_height(), 0);
  EXPECT_GT(md.input_width(), 0);
}

}  // namespace
}  // namespace mediapipe
