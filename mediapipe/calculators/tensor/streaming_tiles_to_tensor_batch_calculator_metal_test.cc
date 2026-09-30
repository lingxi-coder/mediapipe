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
// Runs on this Mac (Metal). Drives StreamingTilesToTensorBatchCalculator's Metal
// zero-copy path through a real CalculatorGraph with GpuResources: GPU buffer +
// tile plan -> logical [N,H,W,C] batch tensor, asserting shape,
// no-CPU-readback, and multi-batch semantics. On non-Metal configs the whole TU
// compiles to nothing.

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_METAL_ENABLED

#include <cstring>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.pb.h"
#include "mediapipe/framework/formats/tiling_cache_stats.h"
#include "mediapipe/framework/formats/tiling_types.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/util/image_test_utils.h"

namespace mediapipe {
namespace {

TileGeometry NormTile(int index, float x0, float y0, float w, float h) {
  TileGeometry g;
  g.tile_index = index;
  g.width = w;
  g.height = h;
  g.x_center = x0 + w / 2.0f;
  g.y_center = y0 + h / 2.0f;
  return g;
}

struct RunResult {
  std::vector<Packet> tensors;
  std::vector<Packet> info;
};

absl::Status RunZeroCopy(std::shared_ptr<GpuResources> gpu_resources,
                         const std::string& options_body, int out_w, int out_h,
                         int channels, int batch_capacity, int in_w, int in_h,
                         const std::vector<TileGeometry>& tiles,
                         RunResult* out,
                         GpuBufferFormat input_format = GpuBufferFormat::kRGBA32) {
  InferenceMetadata meta;
  meta.set_input_height(out_h);
  meta.set_input_width(out_w);
  meta.set_input_channels(channels);
  meta.set_batch_capacity(batch_capacity);

  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image_gpu"
    input_stream: "tile_plan"
    output_stream: "tensors"
    output_stream: "info"
    node {
      calculator: "StreamingTilesToTensorBatchCalculator"
      input_stream: "IMAGE_GPU:image_gpu"
      input_stream: "TILE_PLAN:tile_plan"
      input_side_packet: "METADATA:meta"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:info"
      options { [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {} }
    }
  )pb");
  config.mutable_node(0)
      ->mutable_options()
      ->MutableExtension(StreamingTilesToTensorBatchCalculatorOptions::ext)
      ->MergeFrom(
          ParseTextProtoOrDie<StreamingTilesToTensorBatchCalculatorOptions>(
              options_body));

  CalculatorGraph graph;
  ABSL_RETURN_IF_ERROR(graph.Initialize(config));
  ABSL_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  ABSL_RETURN_IF_ERROR(graph.ObserveOutputStream("tensors", [out](const Packet& p) {
    out->tensors.push_back(p);
    return absl::OkStatus();
  }));
  ABSL_RETURN_IF_ERROR(graph.ObserveOutputStream("info", [out](const Packet& p) {
    out->info.push_back(p);
    return absl::OkStatus();
  }));
  ABSL_RETURN_IF_ERROR(
      graph.StartRun({{"meta", MakePacket<InferenceMetadata>(meta)}}));
  GpuBuffer input;
  if (input_format == GpuBufferFormat::kRGBA32) {
    input = CreateTestRgba8GpuBuffer(in_w, in_h);
  } else if (input_format == GpuBufferFormat::kBGRA32) {
    // Materialize a native CVPixelBuffer while preserving the RGB values.
    GpuBuffer rgba = CreateTestRgba8GpuBuffer(in_w, in_h);
    input = GpuBuffer(GetCVPixelBufferRef(rgba));
  } else {
    input = GpuBuffer(in_w, in_h, input_format);
  }
  TilePlan plan;
  plan.tiles = tiles;
  ABSL_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "image_gpu", MakePacket<GpuBuffer>(input).At(Timestamp(0))));
  ABSL_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "tile_plan", MakePacket<TilePlan>(plan).At(Timestamp(0))));
  ABSL_RETURN_IF_ERROR(graph.WaitUntilIdle());
  ABSL_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  return graph.WaitUntilDone();
}

// Minimal GPU test fixture (replicates gpu_test_base's GpuResources setup,
// which is a package-private target in //mediapipe/gpu).
class StreamingTilesMetalTest : public testing::Test {
 protected:
  GpuSharedData gpu_shared_;
  std::shared_ptr<GpuResources> gpu_resources_ = gpu_shared_.gpu_resources;
};

TEST(StreamingTilesMetalOpenTest, MissingGpuServiceReturnsError) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image_gpu"
    input_stream: "tile_plan"
    node {
      calculator: "StreamingTilesToTensorBatchCalculator"
      input_stream: "IMAGE_GPU:image_gpu"
      input_stream: "TILE_PLAN:tile_plan"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:info"
      options {
        [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
          enable_gpu_zero_copy: true
          max_gpu_tensor_buffers: 2
          metadata_batch_capacity: 1
          metadata_input_height: 8
          metadata_input_width: 8
          metadata_input_channels: 3
        }
      }
    }
  )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.DisallowServiceDefaultInitialization());
  MP_ASSERT_OK(graph.Initialize(config));
  absl::Status status = graph.StartRun({});
  if (status.ok()) status = graph.WaitUntilIdle();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), testing::HasSubstr("GPU service not available"));
}

TEST_F(StreamingTilesMetalTest, ZeroCopyEmitsLogicalBatchNoReadback) {
  RunResult r;
  MP_ASSERT_OK(RunZeroCopy(
      gpu_resources_,
      "enable_gpu_zero_copy: true max_gpu_tensor_buffers: 2 dynamic_batch: true",
      /*out_w=*/8, /*out_h=*/8, /*channels=*/3, /*batch_capacity=*/4,
      /*in_w=*/64, /*in_h=*/48,
      {NormTile(0, 0.0f, 0.0f, 0.5f, 1.0f), NormTile(1, 0.5f, 0.0f, 0.5f, 1.0f)},
      &r));
  ASSERT_EQ(r.tensors.size(), 1);
  const auto& t = r.tensors[0].Get<std::vector<Tensor>>()[0];
  const Tensor::Shape& shape = t.shape();
  ASSERT_EQ(shape.dims.size(), 4);
  EXPECT_EQ(shape.dims[0], 2);  // dynamic batch: N == valid tiles
  EXPECT_EQ(shape.dims[1], 8);
  EXPECT_EQ(shape.dims[2], 8);
  EXPECT_EQ(shape.dims[3], 3);  // logical BHWC (model input channels)

  // No-readback: GPU-resident, not CPU-materialized, before any explicit read.
  EXPECT_TRUE(t.ready_on_gpu());
  EXPECT_FALSE(t.ready_on_cpu());

  // Explicit test-side readback: normalized [0,1].
  auto view = t.GetCpuReadView();
  const float* buf = view.buffer<float>();
  const int n = shape.num_elements();
  for (int i = 0; i < n; ++i) {
    EXPECT_GE(buf[i], 0.0f) << "elem " << i;
    EXPECT_LE(buf[i], 1.0001f) << "elem " << i;
  }
}

TEST_F(StreamingTilesMetalTest, MultiBatchEmitsCeil) {
  RunResult r;
  MP_ASSERT_OK(RunZeroCopy(
      gpu_resources_,
      "enable_gpu_zero_copy: true max_gpu_tensor_buffers: 4 dynamic_batch: true",
      /*out_w=*/8, /*out_h=*/8, /*channels=*/3, /*batch_capacity=*/2,
      /*in_w=*/64, /*in_h=*/48,
      {NormTile(0, 0.0f, 0.0f, 0.33f, 1.0f), NormTile(1, 0.33f, 0.0f, 0.33f, 1.0f),
       NormTile(2, 0.66f, 0.0f, 0.34f, 1.0f)},
      &r));
  EXPECT_EQ(r.tensors.size(), 2);  // ceil(3/2)
  ASSERT_GE(r.info.size(), 2u);
  EXPECT_EQ(r.info[0].Get<TensorBatchInfo>().valid_count, 2);
  EXPECT_EQ(r.info[1].Get<TensorBatchInfo>().valid_count, 1);
}

// Verifies that metal_direct_delegate_input=true emits a physical PHWC4 tensor
// of shape [1,H,W,4], is GPU-resident, and is not CPU-materialized. Uses a
// single tile and batch_capacity=1 (N==1 is required by this mode).
TEST_F(StreamingTilesMetalTest, DirectDelegateInputEmitsPhysicalPhwc4) {
  RunResult r;
  MP_ASSERT_OK(RunZeroCopy(
      gpu_resources_,
      "enable_gpu_zero_copy: true max_gpu_tensor_buffers: 2 "
      "metal_direct_delegate_input: true",
      /*out_w=*/8, /*out_h=*/8, /*channels=*/3, /*batch_capacity=*/1,
      /*in_w=*/64, /*in_h=*/48,
      {NormTile(0, 0.0f, 0.0f, 1.0f, 1.0f)},
      &r));
  ASSERT_EQ(r.tensors.size(), 1);
  const auto& t = r.tensors[0].Get<std::vector<Tensor>>()[0];
  const Tensor::Shape& shape = t.shape();
  ASSERT_EQ(shape.dims.size(), 4);
  // Physical PHWC4: [1, H, W, 4] — batch-1, 4 channels (RGB + 0-pad).
  EXPECT_EQ(shape.dims[0], 1);
  EXPECT_EQ(shape.dims[1], 8);
  EXPECT_EQ(shape.dims[2], 8);
  EXPECT_EQ(shape.dims[3], 4);

  // GPU-resident, not CPU-materialized before any explicit read.
  EXPECT_TRUE(t.ready_on_gpu());
  EXPECT_FALSE(t.ready_on_cpu());

  // Explicit test-side readback: all values in [0,1] (normalized RGB + 0-pad).
  auto view = t.GetCpuReadView();
  const float* buf = view.buffer<float>();
  const int n = shape.num_elements();
  for (int i = 0; i < n; ++i) {
    EXPECT_GE(buf[i], 0.0f) << "elem " << i;
    EXPECT_LE(buf[i], 1.0001f) << "elem " << i;
  }

  // Exactly one batch info packet matching the shape.
  ASSERT_EQ(r.info.size(), 1u);
  const TensorBatchInfo& info = r.info[0].Get<TensorBatchInfo>();
  EXPECT_EQ(info.valid_count, 1);
  EXPECT_EQ(info.total_batches, 1);
  EXPECT_EQ(info.batch_capacity, 1);
}

// When zero-copy is enabled (IMAGE_GPU wired, Metal writer initialized) but a
// frame arrives WITHOUT a GPU packet, the calculator serves it on the CPU
// path. That silent fallback must be counted in
// CACHE_STATS.gpu_to_cpu_fallbacks — the zero-copy e2e tests assert "== 0" on
// this counter, which is only meaningful if real fallbacks increment it.
TEST_F(StreamingTilesMetalTest, CpuServedFrameCountsGpuToCpuFallback) {
  InferenceMetadata meta;
  meta.set_input_height(8);
  meta.set_input_width(8);
  meta.set_input_channels(3);
  meta.set_batch_capacity(2);

  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    input_stream: "image_gpu"
    input_stream: "tile_plan"
    output_stream: "tensors"
    output_stream: "stats"
    node {
      calculator: "StreamingTilesToTensorBatchCalculator"
      input_stream: "IMAGE:image"
      input_stream: "IMAGE_GPU:image_gpu"
      input_stream: "TILE_PLAN:tile_plan"
      input_side_packet: "METADATA:meta"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:info"
      output_stream: "CACHE_STATS:stats"
      options {
        [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
          enable_gpu_zero_copy: true
          max_gpu_tensor_buffers: 2
          emit_cache_stats: true
        }
      }
    }
  )pb");

  std::vector<Packet> tensors;
  std::vector<Packet> stats;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.SetGpuResources(gpu_resources_));
  MP_ASSERT_OK(graph.ObserveOutputStream("tensors", [&](const Packet& p) {
    tensors.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.ObserveOutputStream("stats", [&](const Packet& p) {
    stats.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({{"meta", MakePacket<InferenceMetadata>(meta)}}));

  // CPU frame only — no IMAGE_GPU packet at this timestamp.
  auto frame = std::make_unique<ImageFrame>(ImageFormat::SRGB, 32, 32);
  std::memset(frame->MutablePixelData(), 128,
              frame->Height() * frame->WidthStep());
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image", Adopt(frame.release()).At(Timestamp(0))));
  TilePlan plan;
  plan.tiles = {NormTile(0, 0.0f, 0.0f, 1.0f, 1.0f)};
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tile_plan", MakePacket<TilePlan>(plan).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  // The frame was served on the CPU path...
  ASSERT_EQ(tensors.size(), 1u);
  EXPECT_TRUE(tensors[0].Get<std::vector<Tensor>>()[0].ready_on_cpu());
  // ...and the fallback was counted.
  ASSERT_GE(stats.size(), 1u);
  EXPECT_EQ(stats.back().Get<TilingCacheStats>().gpu_to_cpu_fallbacks, 1);
}

TEST_F(StreamingTilesMetalTest, RejectsNonRgbGpuFormatsBeforeTextureMapping) {
  for (const auto format : {GpuBufferFormat::kBiPlanar420YpCbCr8FullRange,
                            GpuBufferFormat::kGrayFloat32,
                            GpuBufferFormat::kRGB24,
                            GpuBufferFormat::kRGBAFloat128}) {
    RunResult result;
    const absl::Status status = RunZeroCopy(
        gpu_resources_, "enable_gpu_zero_copy: true max_gpu_tensor_buffers: 2",
        /*out_w=*/8, /*out_h=*/8, /*channels=*/3, /*batch_capacity=*/1,
        /*in_w=*/32, /*in_h=*/32, {NormTile(0, 0, 0, 1, 1)}, &result, format);
    EXPECT_FALSE(status.ok());
    EXPECT_THAT(status.message(), testing::HasSubstr("RGB or RGBA input format"));
    EXPECT_TRUE(result.tensors.empty());
  }
}

TEST_F(StreamingTilesMetalTest, AcceptsRgbaImageFrameAndBgraPixelBufferStorage) {
  for (const auto format : {GpuBufferFormat::kRGBA32, GpuBufferFormat::kBGRA32}) {
    SCOPED_TRACE(static_cast<uint32_t>(format));
    RunResult result;
    MP_ASSERT_OK(RunZeroCopy(
        gpu_resources_, "enable_gpu_zero_copy: true max_gpu_tensor_buffers: 2",
        /*out_w=*/8, /*out_h=*/8, /*channels=*/3, /*batch_capacity=*/1,
        /*in_w=*/8, /*in_h=*/8, {NormTile(0, 0, 0, 1, 1)}, &result, format));
    ASSERT_EQ(result.tensors.size(), 1);
    const Tensor& tensor = result.tensors[0].Get<std::vector<Tensor>>()[0];
    auto read = tensor.GetCpuReadView();
    const ImageFrame expected = CreateTestRgba8ImageFrame(8, 8);
    const float* values = read.buffer<float>();
    for (int y = 0; y < 8; ++y) {
      const uint8_t* row = expected.PixelData() + y * expected.WidthStep();
      for (int x = 0; x < 8; ++x) {
        for (int c = 0; c < 3; ++c) {
          EXPECT_NEAR(values[(y * 8 + x) * 3 + c], row[x * 4 + c] / 255.0f,
                      1.0f / 255.0f);
        }
      }
    }
  }
}

TEST_F(StreamingTilesMetalTest, CpuFallbackPreservesDirectDelegateInputLayout) {
  std::vector<float> logical_result;
  for (const bool direct : {false, true}) {
    SCOPED_TRACE(direct);
    auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
      input_stream: "image"
      input_stream: "image_gpu"
      input_stream: "tile_plan"
      output_stream: "preprocessed"
      output_stream: "out"
      node {
        calculator: "StreamingTilesToTensorBatchCalculator"
        input_stream: "IMAGE:image"
        input_stream: "IMAGE_GPU:image_gpu"
        input_stream: "TILE_PLAN:tile_plan"
        output_stream: "TENSORS:preprocessed"
        output_stream: "BATCH_INFO:info"
        options {
          [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
            enable_gpu_zero_copy: true
            max_gpu_tensor_buffers: 2
            metadata_batch_capacity: 1
            metadata_input_height: 256
            metadata_input_width: 256
            metadata_input_channels: 3
          }
        }
      }
      node {
        calculator: "InferenceCalculator"
        input_stream: "TENSORS:preprocessed"
        output_stream: "TENSORS:out"
        options {
          [mediapipe.InferenceCalculatorOptions.ext] {
            model_path: "mediapipe/calculators/tensor/testdata/1x256x256x3_softmax.tflite"
            delegate { gpu { allow_precision_loss: false } }
          }
        }
      }
    )pb");
    config.mutable_node(0)->mutable_options()->MutableExtension(
        StreamingTilesToTensorBatchCalculatorOptions::ext)
        ->set_metal_direct_delegate_input(direct);
    // Keep the inference option in text so this test uses the same graph
    // registration path as a caller loading a graph configuration.
    if (direct) {
      config.mutable_node(1)->mutable_options()->MergeFrom(
          ParseTextProtoOrDie<CalculatorOptions>(R"pb(
            [mediapipe.InferenceCalculatorOptions.ext] {
              delegate { gpu { metal_external_input_zero_copy: true } }
            }
          )pb"));
    }
    CalculatorGraph graph;
    MP_ASSERT_OK(graph.Initialize(config));
    MP_ASSERT_OK(graph.SetGpuResources(gpu_resources_));
    std::vector<Packet> inputs;
    std::vector<Packet> outputs;
    MP_ASSERT_OK(graph.ObserveOutputStream("preprocessed", [&](const Packet& p) {
      inputs.push_back(p);
      return absl::OkStatus();
    }));
    MP_ASSERT_OK(graph.ObserveOutputStream("out", [&](const Packet& p) {
      outputs.push_back(p);
      return absl::OkStatus();
    }));
    MP_ASSERT_OK(graph.StartRun({}));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image", MakePacket<ImageFrame>(CreateTestRgb8ImageFrame(32, 32))
                     .At(Timestamp(0))));
    TilePlan plan;
    plan.tiles = {NormTile(0, 0, 0, 1, 1)};
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "tile_plan", MakePacket<TilePlan>(plan).At(Timestamp(0))));
    // No GPU packet: closing that stream makes the CPU fallback frame ready.
    MP_ASSERT_OK(graph.CloseAllInputStreams());
    MP_ASSERT_OK(graph.WaitUntilDone());
    ASSERT_EQ(inputs.size(), 1);
    const Tensor& input = inputs[0].Get<std::vector<Tensor>>()[0];
    EXPECT_EQ(input.shape().dims, (std::vector<int>{1, 256, 256, direct ? 4 : 3}));
    if (direct) {
      auto read = input.GetCpuReadView();
      const float* pixels = read.buffer<float>();
      for (int i = 0; i < 256 * 256; ++i) EXPECT_EQ(pixels[i * 4 + 3], 0.0f);
    }
    ASSERT_EQ(outputs.size(), 1);
    const Tensor& output = outputs[0].Get<std::vector<Tensor>>()[0];
    auto read = output.GetCpuReadView();
    const float* values = read.buffer<float>();
    if (!direct) {
      logical_result.assign(values, values + output.shape().num_elements());
    } else {
      ASSERT_EQ(logical_result.size(), output.shape().num_elements());
      for (int i = 0; i < logical_result.size(); ++i) {
        EXPECT_NEAR(values[i], logical_result[i], 1e-4f) << "element " << i;
      }
    }
  }
}

}  // namespace
}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
