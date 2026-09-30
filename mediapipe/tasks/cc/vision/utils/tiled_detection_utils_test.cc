/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/vision/utils/tiled_detection_utils.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "flatbuffers/flatbuffers.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/vision/utils/detection_label_resolution.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "tflite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace {

struct TestModelOptions {
  std::vector<int32_t> shape = {2, 4, 5, 3};
  std::vector<int32_t> shape_signature;
  tflite::TensorType tensor_type = tflite::TensorType_FLOAT32;
  int input_count = 1;
  std::optional<int> input_index_override;
  bool include_metadata = true;
  int metadata_subgraph_count = 1;
  bool include_normalization = true;
  std::optional<std::vector<float>> mean = std::vector<float>{0.0f};
  std::optional<std::vector<float>> std = std::vector<float>{255.0f};
  tflite::ColorSpaceType color_space = tflite::ColorSpaceType_RGB;
};

// A small, valid flatbuffer model with input forwarded to output and no ops.
// This exercises ModelResources and the real metadata parser without fixtures
// or a TFLite inference session.
absl::StatusOr<std::unique_ptr<tasks::core::ModelResources>> CreateTestModel(
    const TestModelOptions& options) {
  flatbuffers::FlatBufferBuilder metadata_builder;
  std::vector<flatbuffers::Offset<tflite::ProcessUnit>> process_units;
  if (options.include_normalization) {
    const auto mean = options.mean.has_value()
                          ? metadata_builder.CreateVector(*options.mean)
                          : 0;
    const auto std = options.std.has_value()
                         ? metadata_builder.CreateVector(*options.std)
                         : 0;
    const auto normalization =
        tflite::CreateNormalizationOptions(metadata_builder, mean, std);
    process_units.push_back(tflite::CreateProcessUnit(
        metadata_builder, tflite::ProcessUnitOptions_NormalizationOptions,
        normalization.Union()));
  }
  const auto image_properties =
      tflite::CreateImageProperties(metadata_builder, options.color_space);
  const auto content = tflite::CreateContent(
      metadata_builder, tflite::ContentProperties_ImageProperties,
      image_properties.Union());
  const auto tensor_metadata = tflite::CreateTensorMetadata(
      metadata_builder, 0, 0, 0, content,
      metadata_builder.CreateVector(process_units));
  const auto inputs_metadata = metadata_builder.CreateVector(
      std::vector<flatbuffers::Offset<tflite::TensorMetadata>>(
          options.input_count, tensor_metadata));
  const auto subgraph_metadata =
      tflite::CreateSubGraphMetadata(metadata_builder, 0, 0, inputs_metadata);
  const auto model_metadata = tflite::CreateModelMetadata(
      metadata_builder, 0, 0, 0,
      metadata_builder.CreateVector(
          std::vector<flatbuffers::Offset<tflite::SubGraphMetadata>>(
              options.metadata_subgraph_count, subgraph_metadata)));
  tflite::FinishModelMetadataBuffer(metadata_builder, model_metadata);

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(builder)};
  std::vector<flatbuffers::Offset<tflite::Metadata>> metadata;
  if (options.include_metadata) {
    buffers.push_back(tflite::CreateBuffer(
        builder, builder.CreateVector(metadata_builder.GetBufferPointer(),
                                      metadata_builder.GetSize())));
    metadata.push_back(tflite::CreateMetadata(
        builder, builder.CreateString("TFLITE_METADATA"), 1));
  }
  const auto shape = builder.CreateVector(options.shape);
  const auto shape_signature = builder.CreateVector(options.shape_signature);
  const auto tensor = tflite::CreateTensor(builder, shape, options.tensor_type,
                                           0, builder.CreateString("image"), 0,
                                           false, 0, shape_signature);
  // Keep a tensor even for zero-input tests so the output remains valid.
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors(
      options.input_count > 0 ? options.input_count : 1, tensor);
  std::vector<int32_t> inputs;
  for (int i = 0; i < options.input_count; ++i) {
    inputs.push_back(i);
  }
  if (inputs.size() == 1 && options.input_index_override.has_value()) {
    inputs[0] = *options.input_index_override;
  }
  const auto subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors), builder.CreateVector(inputs),
      builder.CreateVector(std::vector<int32_t>{0}),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<tflite::Operator>>{}));
  const auto model = tflite::CreateModel(
      builder, 3,
      builder.CreateVector(
          std::vector<flatbuffers::Offset<tflite::OperatorCode>>{}),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<tflite::SubGraph>>{subgraph}),
      0, builder.CreateVector(buffers), 0, builder.CreateVector(metadata));
  tflite::FinishModelBuffer(builder, model);

  auto model_file = std::make_unique<tasks::core::proto::ExternalFile>();
  model_file->set_file_content(builder.GetBufferPointer(), builder.GetSize());
  return tasks::core::ModelResources::Create("tiled_test",
                                             std::move(model_file));
}

TEST(ValidateTiledModelInputTest,
     AcceptsFixedAndDynamicBatchesWithUnitNormalization) {
  for (const int batch : {1, 2}) {
    for (const bool dynamic : {false, true}) {
      SCOPED_TRACE(batch);
      SCOPED_TRACE(dynamic);
      TestModelOptions options;
      options.shape[0] = batch;
      if (dynamic) options.shape_signature = {-1, 4, 5, 3};
      MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
      MP_ASSERT_OK_AND_ASSIGN(auto dims,
                              ValidateTiledModelInputAndGetDims(*model));
      EXPECT_EQ(dims.batch, dynamic ? 1 : batch);
      EXPECT_EQ(dims.height, 4);
      EXPECT_EQ(dims.width, 5);
      EXPECT_EQ(dims.channels, 3);
      EXPECT_EQ(dims.is_dynamic_batch, dynamic);
    }
  }
}

TEST(ValidateTiledModelInputTest,
     RejectsIncompatibleNormalizationForEveryBatch) {
  for (const int batch : {1, 2}) {
    for (const bool dynamic : {false, true}) {
      SCOPED_TRACE(batch);
      SCOPED_TRACE(dynamic);
      TestModelOptions options;
      options.shape[0] = batch;
      if (dynamic) options.shape_signature = {-1, 4, 5, 3};
      options.mean = {127.5f};
      options.std = {127.5f};
      MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
      EXPECT_EQ(ValidateTiledModelInputAndGetDims(*model).status().code(),
                absl::StatusCode::kInvalidArgument);
    }
  }
}

TEST(ValidateTiledModelInputTest, AllowsAbsentNormalizationAndAbsentMetadata) {
  for (const bool include_metadata : {false, true}) {
    TestModelOptions options;
    options.include_metadata = include_metadata;
    options.include_normalization = false;
    MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
    MP_EXPECT_OK(ValidateTiledModelInputAndGetDims(*model));
  }
}

TEST(ValidateTiledModelInputTest, AcceptsPerChannelUnitNormalization) {
  TestModelOptions options;
  options.mean = {0.0f, 0.0f, 0.0f};
  options.std = {255.0f, 255.0f, 255.0f};
  MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
  MP_EXPECT_OK(ValidateTiledModelInputAndGetDims(*model));
}

TEST(ValidateTiledModelInputTest,
     AcceptsGrayscaleWithScalarOrAbsentNormalization) {
  for (bool include_metadata : {false, true}) {
    for (bool include_normalization : {false, true}) {
      TestModelOptions options;
      options.shape[3] = 1;
      options.color_space = tflite::ColorSpaceType_GRAYSCALE;
      options.include_metadata = include_metadata;
      options.include_normalization = include_normalization;
      MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
      MP_ASSERT_OK_AND_ASSIGN(auto dims,
                              ValidateTiledModelInputAndGetDims(*model));
      EXPECT_EQ(dims.channels, 1);
      EXPECT_EQ(dims.batch, 2);
    }
  }
}

TEST(ValidateTiledModelInputTest, RejectsPerChannelNormalizationForGrayscale) {
  TestModelOptions options;
  options.shape[3] = 1;
  options.color_space = tflite::ColorSpaceType_GRAYSCALE;
  options.mean = {0.0f, 0.0f, 0.0f};
  options.std = {255.0f, 255.0f, 255.0f};
  MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
  EXPECT_EQ(ValidateTiledModelInputAndGetDims(*model).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ValidateTiledModelInputTest, RejectsMalformedOrNonFiniteNormalization) {
  std::vector<TestModelOptions> cases(6);
  cases[0].mean = std::nullopt;
  cases[1].std = std::nullopt;
  cases[2].mean = {0.0f, 0.0f};
  cases[2].std = {255.0f, 255.0f};
  cases[3].std = {255.0f, 255.0f, 255.0f};
  cases[4].mean = {std::numeric_limits<float>::quiet_NaN()};
  cases[5].std = {std::numeric_limits<float>::infinity()};
  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(i);
    MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(cases[i]));
    EXPECT_EQ(ValidateTiledModelInputAndGetDims(*model).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(ValidateTiledModelInputTest, RejectsInvalidInputDimensionsAndType) {
  const std::vector<std::vector<int32_t>> invalid_shapes = {
      {2, 4, 5},    {2, 4, 5, 3, 1}, {2, 0, 5, 3},
      {2, 4, 0, 3}, {2, 4, 5, 0},    {2, 4, 5, 2}};
  for (const auto& shape : invalid_shapes) {
    TestModelOptions options;
    options.shape = shape;
    MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
    EXPECT_EQ(ValidateTiledModelInputAndGetDims(*model).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  TestModelOptions options;
  options.tensor_type = tflite::TensorType_UINT8;
  MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
  EXPECT_EQ(ValidateTiledModelInputAndGetDims(*model).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ValidateTiledModelInputTest, RejectsZeroOrMultipleModelInputs) {
  for (int input_count : {0, 2}) {
    TestModelOptions options;
    options.input_count = input_count;
    MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
    EXPECT_EQ(ValidateTiledModelInputAndGetDims(*model).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(ValidateTiledModelInputTest, RejectsOutOfRangeInputTensorIndices) {
  for (int input_index : {-1, 1}) {
    TestModelOptions options;
    options.input_index_override = input_index;
    MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
    EXPECT_EQ(ValidateTiledModelInputAndGetDims(*model).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(ValidateTiledModelInputTest, RejectsMalformedShapeSignature) {
  TestModelOptions options;
  options.shape_signature = {-1, 4, 5};
  MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
  EXPECT_EQ(ValidateTiledModelInputAndGetDims(*model).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ValidateTiledModelInputTest,
     RejectsEmptyOrMultipleSubgraphMetadataEntries) {
  for (int metadata_subgraph_count : {0, 2}) {
    TestModelOptions options;
    options.metadata_subgraph_count = metadata_subgraph_count;
    MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
    EXPECT_EQ(ValidateTiledModelInputAndGetDims(*model).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(ValidateTiledModelInputTest, RejectsMismatchedColorMetadata) {
  for (int channels : {1, 3, 4}) {
    TestModelOptions options;
    options.shape[3] = channels;
    options.color_space = channels == 1 ? tflite::ColorSpaceType_RGB
                                        : tflite::ColorSpaceType_GRAYSCALE;
    MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
    EXPECT_EQ(ValidateTiledModelInputAndGetDims(*model).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(DetectionLabelMetadataTest, RejectsEmptyOrMultipleSubgraphMetadataEntries) {
  for (int metadata_subgraph_count : {0, 2}) {
    TestModelOptions options;
    options.metadata_subgraph_count = metadata_subgraph_count;
    MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
    EXPECT_EQ(GetLabelItemsFromMetadata(*model, "en").status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(DetectionLabelMetadataTest, ReturnsEmptyLabelsWhenMetadataOrLabelsAbsent) {
  for (bool include_metadata : {false, true}) {
    TestModelOptions options;
    options.include_metadata = include_metadata;
    MP_ASSERT_OK_AND_ASSIGN(auto model, CreateTestModel(options));
    MP_ASSERT_OK_AND_ASSIGN(auto labels,
                           GetLabelItemsFromMetadata(*model, "en"));
    EXPECT_TRUE(labels.empty());
  }
}

TEST(TilingEnabledTest, HandlesLargeDimensionsWithoutMultiplyingThem) {
  struct TilingOptions {
    int rows;
    int cols;
    int explicit_count;
    int tile_rows() const { return rows; }
    int tile_cols() const { return cols; }
    int explicit_tiles_size() const { return explicit_count; }
  };
  const int max = std::numeric_limits<int>::max();
  EXPECT_TRUE(TilingEnabled(TilingOptions{65536, 65536, 0}));
  EXPECT_FALSE(TilingEnabled(TilingOptions{1, 1, 0}));
  EXPECT_FALSE(TilingEnabled(TilingOptions{0, max, 0}));
  EXPECT_FALSE(TilingEnabled(TilingOptions{max, 0, 0}));
  EXPECT_TRUE(TilingEnabled(TilingOptions{1, 1, 1}));
}

// A non-positive (dynamic/unspecified) model batch dim is normalized to a
// dynamic batch with capacity 1, so it never trips the positive-capacity
// RET_CHECK downstream and the front emits a variable (valid-count) batch.
TEST(NormalizeTiledBatchDimTest, NonPositiveBecomesDynamicCapacityOne) {
  for (int raw : {0, -1, -8}) {
    const NormalizedBatchDim nb = NormalizeTiledBatchDim(raw);
    EXPECT_EQ(nb.batch_capacity, 1) << "raw=" << raw;
    EXPECT_TRUE(nb.is_dynamic) << "raw=" << raw;
  }
}

// A positive batch dim is a fixed batch: passed through unchanged, not dynamic.
TEST(NormalizeTiledBatchDimTest, PositiveIsFixedPassthrough) {
  for (int raw : {1, 2, 4, 16}) {
    const NormalizedBatchDim nb = NormalizeTiledBatchDim(raw);
    EXPECT_EQ(nb.batch_capacity, raw) << "raw=" << raw;
    EXPECT_FALSE(nb.is_dynamic) << "raw=" << raw;
  }
}

}  // namespace
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
