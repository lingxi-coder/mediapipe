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

#include <cmath>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "flatbuffers/flatbuffers.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "tflite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {

NormalizedBatchDim NormalizeTiledBatchDim(int raw_batch) {
  if (raw_batch <= 0) {
    return NormalizedBatchDim{/*batch_capacity=*/1, /*is_dynamic=*/true};
  }
  return NormalizedBatchDim{/*batch_capacity=*/raw_batch, /*is_dynamic=*/false};
}

absl::StatusOr<TiledModelInputDims> ValidateTiledModelInputAndGetDims(
    const tasks::core::ModelResources& model_resources) {
  const auto& model = *model_resources.GetTfLiteModel();
  if (model.subgraphs() == nullptr || model.subgraphs()->size() != 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiled mode expects exactly one model subgraph",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  const tflite::SubGraph* sg = model.subgraphs()->Get(0);
  if (sg->inputs() == nullptr || sg->inputs()->size() != 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiled mode expects exactly one image input tensor",
        MediaPipeTasksStatus::kInvalidNumInputTensorsError);
  }
  const int input_index = sg->inputs()->Get(0);
  if (sg->tensors() == nullptr || input_index < 0 ||
      static_cast<size_t>(input_index) >= sg->tensors()->size()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiled mode image input tensor index is out of range",
        MediaPipeTasksStatus::kInvalidNumInputTensorsError);
  }
  const auto* input_tensor = sg->tensors()->Get(input_index);
  const auto* dims = input_tensor->shape();
  if (dims == nullptr || dims->size() != 4) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiled mode expects a [N,H,W,C] image input tensor",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (input_tensor->type() != tflite::TensorType_FLOAT32) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiled mode currently supports float32 image input only",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (dims->Get(1) <= 0 || dims->Get(2) <= 0 ||
      (dims->Get(3) != 1 && dims->Get(3) != 3 && dims->Get(3) != 4)) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiled mode expects positive image height and width and 1, 3, or 4 "
        "channels",
        MediaPipeTasksStatus::kInvalidInputTensorDimensionsError);
  }

  // The general image preprocessing spec builder requires batch size one and
  // normalization metadata. Tiled preprocessing has a different contract:
  // fixed/dynamic batches are supported, and missing normalization means /255.
  // Parse the metadata directly so neither difference can hide invalid
  // metadata.
  ABSL_ASSIGN_OR_RETURN(
      const auto* tensor_metadata,
      GetImageTensorMetadataIfAny(*model_resources.GetMetadataExtractor(), 0));
  std::optional<NormalizationOptions> normalization_options;
  if (tensor_metadata != nullptr) {
    ABSL_ASSIGN_OR_RETURN(const auto* image_properties,
                          GetImagePropertiesIfAny(*tensor_metadata));
    const auto expected_color_space = dims->Get(3) == 1
                                          ? tflite::ColorSpaceType_GRAYSCALE
                                          : tflite::ColorSpaceType_RGB;
    if (image_properties != nullptr &&
        image_properties->color_space() != expected_color_space) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "tiled mode image color metadata must match the input channels: "
          "GRAYSCALE for 1 channel, RGB for 3 or 4 channels",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    ABSL_ASSIGN_OR_RETURN(normalization_options,
                          GetNormalizationOptionsIfAny(*tensor_metadata));
  }
  if (normalization_options.has_value()) {
    const auto& norm = *normalization_options;
    if (dims->Get(3) == 1 && norm.num_values != 1) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "tiled grayscale input requires scalar normalization parameters",
          MediaPipeTasksStatus::kMetadataInvalidProcessUnitsError);
    }
    for (int i = 0; i < norm.num_values; ++i) {
      if (!std::isfinite(norm.mean_values[i]) ||
          !std::isfinite(norm.std_values[i]) ||
          std::abs(norm.mean_values[i]) > 1e-3f ||
          std::abs(norm.std_values[i] - 255.0f) > 1e-3f) {
        return CreateStatusWithPayload(
            absl::StatusCode::kInvalidArgument,
            "tiled mode supports only (mean=0, std=255) input "
            "normalization (implicit x/255), but the model metadata "
            "requests a different normalization",
            MediaPipeTasksStatus::kInvalidArgumentError);
      }
    }
  }
  const auto* shape_signature = input_tensor->shape_signature();
  if (shape_signature != nullptr && shape_signature->size() != 0 &&
      shape_signature->size() != 4) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiled mode expects a [N,H,W,C] image input shape signature",
        MediaPipeTasksStatus::kInvalidInputTensorDimensionsError);
  }
  // TFLite commonly stores a positive allocated batch in shape() while -1 in
  // shape_signature() marks that axis as resizable.
  const int batch_dim = shape_signature != nullptr &&
                                shape_signature->size() == 4 &&
                                shape_signature->Get(0) <= 0
                            ? shape_signature->Get(0)
                            : dims->Get(0);
  const NormalizedBatchDim nb = NormalizeTiledBatchDim(batch_dim);
  return TiledModelInputDims{/*batch=*/nb.batch_capacity,
                             /*height=*/dims->Get(1), /*width=*/dims->Get(2),
                             /*channels=*/dims->Get(3),
                             /*is_dynamic_batch=*/nb.is_dynamic};
}

absl::Status CheckTiledImageProcessingOptions(
    const std::optional<core::ImageProcessingOptions>&
        image_processing_options) {
  if (!image_processing_options.has_value()) {
    return absl::OkStatus();
  }
  if (image_processing_options->region_of_interest.has_value()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiling and ROI are mutually exclusive",
        MediaPipeTasksStatus::kImageProcessingInvalidArgumentError);
  }
  if (image_processing_options->rotation_degrees != 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiling does not support rotation_degrees",
        MediaPipeTasksStatus::kImageProcessingInvalidArgumentError);
  }
  return absl::OkStatus();
}

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
