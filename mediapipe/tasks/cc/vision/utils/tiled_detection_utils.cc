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
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "flatbuffers/flatbuffers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {

absl::StatusOr<const flatbuffers::Vector<int32_t>*>
ValidateTiledModelInputAndGetDims(const core::ModelResources& model_resources) {
  const auto& model = *model_resources.GetTfLiteModel();
  const tflite::SubGraph* sg = model.subgraphs()->Get(0);
  const auto* input_tensor = sg->tensors()->Get(sg->inputs()->Get(0));
  const auto* dims = input_tensor->shape();
  RET_CHECK(dims != nullptr && dims->size() == 4)
      << "tiled mode expects a [N,H,W,C] image input tensor";
  RET_CHECK_EQ(input_tensor->type(), tflite::TensorType_FLOAT32)
      << "tiled mode currently supports float32 image input only";
  auto specs_or = BuildInputImageTensorSpecs(model_resources);
  if (specs_or.ok() && specs_or->normalization_options.has_value()) {
    const auto& norm = *specs_or->normalization_options;
    for (int i = 0; i < norm.num_values; ++i) {
      if (std::abs(norm.mean_values[i]) > 1e-3f ||
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
  return dims;
}

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
