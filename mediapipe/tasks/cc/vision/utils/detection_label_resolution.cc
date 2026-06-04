/* Copyright 2024 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/vision/utils/detection_label_resolution.h"

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/label_map_util.h"

namespace mediapipe {
namespace tasks {
namespace vision {

namespace {

using ::mediapipe::tasks::metadata::ModelMetadataExtractor;

}  // namespace

absl::StatusOr<mediapipe::proto_ns::Map<int64_t, mediapipe::LabelMapItem>>
GetLabelItemsFromMetadata(const core::ModelResources& model_resources,
                          absl::string_view display_names_locale) {
  using LabelItems = mediapipe::proto_ns::Map<int64_t, mediapipe::LabelMapItem>;

  const ModelMetadataExtractor* metadata_extractor =
      model_resources.GetMetadataExtractor();

  // If there is no output tensor metadata at all, return empty map.
  const auto* output_tensors_metadata =
      metadata_extractor->GetOutputTensorMetadata();
  if (output_tensors_metadata == nullptr ||
      output_tensors_metadata->size() == 0) {
    return LabelItems();
  }

  // Use the first output tensor (index 0), which is appropriate for
  // single-output-tensor YOLO/OBB models that pack all scores/classes into
  // one combined output tensor.
  const tflite::TensorMetadata* tensor_metadata =
      output_tensors_metadata->Get(0);
  if (tensor_metadata == nullptr) {
    return LabelItems();
  }

  // Find the TENSOR_AXIS_LABELS associated file name (empty = no label file).
  const std::string labels_filename =
      ModelMetadataExtractor::FindFirstAssociatedFileName(
          *tensor_metadata,
          tflite::AssociatedFileType_TENSOR_AXIS_LABELS);
  if (labels_filename.empty()) {
    return LabelItems();
  }

  // Read label file contents.
  MP_ASSIGN_OR_RETURN(absl::string_view labels_file,
                      metadata_extractor->GetAssociatedFile(labels_filename));

  // Optionally read locale-matched display names file.
  const std::string display_names_filename =
      ModelMetadataExtractor::FindFirstAssociatedFileName(
          *tensor_metadata,
          tflite::AssociatedFileType_TENSOR_AXIS_LABELS,
          display_names_locale);
  absl::string_view display_names_file;
  if (!display_names_filename.empty()) {
    MP_ASSIGN_OR_RETURN(
        display_names_file,
        metadata_extractor->GetAssociatedFile(display_names_filename));
  }

  return mediapipe::BuildLabelMapFromFiles(labels_file, display_names_file);
}

absl::StatusOr<absl::flat_hash_set<int>> ResolveCategoryIndices(
    const mediapipe::proto_ns::Map<int64_t, mediapipe::LabelMapItem>&
        label_items,
    const google::protobuf::RepeatedPtrField<std::string>& allowlist,
    const google::protobuf::RepeatedPtrField<std::string>& denylist) {
  absl::flat_hash_set<int> category_indices;

  // No filtering requested: return empty set (no-op).
  if (allowlist.empty() && denylist.empty()) {
    return category_indices;
  }

  // Filter requested but no label metadata available.
  if (label_items.empty()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Using `category_allowlist` or `category_denylist` requires "
        "labels to be present in the TFLite Model Metadata but none was found.",
        MediaPipeTasksStatus::kMetadataMissingLabelsError);
  }

  // allowlist takes priority if both are set (caller contract: only one should
  // be set, but be robust).
  const auto& category_list =
      !allowlist.empty() ? allowlist : denylist;

  for (const auto& category_name : category_list) {
    int index = -1;
    for (int i = 0; i < static_cast<int>(label_items.size()); ++i) {
      if (label_items.at(i).name() == category_name) {
        index = i;
        break;
      }
    }
    // Ignore unknown or duplicate names.
    if (index < 0) {
      continue;
    }
    category_indices.insert(index);
  }
  return category_indices;
}

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
