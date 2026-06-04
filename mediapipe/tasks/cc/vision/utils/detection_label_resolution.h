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

#ifndef MEDIAPIPE_TASKS_CC_VISION_UTILS_DETECTION_LABEL_RESOLUTION_H_
#define MEDIAPIPE_TASKS_CC_VISION_UTILS_DETECTION_LABEL_RESOLUTION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/repeated_field.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/util/label_map.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {

// Reads the output-tensor TENSOR_AXIS_LABELS associated label file (+ locale
// display names if present) from the model metadata into a label-items map.
// Returns an empty map if the model has no label file or no output tensor
// metadata.
//
// Uses output tensor index 0 (appropriate for single-output-tensor YOLO/OBB
// models). The `display_names_locale` may be empty; locale display names are
// optional.
absl::StatusOr<mediapipe::proto_ns::Map<int64_t, mediapipe::LabelMapItem>>
GetLabelItemsFromMetadata(const core::ModelResources& model_resources,
                          absl::string_view display_names_locale);

// Resolves category allow/deny NAMES -> class INDEX set using label_items.
//
// Semantics:
//   - Both lists empty  -> empty set returned (no filtering).
//   - allowlist/denylist non-empty but label_items empty ->
//       kMetadataMissingLabelsError (caller must provide label metadata).
//   - Unknown/duplicate names are silently ignored; an all-unknown allowlist
//       yields an empty set (treated as no-op by the downstream calculator).
//   - Only one of allowlist/denylist should be non-empty (caller contract);
//       allowlist takes priority when both are unexpectedly set.
absl::StatusOr<absl::flat_hash_set<int>> ResolveCategoryIndices(
    const mediapipe::proto_ns::Map<int64_t, mediapipe::LabelMapItem>&
        label_items,
    const google::protobuf::RepeatedPtrField<std::string>& allowlist,
    const google::protobuf::RepeatedPtrField<std::string>& denylist);

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_UTILS_DETECTION_LABEL_RESOLUTION_H_
