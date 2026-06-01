# Copyright 2026 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "oriented_object_detector_graph",
    srcs = ["oriented_object_detector_graph.cc"],
    deps = [
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:yolo_obb_batch_detections_to_single_calculator",
        "//mediapipe/calculators/tensor:yolo_obb_tensors_to_oriented_detections_calculator",
        "//mediapipe/calculators/tensor:yolo_obb_tensors_to_oriented_detections_calculator_cc_proto",
        "//mediapipe/calculators/util:oriented_detection_projection_calculator",
        "//mediapipe/calculators/util:rotated_non_max_suppression_calculator",
        "//mediapipe/calculators/util:rotated_non_max_suppression_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:oriented_detection_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/processors:image_preprocessing_graph",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "//mediapipe/tasks/cc/vision/oriented_object_detector/proto:oriented_object_detector_options_cc_proto",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)
