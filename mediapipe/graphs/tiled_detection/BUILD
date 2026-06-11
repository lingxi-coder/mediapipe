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
#

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_proto_library(
    name = "tiled_detection_graphs_proto",
    srcs = ["tiled_detection_graphs.proto"],
    deps = [
        "//mediapipe/calculators/tensor:tile_grid_calculator_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "tiled_detection_front_graph",
    srcs = ["tiled_detection_front_graph.cc"],
    deps = [
        ":tiled_detection_graphs_cc_proto",
        "//mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator",
        "//mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tile_grid_calculator",
        "//mediapipe/calculators/tensor:tile_grid_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tile_spec_to_tile_plan_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:image_frame",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tiled_obb_merge_graph",
    srcs = ["tiled_obb_merge_graph.cc"],
    deps = [
        ":tiled_detection_graphs_cc_proto",
        "//mediapipe/calculators/tensor:merge_tile_detections_accumulator_calculator",
        "//mediapipe/calculators/util:rotated_non_max_suppression_calculator",
        "//mediapipe/calculators/util:rotated_non_max_suppression_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tiled_box_merge_graph",
    srcs = ["tiled_box_merge_graph.cc"],
    deps = [
        ":tiled_detection_graphs_cc_proto",
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator_cc_proto",
        "//mediapipe/calculators/tensor:merge_tile_box_detections_accumulator_calculator",
        "//mediapipe/calculators/tensor:tiled_frame_suppression_calculator",
        "//mediapipe/calculators/tensor:tiled_frame_suppression_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_test(
    name = "tiled_detection_graphs_test",
    size = "small",
    srcs = ["tiled_detection_graphs_test.cc"],
    deps = [
        ":tiled_box_merge_graph",
        ":tiled_detection_front_graph",
        ":tiled_detection_graphs_cc_proto",
        ":tiled_obb_merge_graph",
        "//mediapipe/calculators/tensor:tiling_matrix_utils",
        "//mediapipe/calculators/tensor:tiling_types",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:oriented_detection_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
        "@com_google_absl//absl/status",
    ],
)
