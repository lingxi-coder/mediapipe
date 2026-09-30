# Copyright 2022 The MediaPipe Authors.
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

load("@litert//tflite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "classifier_options",
    srcs = ["classifier_options.cc"],
    hdrs = ["classifier_options.h"],
    deps = ["//mediapipe/tasks/cc/components/processors/proto:classifier_options_cc_proto"],
)

cc_library(
    name = "classification_postprocessing_graph",
    srcs = ["classification_postprocessing_graph.cc"],
    hdrs = ["classification_postprocessing_graph.h"],
    deps = [
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/core:split_vector_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tensors_dequantization_calculator",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/calculators:classification_aggregation_calculator",
        "//mediapipe/tasks/cc/components/calculators:classification_aggregation_calculator_cc_proto",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_calculator",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_calculator_cc_proto",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_utils",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:classification_postprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "//mediapipe/util:label_map_cc_proto",
        "//mediapipe/util:label_map_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@litert//tflite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

cc_library(
    name = "embedder_options",
    srcs = ["embedder_options.cc"],
    hdrs = ["embedder_options.h"],
    deps = ["//mediapipe/tasks/cc/components/processors/proto:embedder_options_cc_proto"],
)

cc_library(
    name = "embedding_postprocessing_graph",
    srcs = ["embedding_postprocessing_graph.cc"],
    hdrs = ["embedding_postprocessing_graph.h"],
    deps = [
        "//mediapipe/calculators/tensor:tensors_dequantization_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/tool:options_map",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/calculators:embedding_aggregation_calculator",
        "//mediapipe/tasks/cc/components/calculators:tensors_to_embeddings_calculator",
        "//mediapipe/tasks/cc/components/calculators:tensors_to_embeddings_calculator_cc_proto",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:embedder_options_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:embedding_postprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@litert//tflite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "image_preprocessing_graph",
    srcs = ["image_preprocessing_graph.cc"],
    hdrs = ["image_preprocessing_graph.h"],
    tflite_deps = [
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/vision/utils:image_tensor_specs",
    ],
    deps = [
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/calculators/image:image_clone_calculator",
        "//mediapipe/calculators/image:image_clone_calculator_cc_proto",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:inference_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/gpu:gpu_origin_cc_proto",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/processors/proto:image_preprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/core/proto:acceleration_cc_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@litert//tflite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

# TODO: Enable this test

cc_library(
    name = "text_preprocessing_graph",
    srcs = ["text_preprocessing_graph.cc"],
    hdrs = ["text_preprocessing_graph.h"],
    deps = [
        "//mediapipe/calculators/tensor:bert_preprocessor_calculator",
        "//mediapipe/calculators/tensor:bert_preprocessor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:regex_preprocessor_calculator",
        "//mediapipe/calculators/tensor:regex_preprocessor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:sentencepiece_preprocessor_calculator",
        "//mediapipe/calculators/tensor:sentencepiece_preprocessor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:text_to_tensor_calculator",
        "//mediapipe/calculators/tensor:universal_sentence_encoder_preprocessor_calculator",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/processors/proto:text_model_type_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:text_preprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "//mediapipe/tasks/cc/text/utils:text_model_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "detection_postprocessing_graph",
    srcs = ["detection_postprocessing_graph.cc"],
    hdrs = ["detection_postprocessing_graph.h"],
    deps = [
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/core:split_vector_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tensors_dequantization_calculator",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator_cc_proto",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator_cc_proto",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator_cc_proto",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/formats/object_detection:anchor_cc_proto",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_calculator",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_calculator_cc_proto",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_utils",
        "//mediapipe/tasks/cc/components/processors/proto:detection_postprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:detector_options_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "//mediapipe/tasks/cc/vision/object_detector/proto:object_detector_options_cc_proto",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "//mediapipe/tasks/metadata:object_detector_metadata_schema_cc",
        "//mediapipe/util:label_map_cc_proto",
        "//mediapipe/util:label_map_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers//:runtime_cc",
        "@litert//tflite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

# Metal env required. End-to-end: Metal tiled zero-copy preprocessing ->
# InferenceCalculatorMetal direct external-input, at batch N=1. On non-apple:
# empty TU.
cc_test(
    name = "streaming_to_inference_metal_zero_copy_test",
    srcs = ["streaming_to_inference_metal_zero_copy_test.cc"],
    data = select({
        "//mediapipe:apple": ["//mediapipe/calculators/tensor:testdata/1x256x256x3_softmax.tflite"],
        "//conditions:default": [],
    }),
    deps = [
        "//mediapipe/framework:port",
        "//mediapipe/framework/port:gtest_main",
    ] + select({
        "//mediapipe:apple": [
            "//mediapipe/calculators/tensor:inference_calculator",
            "//mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator",
            "//mediapipe/framework/formats:tiling_types",
            "//mediapipe/framework:calculator_framework",
            "//mediapipe/framework/formats:inference_metadata_cc_proto",
            "//mediapipe/framework/formats:tensor",
            "//mediapipe/framework/port:parse_text_proto",
            "//mediapipe/framework/port:status_matchers",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:gpu_shared_data_internal",
            "//mediapipe/util:image_test_utils",
            "@com_google_absl//absl/status",
        ],
        "//conditions:default": [],
    }),
)

# Metal env required (runs on this Mac, Apple M2 Max). END-TO-END proof of TRUE
# Metal zero-copy through a real tiled OBB detection pipeline: GPU frame ->
# direct-PHWC4 tiled front -> direct Metal delegate input (converter skipped) ->
# OBB decode -> tile->global merge -> global rotated NMS. Asserts converter-
# skipped direct bind + buffer identity (ZERO_COPY_DEBUG), no CPU fallback, one
# merged result, equivalence vs the CPU-front Metal-inference control, and a
# ship. Gated on the yolov8n-obb.tflite + boats.jpg fixtures. On non-apple the
# source compiles to an empty TU.
cc_test(
    name = "tiled_zero_copy_detection_metal_test",
    srcs = ["tiled_zero_copy_detection_metal_test.cc"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",
        ],
        "//conditions:default": [],
    }),
    # yolo_obb_test_model globs BOTH yolov8n-obb.tflite AND boats.jpg, and is
    # the fixture target whose visibility was widened to this package. (The
    # general :test_images group keeps the package-default //mediapipe/tasks
    # visibility, so it is intentionally NOT used here.)
    data = select({
        "//mediapipe:apple": [
            "//mediapipe/tasks/testdata/vision:yolo_obb_test_model",
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "//mediapipe:apple": [
            "-framework Metal",
            "-framework Foundation",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    deps = [
        "//mediapipe/framework:port",
        "//mediapipe/framework/port:gtest_main",
    ] + select({
        "//mediapipe:apple": [
            "//mediapipe/calculators/tensor:inference_calculator",
            "//mediapipe/calculators/util:merge_tile_detections_accumulator_calculator",
            "//mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator",
            "//mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_cc_proto",
            "//mediapipe/framework/formats:tiling_cache_stats",
            "//mediapipe/framework/formats:tiling_types",
            "//mediapipe/calculators/tensor:yolo_obb_tensors_to_oriented_detections_calculator",
            "//mediapipe/calculators/core:pass_through_calculator",
            "//mediapipe/calculators/util:rotated_non_max_suppression_calculator",
            "//mediapipe/framework:calculator_framework",
            "//mediapipe/framework/formats:image_frame",
            "//mediapipe/framework/formats:inference_metadata_cc_proto",
            "//mediapipe/framework/formats:oriented_detection_cc_proto",
            "//mediapipe/framework/formats:tensor",
            "//mediapipe/framework/port:file_helpers",
            "//mediapipe/framework/port:opencv_core",
            "//mediapipe/framework/port:opencv_imgcodecs",
            "//mediapipe/framework/port:parse_text_proto",
            "//mediapipe/framework/port:status",
            "//mediapipe/framework/port:status_matchers",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:gpu_buffer_format",
            "//mediapipe/gpu:gpu_shared_data_internal",
            "//mediapipe/util:image_test_utils",
            "@com_google_absl//absl/status",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "video_tile_scheduler_pipeline_test",
    srcs = ["video_tile_scheduler_pipeline_test.cc"],
    size = "small",
    deps = [
        "//mediapipe/calculators/util:merge_tile_detections_accumulator_calculator",
        "//mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator",
        "//mediapipe/calculators/util:tile_spec_to_tile_plan_calculator",
        "//mediapipe/framework/formats:tiling_cache_stats",
        "//mediapipe/framework/formats:tiling_types",
        "//mediapipe/calculators/video:video_tile_scheduler_calculator",
        "//mediapipe/calculators/tensor:yolo_obb_tensors_to_oriented_detections_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:inference_metadata_cc_proto",
        "//mediapipe/framework/formats:oriented_detection_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
    ],
)

cc_test(
    name = "tiled_obb_pipeline_test",
    srcs = ["tiled_obb_pipeline_test.cc"],
    size = "small",
    deps = [
        "//mediapipe/calculators/util:merge_tile_detections_accumulator_calculator",
        "//mediapipe/util:tiling_matrix_utils",
        "//mediapipe/framework/formats:tiling_types",
        "//mediapipe/calculators/util:rotated_non_max_suppression_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:oriented_detection_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
    ],
)

cc_test(
    name = "video_tile_scheduler_e2e_test",
    srcs = ["video_tile_scheduler_e2e_test.cc"],
    size = "small",
    deps = [
        "//mediapipe/calculators/util:tiled_frame_suppression_calculator",
        "//mediapipe/calculators/video:video_tile_scheduler_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
        "//mediapipe/util/tracking:flow_packager_cc_proto",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "tiled_detection_front_graph",
    srcs = ["tiled_detection_front_graph.cc"],
    deps = [
        "//mediapipe/tasks/cc/components/processors/proto:tiled_detection_graph_options_cc_proto",
        "//mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator",
        "//mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_cc_proto",
        "//mediapipe/calculators/util:tile_grid_calculator",
        "//mediapipe/calculators/util:tile_grid_calculator_cc_proto",
        "//mediapipe/calculators/util:tile_spec_to_tile_plan_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:image_frame",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tiled_detection_stream_front_graph",
    srcs = ["tiled_detection_stream_front_graph.cc"],
    deps = [
        "//mediapipe/graphs/tracking/subgraphs:optical_flow_tracking_graph",
        "//mediapipe/tasks/cc/components/processors/proto:tiled_detection_graph_options_cc_proto",
        "//mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator",
        "//mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_cc_proto",
        "//mediapipe/calculators/util:tile_grid_calculator",
        "//mediapipe/calculators/util:tile_grid_calculator_cc_proto",
        "//mediapipe/calculators/util:tile_spec_to_tile_plan_calculator",
        "//mediapipe/calculators/video:video_tile_scheduler_calculator",
        "//mediapipe/calculators/video:video_tile_scheduler_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tiled_obb_merge_graph",
    srcs = [
        "tiled_obb_merge_graph.cc",
        "tiled_obb_track_merge_graph.cc",
    ],
    deps = [
        "//mediapipe/tasks/cc/components/processors/proto:tiled_detection_graph_options_cc_proto",
        "//mediapipe/calculators/video:botsort_tracking_calculator_cc_proto",
        "//mediapipe/calculators/util:merge_tile_detections_accumulator_calculator",
        "//mediapipe/calculators/video:oriented_botsort_tracking_calculator",
        "//mediapipe/calculators/util:rotated_non_max_suppression_calculator",
        "//mediapipe/calculators/util:rotated_non_max_suppression_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:oriented_detection_cc_proto",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tiled_box_merge_graph",
    srcs = ["tiled_box_merge_graph.cc"],
    deps = [
        "//mediapipe/tasks/cc/components/processors/proto:tiled_detection_graph_options_cc_proto",
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator_cc_proto",
        "//mediapipe/calculators/util:merge_tile_box_detections_accumulator_calculator",
        "//mediapipe/framework/formats:tiling_types",
        "//mediapipe/calculators/util:tiled_frame_suppression_calculator",
        "//mediapipe/calculators/util:tiled_frame_suppression_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tiled_box_track_merge_graph",
    srcs = ["tiled_box_track_merge_graph.cc"],
    deps = [
        "//mediapipe/tasks/cc/components/processors/proto:tiled_detection_graph_options_cc_proto",
        ":tiled_tracking_graph",
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator_cc_proto",
        "//mediapipe/calculators/util:merge_tile_box_detections_accumulator_calculator",
        "//mediapipe/calculators/util:tiled_frame_suppression_calculator",
        "//mediapipe/calculators/util:tiled_frame_suppression_calculator_cc_proto",
        "//mediapipe/framework/formats:tiling_types",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tiled_tracking_graph",
    srcs = ["tiled_tracking_graph.cc"],
    deps = [
        "//mediapipe/tasks/cc/components/processors/proto:tiled_detection_graph_options_cc_proto",
        "//mediapipe/calculators/video:botsort_tracking_calculator",
        "//mediapipe/calculators/video:botsort_tracking_calculator_cc_proto",
        "//mediapipe/calculators/util:detection_label_id_codec_calculator",
        "//mediapipe/calculators/util:detection_label_id_codec_calculator_cc_proto",
        "//mediapipe/calculators/util:detections_tick_gate_calculator",
        "//mediapipe/framework/formats:tiling_types",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/graphs/tracking/subgraphs:object_tracking_cpu",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_test(
    name = "tiled_tracking_graphs_test",
    size = "medium",
    srcs = ["tiled_tracking_graphs_test.cc"],
    deps = [
        ":tiled_box_track_merge_graph",
        ":tiled_detection_front_graph",
        "//mediapipe/tasks/cc/components/processors/proto:tiled_detection_graph_options_cc_proto",
        ":tiled_detection_stream_front_graph",
        ":tiled_obb_merge_graph",
        ":tiled_tracking_graph",
        "//mediapipe/calculators/core:packet_presence_calculator",
        "//mediapipe/calculators/video:botsort_tracking_calculator",
        "//mediapipe/calculators/video:oriented_botsort_tracking_calculator",
        "//mediapipe/util:tiling_matrix_utils",
        "//mediapipe/framework/formats:tiling_types",
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
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "tiled_detection_graphs_test",
    size = "small",
    srcs = ["tiled_detection_graphs_test.cc"],
    deps = [
        ":tiled_box_merge_graph",
        ":tiled_detection_front_graph",
        "//mediapipe/tasks/cc/components/processors/proto:tiled_detection_graph_options_cc_proto",
        ":tiled_obb_merge_graph",
        "//mediapipe/util:tiling_matrix_utils",
        "//mediapipe/framework/formats:tiling_types",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:oriented_detection_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:status_matchers",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "tiled_video_scheduler_graphs_test",
    size = "medium",
    srcs = ["tiled_video_scheduler_graphs_test.cc"],
    deps = [
        "//mediapipe/graphs/tracking/subgraphs:optical_flow_tracking_graph",
        ":tiled_detection_stream_front_graph",
        "//mediapipe/framework/formats:tiling_types",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
        "//mediapipe/util/tracking:tracking",
        "@com_google_absl//absl/status",
    ],
)
