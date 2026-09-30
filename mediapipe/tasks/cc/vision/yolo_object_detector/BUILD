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
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "yolo_object_detector",
    srcs = ["yolo_object_detector.cc"],
    hdrs = ["yolo_object_detector.h"],
    deps = [
        ":yolo_object_detector_graph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/containers:detection_result",
        "//mediapipe/tasks/cc/core:base_options",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/core/proto:base_options_cc_proto",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "//mediapipe/tasks/cc/vision/core:base_vision_task_api",
        "//mediapipe/tasks/cc/vision/core:image_processing_options",
        "//mediapipe/tasks/cc/vision/core:running_mode",
        "//mediapipe/tasks/cc/vision/core:vision_task_api_factory",
        "//mediapipe/tasks/cc/vision/utils:tiled_detection_utils",
        "//mediapipe/tasks/cc/vision/yolo_object_detector/proto:yolo_object_detector_options_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//tflite/core/api:op_resolver",
    ],
)

cc_library(
    name = "yolo_object_detector_graph",
    srcs = ["yolo_object_detector_graph.cc"],
    deps = [
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/calculators/core:packet_presence_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/util:yolo_batch_detections_to_single_calculator",
        "//mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator",
        "//mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_cc_proto",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator_cc_proto",
        "//mediapipe/calculators/util:detection_projection_calculator",
        "//mediapipe/calculators/util:detection_transformation_calculator",
        "//mediapipe/calculators/util:detections_deduplicate_calculator",
        "//mediapipe/calculators/util:from_image_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:status",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/tasks/cc/components/processors:tiled_box_merge_graph",
        "//mediapipe/tasks/cc/components/processors:tiled_box_track_merge_graph",
        "//mediapipe/tasks/cc/components/processors:tiled_detection_front_graph",
        "//mediapipe/tasks/cc/components/processors:tiled_detection_stream_front_graph",
        "//mediapipe/tasks/cc/components/processors/proto:tiled_detection_graph_options_cc_proto",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/processors:image_preprocessing_graph",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "//mediapipe/tasks/cc/vision/utils:detection_label_resolution",
        "//mediapipe/tasks/cc/vision/utils:tiled_detection_utils",
        "//mediapipe/tasks/cc/vision/yolo_object_detector/proto:yolo_object_detector_options_cc_proto",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_test(
    name = "yolo_object_detector_test",
    srcs = ["yolo_object_detector_test.cc"],
    data = [
        # Image fixture (always present — fetched from GCS via mediapipe_files).
        "//mediapipe/tasks/testdata/vision:test_images",
        # Model fixture: empty unless exported locally (see export_yolov8n_tflite.py).
        "//mediapipe/tasks/testdata/vision:yolo_test_model",
    ],
    deps = [
        ":yolo_object_detector",
        "//mediapipe/framework/deps:file_helpers",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/tasks/cc/components/containers:detection_result",
        "//mediapipe/tasks/cc/vision/utils:image_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
