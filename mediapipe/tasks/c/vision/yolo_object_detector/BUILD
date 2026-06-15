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

licenses(["notice"])

YOLO_OBJECT_DETECTOR_DEPS = [
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/log:absl_check",
    "//mediapipe/framework/formats:image",
    "//mediapipe/tasks/c/components/containers:detection_result",
    "//mediapipe/tasks/c/components/containers:detection_result_converter",
    "//mediapipe/tasks/c/core:base_options",
    "//mediapipe/tasks/c/core:base_options_converter",
    "//mediapipe/tasks/c/core:common",
    "//mediapipe/tasks/c/core:mp_status",
    "//mediapipe/tasks/c/core:mp_status_converter",
    "//mediapipe/tasks/c/vision/core:image",
    "//mediapipe/tasks/c/vision/core:image_frame_util",
    "//mediapipe/tasks/c/vision/core:image_processing_options",
    "//mediapipe/tasks/c/vision/core:image_processing_options_converter",
    "//mediapipe/tasks/cc/vision/core:image_processing_options",
    "//mediapipe/tasks/cc/vision/core:running_mode",
    "//mediapipe/tasks/cc/vision/yolo_object_detector",
]

cc_library(
    name = "yolo_object_detector_lib",
    srcs = [
        "tiling_options_converter.cc",
        "yolo_object_detector.cc",
    ],
    hdrs = [
        "tiling_options_converter.h",
        "yolo_object_detector.h",
    ],
    visibility = ["//visibility:public"],
    deps = YOLO_OBJECT_DETECTOR_DEPS,
)

cc_test(
    name = "yolo_object_detector_test",
    srcs = ["yolo_object_detector_test.cc"],
    data = [
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/tasks/testdata/vision:test_images",
        # Uncomment when yolov8n.tflite is added to testdata/vision/BUILD:
        # "//mediapipe/tasks/testdata/vision:yolo_test_models",
    ],
    linkstatic = 1,
    deps = [
        ":yolo_object_detector_lib",
        "//mediapipe/framework/deps:file_helpers",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/c/core:common",
        "//mediapipe/tasks/c/core:mp_status",
        "//mediapipe/tasks/c/vision/core:image",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tiling_options_converter_test",
    srcs = ["tiling_options_converter_test.cc"],
    deps = [
        ":yolo_object_detector_lib",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/vision/yolo_object_detector",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "yolo_object_detector_c_lib",
    srcs = [
        "tiling_options_converter.cc",
        "yolo_object_detector.cc",
    ],
    hdrs = [
        "tiling_options_converter.h",
        "yolo_object_detector.h",
    ],
    visibility = ["//mediapipe/tasks/c:__subpackages__"],
    deps = YOLO_OBJECT_DETECTOR_DEPS,
    alwayslink = 1,
)
