# Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["notice"])

YOLO_OBJECT_DETECTOR_DEPS = [
    "@com_google_absl//absl/log:absl_log",
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
    srcs = ["yolo_object_detector.cc"],
    hdrs = ["yolo_object_detector.h"],
    visibility = ["//visibility:public"],
    deps = YOLO_OBJECT_DETECTOR_DEPS,
)

cc_library(
    name = "yolo_object_detector_c_lib",
    srcs = ["yolo_object_detector.cc"],
    hdrs = ["yolo_object_detector.h"],
    visibility = ["//mediapipe/tasks/c:__subpackages__"],
    deps = YOLO_OBJECT_DETECTOR_DEPS,
    alwayslink = 1,
)
