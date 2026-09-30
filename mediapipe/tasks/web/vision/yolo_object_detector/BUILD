# This contains the MediaPipe YOLO Object Detector Task.
#
# This task requires a custom Wasm binary that registers
# mediapipe.tasks.vision.yolo_object_detector.YoloObjectDetectorGraph.

load("//mediapipe/framework/port:build_config.bzl", "ts_declaration", "ts_library")
load("//mediapipe/tasks/web:build_defs.bzl", "jasmine_node_test")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

ts_library(
    name = "yolo_object_detector",
    srcs = ["yolo_object_detector.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":yolo_object_detector_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/framework/formats:detection_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/yolo_object_detector/proto:yolo_object_detector_options_jspb_proto",
        "//mediapipe/tasks/web/components/processors:detection_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/vision/core:custom_vision_wasm_fileset",
        "//mediapipe/tasks/web/vision/core:image_processing_options",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

ts_declaration(
    name = "yolo_object_detector_types",
    srcs = [
        "yolo_object_detector_options.d.ts",
        "yolo_object_detector_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/components/containers:detection_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/vision/core:vision_task_options",
    ],
)

ts_library(
    name = "yolo_object_detector_test_lib",
    testonly = True,
    srcs = ["yolo_object_detector_test.ts"],
    deps = [
        ":yolo_object_detector",
        ":yolo_object_detector_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework/formats:detection_jspb_proto",
        "//mediapipe/framework/formats:location_data_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
        "//mediapipe/tasks/web/vision/core:custom_vision_wasm_fileset",
    ],
)

jasmine_node_test(
    name = "yolo_object_detector_test",
    srcs = [":yolo_object_detector_test_lib"],
    tags = ["nomsan"],
)
