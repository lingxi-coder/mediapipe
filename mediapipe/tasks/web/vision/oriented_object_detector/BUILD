# This contains the MediaPipe oriented object detector task.
#
# This task requires a custom Wasm binary that registers
# mediapipe.tasks.vision.oriented_object_detector.OrientedObjectDetectorGraph.

load("//mediapipe/framework/port:build_config.bzl", "ts_declaration", "ts_library")
load("//mediapipe/tasks/web:build_defs.bzl", "jasmine_node_test")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

ts_library(
    name = "oriented_object_detector",
    srcs = ["oriented_object_detector.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":oriented_object_detector_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/framework/formats:oriented_detection_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/oriented_object_detector/proto:oriented_object_detector_options_jspb_proto",
        "//mediapipe/tasks/web/components/processors:oriented_detection_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/vision/core:custom_vision_wasm_fileset",
        "//mediapipe/tasks/web/vision/core:image_processing_options",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

ts_declaration(
    name = "oriented_object_detector_types",
    srcs = [
        "oriented_object_detector_options.d.ts",
        "oriented_object_detector_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/components/containers:oriented_detection_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/vision/core:vision_task_options",
    ],
)

ts_library(
    name = "oriented_object_detector_test_lib",
    testonly = True,
    srcs = ["oriented_object_detector_test.ts"],
    deps = [
        ":oriented_object_detector",
        ":oriented_object_detector_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework/formats:oriented_detection_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
        "//mediapipe/tasks/web/vision/core:custom_vision_wasm_fileset",
    ],
)

jasmine_node_test(
    name = "oriented_object_detector_test",
    srcs = [":oriented_object_detector_test_lib"],
    tags = ["nomsan"],
)
