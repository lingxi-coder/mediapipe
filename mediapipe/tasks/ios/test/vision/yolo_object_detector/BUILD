load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load(
    "@litert//tflite:special_rules.bzl",
    "tflite_ios_lab_runner",
)
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//mediapipe/framework/tool:ios.bzl",
    "MPP_TASK_MINIMUM_OS_VERSION",
)

licenses(["notice"])

TFL_DEFAULT_TAGS = [
    "apple",
]

TFL_DISABLED_SANITIZER_TAGS = [
    "noasan",
    "nomsan",
    "notsan",
]

objc_library(
    name = "MPPYoloObjectDetectorObjcTestLibrary",
    testonly = True,
    srcs = ["MPPYoloObjectDetectorTests.m"],
    deps = [
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/vision/yolo_object_detector:MPPYoloObjectDetector",
        "//third_party/apple_frameworks:XCTest",
    ],
)

ios_unit_test(
    name = "MPPYoloObjectDetectorObjcTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":MPPYoloObjectDetectorObjcTestLibrary",
    ],
)
