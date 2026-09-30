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

load("@rules_cc//cc:objc_library.bzl", "objc_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

exports_files(glob(["sources/*.h"]))

objc_library(
    name = "MPPYoloObjectDetectorOptions",
    srcs = ["sources/MPPYoloObjectDetectorOptions.m"],
    hdrs = ["sources/MPPYoloObjectDetectorOptions.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/vision/core:MPPRunningMode",
        "//mediapipe/tasks/ios/vision/object_detector:MPPObjectDetectorResult",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPYoloObjectDetector",
    srcs = ["sources/MPPYoloObjectDetector.mm"],
    hdrs = ["sources/MPPYoloObjectDetector.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    deps = [
        ":MPPYoloObjectDetectorOptions",
        "//mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_graph",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core:MPPVisionPacketCreator",
        "//mediapipe/tasks/ios/vision/core:MPPVisionTaskRunner",
        "//mediapipe/tasks/ios/vision/object_detector/utils:MPPObjectDetectorResultHelpers",
        "//mediapipe/tasks/ios/vision/yolo_object_detector/utils:MPPYoloObjectDetectorOptionsHelpers",
        "//third_party/apple_frameworks:Foundation",
    ],
)
