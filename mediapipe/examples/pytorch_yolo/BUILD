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

# Use pybind11_bazel's own extension rule so a single, consistent pybind11
# (with numpy.h) is used — avoids the version mismatch from TensorFlow's
# bundled pybind11.
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# Python-importable C++ extension that runs the real MediaPipe YOLO decode+NMS
# CalculatorGraph over a raw YOLOv8 output tensor. Deliberately avoids
# //mediapipe/python:builtin_task_graphs (which pulls a missing tasks package);
# it only needs the framework + the YOLO/NMS calculators.
pybind_extension(
    name = "_yolo_pt_graph",
    srcs = ["yolo_pt_graph_bridge.cc"],
    deps = [
        "//mediapipe/calculators/tensor:yolo_batch_detections_to_single_calculator",
        "//mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:parse_text_proto",
        "//third_party:opencv",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)
