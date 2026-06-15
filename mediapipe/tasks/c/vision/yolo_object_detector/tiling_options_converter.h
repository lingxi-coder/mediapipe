/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_TILING_OPTIONS_CONVERTER_H_
#define MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_TILING_OPTIONS_CONVERTER_H_

#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

namespace mediapipe::tasks::c::vision::yolo_object_detector {

// Copies a plain-C MpTilingOptions into the C++ TilingOptions sub-struct.
// Scalars are copied 1:1; explicit_tiles (a caller-owned MpTileRect array) is
// copied element-by-element into a fresh std::vector (the caller retains
// ownership of the input array). Declared here so unit tests can verify the
// C->C++ mapping without constructing a live graph.
void CppConvertToTilingOptions(
    const MpTilingOptions& in,
    ::mediapipe::tasks::vision::yolo_object_detector::
        YoloObjectDetectorOptions::TilingOptions* out);

}  // namespace mediapipe::tasks::c::vision::yolo_object_detector

#endif  // MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_TILING_OPTIONS_CONVERTER_H_
