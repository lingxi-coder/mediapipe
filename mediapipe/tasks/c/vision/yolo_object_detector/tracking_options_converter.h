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

#ifndef MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_TRACKING_OPTIONS_CONVERTER_H_
#define MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_TRACKING_OPTIONS_CONVERTER_H_

#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

namespace mediapipe::tasks::c::vision::yolo_object_detector {

// Copies a plain-C MpTrackingOptions into the C++ TrackingOptions sub-struct.
// tracker_type maps 1:1 onto the C++ TrackerType enum (kBoxTracker=1,
// kBotsort=2; 0 = unspecified, treated as BOX_TRACKER by the cc layer); the
// remaining knobs except nominal_frame_rate are copied verbatim. The legacy
// conversion deliberately defaults nominal_frame_rate to 30 without reading
// it because that field occupied padding in the previous public ABI.
void CppConvertToTrackingOptions(
    const MpTrackingOptions& in,
    ::mediapipe::tasks::vision::yolo_object_detector::
        YoloObjectDetectorOptions::TrackingOptions* out);

// V2 conversion reads nominal_frame_rate; zero retains the 30 FPS default.
void CppConvertToTrackingOptionsV2(
    const MpTrackingOptions& in,
    ::mediapipe::tasks::vision::yolo_object_detector::
        YoloObjectDetectorOptions::TrackingOptions* out);

}  // namespace mediapipe::tasks::c::vision::yolo_object_detector

#endif  // MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_TRACKING_OPTIONS_CONVERTER_H_
