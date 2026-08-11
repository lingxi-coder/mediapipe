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

#include "mediapipe/tasks/c/vision/yolo_object_detector/tracking_options_converter.h"

#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

namespace mediapipe::tasks::c::vision::yolo_object_detector {

namespace YoloNs = ::mediapipe::tasks::vision::yolo_object_detector;

namespace {

void CopyCommonTrackingOptions(
    const MpTrackingOptions& in,
    YoloNs::YoloObjectDetectorOptions::TrackingOptions* out) {
  out->tracker_type =
      static_cast<YoloNs::YoloObjectDetectorOptions::TrackingOptions::
                      TrackerType>(in.tracker_type);
  out->track_high_threshold = in.track_high_threshold;
  out->track_low_threshold = in.track_low_threshold;
  out->new_track_threshold = in.new_track_threshold;
  out->track_buffer = in.track_buffer;
  out->match_threshold = in.match_threshold;
  out->enable_gmc = in.enable_gmc;
}

}  // namespace

void CppConvertToTrackingOptions(
    const MpTrackingOptions& in,
    YoloNs::YoloObjectDetectorOptions::TrackingOptions* out) {
  // tracker_type is copied verbatim: 0/1/2 are numerically equal to the C++
  // enum (kBoxTracker=1, kBotsort=2; 0 = unspecified -> treated as BOX_TRACKER
  // by the cc layer). Knobs are copied 1:1; honored only for BOTSORT.
  CopyCommonTrackingOptions(in, out);
  // This offset was padding in the previous public struct layout. Do not read
  // it through the legacy entry point because old callers may leave it
  // indeterminate even though sizeof(the parent options struct) is unchanged.
  out->nominal_frame_rate = 30;
}

void CppConvertToTrackingOptionsV2(
    const MpTrackingOptions& in,
    YoloNs::YoloObjectDetectorOptions::TrackingOptions* out) {
  CopyCommonTrackingOptions(in, out);
  out->nominal_frame_rate =
      in.nominal_frame_rate == 0 ? 30 : in.nominal_frame_rate;
}

}  // namespace mediapipe::tasks::c::vision::yolo_object_detector
