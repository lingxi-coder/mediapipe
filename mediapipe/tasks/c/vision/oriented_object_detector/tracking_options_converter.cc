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

#include "mediapipe/tasks/c/vision/oriented_object_detector/tracking_options_converter.h"

#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"
#include "mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h"

namespace mediapipe::tasks::c::vision::oriented_object_detector {

namespace ObbNs = ::mediapipe::tasks::vision::oriented_object_detector;

namespace {

void CopyCommonTrackingOptions(
    const MpOrientedTrackingOptions& in,
    ObbNs::OrientedObjectDetectorOptions::TrackingOptions* out) {
  out->tracker_type =
      static_cast<ObbNs::OrientedObjectDetectorOptions::TrackingOptions::
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
    const MpOrientedTrackingOptions& in,
    ObbNs::OrientedObjectDetectorOptions::TrackingOptions* out) {
  // tracker_type copied verbatim: 0/1/2 == unspecified/kBoxTracker/kBotsort.
  // Knobs copied 1:1; honored only for BOTSORT. BOX_TRACKER is rejected at the
  // cc Create(), not here.
  CopyCommonTrackingOptions(in, out);
  // This offset was padding in the previous public struct layout. Do not read
  // it through the legacy entry point.
  out->nominal_frame_rate = 30;
}

void CppConvertToTrackingOptionsV2(
    const MpOrientedTrackingOptions& in,
    ObbNs::OrientedObjectDetectorOptions::TrackingOptions* out) {
  CopyCommonTrackingOptions(in, out);
  out->nominal_frame_rate =
      in.nominal_frame_rate == 0 ? 30 : in.nominal_frame_rate;
}

}  // namespace mediapipe::tasks::c::vision::oriented_object_detector
