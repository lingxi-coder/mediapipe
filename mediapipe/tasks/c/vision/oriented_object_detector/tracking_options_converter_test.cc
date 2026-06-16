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

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"
#include "mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h"

namespace mediapipe::tasks::c::vision::oriented_object_detector {
namespace {

using CppTrackingOptions = ::mediapipe::tasks::vision::oriented_object_detector::
    OrientedObjectDetectorOptions::TrackingOptions;

TEST(OrientedTrackingOptionsConverterTest, CopiesAllScalarFields) {
  MpOrientedTrackingOptions in = {};
  in.tracker_type = 2;  // BOTSORT
  in.track_high_threshold = 0.55f;
  in.track_low_threshold = 0.15f;
  in.new_track_threshold = 0.65f;
  in.track_buffer = 25;
  in.match_threshold = 0.75f;
  in.enable_gmc = true;

  CppTrackingOptions out;
  CppConvertToTrackingOptions(in, &out);

  EXPECT_EQ(out.tracker_type, CppTrackingOptions::kBotsort);
  EXPECT_FLOAT_EQ(out.track_high_threshold, 0.55f);
  EXPECT_FLOAT_EQ(out.track_low_threshold, 0.15f);
  EXPECT_FLOAT_EQ(out.new_track_threshold, 0.65f);
  EXPECT_EQ(out.track_buffer, 25);
  EXPECT_FLOAT_EQ(out.match_threshold, 0.75f);
  EXPECT_TRUE(out.enable_gmc);
}

TEST(OrientedTrackingOptionsConverterTest, MapsTrackerTypeEnumValues) {
  CppTrackingOptions out;
  MpOrientedTrackingOptions box = {};
  box.tracker_type = 1;  // BOX_TRACKER
  CppConvertToTrackingOptions(box, &out);
  EXPECT_EQ(out.tracker_type, CppTrackingOptions::kBoxTracker);

  MpOrientedTrackingOptions uns = {};
  uns.tracker_type = 0;
  CppConvertToTrackingOptions(uns, &out);
  EXPECT_EQ(out.tracker_type, CppTrackingOptions::kTrackerUnspecified);
}

}  // namespace
}  // namespace mediapipe::tasks::c::vision::oriented_object_detector
