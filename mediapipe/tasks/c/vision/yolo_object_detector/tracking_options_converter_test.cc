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

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

namespace mediapipe::tasks::c::vision::yolo_object_detector {
namespace {

using CppTrackingOptions = ::mediapipe::tasks::vision::yolo_object_detector::
    YoloObjectDetectorOptions::TrackingOptions;

TEST(TrackingOptionsConverterTest, V2CopiesAllScalarFields) {
  MpTrackingOptions in = {};
  in.tracker_type = 2;  // BOTSORT
  in.track_high_threshold = 0.55f;
  in.track_low_threshold = 0.15f;
  in.new_track_threshold = 0.65f;
  in.track_buffer = 25;
  in.match_threshold = 0.75f;
  in.enable_gmc = true;
  in.nominal_frame_rate = 60;

  CppTrackingOptions out;
  CppConvertToTrackingOptionsV2(in, &out);

  EXPECT_EQ(out.tracker_type, CppTrackingOptions::kBotsort);
  EXPECT_FLOAT_EQ(out.track_high_threshold, 0.55f);
  EXPECT_FLOAT_EQ(out.track_low_threshold, 0.15f);
  EXPECT_FLOAT_EQ(out.new_track_threshold, 0.65f);
  EXPECT_EQ(out.track_buffer, 25);
  EXPECT_FLOAT_EQ(out.match_threshold, 0.75f);
  EXPECT_TRUE(out.enable_gmc);
  EXPECT_EQ(out.nominal_frame_rate, 60);
}

TEST(TrackingOptionsConverterTest, LegacyIgnoresNominalFrameRate) {
  MpTrackingOptions in = {};
  in.tracker_type = 2;
  in.nominal_frame_rate = 240;

  CppTrackingOptions out;
  CppConvertToTrackingOptions(in, &out);

  EXPECT_EQ(out.tracker_type, CppTrackingOptions::kBotsort);
  EXPECT_EQ(out.nominal_frame_rate, 30);
}

TEST(TrackingOptionsConverterTest, V2DefaultsZeroNominalFrameRate) {
  MpTrackingOptions in = {};
  CppTrackingOptions out;

  CppConvertToTrackingOptionsV2(in, &out);

  EXPECT_EQ(out.nominal_frame_rate, 30);
}

TEST(TrackingOptionsConverterTest, MapsTrackerTypeEnumValues) {
  CppTrackingOptions out;

  MpTrackingOptions box = {};
  box.tracker_type = 1;  // BOX_TRACKER
  CppConvertToTrackingOptions(box, &out);
  EXPECT_EQ(out.tracker_type, CppTrackingOptions::kBoxTracker);

  MpTrackingOptions unspecified = {};
  unspecified.tracker_type = 0;  // unspecified -> 0 (treated as BOX_TRACKER downstream)
  CppConvertToTrackingOptions(unspecified, &out);
  EXPECT_EQ(static_cast<int>(out.tracker_type), 0);
  EXPECT_EQ(out.nominal_frame_rate, 30);
}

}  // namespace
}  // namespace mediapipe::tasks::c::vision::yolo_object_detector
