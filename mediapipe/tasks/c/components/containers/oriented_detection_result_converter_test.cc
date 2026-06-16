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

#include "mediapipe/tasks/c/components/containers/oriented_detection_result_converter.h"

#include <optional>
#include <string>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/oriented_detection_result.h"
#include "mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h"

namespace mediapipe::tasks::c::components::containers {
namespace {

namespace cc = ::mediapipe::tasks::components::containers;

TEST(OrientedDetectionResultConverterTest, ConvertsTrackId) {
  cc::OrientedObjectDetection in;
  in.categories.push_back({/*index=*/1, /*score=*/0.9f, "", ""});
  in.cx = 1.0f;
  in.cy = 2.0f;
  in.width = 3.0f;
  in.height = 4.0f;
  in.rotation = 0.3f;
  in.track_id = std::string("42");

  MpOrientedDetection out;
  CppConvertToOrientedDetection(in, &out);
  ASSERT_NE(out.track_id, nullptr);
  EXPECT_STREQ(out.track_id, "42");
  CppCloseOrientedDetection(&out);
  EXPECT_EQ(out.track_id, nullptr);
}

TEST(OrientedDetectionResultConverterTest, NoTrackIdIsNull) {
  cc::OrientedObjectDetection in;
  in.categories.push_back({/*index=*/1, /*score=*/0.9f, "", ""});
  MpOrientedDetection out;
  CppConvertToOrientedDetection(in, &out);
  EXPECT_EQ(out.track_id, nullptr);
  CppCloseOrientedDetection(&out);
}

}  // namespace
}  // namespace mediapipe::tasks::c::components::containers
