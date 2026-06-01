/* Copyright 2024 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h"

#include <vector>

#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::tasks::components::containers {
namespace {

TEST(OrientedObjectDetectionResultTest, ConvertsNormalizedProtoToPixels) {
  mediapipe::OrientedDetection d;
  d.set_cx(0.5f); d.set_cy(0.25f);
  d.set_width(0.4f); d.set_height(0.2f);
  d.set_rotation(0.3f);
  d.add_score(0.9f); d.add_label_id(7);

  OrientedObjectDetectionResult result =
      ConvertToOrientedObjectDetectionResult({d}, /*image_size=*/{200, 100});

  ASSERT_EQ(result.detections.size(), 1);
  const OrientedObjectDetection& od = result.detections[0];
  EXPECT_NEAR(od.cx, 100.0f, 1e-4);
  EXPECT_NEAR(od.cy, 25.0f, 1e-4);
  EXPECT_NEAR(od.width, 80.0f, 1e-4);
  EXPECT_NEAR(od.height, 20.0f, 1e-4);
  EXPECT_NEAR(od.rotation, 0.3f, 1e-6);
  ASSERT_EQ(od.categories.size(), 1);
  EXPECT_EQ(od.categories[0].index, 7);
  EXPECT_NEAR(od.categories[0].score, 0.9f, 1e-6);
  EXPECT_FALSE(od.categories[0].category_name.has_value());
  EXPECT_FALSE(od.categories[0].display_name.has_value());
}

}  // namespace
}  // namespace mediapipe::tasks::components::containers
