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

// Integration test for the YOLO object detector C API.
//
// The assertions inside the TEST are gated on the presence of a yolov8n.tflite
// fixture.  If the fixture is absent the test calls GTEST_SKIP() and exits
// cleanly (neither failing nor faking a pass).  To enable the assertions:
//   1.  Place the exported yolov8n.tflite file at
//       mediapipe/tasks/testdata/vision/yolov8n.tflite.
//   2.  Add it to mediapipe/tasks/testdata/vision/BUILD (mediapipe_files +
//       filegroup) and uncomment the data dep in this package's BUILD rule.
//   3.  Re-run the test.

#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_helpers.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"

namespace {

using ::mediapipe::file::JoinPath;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kYoloModel[] = "yolov8n.tflite";
constexpr char kImageFile[] = "cats_and_dogs.jpg";

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

// RAII wrapper for MpImagePtr.
struct ScopedMpImage {
  explicit ScopedMpImage(MpImagePtr p) : ptr(p) {}
  ~ScopedMpImage() {
    if (ptr) MpImageFree(ptr);
  }
  ScopedMpImage(const ScopedMpImage&) = delete;
  ScopedMpImage& operator=(const ScopedMpImage&) = delete;
  MpImagePtr get() const { return ptr; }
  MpImagePtr ptr;
};

TEST(YoloObjectDetectorCApiTest, ImageMode) {
  const std::string model_path = GetFullPath(kYoloModel);

  // Skip cleanly when the fixture is not present.  This is the intentional
  // behaviour: the target must build and run, but real assertions are gated on
  // the model fixture.
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "YOLO model fixture not available at " << model_path
                 << "; integration assertions gated until yolov8n.tflite is "
                    "added to mediapipe/tasks/testdata/vision/.";
  }

  MpYoloObjectDetectorOptions options = {};
  options.base_options.model_asset_path = model_path.c_str();
  options.running_mode = MpRunningMode::MP_RUNNING_MODE_IMAGE;
  options.max_results = 10;
  options.score_threshold = 0.25f;
  options.iou_threshold = 0.45f;
  options.num_classes = 80;
  options.layout = 2;  // CHANNELS_LAST

  MpYoloObjectDetectorPtr detector = nullptr;
  char* error_msg = nullptr;
  ASSERT_EQ(MpYoloObjectDetectorCreate(&options, &detector, &error_msg),
            kMpOk);
  EXPECT_NE(detector, nullptr);

  MpImagePtr raw_image = nullptr;
  ASSERT_EQ(
      MpImageCreateFromFile(GetFullPath(kImageFile).c_str(), &raw_image,
                            &error_msg),
      kMpOk);
  ScopedMpImage image(raw_image);

  MpYoloObjectDetectorResult result;
  ASSERT_EQ(MpYoloObjectDetectorDetectImage(detector, image.get(),
                                            /*options=*/nullptr, &result,
                                            &error_msg),
            kMpOk);

  EXPECT_GT(result.detections_count, 0u);
  for (uint32_t i = 0; i < result.detections_count; ++i) {
    EXPECT_EQ(result.detections[i].categories_count, 1u);
    EXPECT_GT(result.detections[i].categories[0].score, 0.0f);
  }

  MpYoloObjectDetectorCloseResult(&result);
  EXPECT_EQ(MpYoloObjectDetectorClose(detector, &error_msg), kMpOk);
}

}  // namespace
