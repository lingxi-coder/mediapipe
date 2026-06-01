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

// Integration test for YoloObjectDetector.
//
// The assertions inside each TEST are gated on the presence of a yolov8n.tflite
// fixture.  If the fixture is absent the test calls GTEST_SKIP() and exits
// cleanly (neither failing nor faking a pass).  To enable the assertions:
//   1.  Place the exported yolov8n.tflite file at
//       mediapipe/tasks/testdata/vision/yolov8n.tflite.
//   2.  Add it to mediapipe/tasks/testdata/vision/BUILD (mediapipe_files +
//       filegroup) and uncomment the data dep in this package's BUILD rule.
//   3.  Re-run the test.

#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

#include <memory>
#include <string>

#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/components/containers/detection_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace yolo_object_detector {
namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::components::containers::DetectionResult;

// Path prefix used by all testdata files (mirrors the sibling task tests).
constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";

// The fixture gating these integration tests.
// Place the actual model here to enable assertions:
//   mediapipe/tasks/testdata/vision/yolov8n.tflite
constexpr char kYoloModel[] = "yolov8n.tflite";

// An existing test image from //mediapipe/tasks/testdata/vision:test_images.
// cats_and_dogs.jpg is a standard object-detection scene used by the sibling
// object_detector_test and is always available when the data dep is declared.
constexpr char kTestImage[] = "cats_and_dogs.jpg";

// Returns the absolute path to the YOLO fixture, using the same path
// convention as the sibling object_detector tests ("./", prefix, filename).
std::string ModelPath() {
  return JoinPath("./", kTestDataDirectory, kYoloModel);
}

// Returns the absolute path to the test input image.
std::string ImagePath() {
  return JoinPath("./", kTestDataDirectory, kTestImage);
}

// ---------------------------------------------------------------------------
// Image-mode test
// ---------------------------------------------------------------------------
TEST(YoloObjectDetectorTest, DetectOnImage) {
  const std::string model_path = ModelPath();

  // Skip cleanly when the fixture is not present.  This is the honest,
  // intentional behaviour: the target must build and run, but real assertions
  // are gated on the model fixture.
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "YOLO model fixture not available at " << model_path
                 << "; integration assertions gated until yolov8n.tflite is "
                    "added to mediapipe/tasks/testdata/vision/.";
  }

  // --- real assertions (executed only once the fixture is present) ----------

  // Build options for image mode.
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::IMAGE;
  options->max_results = 10;
  // Standard COCO 80-class model; adjust num_classes if a different variant is
  // used as the fixture.
  options->num_classes = 80;
  options->score_threshold = 0.25f;
  options->iou_threshold = 0.45f;
  // YOLOv8n exported to TFLite typically uses CHANNELS_LAST layout.
  options->layout = YoloObjectDetectorOptions::kChannelsLast;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                           YoloObjectDetector::Create(std::move(options)));

  // Decode the test image (cats and dogs: a well-known object-detection scene).
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));

  MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result,
                           detector->Detect(image));

  // Verify that at least one detection was returned.
  EXPECT_FALSE(result.detections.empty())
      << "Expected at least one detection on cats_and_dogs.jpg.";

  // Each detection must have exactly one category and a positive score.
  for (const auto& det : result.detections) {
    ASSERT_EQ(det.categories.size(), 1u);
    EXPECT_GT(det.categories[0].score, 0.0f);
  }

  MP_ASSERT_OK(detector->Close());
}

// ---------------------------------------------------------------------------
// Video-mode test
// ---------------------------------------------------------------------------
TEST(YoloObjectDetectorTest, DetectForVideo) {
  const std::string model_path = ModelPath();

  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "YOLO model fixture not available at " << model_path
                 << "; integration assertions gated until yolov8n.tflite is "
                    "added to mediapipe/tasks/testdata/vision/.";
  }

  // --- real assertions (executed only once the fixture is present) ----------

  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::VIDEO;
  options->max_results = 10;
  options->num_classes = 80;
  options->score_threshold = 0.25f;
  options->iou_threshold = 0.45f;
  options->layout = YoloObjectDetectorOptions::kChannelsLast;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                           YoloObjectDetector::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));

  // Feed two consecutive frames.
  MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result0,
                           detector->DetectForVideo(image, /*timestamp_ms=*/0));
  MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result1,
                           detector->DetectForVideo(image, /*timestamp_ms=*/33));

  // Results from both frames must be non-empty.
  EXPECT_FALSE(result0.detections.empty());
  EXPECT_FALSE(result1.detections.empty());

  MP_ASSERT_OK(detector->Close());
}

}  // namespace
}  // namespace yolo_object_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
