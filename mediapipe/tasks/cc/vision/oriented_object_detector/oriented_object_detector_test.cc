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

// Integration test for OrientedObjectDetector.
//
// The assertions inside each TEST are gated on the presence of a
// yolov8n-obb.tflite fixture.  If the fixture is absent the test calls
// GTEST_SKIP() and exits cleanly (neither failing nor faking a pass).  To
// enable the assertions:
//   1.  Place the exported yolov8n-obb.tflite file at
//       mediapipe/tasks/testdata/vision/yolov8n-obb.tflite.
//   2.  Add it to mediapipe/tasks/testdata/vision/BUILD (mediapipe_files +
//       filegroup) and uncomment the data dep in this package's BUILD rule.
//   3.  Re-run the test.

#include "mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h"

#include <cmath>
#include <memory>
#include <string>

#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace oriented_object_detector {
namespace {

using ::mediapipe::file::JoinPath;

// Path prefix used by all testdata files (mirrors the sibling task tests).
constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";

// The fixture gating these integration tests. Place the actual model here to
// enable assertions: mediapipe/tasks/testdata/vision/yolov8n-obb.tflite
constexpr char kOrientedModel[] = "yolov8n-obb.tflite";

// An existing test image from //mediapipe/tasks/testdata/vision:test_images.
constexpr char kTestImage[] = "cats_and_dogs.jpg";

// Returns the absolute path to the OBB fixture, using the same path
// convention as the sibling object_detector tests ("./", prefix, filename).
std::string ModelPath() {
  return JoinPath("./", kTestDataDirectory, kOrientedModel);
}

// Returns the absolute path to the test input image.
std::string ImagePath() {
  return JoinPath("./", kTestDataDirectory, kTestImage);
}

// ---------------------------------------------------------------------------
// Image-mode test
// ---------------------------------------------------------------------------
TEST(OrientedObjectDetectorTest, DetectOnImage) {
  const std::string model_path = ModelPath();

  // Skip cleanly when the fixture is not present.  This is the honest,
  // intentional behaviour: the target must build and run, but real assertions
  // are gated on the model fixture.
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "OBB model fixture not available at " << model_path
                 << "; integration assertions gated until yolov8n-obb.tflite is "
                    "added to mediapipe/tasks/testdata/vision/.";
  }

  // --- real assertions (executed only once the fixture is present) ----------

  auto options = std::make_unique<OrientedObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::IMAGE;
  options->max_results = 10;
  // YOLOv8-obb is trained on DOTAv1 (15 classes); adjust if a different OBB
  // variant is used as the fixture.
  options->num_classes = 15;
  options->score_threshold = 0.25f;
  options->iou_threshold = 0.45f;
  // YOLOv8 exported to TFLite typically uses CHANNELS_LAST layout.
  options->layout = OrientedObjectDetectorOptions::kChannelsLast;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                           OrientedObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));
  MP_ASSERT_OK_AND_ASSIGN(OrientedObjectDetectorResult result,
                           detector->Detect(image));

  EXPECT_FALSE(result.detections.empty())
      << "Expected at least one oriented detection on " << kTestImage << ".";
  for (const auto& det : result.detections) {
    // Pixel-unit box dimensions must be positive and the angle finite.
    EXPECT_GT(det.width, 0.0f);
    EXPECT_GT(det.height, 0.0f);
    EXPECT_TRUE(std::isfinite(det.rotation));
    ASSERT_EQ(det.categories.size(), 1u);
    EXPECT_GT(det.categories[0].score, 0.0f);
  }
  MP_ASSERT_OK(detector->Close());
}

// ---------------------------------------------------------------------------
// Video-mode test
// ---------------------------------------------------------------------------
TEST(OrientedObjectDetectorTest, DetectForVideo) {
  const std::string model_path = ModelPath();

  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "OBB model fixture not available at " << model_path
                 << "; integration assertions gated until yolov8n-obb.tflite is "
                    "added to mediapipe/tasks/testdata/vision/.";
  }

  // --- real assertions (executed only once the fixture is present) ----------

  auto options = std::make_unique<OrientedObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::VIDEO;
  options->max_results = 10;
  options->num_classes = 15;
  options->score_threshold = 0.25f;
  options->iou_threshold = 0.45f;
  options->layout = OrientedObjectDetectorOptions::kChannelsLast;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                           OrientedObjectDetector::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));

  // Feed two consecutive frames.
  MP_ASSERT_OK_AND_ASSIGN(OrientedObjectDetectorResult result0,
                           detector->DetectForVideo(image, /*timestamp_ms=*/0));
  MP_ASSERT_OK_AND_ASSIGN(OrientedObjectDetectorResult result1,
                           detector->DetectForVideo(image, /*timestamp_ms=*/33));

  // Results from both frames must be non-empty.
  EXPECT_FALSE(result0.detections.empty());
  EXPECT_FALSE(result1.detections.empty());

  MP_ASSERT_OK(detector->Close());
}

}  // namespace
}  // namespace oriented_object_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
