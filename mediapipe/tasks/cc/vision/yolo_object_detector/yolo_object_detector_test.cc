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
#include <set>
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
  options->num_classes = 80;  // COCO — REQUIRED by the graph (RET_CHECK_GT).
  options->score_threshold = 0.25f;
  options->iou_threshold = 0.45f;
  options->max_results = 10;
  // NOTE: leave layout at its default (kChannelsFirst). yolov8n.tflite emits a
  // CHANNELS_FIRST [1,84,8400] tensor; setting kChannelsLast would misread it.

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));
  MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result,
                          detector->Detect(image));

  ASSERT_FALSE(result.detections.empty())
      << "expected detections on " << kTestImage;
  EXPECT_LE(result.detections.size(), 10u);  // max_results respected
  const int w = image.width(), h = image.height();
  std::set<int> labels;
  for (const auto& det : result.detections) {
    ASSERT_EQ(det.categories.size(), 1u);
    const auto& cat = det.categories[0];
    EXPECT_GE(cat.score, 0.25f);
    EXPECT_GE(cat.index, 0);
    EXPECT_LT(cat.index, 80);
    labels.insert(cat.index);
    const auto& bb = det.bounding_box;  // Rect{left,top,right,bottom} in pixels
    EXPECT_GE(bb.left, 0);
    EXPECT_GE(bb.top, 0);
    EXPECT_LE(bb.right, w + 1);
    EXPECT_LE(bb.bottom, h + 1);
    EXPECT_GT(bb.right, bb.left);
    EXPECT_GT(bb.bottom, bb.top);
  }
  // cats_and_dogs.jpg: model detects a cat (15) and/or dog (16) (Step 1 oracle
  // reports 4x cat + 1x dog at conf>=0.25 on this 1200x600 image).
  EXPECT_TRUE(labels.count(15) || labels.count(16)) << "expected a cat or dog";
  MP_ASSERT_OK(detector->Close());
}

// ---------------------------------------------------------------------------
// Filtering test: score_threshold and max_results must shrink the result set.
// ---------------------------------------------------------------------------
TEST(YoloObjectDetectorTest, ScoreThresholdAndMaxResultsFilter) {
  const std::string model_path = ModelPath();
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "YOLO model fixture not available at " << model_path;
  }
  auto base = [&]() {
    auto o = std::make_unique<YoloObjectDetectorOptions>();
    o->base_options.model_asset_path = model_path;
    o->running_mode = core::RunningMode::IMAGE;
    o->num_classes = 80;
    o->iou_threshold = 0.45f;
    return o;
  };
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));

  auto lo = base();
  lo->score_threshold = 0.25f;
  MP_ASSERT_OK_AND_ASSIGN(auto det_lo, YoloObjectDetector::Create(std::move(lo)));
  MP_ASSERT_OK_AND_ASSIGN(auto r_lo, det_lo->Detect(image));
  MP_ASSERT_OK(det_lo->Close());

  auto hi = base();
  hi->score_threshold = 0.9f;
  MP_ASSERT_OK_AND_ASSIGN(auto det_hi, YoloObjectDetector::Create(std::move(hi)));
  MP_ASSERT_OK_AND_ASSIGN(auto r_hi, det_hi->Detect(image));
  MP_ASSERT_OK(det_hi->Close());
  EXPECT_LE(r_hi.detections.size(), r_lo.detections.size());
  for (const auto& d : r_hi.detections) EXPECT_GE(d.categories[0].score, 0.9f);

  auto cap = base();
  cap->score_threshold = 0.25f;
  cap->max_results = 1;
  MP_ASSERT_OK_AND_ASSIGN(auto det_cap,
                          YoloObjectDetector::Create(std::move(cap)));
  MP_ASSERT_OK_AND_ASSIGN(auto r_cap, det_cap->Detect(image));
  MP_ASSERT_OK(det_cap->Close());
  EXPECT_LE(r_cap.detections.size(), 1u);
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
  options->num_classes = 80;  // COCO — REQUIRED by the graph (RET_CHECK_GT).
  options->score_threshold = 0.25f;
  options->iou_threshold = 0.45f;
  // Leave layout at its CHANNELS_FIRST default (matches yolov8n.tflite).

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
