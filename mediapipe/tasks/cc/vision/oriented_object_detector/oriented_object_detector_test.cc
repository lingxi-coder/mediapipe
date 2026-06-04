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
#include <set>
#include <string>
#include <vector>

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
using ::mediapipe::tasks::vision::oriented_object_detector::
    ConvertOrientedObjectDetectorOptionsToProto;

// Path prefix used by all testdata files (mirrors the sibling task tests).
constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";

// The fixture gating these integration tests. Place the actual model here to
// enable assertions: mediapipe/tasks/testdata/vision/yolov8n-obb.tflite
constexpr char kOrientedModel[] = "yolov8n-obb.tflite";

// An existing test image from //mediapipe/tasks/testdata/vision:test_images.
constexpr char kTestImage[] = "boats.jpg";

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
// Options passthrough test — no model fixture required.
// Verifies that display_names_locale, category_allowlist, and
// category_denylist are copied correctly into the options proto by
// ConvertOrientedObjectDetectorOptionsToProto.
// ---------------------------------------------------------------------------
TEST(OrientedObjectDetectorOptionsTest, CopiesCategoryFieldsToProto) {
  auto opts = std::make_unique<OrientedObjectDetectorOptions>();
  opts->display_names_locale = "fr";
  opts->category_allowlist = {"ship", "plane"};
  opts->category_denylist = {"helicopter"};

  auto proto = ConvertOrientedObjectDetectorOptionsToProto(opts.get());

  EXPECT_EQ(proto->display_names_locale(), "fr");
  ASSERT_EQ(proto->category_allowlist_size(), 2);
  EXPECT_EQ(proto->category_allowlist(0), "ship");
  EXPECT_EQ(proto->category_allowlist(1), "plane");
  ASSERT_EQ(proto->category_denylist_size(), 1);
  EXPECT_EQ(proto->category_denylist(0), "helicopter");
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

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                           OrientedObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));
  MP_ASSERT_OK_AND_ASSIGN(OrientedObjectDetectorResult result,
                           detector->Detect(image));

  ASSERT_FALSE(result.detections.empty())
      << "Expected oriented detections on " << kTestImage << ".";
  EXPECT_LE(result.detections.size(), 10u);  // max_results respected
  std::set<int> labels;
  for (const auto& det : result.detections) {
    ASSERT_EQ(det.categories.size(), 1u);
    const auto& cat = det.categories[0];
    EXPECT_GE(cat.score, 0.25f);
    EXPECT_GE(cat.index, 0);
    EXPECT_LT(cat.index, 15);
    labels.insert(cat.index);
    // Names are populated in-graph from model metadata; index is preserved
    // (keep_label_id=true). DOTA: ship=1.
    ASSERT_TRUE(cat.category_name.has_value());
    EXPECT_FALSE(cat.category_name->empty());
    if (cat.index == 1) EXPECT_EQ(*cat.category_name, "ship");
    EXPECT_GT(det.width, 0.0f);
    EXPECT_GT(det.height, 0.0f);
    EXPECT_TRUE(std::isfinite(det.rotation));
  }
  // boats.jpg -> DOTA "ship" (class 1) per Step 1 oracle.
  EXPECT_TRUE(labels.count(1)) << "expected a ship detection";
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

// ---------------------------------------------------------------------------
// category_allowlist / category_denylist filter by class name.
// ---------------------------------------------------------------------------
TEST(OrientedObjectDetectorTest, CategoryAllowlistAndDenylistFilterByName) {
  const std::string model_path = ModelPath();
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "OBB model fixture not available at " << model_path;
  }
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));
  auto base = [&]() {
    auto o = std::make_unique<OrientedObjectDetectorOptions>();
    o->base_options.model_asset_path = model_path;
    o->running_mode = core::RunningMode::IMAGE;
    o->num_classes = 15;
    o->score_threshold = 0.25f;
    o->iou_threshold = 0.45f;
    o->max_results = 10;
    return o;
  };

  // Allowlist {"ship"}: only ships (index 1, name "ship") may survive.
  auto allow = base();
  allow->category_allowlist = {"ship"};
  MP_ASSERT_OK_AND_ASSIGN(auto det_allow,
                          OrientedObjectDetector::Create(std::move(allow)));
  MP_ASSERT_OK_AND_ASSIGN(auto r_allow, det_allow->Detect(image));
  MP_ASSERT_OK(det_allow->Close());
  EXPECT_FALSE(r_allow.detections.empty()) << "allowlist {ship} dropped all";
  for (const auto& det : r_allow.detections) {
    EXPECT_EQ(det.categories[0].index, 1);
    ASSERT_TRUE(det.categories[0].category_name.has_value());
    EXPECT_EQ(*det.categories[0].category_name, "ship");
  }

  // Denylist {"ship"}: ships (index 1) must be excluded.
  auto deny = base();
  deny->category_denylist = {"ship"};
  MP_ASSERT_OK_AND_ASSIGN(auto det_deny,
                          OrientedObjectDetector::Create(std::move(deny)));
  MP_ASSERT_OK_AND_ASSIGN(auto r_deny, det_deny->Detect(image));
  MP_ASSERT_OK(det_deny->Close());
  for (const auto& det : r_deny.detections) {
    EXPECT_NE(det.categories[0].index, 1);
  }
}

}  // namespace
}  // namespace oriented_object_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
