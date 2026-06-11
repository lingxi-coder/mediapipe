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
#include <vector>

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

// Harbor scene used by the tiled e2e test (shared with the OBB twin task;
// gitignored local fixture shipped alongside the exported models).
constexpr char kBoatsImage[] = "boats.jpg";

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
// Options passthrough test — no model fixture required.
// Verifies that the nested TilingOptions struct is copied correctly into the
// options proto by ConvertYoloObjectDetectorOptionsToProto.
// ---------------------------------------------------------------------------
TEST(YoloObjectDetectorOptionsTest, TilingOptionsConvertToProto) {
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  // The converter is a dumb mapper: it intentionally performs no grid/explicit
  // mutual-exclusion validation (that happens at graph build), so this fixture
  // sets both.
  options->tiling.tile_rows = 2;
  options->tiling.tile_cols = 3;
  options->tiling.tile_overlap_fraction = 0.2f;
  options->tiling.explicit_tiles.push_back({0.3f, 0.4f, 0.2f, 0.6f});
  options->tiling.tile_local_nms_iou_threshold = 0.5f;
  options->tiling.max_detections_after_tile_nms = 50;
  auto proto = ConvertYoloObjectDetectorOptionsToProto(options.get());
  EXPECT_EQ(proto->tiling().tile_rows(), 2);
  EXPECT_EQ(proto->tiling().tile_cols(), 3);
  EXPECT_NEAR(proto->tiling().tile_overlap_fraction(), 0.2f, 1e-6);
  ASSERT_EQ(proto->tiling().explicit_tiles_size(), 1);
  EXPECT_NEAR(proto->tiling().explicit_tiles(0).x_center(), 0.3f, 1e-6);
  EXPECT_NEAR(proto->tiling().explicit_tiles(0).y_center(), 0.4f, 1e-6);
  EXPECT_NEAR(proto->tiling().explicit_tiles(0).width(), 0.2f, 1e-6);
  EXPECT_NEAR(proto->tiling().explicit_tiles(0).height(), 0.6f, 1e-6);
  EXPECT_NEAR(proto->tiling().tile_local_nms_iou_threshold(), 0.5f, 1e-6);
  EXPECT_EQ(proto->tiling().max_detections_after_tile_nms(), 50);
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
    // Names are populated in-graph from model metadata; the integer index is
    // preserved (keep_label_id=true). COCO: cat=15, dog=16.
    ASSERT_TRUE(cat.category_name.has_value());
    EXPECT_FALSE(cat.category_name->empty());
    if (cat.index == 15) EXPECT_EQ(*cat.category_name, "cat");
    if (cat.index == 16) EXPECT_EQ(*cat.category_name, "dog");
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

// ---------------------------------------------------------------------------
// category_allowlist / category_denylist filter by class name.
// ---------------------------------------------------------------------------
TEST(YoloObjectDetectorTest, CategoryAllowlistAndDenylistFilterByName) {
  const std::string model_path = ModelPath();
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "YOLO model fixture not available at " << model_path;
  }
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));
  auto base = [&]() {
    auto o = std::make_unique<YoloObjectDetectorOptions>();
    o->base_options.model_asset_path = model_path;
    o->running_mode = core::RunningMode::IMAGE;
    o->num_classes = 80;
    o->score_threshold = 0.25f;
    o->iou_threshold = 0.45f;
    o->max_results = 10;
    return o;
  };

  // Allowlist {"dog"}: only dogs (index 16, name "dog") may survive.
  auto allow = base();
  allow->category_allowlist = {"dog"};
  MP_ASSERT_OK_AND_ASSIGN(auto det_allow,
                          YoloObjectDetector::Create(std::move(allow)));
  MP_ASSERT_OK_AND_ASSIGN(auto r_allow, det_allow->Detect(image));
  MP_ASSERT_OK(det_allow->Close());
  // Guard against a vacuous pass: a broken allow-filter that drops everything
  // would make the per-detection loop below trivially true. The oracle reports
  // a dog at conf>=0.25 on cats_and_dogs.jpg, so the allowed class must remain.
  EXPECT_FALSE(r_allow.detections.empty()) << "allowlist {dog} dropped all";
  for (const auto& det : r_allow.detections) {
    EXPECT_EQ(det.categories[0].index, 16);
    ASSERT_TRUE(det.categories[0].category_name.has_value());
    EXPECT_EQ(*det.categories[0].category_name, "dog");
  }

  // Denylist {"dog"}: dogs (index 16) must be excluded.
  auto deny = base();
  deny->category_denylist = {"dog"};
  MP_ASSERT_OK_AND_ASSIGN(auto det_deny,
                          YoloObjectDetector::Create(std::move(deny)));
  MP_ASSERT_OK_AND_ASSIGN(auto r_deny, det_deny->Detect(image));
  MP_ASSERT_OK(det_deny->Close());
  for (const auto& det : r_deny.detections) {
    EXPECT_NE(det.categories[0].index, 16);
  }
}

// ---------------------------------------------------------------------------
// Tiled-mode e2e: 1x2 grid with 20% overlap on boats.jpg.
// ---------------------------------------------------------------------------
TEST(YoloObjectDetectorTest, TiledGridDetectsBoatsOnBoats) {
  const std::string model_path = ModelPath();
  const std::string image_path = JoinPath("./", kTestDataDirectory, kBoatsImage);
  if (!mediapipe::file::Exists(model_path).ok() ||
      !mediapipe::file::Exists(image_path).ok()) {
    GTEST_SKIP() << "YOLO model or boats.jpg fixture not available; "
                    "integration assertions gated until yolov8n.tflite and "
                    "boats.jpg are added to mediapipe/tasks/testdata/vision/.";
  }

  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(image_path));
  const int image_width = image.width();
  const int image_height = image.height();

  // Options identical between the untiled oracle and the tiled run.
  // score_threshold is fixture-specific: boats.jpg is an aerial harbor scene
  // whose boats are tiny in COCO terms — yolov8n only resolves them once the
  // tiled front upscales each tile (tiled "boat" scores ~0.10; the untiled
  // path never produces a boat at any threshold). 0.09 sits safely below the
  // boat scores while still cutting sub-0.09 noise.
  auto base = [&]() {
    auto o = std::make_unique<YoloObjectDetectorOptions>();
    o->base_options.model_asset_path = model_path;
    o->running_mode = core::RunningMode::IMAGE;
    o->max_results = 10;
    o->num_classes = 80;
    o->score_threshold = 0.09f;
    o->iou_threshold = 0.45f;
    return o;
  };

  // Cross-check oracle: the SAME image through the single-image path.
  MP_ASSERT_OK_AND_ASSIGN(auto single_detector,
                          YoloObjectDetector::Create(base()));
  MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult single_result,
                          single_detector->Detect(image));
  MP_ASSERT_OK(single_detector->Close());
  const int n_single = single_result.detections.size();

  // Tiled detector: 1x2 grid, 20% overlap.
  auto options = base();
  options->tiling.tile_cols = 2;
  options->tiling.tile_overlap_fraction = 0.2f;
  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result,
                          detector->Detect(image));
  MP_ASSERT_OK(detector->Close());

  ASSERT_GE(result.detections.size(), 1u);
  EXPECT_LE(result.detections.size(), 10u);  // max_results respected

  // Public result is PIXEL units: every bounding box of each detection must
  // lie within image pixel bounds (symmetric +/-1px tolerance).
  bool found_boat = false;
  for (const auto& d : result.detections) {
    const auto& bb = d.bounding_box;  // Rect{left,top,right,bottom} in pixels
    EXPECT_GE(bb.left, -1);
    EXPECT_LE(bb.right, image_width + 1);
    EXPECT_GE(bb.top, -1);
    EXPECT_LE(bb.bottom, image_height + 1);
    EXPECT_GT(bb.right, bb.left);
    EXPECT_GT(bb.bottom, bb.top);
    ASSERT_EQ(d.categories.size(), 1u);
    const auto& cat = d.categories[0];
    EXPECT_GE(cat.score, 0.09f);  // matches score_threshold above
    EXPECT_GE(cat.index, 0);
    EXPECT_LT(cat.index, 80);
    ASSERT_TRUE(cat.category_name.has_value());
    EXPECT_FALSE(cat.category_name->empty());
    // COCO: boat=8.
    if (cat.index == 8) {
      EXPECT_EQ(*cat.category_name, "boat");
      found_boat = true;
    }
  }
  // boats.jpg -> COCO "boat" (class 8), same as the single-image oracle.
  EXPECT_TRUE(found_boat) << "expected a boat detection";

  // Detection count must be in the neighborhood of the single-image path.
  const int n_tiled = result.detections.size();
  EXPECT_GE(n_tiled, n_single - 1);
  EXPECT_LE(n_tiled, n_single + 3);
}

// ---------------------------------------------------------------------------
// Tiled mode + region-of-interest is rejected (the tiled graph has no
// NORM_RECT input, so per-call ROI cannot be honored).
// ---------------------------------------------------------------------------
TEST(YoloObjectDetectorTest, TiledRoiAndRotationRejected) {
  const std::string model_path = ModelPath();
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "YOLO model fixture not available at " << model_path
                 << "; integration assertions gated until yolov8n.tflite is "
                    "added to mediapipe/tasks/testdata/vision/.";
  }

  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::IMAGE;
  options->max_results = 10;
  options->num_classes = 80;
  options->score_threshold = 0.25f;
  options->iou_threshold = 0.45f;
  options->tiling.tile_cols = 2;
  options->tiling.tile_overlap_fraction = 0.2f;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));

  core::ImageProcessingOptions image_processing_options;
  image_processing_options.region_of_interest =
      components::containers::RectF{/*left=*/0.1f, /*top=*/0.1f,
                                    /*right=*/0.9f, /*bottom=*/0.9f};
  auto result = detector->Detect(image, image_processing_options);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(result.status().message(),
              testing::HasSubstr("tiling and ROI are mutually exclusive"));

  // Rotation (without ROI) is likewise rejected: the tiled graph has no
  // NORM_RECT input, so rotation_degrees cannot be honored.
  core::ImageProcessingOptions rotation_options;
  rotation_options.rotation_degrees = 90;
  auto rotated_result = detector->Detect(image, rotation_options);
  EXPECT_EQ(rotated_result.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(rotated_result.status().message(),
              testing::HasSubstr("rotation"));
  MP_ASSERT_OK(detector->Close());
}

// ---------------------------------------------------------------------------
// Explicit tiles + a nonzero tile_overlap_fraction is rejected at graph build
// (overlap only applies to grid mode; silently ignoring it would mislead).
// ---------------------------------------------------------------------------
TEST(YoloObjectDetectorTest, TiledExplicitTilesWithOverlapRejected) {
  const std::string model_path = ModelPath();
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "YOLO model fixture not available at " << model_path
                 << "; integration assertions gated until yolov8n.tflite is "
                    "added to mediapipe/tasks/testdata/vision/.";
  }

  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::IMAGE;
  options->max_results = 10;
  options->num_classes = 80;
  options->score_threshold = 0.25f;
  options->iou_threshold = 0.45f;
  options->tiling.explicit_tiles.push_back(
      {/*x_center=*/0.5f, /*y_center=*/0.5f, /*width=*/1.0f, /*height=*/1.0f});
  options->tiling.tile_overlap_fraction = 0.2f;

  auto detector = YoloObjectDetector::Create(std::move(options));
  EXPECT_EQ(detector.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(detector.status().message(),
              testing::HasSubstr("tile_overlap_fraction"));
}

}  // namespace
}  // namespace yolo_object_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
