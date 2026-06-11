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

#include <array>
#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <utility>
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

// Returns the 4 rotated corners (x, y) of an oriented box given center
// (cx, cy), full extents (w, h), and rotation theta in radians:
//   x = cx + (±w/2)·cosθ − (±h/2)·sinθ
//   y = cy + (±w/2)·sinθ + (±h/2)·cosθ
std::array<std::pair<float, float>, 4> ObbCorners(float cx, float cy, float w,
                                                  float h, float theta) {
  const float cos_t = std::cos(theta);
  const float sin_t = std::sin(theta);
  std::array<std::pair<float, float>, 4> corners;
  int i = 0;
  for (float sx : {-0.5f, 0.5f}) {
    for (float sy : {-0.5f, 0.5f}) {
      const float dx = sx * w;
      const float dy = sy * h;
      corners[i++] = {cx + dx * cos_t - dy * sin_t,
                      cy + dx * sin_t + dy * cos_t};
    }
  }
  return corners;
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

TEST(OrientedObjectDetectorOptionsTest, TilingOptionsConvertToProto) {
  auto options = std::make_unique<OrientedObjectDetectorOptions>();
  // The converter is a dumb mapper: it intentionally performs no grid/explicit
  // mutual-exclusion validation (that happens at graph build), so this fixture
  // sets both.
  options->tiling.tile_rows = 2;
  options->tiling.tile_cols = 3;
  options->tiling.tile_overlap_fraction = 0.2f;
  options->tiling.explicit_tiles.push_back({0.3f, 0.4f, 0.2f, 0.6f});
  options->tiling.tile_local_nms_iou_threshold = 0.5f;
  options->tiling.max_detections_after_tile_nms = 50;
  auto proto = ConvertOrientedObjectDetectorOptionsToProto(options.get());
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

// ---------------------------------------------------------------------------
// Tiled-mode e2e: 1x2 grid with 20% overlap on boats.jpg.
// ---------------------------------------------------------------------------
TEST(OrientedObjectDetectorTest, TiledGridDetectsShipsOnBoats) {
  const std::string model_path = ModelPath();
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "OBB model fixture not available at " << model_path
                 << "; integration assertions gated until yolov8n-obb.tflite is "
                    "added to mediapipe/tasks/testdata/vision/.";
  }

  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));
  const float image_width = image.width();
  const float image_height = image.height();

  // Options identical to the adjacent single-image boats test.
  auto base = [&]() {
    auto o = std::make_unique<OrientedObjectDetectorOptions>();
    o->base_options.model_asset_path = model_path;
    o->running_mode = core::RunningMode::IMAGE;
    o->max_results = 10;
    o->num_classes = 15;
    o->score_threshold = 0.25f;
    o->iou_threshold = 0.45f;
    return o;
  };

  // Cross-check oracle: the SAME image through the single-image path.
  MP_ASSERT_OK_AND_ASSIGN(auto single_detector,
                          OrientedObjectDetector::Create(base()));
  MP_ASSERT_OK_AND_ASSIGN(OrientedObjectDetectorResult single_result,
                          single_detector->Detect(image));
  MP_ASSERT_OK(single_detector->Close());
  const int n_single = single_result.detections.size();

  // Tiled detector: 1x2 grid, 20% overlap.
  auto options = base();
  options->tiling.tile_cols = 2;
  options->tiling.tile_overlap_fraction = 0.2f;
  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          OrientedObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(OrientedObjectDetectorResult result,
                          detector->Detect(image));
  MP_ASSERT_OK(detector->Close());

  ASSERT_GE(result.detections.size(), 1u);
  EXPECT_LE(result.detections.size(), 10u);  // max_results respected

  // Public result is PIXEL units: every rotated corner of each OBB must lie
  // within image pixel bounds (symmetric +/-1px tolerance).
  bool found_ship = false;
  for (const auto& d : result.detections) {
    for (const auto& [corner_x, corner_y] :
         ObbCorners(d.cx, d.cy, d.width, d.height, d.rotation)) {
      EXPECT_GE(corner_x, -1.0f);
      EXPECT_LE(corner_x, image_width + 1.0f);
      EXPECT_GE(corner_y, -1.0f);
      EXPECT_LE(corner_y, image_height + 1.0f);
    }
    EXPECT_GT(d.width, 0.0f);
    EXPECT_GT(d.height, 0.0f);
    EXPECT_TRUE(std::isfinite(d.rotation));
    ASSERT_EQ(d.categories.size(), 1u);
    const auto& cat = d.categories[0];
    EXPECT_GE(cat.score, 0.25f);
    EXPECT_GE(cat.index, 0);
    EXPECT_LT(cat.index, 15);
    ASSERT_TRUE(cat.category_name.has_value());
    EXPECT_FALSE(cat.category_name->empty());
    if (cat.index == 1) {
      EXPECT_EQ(*cat.category_name, "ship");
      found_ship = true;
    }
  }
  // boats.jpg -> DOTA "ship" (class 1), same as the single-image oracle.
  EXPECT_TRUE(found_ship) << "expected a ship detection";

  // Detection count must be in the neighborhood of the single-image path.
  const int n_tiled = result.detections.size();
  EXPECT_GE(n_tiled, n_single - 1);
  EXPECT_LE(n_tiled, n_single + 3);
}

// ---------------------------------------------------------------------------
// Tiled mode + region-of-interest is rejected (the tiled graph has no
// NORM_RECT input, so per-call ROI cannot be honored).
// ---------------------------------------------------------------------------
TEST(OrientedObjectDetectorTest, TiledWithRoiRejected) {
  const std::string model_path = ModelPath();
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "OBB model fixture not available at " << model_path
                 << "; integration assertions gated until yolov8n-obb.tflite is "
                    "added to mediapipe/tasks/testdata/vision/.";
  }

  auto options = std::make_unique<OrientedObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::IMAGE;
  options->max_results = 10;
  options->num_classes = 15;
  options->score_threshold = 0.25f;
  options->iou_threshold = 0.45f;
  options->tiling.tile_cols = 2;
  options->tiling.tile_overlap_fraction = 0.2f;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          OrientedObjectDetector::Create(std::move(options)));
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
TEST(OrientedObjectDetectorTest, TiledExplicitTilesWithOverlapRejected) {
  const std::string model_path = ModelPath();
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "OBB model fixture not available at " << model_path
                 << "; integration assertions gated until yolov8n-obb.tflite is "
                    "added to mediapipe/tasks/testdata/vision/.";
  }

  auto options = std::make_unique<OrientedObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::IMAGE;
  options->max_results = 10;
  options->num_classes = 15;
  options->score_threshold = 0.25f;
  options->iou_threshold = 0.45f;
  options->tiling.explicit_tiles.push_back(
      {/*x_center=*/0.5f, /*y_center=*/0.5f, /*width=*/1.0f, /*height=*/1.0f});
  options->tiling.tile_overlap_fraction = 0.2f;

  auto detector = OrientedObjectDetector::Create(std::move(options));
  EXPECT_EQ(detector.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(detector.status().message(),
              testing::HasSubstr("tile_overlap_fraction"));
}

}  // namespace
}  // namespace oriented_object_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
