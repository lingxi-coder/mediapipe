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

#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
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
// The generated options proto (mirrors the alias used in the .cc); needed to
// name the proto-side TrackingOptions enum in the converter round-trip test.
using YoloObjectDetectorOptionsProto =
    ::mediapipe::tasks::vision::yolo_object_detector::proto::
        YoloObjectDetectorOptions;

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

// Translates `src` by (dx, dy) pixels (border replicated), returning a new
// SRGB Image. Produces real, trackable motion for the optical-flow tracker.
Image TranslateImage(const Image& src, int dx, int dy) {
  cv::Mat in = mediapipe::formats::MatView(src.GetImageFrameSharedPtr().get());
  cv::Mat shifted;
  cv::Mat m = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
  cv::warpAffine(in, shifted, m, in.size(), cv::INTER_LINEAR,
                 cv::BORDER_REPLICATE);
  cv::Mat out = shifted.clone();  // own the buffer
  mediapipe::ImageFrame frame(mediapipe::ImageFormat::SRGB, out.cols, out.rows,
                              out.step, out.data, [out](uint8_t[]) {});
  return Image(std::make_shared<mediapipe::ImageFrame>(std::move(frame)));
}

// ---------------------------------------------------------------------------
// Options passthrough test — no model fixture required.
// Verifies that the nested TilingOptions struct is copied correctly into the
// options proto by ConvertYoloObjectDetectorOptionsToProto.
// ---------------------------------------------------------------------------
TEST(YoloObjectDetectorOptionsTest, OversizedTileGridRejectedBeforeLoadingModel) {
  for (const auto [rows, cols] : {std::pair<int, int>{65536, 65536},
                                  std::pair<int, int>{46341, 46341},
                                  std::pair<int, int>{std::numeric_limits<int>::max(), 2}}) {
    auto options = std::make_unique<YoloObjectDetectorOptions>();
    options->base_options.model_asset_path = "/missing/model.tflite";
    options->num_classes = 80;
    options->tiling.tile_rows = rows;
    options->tiling.tile_cols = cols;
    auto result = YoloObjectDetector::Create(std::move(options));
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(result.status().message(), testing::HasSubstr(
        "tiling.tile_rows * tiling.tile_cols must be <= 2147483647"));
  }
}

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

  // Sub-project B: motion-scheduling fields round-trip through the converter.
  options->tiling.enable_motion_scheduling = true;
  options->tiling.max_scheduled_tiles = 4;
  auto proto2 = ConvertYoloObjectDetectorOptionsToProto(options.get());
  EXPECT_TRUE(proto2->tiling().enable_motion_scheduling());
  EXPECT_EQ(proto2->tiling().max_scheduled_tiles(), 4);
}

// ---------------------------------------------------------------------------
// Options passthrough test — no model fixture required.
// Verifies that the nested TrackingOptions struct is mapped into the options
// proto by ConvertYoloObjectDetectorOptionsToProto (tracker selection +
// thresholds), including the struct-enum <-> proto-enum static_cast.
// ---------------------------------------------------------------------------
TEST(YoloObjectDetectorOptionsConverterTest, MapsTrackingOptions) {
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->tracking.tracker_type =
      YoloObjectDetectorOptions::TrackingOptions::kBotsort;
  options->tracking.track_high_threshold = 0.55f;
  options->tracking.track_low_threshold = 0.15f;
  options->tracking.new_track_threshold = 0.65f;
  options->tracking.track_buffer = 25;
  options->tracking.match_threshold = 0.75f;
  options->tracking.enable_gmc = true;
  options->tracking.nominal_frame_rate = 60;

  auto proto = ConvertYoloObjectDetectorOptionsToProto(options.get());

  EXPECT_EQ(proto->tracking().tracker_type(),
            YoloObjectDetectorOptionsProto::TrackingOptions::BOTSORT);
  EXPECT_FLOAT_EQ(proto->tracking().track_high_threshold(), 0.55f);
  EXPECT_FLOAT_EQ(proto->tracking().track_low_threshold(), 0.15f);
  EXPECT_FLOAT_EQ(proto->tracking().new_track_threshold(), 0.65f);
  EXPECT_EQ(proto->tracking().track_buffer(), 25);
  EXPECT_FLOAT_EQ(proto->tracking().match_threshold(), 0.75f);
  EXPECT_TRUE(proto->tracking().enable_gmc());
  EXPECT_EQ(proto->tracking().nominal_frame_rate(), 60);
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

// Tiled VIDEO mode with the BoxTracker: pan boats.jpg a few px/frame so the
// optical-flow tracker has real motion to follow, and assert boat detections
// persist across frames (temporal recall), not just on the first frame.
TEST(YoloObjectDetectorTest, TiledVideoTracksBoatsWhilePanning) {
  const std::string model_path = ModelPath();
  const std::string image_path = JoinPath("./", kTestDataDirectory, kBoatsImage);
  if (!mediapipe::file::Exists(model_path).ok() ||
      !mediapipe::file::Exists(image_path).ok()) {
    GTEST_SKIP() << "YOLO model or boats.jpg fixture not available.";
  }

  MP_ASSERT_OK_AND_ASSIGN(Image base_image, DecodeImageFromFile(image_path));

  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::VIDEO;
  options->max_results = 10;
  options->num_classes = 80;
  options->score_threshold = 0.09f;
  options->iou_threshold = 0.45f;
  options->tiling.tile_cols = 2;
  options->tiling.tile_overlap_fraction = 0.2f;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));

  int frames_with_boat = 0;
  const int kFrames = 8;
  for (int i = 0; i < kFrames; ++i) {
    Image frame = TranslateImage(base_image, /*dx=*/2 * i, /*dy=*/0);
    MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result,
                            detector->DetectForVideo(frame, /*timestamp_ms=*/i));
    for (const auto& d : result.detections) {
      ASSERT_EQ(d.categories.size(), 1u);
      if (d.categories[0].index == 8) {
        ++frames_with_boat;
        break;
      }
    }
  }
  MP_ASSERT_OK(detector->Close());

  EXPECT_GE(frames_with_boat, kFrames / 2)
      << "expected boats tracked across most panning frames";
}

// Tiled VIDEO mode selecting the BoTSORT tracker (tracker_type=BOTSORT).
// Mirrors TiledVideoTracksBoatsWhilePanning exactly except for the tracker
// selection: BoTSORT runs on every inferred frame, so the tiled boats scene
// must still yield non-empty detections by the last frame. Skips when the
// model/image fixtures are absent, like the sibling model-gated tests.
TEST(YoloObjectDetectorTest, TiledVideoTracksBoatsWithBotsort) {
  const std::string model_path = ModelPath();
  const std::string image_path = JoinPath("./", kTestDataDirectory, kBoatsImage);
  if (!mediapipe::file::Exists(model_path).ok() ||
      !mediapipe::file::Exists(image_path).ok()) {
    GTEST_SKIP() << "YOLO model or boats.jpg fixture not available.";
  }

  MP_ASSERT_OK_AND_ASSIGN(Image base_image, DecodeImageFromFile(image_path));

  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::VIDEO;
  options->max_results = 10;
  options->num_classes = 80;
  options->score_threshold = 0.09f;
  options->iou_threshold = 0.45f;
  options->tiling.tile_cols = 2;
  options->tiling.tile_overlap_fraction = 0.2f;
  // Select the BoTSORT tracker instead of the default optical-flow BoxTracker.
  options->tracking.tracker_type =
      YoloObjectDetectorOptions::TrackingOptions::kBotsort;
  // The boats scene is detected at a low score_threshold (0.09), so the boat
  // detections sit well below BoTSORT's default new_track_threshold (0.7) and
  // track_high_threshold (0.6). With the defaults BoTSORT confirms zero tracks
  // and never emits a track_id. Lower the confirmation thresholds beneath the
  // detection scores so tracks actually form and persist across frames.
  // NOTE: these tuned values are chosen for this synthetic low-confidence
  // fixture and are NOT representative production defaults.
  options->tracking.track_high_threshold = 0.05f;
  options->tracking.new_track_threshold = 0.05f;
  options->tracking.track_low_threshold = 0.02f;
  // Relax the association gate and widen the lost-track buffer so confirmed
  // tracks survive the per-frame panning jitter (dx=2px) and the tile-merge
  // box noise rather than dying after a single unmatched frame; otherwise the
  // tracker emits an empty set on some frames and no id reaches the result.
  options->tracking.match_threshold = 0.95f;
  options->tracking.track_buffer = 60;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));

  int frames_with_boat = 0;
  const int kFrames = 8;
  std::vector<std::set<std::string>> per_frame_track_ids;
  for (int i = 0; i < kFrames; ++i) {
    Image frame = TranslateImage(base_image, /*dx=*/2 * i, /*dy=*/0);
    MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result,
                            detector->DetectForVideo(frame, /*timestamp_ms=*/i));
    for (const auto& d : result.detections) {
      ASSERT_EQ(d.categories.size(), 1u);
      if (d.categories[0].index == 8) {
        ++frames_with_boat;
        break;
      }
    }
    std::set<std::string> ids_this_frame;
    for (const auto& det : result.detections) {
      if (det.track_id.has_value()) ids_this_frame.insert(*det.track_id);
    }
    per_frame_track_ids.push_back(ids_this_frame);
  }
  MP_ASSERT_OK(detector->Close());

  // BoTSORT runs on every inferred frame; the tiled boats scene must yield
  // boat content (category index 8) across most frames, mirroring the
  // sibling TiledVideoTracksBoatsWhilePanning flake-resistant assertion.
  EXPECT_GE(frames_with_boat, kFrames / 2)
      << "expected boats tracked across most BoTSORT tiled frames";

  // BoTSORT track ids survive the full pipeline to the public result, and at
  // least one object keeps a stable id across consecutive frames.
  ASSERT_GE(per_frame_track_ids.size(), 2u);
  const auto& last_ids = per_frame_track_ids.back();
  EXPECT_FALSE(last_ids.empty()) << "final frame produced no track ids";
  bool stable = false;
  for (size_t f = 1; f < per_frame_track_ids.size() && !stable; ++f) {
    for (const auto& id : per_frame_track_ids[f]) {
      if (per_frame_track_ids[f - 1].count(id)) { stable = true; break; }
    }
  }
  EXPECT_TRUE(stable)
      << "expected at least one track id stable across consecutive frames";
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
// enable_motion_scheduling requires VIDEO/LIVE_STREAM; IMAGE mode is rejected
// at Create() (the scheduler is meaningless without temporal continuity).
// This check fires before model loading, so no model fixture is required.
// ---------------------------------------------------------------------------
TEST(YoloObjectDetectorTest, MotionSchedulingInImageModeRejected) {
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = ModelPath();
  options->running_mode = core::RunningMode::IMAGE;
  options->num_classes = 80;
  options->tiling.tile_cols = 2;
  options->tiling.enable_motion_scheduling = true;
  auto detector = YoloObjectDetector::Create(std::move(options));
  EXPECT_EQ(detector.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(detector.status().message(),
              testing::HasSubstr("motion scheduling"));
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

// Tiled VIDEO with motion scheduling ON: panning boats. The cyclic
// scheduler<-loopback graph must run to completion (no deadlock) and boats
// must persist across frames (DETECT frames detect; SKIP frames are filled by
// the tracker).
TEST(YoloObjectDetectorTest, TiledVideoSchedulingPanningKeepsBoats) {
  const std::string model_path = ModelPath();
  const std::string image_path = JoinPath("./", kTestDataDirectory, kBoatsImage);
  if (!mediapipe::file::Exists(model_path).ok() ||
      !mediapipe::file::Exists(image_path).ok()) {
    GTEST_SKIP() << "YOLO model or boats.jpg fixture not available.";
  }
  MP_ASSERT_OK_AND_ASSIGN(Image base_image, DecodeImageFromFile(image_path));

  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::VIDEO;
  options->max_results = 10;
  options->num_classes = 80;
  options->score_threshold = 0.09f;
  options->iou_threshold = 0.45f;
  options->tiling.tile_cols = 2;
  options->tiling.tile_overlap_fraction = 0.2f;
  options->tiling.enable_motion_scheduling = true;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));

  int frames_with_boat = 0;
  const int kFrames = 8;
  for (int i = 0; i < kFrames; ++i) {
    Image frame = TranslateImage(base_image, /*dx=*/2 * i, /*dy=*/0);
    MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result,
                            detector->DetectForVideo(frame, /*timestamp_ms=*/i));
    for (const auto& d : result.detections) {
      ASSERT_EQ(d.categories.size(), 1u);
      if (d.categories[0].index == 8) {
        ++frames_with_boat;
        break;
      }
    }
  }
  MP_ASSERT_OK(detector->Close());
  EXPECT_GE(frames_with_boat, kFrames / 2)
      << "boats should persist across most frames with scheduling on";
}

// SKIP -> tracker-fill at the public level: after a first DETECT frame, feed
// IDENTICAL frames (zero motion). Each DetectForVideo call returns that frame's
// result synchronously; boats must still be present (the tracker fills any
// SKIPped frame). The deterministic "inference actually skipped" proof is the
// graph-level video_tile_scheduler_pipeline_test.
TEST(YoloObjectDetectorTest, TiledVideoSchedulingStaticFramesKeepBoats) {
  const std::string model_path = ModelPath();
  const std::string image_path = JoinPath("./", kTestDataDirectory, kBoatsImage);
  if (!mediapipe::file::Exists(model_path).ok() ||
      !mediapipe::file::Exists(image_path).ok()) {
    GTEST_SKIP() << "YOLO model or boats.jpg fixture not available.";
  }
  MP_ASSERT_OK_AND_ASSIGN(Image base_image, DecodeImageFromFile(image_path));

  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::VIDEO;
  options->max_results = 10;
  options->num_classes = 80;
  options->score_threshold = 0.09f;
  options->iou_threshold = 0.45f;
  options->tiling.tile_cols = 2;
  options->tiling.tile_overlap_fraction = 0.2f;
  options->tiling.enable_motion_scheduling = true;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));

  const int kFrames = 6;
  int frames_with_boat = 0;
  for (int i = 0; i < kFrames; ++i) {
    Image frame = TranslateImage(base_image, /*dx=*/0, /*dy=*/0);  // identical
    MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result,
                            detector->DetectForVideo(frame, /*timestamp_ms=*/i));
    for (const auto& d : result.detections) {
      if (!d.categories.empty() && d.categories[0].index == 8) {
        ++frames_with_boat;
        break;
      }
    }
  }
  MP_ASSERT_OK(detector->Close());
  EXPECT_GE(frames_with_boat, kFrames - 1)
      << "tracker should keep boats present across static (SKIPped) frames";
}

// ---------------------------------------------------------------------------
// BOTSORT tracker-selection validation — all fire at Create() before the model
// loads (no model_asset_path set), so no fixture is required. BOTSORT requires
// stream mode + tiling + no motion-scheduling + num_classes in [1, 256].
// ---------------------------------------------------------------------------
TEST(YoloObjectDetectorTrackingValidationTest, BotsortInImageModeRejected) {
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->running_mode = core::RunningMode::IMAGE;
  options->num_classes = 80;
  options->tiling.tile_rows = 2;
  options->tiling.tile_cols = 2;
  options->tracking.tracker_type =
      YoloObjectDetectorOptions::TrackingOptions::kBotsort;
  auto result = YoloObjectDetector::Create(std::move(options));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  // Assert on the BOTSORT message so the check is not vacuous (no model is
  // loaded; a missing model also yields kInvalidArgument, so the code alone
  // cannot prove our validation fired).
  EXPECT_THAT(result.status().message(),
              testing::HasSubstr("IMAGE mode"));
}

TEST(YoloObjectDetectorTrackingValidationTest, BotsortWithoutTilingRejected) {
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback = [](absl::StatusOr<YoloObjectDetectorResult>,
                                const Image&, int64_t) {};
  options->num_classes = 80;
  // tiling left at default 1x1 (disabled)
  options->tracking.tracker_type =
      YoloObjectDetectorOptions::TrackingOptions::kBotsort;
  auto result = YoloObjectDetector::Create(std::move(options));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(result.status().message(),
              testing::HasSubstr("requires tiling"));
}

TEST(YoloObjectDetectorTrackingValidationTest,
     BotsortWithMotionSchedulingAccepted) {
  const std::string model_path = ModelPath();
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "YOLO model fixture not available at " << model_path;
  }
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::VIDEO;
  options->num_classes = 80;
  options->tiling.tile_rows = 2;
  options->tiling.tile_cols = 2;
  options->tiling.enable_motion_scheduling = true;
  options->tracking.tracker_type =
      YoloObjectDetectorOptions::TrackingOptions::kBotsort;
  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK(detector->Close());
}

TEST(YoloObjectDetectorTrackingValidationTest,
     BotsortWithTooManyClassesRejected) {
  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->running_mode = core::RunningMode::VIDEO;
  options->num_classes = 300;  // > 256: uint8_t class-id limit
  options->tiling.tile_rows = 2;
  options->tiling.tile_cols = 2;
  options->tracking.tracker_type =
      YoloObjectDetectorOptions::TrackingOptions::kBotsort;
  auto result = YoloObjectDetector::Create(std::move(options));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(result.status().message(),
              testing::HasSubstr("num_classes in [1, 256]"));
}

}  // namespace
}  // namespace yolo_object_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
