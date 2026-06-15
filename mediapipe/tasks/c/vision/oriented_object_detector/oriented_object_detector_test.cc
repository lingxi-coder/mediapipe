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

// Integration test for the oriented (OBB) object detector C API.
//
// The assertions inside the TEST are gated on the presence of a
// yolov8n-obb.tflite fixture.  If the fixture is absent the test calls
// GTEST_SKIP() and exits cleanly (neither failing nor faking a pass).  To
// enable the assertions:
//   1.  Place the exported yolov8n-obb.tflite file at
//       mediapipe/tasks/testdata/vision/yolov8n-obb.tflite.
//   2.  Add it to mediapipe/tasks/testdata/vision/BUILD (mediapipe_files +
//       filegroup) and uncomment the data dep in this package's BUILD rule.
//   3.  Re-run the test.

#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"

#include <cmath>
#include <cstdint>
#include <cstring>
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
constexpr char kObbModel[] = "yolov8n-obb.tflite";
constexpr char kImageFile[] = "boats.jpg";

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

// RAII guard: closes the detector on scope exit.
struct ScopedMpOrientedObjectDetector {
  MpOrientedObjectDetectorPtr ptr = nullptr;
  ~ScopedMpOrientedObjectDetector() {
    if (ptr) MpOrientedObjectDetectorClose(ptr, /*error_msg=*/nullptr);
  }
  ScopedMpOrientedObjectDetector() = default;
  ScopedMpOrientedObjectDetector(const ScopedMpOrientedObjectDetector&) =
      delete;
  ScopedMpOrientedObjectDetector& operator=(
      const ScopedMpOrientedObjectDetector&) = delete;
};

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

TEST(OrientedObjectDetectorCApiTest, ImageMode) {
  const std::string model_path = GetFullPath(kObbModel);

  // Skip cleanly when the fixture is not present.  This is the intentional
  // behaviour: the target must build and run, but real assertions are gated on
  // the model fixture.
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "OBB model fixture not available at " << model_path
                 << "; integration assertions gated until yolov8n-obb.tflite "
                    "is added to mediapipe/tasks/testdata/vision/.";
  }

  MpOrientedObjectDetectorOptions options = {};
  options.base_options.model_asset_path = model_path.c_str();
  options.running_mode = MpRunningMode::MP_RUNNING_MODE_IMAGE;
  options.max_results = 10;
  options.score_threshold = 0.25f;
  options.iou_threshold = 0.45f;
  options.num_classes = 15;
  options.layout = 1;  // CHANNELS_FIRST (matches yolov8n-obb.tflite [1,20,8400]).

  MpOrientedObjectDetectorPtr detector = nullptr;
  ASSERT_EQ(
      MpOrientedObjectDetectorCreate(&options, &detector, /*error_msg=*/nullptr),
      kMpOk);
  EXPECT_NE(detector, nullptr);
  ScopedMpOrientedObjectDetector scoped_detector;
  scoped_detector.ptr = detector;

  MpImagePtr raw_image = nullptr;
  ASSERT_EQ(
      MpImageCreateFromFile(GetFullPath(kImageFile).c_str(), &raw_image,
                            /*error_msg=*/nullptr),
      kMpOk);
  ScopedMpImage image(raw_image);

  MpOrientedObjectDetectorResult result;
  ASSERT_EQ(MpOrientedObjectDetectorDetectImage(detector, image.get(),
                                                /*options=*/nullptr, &result,
                                                /*error_msg=*/nullptr),
            kMpOk);

  EXPECT_GT(result.detections_count, 0u);
  bool saw_ship = false;
  for (uint32_t i = 0; i < result.detections_count; ++i) {
    EXPECT_GT(result.detections[i].width, 0.0f);
    EXPECT_GT(result.detections[i].height, 0.0f);
    EXPECT_TRUE(std::isfinite(result.detections[i].rotation));
    ASSERT_EQ(result.detections[i].categories_count, 1u);
    const MpCategory& cat = result.detections[i].categories[0];
    EXPECT_GT(cat.score, 0.0f);
    // Plan A populates category_name in-graph from model metadata; the C
    // converter strdup's it into the result.
    ASSERT_NE(cat.category_name, nullptr);
    if (cat.index == 1) {
      EXPECT_STREQ(cat.category_name, "ship");
      saw_ship = true;
    }
  }
  EXPECT_TRUE(saw_ship) << "expected a 'ship' (DOTA class 1) on boats.jpg";

  MpOrientedObjectDetectorCloseResult(&result);
}

// Runs OBB detection with a 2x2 tiling grid on boats.jpg. The yolo_obb_test_model
// fixture is vendored in this package's BUILD, so this RUNS (does not skip): it
// verifies the tiled C->C++->proto path executes and returns a valid result. It is
// a smoke test, NOT a tiling-efficacy check -- boats.jpg yields a ship in non-tiled
// mode too, so this would still pass if tiling degraded to a no-op; the converter
// unit test (tiling_options_converter_test.cc) is what pins the field mapping. In
// tiled mode the graph drops its NORM_RECT input, so we pass null
// image_processing_options. (A default no-ROI, no-rotation options object is also
// accepted; only a region-of-interest or non-zero rotation is rejected.)
TEST(OrientedObjectDetectorCApiTest, TiledImageMode) {
  const std::string model_path = GetFullPath(kObbModel);

  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "OBB model fixture not available at " << model_path;
  }

  MpOrientedObjectDetectorOptions options = {};
  options.base_options.model_asset_path = model_path.c_str();
  options.running_mode = MpRunningMode::MP_RUNNING_MODE_IMAGE;
  options.max_results = 10;
  options.score_threshold = 0.25f;
  options.iou_threshold = 0.45f;
  options.num_classes = 15;
  options.layout = 1;  // CHANNELS_FIRST
  options.tiling.tile_rows = 2;
  options.tiling.tile_cols = 2;
  options.tiling.tile_overlap_fraction = 0.2f;

  MpOrientedObjectDetectorPtr detector = nullptr;
  ASSERT_EQ(
      MpOrientedObjectDetectorCreate(&options, &detector, /*error_msg=*/nullptr),
      kMpOk);
  EXPECT_NE(detector, nullptr);
  ScopedMpOrientedObjectDetector scoped_detector;
  scoped_detector.ptr = detector;

  MpImagePtr raw_image = nullptr;
  ASSERT_EQ(
      MpImageCreateFromFile(GetFullPath(kImageFile).c_str(), &raw_image,
                            /*error_msg=*/nullptr),
      kMpOk);
  ScopedMpImage image(raw_image);

  MpOrientedObjectDetectorResult result;
  ASSERT_EQ(MpOrientedObjectDetectorDetectImage(detector, image.get(),
                                                /*options=*/nullptr, &result,
                                                /*error_msg=*/nullptr),
            kMpOk);

  EXPECT_GT(result.detections_count, 0u);
  bool saw_ship = false;
  for (uint32_t i = 0; i < result.detections_count; ++i) {
    ASSERT_EQ(result.detections[i].categories_count, 1u);
    if (result.detections[i].categories[0].index == 1) saw_ship = true;
  }
  EXPECT_TRUE(saw_ship) << "expected a 'ship' (DOTA class 1) on boats.jpg tiled";

  MpOrientedObjectDetectorCloseResult(&result);
}

TEST(OrientedObjectDetectorCApiTest, CategoryAllowlistAndDenylistFilterByName) {
  const std::string model_path = GetFullPath(kObbModel);
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "OBB model fixture not available at " << model_path;
  }

  MpImagePtr raw_image = nullptr;
  ASSERT_EQ(MpImageCreateFromFile(GetFullPath(kImageFile).c_str(), &raw_image,
                                  /*error_msg=*/nullptr),
            kMpOk);
  ScopedMpImage image(raw_image);

  auto make_base_options = []() {
    MpOrientedObjectDetectorOptions o = {};
    o.running_mode = MpRunningMode::MP_RUNNING_MODE_IMAGE;
    o.max_results = 10;
    o.score_threshold = 0.25f;
    o.iou_threshold = 0.45f;
    o.num_classes = 15;
    o.layout = 1;  // CHANNELS_FIRST
    return o;
  };

  // Allowlist {"ship"}: only ships (index 1, name "ship") may survive.
  {
    const char* allow[] = {"ship"};
    MpOrientedObjectDetectorOptions options = make_base_options();
    options.base_options.model_asset_path = model_path.c_str();
    options.category_allowlist = allow;
    options.category_allowlist_count = 1;

    MpOrientedObjectDetectorPtr detector = nullptr;
    ASSERT_EQ(MpOrientedObjectDetectorCreate(&options, &detector,
                                             /*error_msg=*/nullptr),
              kMpOk);
    ScopedMpOrientedObjectDetector scoped_detector;
    scoped_detector.ptr = detector;

    MpOrientedObjectDetectorResult result;
    ASSERT_EQ(MpOrientedObjectDetectorDetectImage(detector, image.get(),
                                                  /*options=*/nullptr, &result,
                                                  /*error_msg=*/nullptr),
              kMpOk);
    EXPECT_GT(result.detections_count, 0u) << "allowlist {ship} dropped all";
    for (uint32_t i = 0; i < result.detections_count; ++i) {
      ASSERT_EQ(result.detections[i].categories_count, 1u);
      EXPECT_EQ(result.detections[i].categories[0].index, 1);
      ASSERT_NE(result.detections[i].categories[0].category_name, nullptr);
      EXPECT_STREQ(result.detections[i].categories[0].category_name, "ship");
    }
    MpOrientedObjectDetectorCloseResult(&result);
  }

  // Denylist {"ship"}: ships (index 1) must be excluded.
  {
    const char* deny[] = {"ship"};
    MpOrientedObjectDetectorOptions options = make_base_options();
    options.base_options.model_asset_path = model_path.c_str();
    options.category_denylist = deny;
    options.category_denylist_count = 1;

    MpOrientedObjectDetectorPtr detector = nullptr;
    ASSERT_EQ(MpOrientedObjectDetectorCreate(&options, &detector,
                                             /*error_msg=*/nullptr),
              kMpOk);
    ScopedMpOrientedObjectDetector scoped_detector;
    scoped_detector.ptr = detector;

    MpOrientedObjectDetectorResult result;
    ASSERT_EQ(MpOrientedObjectDetectorDetectImage(detector, image.get(),
                                                  /*options=*/nullptr, &result,
                                                  /*error_msg=*/nullptr),
              kMpOk);
    for (uint32_t i = 0; i < result.detections_count; ++i) {
      ASSERT_EQ(result.detections[i].categories_count, 1u);
      EXPECT_NE(result.detections[i].categories[0].index, 1);
    }
    MpOrientedObjectDetectorCloseResult(&result);
  }
}

TEST(OrientedObjectDetectorCApiTest, RejectsAllowlistAndDenylistTogether) {
  const std::string model_path = GetFullPath(kObbModel);
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "OBB model fixture not available at " << model_path;
  }
  const char* allow[] = {"ship"};
  const char* deny[] = {"ship"};
  MpOrientedObjectDetectorOptions options = {};
  options.base_options.model_asset_path = model_path.c_str();
  options.running_mode = MpRunningMode::MP_RUNNING_MODE_IMAGE;
  options.max_results = 10;
  options.score_threshold = 0.25f;
  options.iou_threshold = 0.45f;
  options.num_classes = 15;
  options.layout = 1;
  options.category_allowlist = allow;
  options.category_allowlist_count = 1;
  options.category_denylist = deny;
  options.category_denylist_count = 1;

  MpOrientedObjectDetectorPtr detector = nullptr;
  EXPECT_NE(MpOrientedObjectDetectorCreate(&options, &detector,
                                           /*error_msg=*/nullptr),
            kMpOk);
  if (detector) MpOrientedObjectDetectorClose(detector, /*error_msg=*/nullptr);
}

}  // namespace
