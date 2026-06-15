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

#include "mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.h"

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

namespace mediapipe::tasks::c::vision::yolo_object_detector {
namespace {

using CppTilingOptions = ::mediapipe::tasks::vision::yolo_object_detector::
    YoloObjectDetectorOptions::TilingOptions;

TEST(TilingOptionsConverterTest, CopiesAllScalarFields) {
  MpTilingOptions in = {};
  in.tile_rows = 3;
  in.tile_cols = 4;
  in.tile_overlap_fraction = 0.25f;
  in.tile_local_nms_iou_threshold = 0.6f;
  in.max_detections_after_tile_nms = 7;
  in.enable_motion_scheduling = true;
  in.max_scheduled_tiles = 5;

  CppTilingOptions out;
  CppConvertToTilingOptions(in, &out);

  EXPECT_EQ(out.tile_rows, 3);
  EXPECT_EQ(out.tile_cols, 4);
  EXPECT_FLOAT_EQ(out.tile_overlap_fraction, 0.25f);
  EXPECT_FLOAT_EQ(out.tile_local_nms_iou_threshold, 0.6f);
  EXPECT_EQ(out.max_detections_after_tile_nms, 7);
  EXPECT_TRUE(out.enable_motion_scheduling);
  EXPECT_EQ(out.max_scheduled_tiles, 5);
  EXPECT_TRUE(out.explicit_tiles.empty());
}

TEST(TilingOptionsConverterTest, CopiesExplicitTilesArray) {
  const MpTileRect tiles[] = {
      {0.1f, 0.2f, 0.5f, 0.4f},
      {0.75f, 0.6f, 0.45f, 0.3f},
  };
  MpTilingOptions in = {};
  in.explicit_tiles = tiles;
  in.explicit_tiles_count = 2;

  CppTilingOptions out;
  CppConvertToTilingOptions(in, &out);

  ASSERT_EQ(out.explicit_tiles.size(), 2u);
  EXPECT_FLOAT_EQ(out.explicit_tiles[0].x_center, 0.1f);
  EXPECT_FLOAT_EQ(out.explicit_tiles[0].y_center, 0.2f);
  EXPECT_FLOAT_EQ(out.explicit_tiles[0].width, 0.5f);
  EXPECT_FLOAT_EQ(out.explicit_tiles[0].height, 0.4f);
  EXPECT_FLOAT_EQ(out.explicit_tiles[1].x_center, 0.75f);
  EXPECT_FLOAT_EQ(out.explicit_tiles[1].y_center, 0.6f);
  EXPECT_FLOAT_EQ(out.explicit_tiles[1].width, 0.45f);
  EXPECT_FLOAT_EQ(out.explicit_tiles[1].height, 0.3f);
}

TEST(TilingOptionsConverterTest, ZeroInitializedDisablesTiling) {
  MpTilingOptions in = {};
  CppTilingOptions out;
  CppConvertToTilingOptions(in, &out);

  EXPECT_EQ(out.tile_rows, 0);
  EXPECT_EQ(out.tile_cols, 0);
  EXPECT_TRUE(out.explicit_tiles.empty());
  EXPECT_FALSE(out.enable_motion_scheduling);
}

TEST(TilingOptionsConverterTest, NullExplicitTilesIsSafe) {
  MpTilingOptions in = {};
  in.explicit_tiles = nullptr;
  in.explicit_tiles_count = 0;

  CppTilingOptions out;
  CppConvertToTilingOptions(in, &out);
  EXPECT_TRUE(out.explicit_tiles.empty());
}

TEST(TilingOptionsConverterTest, ClearsPreexistingExplicitTiles) {
  const MpTileRect tiles[] = {{0.1f, 0.1f, 0.2f, 0.2f}};
  MpTilingOptions in = {};
  in.explicit_tiles = tiles;
  in.explicit_tiles_count = 1;

  CppTilingOptions out;
  out.explicit_tiles.push_back({9.0f, 9.0f, 9.0f, 9.0f});  // stale content
  CppConvertToTilingOptions(in, &out);

  ASSERT_EQ(out.explicit_tiles.size(), 1u);
  EXPECT_FLOAT_EQ(out.explicit_tiles[0].x_center, 0.1f);
}

TEST(TilingOptionsConverterTest, EmptyInputClearsPreexistingExplicitTiles) {
  MpTilingOptions in = {};
  in.explicit_tiles = nullptr;
  in.explicit_tiles_count = 0;

  CppTilingOptions out;
  out.explicit_tiles.push_back({9.0f, 9.0f, 9.0f, 9.0f});  // stale content
  CppConvertToTilingOptions(in, &out);

  EXPECT_TRUE(out.explicit_tiles.empty());
}

}  // namespace
}  // namespace mediapipe::tasks::c::vision::yolo_object_detector
