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

// Pins the byte layout of the OBB C-API tiling structs. The Python ctypes structs
// (MpOrientedTileRectC / MpOrientedTilingOptionsC and the `tiling` field of
// MpOrientedObjectDetectorOptionsC in
// mediapipe/tasks/python/vision/oriented_object_detector.py) MUST match these
// offsets. 64-bit targets (int=4, float=4, pointer=8, native alignment).

#include <cstddef>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"

static_assert(sizeof(MpOrientedTileRect) == 4 * sizeof(float),
              "MpOrientedTileRect must be 4 packed floats");
static_assert(offsetof(MpOrientedTileRect, x_center) == 0, "");
static_assert(offsetof(MpOrientedTileRect, y_center) == 4, "");
static_assert(offsetof(MpOrientedTileRect, width) == 8, "");
static_assert(offsetof(MpOrientedTileRect, height) == 12, "");

static_assert(sizeof(MpOrientedTilingOptions) == 40,
              "MpOrientedTilingOptions layout pinned for the Python ctypes mirror");
static_assert(offsetof(MpOrientedTilingOptions, tile_rows) == 0, "");
static_assert(offsetof(MpOrientedTilingOptions, tile_cols) == 4, "");
static_assert(offsetof(MpOrientedTilingOptions, tile_overlap_fraction) == 8, "");
static_assert(offsetof(MpOrientedTilingOptions, explicit_tiles) == 16, "");
static_assert(offsetof(MpOrientedTilingOptions, explicit_tiles_count) == 24, "");
static_assert(offsetof(MpOrientedTilingOptions, tile_local_nms_iou_threshold) == 28,
              "");
static_assert(offsetof(MpOrientedTilingOptions, max_detections_after_tile_nms) == 32,
              "");

// `tiling` occupies a contiguous block between num_classes and result_callback,
// plus absolute anchors so the Python layout test cross-checks the whole parent
// prefix (base_options + scalars incl. class_agnostic_nms).
static_assert(offsetof(MpOrientedObjectDetectorOptions, tracking) ==
                  offsetof(MpOrientedObjectDetectorOptions, tiling) +
                      sizeof(MpOrientedTilingOptions),
              "tracking must immediately follow tiling");
static_assert(offsetof(MpOrientedObjectDetectorOptions, tiling) == 144,
              "parent prefix size pinned for the Python ctypes mirror");
static_assert(offsetof(MpOrientedObjectDetectorOptions, result_callback) == 216, "");
static_assert(sizeof(MpOrientedObjectDetectorOptions) == 224, "");

namespace {
TEST(OrientedTilingOptionsAbiTest, LayoutPinned) { SUCCEED(); }
}  // namespace
