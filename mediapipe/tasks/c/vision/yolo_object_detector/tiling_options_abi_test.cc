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

// Pins the byte layout of the YOLO C-API tiling structs. The Python ctypes
// structs (MpTileRectC / MpTilingOptionsC and the `tiling` field of
// MpYoloObjectDetectorOptionsC in
// mediapipe/tasks/python/vision/yolo_object_detector.py) MUST match these
// offsets. If the C struct is ever reordered, these static_asserts fail at
// compile time -- a loud signal that the Python binding would otherwise be
// silently corrupted. 64-bit targets (int=4, float=4, bool=1, pointer=8,
// native alignment); stable across arm64 / x86-64 / Windows LLP64.

#include <cstddef>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"

static_assert(sizeof(MpTileRect) == 4 * sizeof(float),
              "MpTileRect must be 4 packed floats");
static_assert(offsetof(MpTileRect, x_center) == 0, "");
static_assert(offsetof(MpTileRect, y_center) == 4, "");
static_assert(offsetof(MpTileRect, width) == 8, "");
static_assert(offsetof(MpTileRect, height) == 12, "");

static_assert(sizeof(MpTilingOptions) == 48,
              "MpTilingOptions layout pinned for the Python ctypes mirror");
static_assert(offsetof(MpTilingOptions, tile_rows) == 0, "");
static_assert(offsetof(MpTilingOptions, tile_cols) == 4, "");
static_assert(offsetof(MpTilingOptions, tile_overlap_fraction) == 8, "");
static_assert(offsetof(MpTilingOptions, explicit_tiles) == 16, "");
static_assert(offsetof(MpTilingOptions, explicit_tiles_count) == 24, "");
static_assert(offsetof(MpTilingOptions, tile_local_nms_iou_threshold) == 28, "");
static_assert(offsetof(MpTilingOptions, max_detections_after_tile_nms) == 32, "");
static_assert(offsetof(MpTilingOptions, enable_motion_scheduling) == 36, "");
static_assert(offsetof(MpTilingOptions, max_scheduled_tiles) == 40, "");
// Pin the bool field's WIDTH (1 byte): a c_bool->c_int swap on the Python side
// would NOT change any offset (the 3 trailing pad bytes absorb the widening),
// so an offset-only check cannot catch it; this size pin can.
static_assert(sizeof(MpTilingOptions::enable_motion_scheduling) == 1,
              "enable_motion_scheduling must stay 1 byte (matches Python c_bool)");

// `tiling` occupies a contiguous block between num_classes and result_callback.
static_assert(offsetof(MpYoloObjectDetectorOptions, tiling) >
                  offsetof(MpYoloObjectDetectorOptions, num_classes),
              "tiling must follow num_classes");
static_assert(offsetof(MpYoloObjectDetectorOptions, tracking) ==
                  offsetof(MpYoloObjectDetectorOptions, tiling) +
                      sizeof(MpTilingOptions),
              "tracking must immediately follow tiling");

// Absolute anchors so the Python layout test cross-checks the WHOLE parent
// prefix (base_options + scalars), not just tiling's relative placement. If the
// prefix ever changes, this fails loudly and the Python ctypes mirror
// (incl. MpBaseOptionsC) must be updated in lockstep.
static_assert(offsetof(MpYoloObjectDetectorOptions, tiling) == 136,
              "parent prefix size pinned for the Python ctypes mirror");
// NOTE: these parent-layout pins (result_callback == 216, sizeof == 224) are
// intentionally duplicated in the sibling tracking_options_abi_test.cc as
// defense-in-depth; a parent reorder must update both.
static_assert(offsetof(MpYoloObjectDetectorOptions, result_callback) == 216, "");
static_assert(sizeof(MpYoloObjectDetectorOptions) == 224, "");

namespace {

// The real guarantees are the compile-time static_asserts above; this gives the
// cc_test target a runtime case to execute.
TEST(TilingOptionsAbiTest, LayoutPinned) { SUCCEED(); }

}  // namespace
