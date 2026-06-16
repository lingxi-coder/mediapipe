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

// Pins the byte layout of the OBB C-API MpOrientedTrackingOptions struct. The
// Python ctypes mirror (MpOrientedTrackingOptionsC and the `tracking` field of
// MpOrientedObjectDetectorOptionsC in
// mediapipe/tasks/python/vision/oriented_object_detector.py) MUST match these
// offsets. If the C struct is ever reordered, these static_asserts fail at
// compile time -- a loud signal that the Python binding would otherwise be
// silently corrupted. 64-bit targets (int=4, float=4, bool=1, pointer=8,
// native alignment); stable across arm64 / x86-64 / Windows LLP64.
//
// NOTE: the parent-layout pins below (tracking == 184, result_callback == 216,
// sizeof == 224) are intentionally duplicated in the sibling
// tiling_options_abi_test.cc as defense-in-depth; a parent reorder must update
// both.

#include <cstddef>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"

static_assert(sizeof(MpOrientedTrackingOptions) == 28,
              "MpOrientedTrackingOptions layout pinned for the Python ctypes mirror");
static_assert(offsetof(MpOrientedTrackingOptions, tracker_type) == 0, "");
static_assert(offsetof(MpOrientedTrackingOptions, track_high_threshold) == 4, "");
static_assert(offsetof(MpOrientedTrackingOptions, track_low_threshold) == 8, "");
static_assert(offsetof(MpOrientedTrackingOptions, new_track_threshold) == 12, "");
static_assert(offsetof(MpOrientedTrackingOptions, track_buffer) == 16, "");
static_assert(offsetof(MpOrientedTrackingOptions, match_threshold) == 20, "");
static_assert(offsetof(MpOrientedTrackingOptions, enable_gmc) == 24, "");
// Pin the bool field's WIDTH (1 byte): a c_bool->c_int swap on the Python side
// would NOT change any offset (the 3 trailing pad bytes absorb the widening),
// so an offset-only check cannot catch it; this size pin can.
static_assert(sizeof(MpOrientedTrackingOptions::enable_gmc) == 1,
              "enable_gmc must stay 1 byte (matches Python c_bool)");

// `tracking` immediately follows `tiling` in the parent options struct.
static_assert(offsetof(MpOrientedObjectDetectorOptions, tracking) ==
                  offsetof(MpOrientedObjectDetectorOptions, tiling) +
                      sizeof(MpOrientedTilingOptions),
              "tracking must immediately follow tiling");

// Absolute anchors so the Python layout test cross-checks the WHOLE parent
// prefix, not just tracking's relative placement. If the prefix ever changes,
// this fails loudly and the Python ctypes mirror must be updated in lockstep.
static_assert(offsetof(MpOrientedObjectDetectorOptions, tracking) == 184,
              "tracking offset pinned for the Python ctypes mirror");
// result_callback is 8-byte aligned; tracking ends at 212 (184+28), so 4 pad
// bytes precede result_callback at 216 (the layout is NOT contiguous at 212).
static_assert(offsetof(MpOrientedObjectDetectorOptions, result_callback) == 216, "");
static_assert(sizeof(MpOrientedObjectDetectorOptions) == 224, "");

namespace {

// The real guarantees are the compile-time static_asserts above; this gives the
// cc_test target a runtime case to execute.
TEST(OrientedTrackingOptionsAbiTest, LayoutPinned) { SUCCEED(); }

}  // namespace
