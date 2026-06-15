# OBB Tiling Bindings (C-API + Python) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose OBB's `TilingOptions` (6 fields, full parity) through the OrientedObjectDetector **C API and Python** wrappers.

**Architecture:** Separate `MpOriented*` C structs + a converter (folded into the OBB libs) wired into the OBB C `CppConvertToDetectorOptions`; mirrored Python ctypes structs + dataclasses + a testable marshalling helper. A compile-time C++ ABI-pin anchors the layout. No YOLO/cc/proto change (OBB tiling already maps to the proto).

**Tech Stack:** C/C++ + Bazel (`--define MEDIAPIPE_DISABLE_GPU=1`), GoogleTest, Python 3 + ctypes, absltest. Spec: `docs/superpowers/specs/2026-06-15-obb-tiling-bindings-design.md`. Reference templates (already shipped): YOLO Phase 1 (`mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.{h,cc}`, `tiling_options_abi_test.cc`) and Phase 2 (`mediapipe/tasks/python/vision/yolo_object_detector.py` tiling structs/dataclasses/`_build_tiling_options_c`).

**CRITICAL build constraint:** The C-API layer (Tasks 1–3) builds and runs here; the OBB C-API test has a **live model fixture** (`yolo_obb_test_model`) so its assertions RUN. The Python layer (Tasks 4–5) does **NOT** build/run here — do NOT run bazel for Python; verify Python via `python3 -m py_compile` + a standalone `python3` ctypes check.

**Standing constraints:** branch `dev` (no new branch); commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`; English comments.

---

## File Structure

| File | Change |
|---|---|
| `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h` | Modify — `MpOrientedTileRect`/`MpOrientedTilingOptions` + `tiling` field |
| `mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_converter.{h,cc}` | Create — `CppConvertToTilingOptions` |
| `mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_converter_test.cc` | Create — model-free converter gate |
| `mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_abi_test.cc` | Create — compile-time ABI-pin |
| `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.cc` | Modify — wire converter |
| `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector_test.cc` | Modify — real tiled e2e |
| `mediapipe/tasks/c/vision/oriented_object_detector/BUILD` | Modify — converter into both libs + 2 new cc_tests |
| `mediapipe/tasks/python/vision/oriented_object_detector.py` | Modify — ctypes structs + dataclasses + helper + marshalling |
| `mediapipe/tasks/python/test/vision/oriented_object_detector_test.py` | Modify — layout/construct/defaults/marshalling/detect tests |

---

## Task 1: OBB C structs + converter + unit test (C-API gate)

**Files:** Create `tiling_options_converter_test.cc`, `tiling_options_converter.{h,cc}`; Modify `oriented_object_detector.h`, `BUILD`.

- [ ] **Step 1: Write the failing converter unit test**

Create `mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_converter_test.cc`:

```cpp
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

#include "mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_converter.h"

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"
#include "mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h"

namespace mediapipe::tasks::c::vision::oriented_object_detector {
namespace {

using CppTilingOptions = ::mediapipe::tasks::vision::oriented_object_detector::
    OrientedObjectDetectorOptions::TilingOptions;

TEST(OrientedTilingOptionsConverterTest, CopiesAllScalarFields) {
  MpOrientedTilingOptions in = {};
  in.tile_rows = 3;
  in.tile_cols = 4;
  in.tile_overlap_fraction = 0.25f;
  in.tile_local_nms_iou_threshold = 0.6f;
  in.max_detections_after_tile_nms = 7;

  CppTilingOptions out;
  CppConvertToTilingOptions(in, &out);

  EXPECT_EQ(out.tile_rows, 3);
  EXPECT_EQ(out.tile_cols, 4);
  EXPECT_FLOAT_EQ(out.tile_overlap_fraction, 0.25f);
  EXPECT_FLOAT_EQ(out.tile_local_nms_iou_threshold, 0.6f);
  EXPECT_EQ(out.max_detections_after_tile_nms, 7);
  EXPECT_TRUE(out.explicit_tiles.empty());
}

TEST(OrientedTilingOptionsConverterTest, CopiesExplicitTilesArray) {
  const MpOrientedTileRect tiles[] = {
      {0.1f, 0.2f, 0.5f, 0.4f},
      {0.75f, 0.6f, 0.45f, 0.3f},
  };
  MpOrientedTilingOptions in = {};
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

TEST(OrientedTilingOptionsConverterTest, ZeroInitializedDisablesTiling) {
  MpOrientedTilingOptions in = {};
  CppTilingOptions out;
  CppConvertToTilingOptions(in, &out);

  EXPECT_EQ(out.tile_rows, 0);
  EXPECT_EQ(out.tile_cols, 0);
  EXPECT_TRUE(out.explicit_tiles.empty());
}

TEST(OrientedTilingOptionsConverterTest, NullExplicitTilesIsSafe) {
  MpOrientedTilingOptions in = {};
  in.explicit_tiles = nullptr;
  in.explicit_tiles_count = 0;

  CppTilingOptions out;
  CppConvertToTilingOptions(in, &out);
  EXPECT_TRUE(out.explicit_tiles.empty());
}

TEST(OrientedTilingOptionsConverterTest, ClearsPreexistingExplicitTiles) {
  const MpOrientedTileRect tiles[] = {{0.1f, 0.1f, 0.2f, 0.2f}};
  MpOrientedTilingOptions in = {};
  in.explicit_tiles = tiles;
  in.explicit_tiles_count = 1;

  CppTilingOptions out;
  out.explicit_tiles.push_back({9.0f, 9.0f, 9.0f, 9.0f});  // stale content
  CppConvertToTilingOptions(in, &out);

  ASSERT_EQ(out.explicit_tiles.size(), 1u);
  EXPECT_FLOAT_EQ(out.explicit_tiles[0].x_center, 0.1f);
}

TEST(OrientedTilingOptionsConverterTest, EmptyInputClearsPreexistingExplicitTiles) {
  MpOrientedTilingOptions in = {};
  in.explicit_tiles = nullptr;
  in.explicit_tiles_count = 0;

  CppTilingOptions out;
  out.explicit_tiles.push_back({9.0f, 9.0f, 9.0f, 9.0f});  // stale content
  CppConvertToTilingOptions(in, &out);

  EXPECT_TRUE(out.explicit_tiles.empty());
}

}  // namespace
}  // namespace mediapipe::tasks::c::vision::oriented_object_detector
```

- [ ] **Step 2: Add the cc_test to BUILD and confirm it fails to compile**

In `mediapipe/tasks/c/vision/oriented_object_detector/BUILD`, add after the `oriented_object_detector_test` target:

```python
cc_test(
    name = "tiling_options_converter_test",
    srcs = ["tiling_options_converter_test.cc"],
    deps = [
        ":oriented_object_detector_lib",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/vision/oriented_object_detector",
        "@com_google_googletest//:gtest_main",
    ],
)
```

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:tiling_options_converter_test`
Expected: FAIL — compile error (`tiling_options_converter.h` not found / `MpOrientedTilingOptions` / `CppConvertToTilingOptions` undefined). This is the red state.

- [ ] **Step 3: Add the C structs to the OBB C header**

In `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h`, insert immediately **before** `struct MpOrientedObjectDetectorOptions {`:

```c
// A frame-normalized tile given by its CENTER point and size. Mirrors
// OrientedObjectDetectorOptions::TilingOptions::TileRect. Named MpOrientedTileRect
// (not MpTileRect) to avoid colliding with the YOLO detector's distinct global
// extern "C" MpTileRect.
struct MpOrientedTileRect {
  float x_center;
  float y_center;
  float width;
  float height;
};

// Static tiling configuration. Mirrors OrientedObjectDetectorOptions::TilingOptions
// field-for-field (6 fields; OBB has no motion-scheduling knobs).
//
// Tiling is ENABLED when tile_rows * tile_cols > 1 or explicit_tiles_count > 0. A
// zero-initialized MpOrientedTilingOptions means tiling DISABLED. To tile with a
// grid, set BOTH tile_rows and tile_cols (each >= 1); a zero in either disables.
struct MpOrientedTilingOptions {
  int tile_rows;
  int tile_cols;
  float tile_overlap_fraction;
  // Explicit (non-grid) tiles. Caller owns this array; it is COPIED during Create
  // and need not outlive the MpOrientedObjectDetectorCreate call (mirrors the
  // category_allowlist pointer+count convention).
  const struct MpOrientedTileRect* explicit_tiles;
  uint32_t explicit_tiles_count;
  float tile_local_nms_iou_threshold;
  int max_detections_after_tile_nms;
};
```

Then, inside `struct MpOrientedObjectDetectorOptions { ... }`, add the field immediately **after** `int num_classes;` and **before** the `result_callback_fn` typedef block:

```c
  // Static tiling configuration. Zero-initialized => tiling disabled.
  struct MpOrientedTilingOptions tiling;
```

- [ ] **Step 4: Create the converter header**

Create `mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_converter.h`:

```cpp
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

#ifndef MEDIAPIPE_TASKS_C_VISION_ORIENTED_OBJECT_DETECTOR_TILING_OPTIONS_CONVERTER_H_
#define MEDIAPIPE_TASKS_C_VISION_ORIENTED_OBJECT_DETECTOR_TILING_OPTIONS_CONVERTER_H_

#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"
#include "mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h"

namespace mediapipe::tasks::c::vision::oriented_object_detector {

// Copies a plain-C MpOrientedTilingOptions into the C++ TilingOptions sub-struct.
// Scalars are copied 1:1; explicit_tiles (a caller-owned MpOrientedTileRect array)
// is copied element-by-element into a fresh std::vector (the caller retains
// ownership). Declared here so unit tests can verify the mapping without a graph.
void CppConvertToTilingOptions(
    const MpOrientedTilingOptions& in,
    ::mediapipe::tasks::vision::oriented_object_detector::
        OrientedObjectDetectorOptions::TilingOptions* out);

}  // namespace mediapipe::tasks::c::vision::oriented_object_detector

#endif  // MEDIAPIPE_TASKS_C_VISION_ORIENTED_OBJECT_DETECTOR_TILING_OPTIONS_CONVERTER_H_
```

- [ ] **Step 5: Create the converter implementation**

Create `mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_converter.cc`:

```cpp
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

#include "mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_converter.h"

#include <cstdint>

#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"
#include "mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h"

namespace mediapipe::tasks::c::vision::oriented_object_detector {

namespace ObbNs = ::mediapipe::tasks::vision::oriented_object_detector;

void CppConvertToTilingOptions(
    const MpOrientedTilingOptions& in,
    ObbNs::OrientedObjectDetectorOptions::TilingOptions* out) {
  out->tile_rows = in.tile_rows;
  out->tile_cols = in.tile_cols;
  out->tile_overlap_fraction = in.tile_overlap_fraction;

  out->explicit_tiles.clear();
  out->explicit_tiles.reserve(in.explicit_tiles_count);
  for (uint32_t i = 0; i < in.explicit_tiles_count; ++i) {
    out->explicit_tiles.push_back({in.explicit_tiles[i].x_center,
                                   in.explicit_tiles[i].y_center,
                                   in.explicit_tiles[i].width,
                                   in.explicit_tiles[i].height});
  }

  out->tile_local_nms_iou_threshold = in.tile_local_nms_iou_threshold;
  out->max_detections_after_tile_nms = in.max_detections_after_tile_nms;
}

}  // namespace mediapipe::tasks::c::vision::oriented_object_detector
```

- [ ] **Step 6: Add the converter to both libs in BUILD**

In `mediapipe/tasks/c/vision/oriented_object_detector/BUILD`, replace the `oriented_object_detector_lib` target:

```python
cc_library(
    name = "oriented_object_detector_lib",
    srcs = [
        "oriented_object_detector.cc",
        "tiling_options_converter.cc",
    ],
    hdrs = [
        "oriented_object_detector.h",
        "tiling_options_converter.h",
    ],
    visibility = ["//visibility:public"],
    deps = ORIENTED_OBJECT_DETECTOR_DEPS,
)
```

…and replace `oriented_object_detector_c_lib`:

```python
cc_library(
    name = "oriented_object_detector_c_lib",
    srcs = [
        "oriented_object_detector.cc",
        "tiling_options_converter.cc",
    ],
    hdrs = [
        "oriented_object_detector.h",
        "tiling_options_converter.h",
    ],
    visibility = ["//mediapipe/tasks/c:__subpackages__"],
    deps = ORIENTED_OBJECT_DETECTOR_DEPS,
    alwayslink = 1,
)
```

(`ORIENTED_OBJECT_DETECTOR_DEPS` already includes `//mediapipe/tasks/cc/vision/oriented_object_detector` — no new dep.)

- [ ] **Step 7: Run the converter test — must pass**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:tiling_options_converter_test`
Expected: PASS (6/6).

- [ ] **Step 8: Verify both libs build**

Run: `bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_lib //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_c_lib`
Expected: exit 0.

- [ ] **Step 9: Commit**

```bash
git add mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h \
        mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_converter.h \
        mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_converter.cc \
        mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_converter_test.cc \
        mediapipe/tasks/c/vision/oriented_object_detector/BUILD
git commit -m "feat(tiling): OBB C-API MpOrientedTilingOptions struct + converter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Wire converter + real tiled e2e

**Files:** Modify `oriented_object_detector.cc` (`CppConvertToDetectorOptions`), `oriented_object_detector_test.cc`.

- [ ] **Step 1: Wire the converter into `CppConvertToDetectorOptions`**

In `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.cc`, add the include alongside the other `tasks/c/vision/oriented_object_detector` includes (immediately before the `mediapipe/tasks/cc/...` includes):

```cpp
#include "mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_converter.h"
```

Then, in `CppConvertToDetectorOptions`, add as the **last** line of the function body (immediately after `out->num_classes = in.num_classes;`):

```cpp
  CppConvertToTilingOptions(in.tiling, &out->tiling);
```

- [ ] **Step 2: Verify the lib builds**

Run: `bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_lib`
Expected: exit 0.

- [ ] **Step 3: Add the real tiled e2e test**

In `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector_test.cc`, add this TEST immediately after the existing `ImageMode` test (after its closing brace, before the next TEST):

```cpp
// Runs OBB detection with a 2x2 tiling grid on boats.jpg. The yolo_obb_test_model
// fixture is vendored in this package's BUILD, so this RUNS (does not skip) and
// genuinely exercises the C->C++->proto tiling path end to end. In tiled mode the
// graph has no NORM_RECT input, so image_processing_options must be null.
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
```

- [ ] **Step 4: Run the OBB C-API test — the tiled e2e RUNS and passes**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_test`
Expected: PASS, all 4 tests run (not skipped — the fixture is vendored), including `TiledImageMode` finding a ship.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.cc \
        mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector_test.cc
git commit -m "feat(tiling): wire OBB C-API tiling converter + real tiled e2e

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: C++ ABI-pin (anchors the Python ctypes layout)

**Files:** Create `tiling_options_abi_test.cc`; Modify `BUILD`.

- [ ] **Step 1: Write the ABI-pin test**

Create `mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_abi_test.cc`:

```cpp
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
static_assert(offsetof(MpOrientedObjectDetectorOptions, result_callback) ==
                  offsetof(MpOrientedObjectDetectorOptions, tiling) +
                      sizeof(MpOrientedTilingOptions),
              "result_callback must immediately follow tiling");
static_assert(offsetof(MpOrientedObjectDetectorOptions, tiling) == 144,
              "parent prefix size pinned for the Python ctypes mirror");
static_assert(offsetof(MpOrientedObjectDetectorOptions, result_callback) == 184, "");
static_assert(sizeof(MpOrientedObjectDetectorOptions) == 192, "");

namespace {
TEST(OrientedTilingOptionsAbiTest, LayoutPinned) { SUCCEED(); }
}  // namespace
```

**IMPORTANT:** the absolute anchors `144 / 184 / 192` are computed estimates (assume `sizeof(MpBaseOptions)`=72). If any of those three fails to compile, the compiler reports the TRUE value — replace it AND record the final three numbers; you MUST use the same numbers in the Python layout test (Task 4). The tiling-struct internals (40 + offsets 0/4/8/16/24/28/32) are robust. Do not force a number to pass.

- [ ] **Step 2: Add the cc_test to BUILD**

In `mediapipe/tasks/c/vision/oriented_object_detector/BUILD`, add after `tiling_options_converter_test`:

```python
cc_test(
    name = "tiling_options_abi_test",
    srcs = ["tiling_options_abi_test.cc"],
    deps = [
        ":oriented_object_detector_lib",
        "//mediapipe/framework/port:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
```

- [ ] **Step 3: Build + run the pin**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:tiling_options_abi_test`
Expected: PASS (compiles → static_asserts hold). If an absolute anchor failed, correct it per the IMPORTANT note and re-run until PASS; record final tiling/result_callback/sizeof numbers.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_abi_test.cc \
        mediapipe/tasks/c/vision/oriented_object_detector/BUILD
git commit -m "test(tiling): C-API ABI-pin for the OBB tiling structs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: OBB Python ctypes structs + layout test

**Files:** Modify `oriented_object_detector.py`, `oriented_object_detector_test.py`. Do NOT run bazel for Python.

- [ ] **Step 1: Add the ctypes structs**

In `mediapipe/tasks/python/vision/oriented_object_detector.py`, immediately **after** the `_C_TYPES_RESULT_CALLBACK = ctypes.CFUNCTYPE(...)` definition and **before** `class MpOrientedObjectDetectorOptionsC(ctypes.Structure):`, add:

```python
class MpOrientedTileRectC(ctypes.Structure):
  """Byte-matches struct MpOrientedTileRect in the OBB C header."""

  _fields_ = [
      ('x_center', ctypes.c_float),
      ('y_center', ctypes.c_float),
      ('width', ctypes.c_float),
      ('height', ctypes.c_float),
  ]


class MpOrientedTilingOptionsC(ctypes.Structure):
  """Byte-matches struct MpOrientedTilingOptions in the OBB C header (6 fields).

  Field order/types MUST stay in sync with the C header (pinned by
  tiling_options_abi_test.cc). OBB has no motion-scheduling fields.
  """

  _fields_ = [
      ('tile_rows', ctypes.c_int),
      ('tile_cols', ctypes.c_int),
      ('tile_overlap_fraction', ctypes.c_float),
      ('explicit_tiles', ctypes.POINTER(MpOrientedTileRectC)),
      ('explicit_tiles_count', ctypes.c_uint32),
      ('tile_local_nms_iou_threshold', ctypes.c_float),
      ('max_detections_after_tile_nms', ctypes.c_int),
  ]
```

- [ ] **Step 2: Add the `tiling` field to `MpOrientedObjectDetectorOptionsC`**

In `MpOrientedObjectDetectorOptionsC._fields_`, insert between `('num_classes', ctypes.c_int),` and `('result_callback', _C_TYPES_RESULT_CALLBACK),`:

```python
      ('num_classes', ctypes.c_int),
      ('tiling', MpOrientedTilingOptionsC),
      ('result_callback', _C_TYPES_RESULT_CALLBACK),
```

- [ ] **Step 3: Syntax-check (runs here)**

Run: `python3 -m py_compile mediapipe/tasks/python/vision/oriented_object_detector.py`
Expected: exit 0.

- [ ] **Step 4: Standalone ctypes layout check (runs here, no mediapipe import)**

```bash
python3 - <<'PY'
import ctypes


class R(ctypes.Structure):
    _fields_ = [('x_center', ctypes.c_float), ('y_center', ctypes.c_float),
                ('width', ctypes.c_float), ('height', ctypes.c_float)]


class T(ctypes.Structure):
    _fields_ = [('tile_rows', ctypes.c_int), ('tile_cols', ctypes.c_int),
                ('tile_overlap_fraction', ctypes.c_float),
                ('explicit_tiles', ctypes.POINTER(R)),
                ('explicit_tiles_count', ctypes.c_uint32),
                ('tile_local_nms_iou_threshold', ctypes.c_float),
                ('max_detections_after_tile_nms', ctypes.c_int)]


assert ctypes.sizeof(R) == 16, ctypes.sizeof(R)
assert ctypes.sizeof(T) == 40, ctypes.sizeof(T)
assert T.tile_rows.offset == 0 and T.tile_cols.offset == 4
assert T.tile_overlap_fraction.offset == 8
assert T.explicit_tiles.offset == 16 and T.explicit_tiles_count.offset == 24
assert T.tile_local_nms_iou_threshold.offset == 28
assert T.max_detections_after_tile_nms.offset == 32
print('OK: OBB ctypes tiling layout matches the C++-pinned numbers (16/40)')
PY
```
Expected: prints the OK line. (Throwaway scratch check — not committed. If an assert fails, the ctypes types in Step 1 are wrong.)

- [ ] **Step 5: Add the layout test**

In `mediapipe/tasks/python/test/vision/oriented_object_detector_test.py`, add to the test class (before the inference test):

```python
  def test_ctypes_tiling_layout_matches_c_abi(self):
    """ctypes tiling structs byte-match the OBB C header (see tiling_options_abi_test.cc)."""
    import ctypes  # pylint: disable=g-import-not-at-top

    rect_c = oriented_object_detector.MpOrientedTileRectC
    self.assertEqual(ctypes.sizeof(rect_c), 16)
    self.assertEqual(rect_c.x_center.offset, 0)
    self.assertEqual(rect_c.y_center.offset, 4)
    self.assertEqual(rect_c.width.offset, 8)
    self.assertEqual(rect_c.height.offset, 12)

    tiling_c = oriented_object_detector.MpOrientedTilingOptionsC
    self.assertEqual(ctypes.sizeof(tiling_c), 40)
    self.assertEqual(tiling_c.tile_rows.offset, 0)
    self.assertEqual(tiling_c.tile_cols.offset, 4)
    self.assertEqual(tiling_c.tile_overlap_fraction.offset, 8)
    self.assertEqual(tiling_c.explicit_tiles.offset, 16)
    self.assertEqual(tiling_c.explicit_tiles_count.offset, 24)
    self.assertEqual(tiling_c.tile_local_nms_iou_threshold.offset, 28)
    self.assertEqual(tiling_c.max_detections_after_tile_nms.offset, 32)

    options_c = oriented_object_detector.MpOrientedObjectDetectorOptionsC
    # Absolute anchors mirroring tiling_options_abi_test.cc (use the SAME numbers
    # the C++ pin compiled with; replace if Task 3 reported different values).
    self.assertEqual(options_c.tiling.offset, 144)
    self.assertEqual(options_c.result_callback.offset, 184)
    self.assertEqual(ctypes.sizeof(options_c), 192)
```

- [ ] **Step 6: Syntax-check the test (runs here)**

Run: `python3 -m py_compile mediapipe/tasks/python/test/vision/oriented_object_detector_test.py`
Expected: exit 0.

- [ ] **Step 7: Commit**

```bash
git add mediapipe/tasks/python/vision/oriented_object_detector.py \
        mediapipe/tasks/python/test/vision/oriented_object_detector_test.py
git commit -m "feat(tiling): OBB Python ctypes tiling structs + layout test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: OBB Python dataclasses + marshalling + tests

**Files:** Modify `oriented_object_detector.py`, `oriented_object_detector_test.py`. Do NOT run bazel for Python. Use the YOLO Phase-2 module (`mediapipe/tasks/python/vision/yolo_object_detector.py`) as the structural template for the dataclasses / helper / `create_from_options` wiring.

- [ ] **Step 1: Add the public dataclasses**

In `mediapipe/tasks/python/vision/oriented_object_detector.py`, immediately **before** the `@dataclasses.dataclass` that precedes `class OrientedObjectDetectorOptions:`, add:

```python
@dataclasses.dataclass
class TileRect:
  """A frame-normalized tile given by its center point and size."""

  x_center: float = 0.0
  y_center: float = 0.0
  width: float = 0.0
  height: float = 0.0


@dataclasses.dataclass
class TilingOptions:
  """Static tiling configuration for the OBB object detector (6 fields).

  Mirrors OrientedObjectDetectorOptions.TilingOptions. Tiling is enabled when
  tile_rows * tile_cols > 1 or explicit_tiles is non-empty. The defaults (1x1, no
  explicit tiles) mean tiling disabled. To tile with a grid set BOTH tile_rows and
  tile_cols (each >= 1); a zero in either disables tiling.

  Attributes:
    tile_rows: Number of grid rows. Mutually exclusive with explicit_tiles.
    tile_cols: Number of grid columns. Mutually exclusive with explicit_tiles.
    tile_overlap_fraction: Fractional overlap added around each grid tile.
    explicit_tiles: Explicit (non-grid) tiles. Mutually exclusive with grid params.
    tile_local_nms_iou_threshold: Per-tile (in-decoder) rotated NMS IoU threshold;
      <= 0 disables.
    max_detections_after_tile_nms: Per-tile cap after tile-local NMS; <= 0 disables.
  """

  tile_rows: int = 1
  tile_cols: int = 1
  tile_overlap_fraction: float = 0.0
  explicit_tiles: Optional[List[TileRect]] = None
  tile_local_nms_iou_threshold: float = 0.0
  max_detections_after_tile_nms: int = 0
```

- [ ] **Step 2: Add the `tiling` field to `OrientedObjectDetectorOptions`**

In the `OrientedObjectDetectorOptions` dataclass, insert between `num_classes: int = 0` and the `result_callback: ...` field:

```python
  num_classes: int = 0
  tiling: TilingOptions = dataclasses.field(default_factory=TilingOptions)
  result_callback: Optional[
      Callable[[OrientedObjectDetectorResult, image_module.Image, int], None]
  ] = None
```

(The `result_callback` annotation above is the existing OBB one — leave it unchanged; only insert the `tiling` line before it.) Also add to that dataclass's docstring Attributes section, after the `num_classes:` line:

```python
    tiling: Static tiling configuration. Defaults to disabled (1x1).
```

- [ ] **Step 3: Add the marshalling helper**

In the same file, immediately **before** `class OrientedObjectDetector:`, add:

```python
def _build_oriented_tiling_options_c(
    tiling: TilingOptions,
) -> tuple['MpOrientedTilingOptionsC', object]:
  """Builds the ctypes MpOrientedTilingOptionsC from a TilingOptions dataclass.

  Returns the populated struct AND the backing explicit_tiles array. The caller
  MUST keep the returned array referenced until the C call that consumes the parent
  options struct returns: explicit_tiles is a raw pointer into it. (ctypes also
  records the array in the parent struct's _objects via the by-value copy; the C
  converter copies the tiles into a std::vector synchronously during Create, so
  outliving the Create call is sufficient.)
  """
  explicit_tiles = tiling.explicit_tiles or []
  tiles_array = (MpOrientedTileRectC * len(explicit_tiles))(
      *[
          MpOrientedTileRectC(t.x_center, t.y_center, t.width, t.height)
          for t in explicit_tiles
      ]
  )
  tiling_c = MpOrientedTilingOptionsC(
      tile_rows=tiling.tile_rows,
      tile_cols=tiling.tile_cols,
      tile_overlap_fraction=tiling.tile_overlap_fraction,
      explicit_tiles=(
          ctypes.cast(tiles_array, ctypes.POINTER(MpOrientedTileRectC))
          if explicit_tiles
          else None
      ),
      explicit_tiles_count=len(explicit_tiles),
      tile_local_nms_iou_threshold=tiling.tile_local_nms_iou_threshold,
      max_detections_after_tile_nms=tiling.max_detections_after_tile_nms,
  )
  return tiling_c, tiles_array
```

- [ ] **Step 4: Wire the helper into `create_from_options`**

In `create_from_options`, immediately **before** the `ctypes_options = MpOrientedObjectDetectorOptionsC(` constructor call (after the `denylist_c = ...` line), add:

```python
    # tiles_keepalive holds the explicit_tiles backing array; it must stay
    # referenced through the MpOrientedObjectDetectorCreate call below (the C
    # converter copies the tiles into a std::vector synchronously during Create).
    tiling_c, tiles_keepalive = _build_oriented_tiling_options_c(options.tiling)
```

Then add `tiling=tiling_c,` to the `MpOrientedObjectDetectorOptionsC(...)` constructor, between `num_classes=options.num_classes,` and `result_callback=c_callback,`. Do NOT delete `tiles_keepalive` (it is the keep-alive; leave it live to the end of the function).

- [ ] **Step 5: Syntax-check the module (runs here)**

Run: `python3 -m py_compile mediapipe/tasks/python/vision/oriented_object_detector.py`
Expected: exit 0.

- [ ] **Step 6: Add the dataclass + marshalling tests**

In `mediapipe/tasks/python/test/vision/oriented_object_detector_test.py`, add to the test class (after the existing options-construct test; use the module-local aliases already defined in the file — `_OrientedObjectDetectorOptions`, `_BaseOptions`, `_RUNNING_MODE` — matching how the existing tests reference them):

```python
  def test_tiling_defaults_disabled(self):
    """A default OrientedObjectDetectorOptions has disabled (1x1) tiling; no model."""
    options = _OrientedObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path='/dummy/model.tflite')
    )
    self.assertEqual(options.tiling.tile_rows, 1)
    self.assertEqual(options.tiling.tile_cols, 1)
    self.assertIsNone(options.tiling.explicit_tiles)
    self.assertAlmostEqual(options.tiling.tile_overlap_fraction, 0.0, places=5)
    self.assertAlmostEqual(
        options.tiling.tile_local_nms_iou_threshold, 0.0, places=5
    )
    self.assertEqual(options.tiling.max_detections_after_tile_nms, 0)

  def test_options_with_tiling_construct_without_model(self):
    """Constructs TilingOptions (grid + explicit_tiles + caps); no model needed."""
    tiling = oriented_object_detector.TilingOptions(
        tile_rows=2,
        tile_cols=2,
        tile_overlap_fraction=0.2,
        explicit_tiles=[
            oriented_object_detector.TileRect(0.1, 0.2, 0.5, 0.4),
            oriented_object_detector.TileRect(0.75, 0.6, 0.45, 0.3),
        ],
        tile_local_nms_iou_threshold=0.5,
        max_detections_after_tile_nms=50,
    )
    options = _OrientedObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path='/dummy/model.tflite'),
        running_mode=_RUNNING_MODE.IMAGE,
        tiling=tiling,
    )
    self.assertEqual(options.tiling.tile_rows, 2)
    self.assertEqual(options.tiling.tile_cols, 2)
    self.assertAlmostEqual(options.tiling.tile_overlap_fraction, 0.2, places=5)
    self.assertLen(options.tiling.explicit_tiles, 2)
    self.assertAlmostEqual(
        options.tiling.explicit_tiles[1].x_center, 0.75, places=5
    )
    self.assertAlmostEqual(
        options.tiling.explicit_tiles[1].height, 0.3, places=5
    )
    self.assertAlmostEqual(
        options.tiling.tile_local_nms_iou_threshold, 0.5, places=5
    )
    self.assertEqual(options.tiling.max_detections_after_tile_nms, 50)

  def test_build_oriented_tiling_options_c_marshalling(self):
    """The dataclass->ctypes marshalling fills the struct; no model needed."""
    tiling = oriented_object_detector.TilingOptions(
        tile_rows=2,
        tile_cols=3,
        tile_overlap_fraction=0.2,
        explicit_tiles=[
            oriented_object_detector.TileRect(0.1, 0.2, 0.5, 0.4),
            oriented_object_detector.TileRect(0.75, 0.6, 0.45, 0.3),
        ],
        tile_local_nms_iou_threshold=0.5,
        max_detections_after_tile_nms=50,
    )
    # keepalive must stay bound while we read explicit_tiles[...] below.
    tiling_c, keepalive = oriented_object_detector._build_oriented_tiling_options_c(  # pylint: disable=protected-access
        tiling
    )
    self.assertEqual(tiling_c.tile_rows, 2)
    self.assertEqual(tiling_c.tile_cols, 3)
    self.assertAlmostEqual(tiling_c.tile_overlap_fraction, 0.2, places=5)
    self.assertEqual(tiling_c.explicit_tiles_count, 2)
    self.assertTrue(bool(tiling_c.explicit_tiles))  # non-null
    self.assertAlmostEqual(tiling_c.explicit_tiles[0].x_center, 0.1, places=5)
    self.assertAlmostEqual(tiling_c.explicit_tiles[1].height, 0.3, places=5)
    self.assertAlmostEqual(tiling_c.tile_local_nms_iou_threshold, 0.5, places=5)
    self.assertEqual(tiling_c.max_detections_after_tile_nms, 50)
    self.assertIsNotNone(keepalive)

  def test_build_oriented_tiling_options_c_empty(self):
    """Empty explicit_tiles -> null pointer + zero count (C-safe)."""
    tiling_c, _ = oriented_object_detector._build_oriented_tiling_options_c(  # pylint: disable=protected-access
        oriented_object_detector.TilingOptions()
    )
    self.assertEqual(tiling_c.explicit_tiles_count, 0)
    self.assertFalse(bool(tiling_c.explicit_tiles))  # null pointer
```

(The aliases `_OrientedObjectDetectorOptions`, `_BaseOptions`, `_RUNNING_MODE` are already defined at the top of this test file.)

- [ ] **Step 7: Add the model-gated tiled detect test**

Add this method after the existing `test_detect_image`. It mirrors `test_detect_image` exactly (same `_IMAGE_FILE` = `cats_and_dogs.jpg`, `_Layout.CHANNELS_FIRST`, `num_classes=15`, `score_threshold=0.25`), only adding the `tiling=` argument:

```python
  @unittest.skipUnless(
      _MODEL_PRESENT,
      'yolov8n-obb.tflite fixture not present; skipping inference test',
  )
  def test_detect_image_tiled(self):
    """Smoke-checks the tiled OBB detect path end-to-end (model-gated).

    cats_and_dogs.jpg is detectable without tiling, so this verifies the tiling
    path runs and returns a valid result, not tiling efficacy.
    """
    model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _MODEL_FILE)
    )
    image_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _IMAGE_FILE)
    )
    image = _Image.create_from_file(image_path)

    options = _OrientedObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RUNNING_MODE.IMAGE,
        score_threshold=0.25,
        layout=_Layout.CHANNELS_FIRST,
        num_classes=15,
        tiling=oriented_object_detector.TilingOptions(
            tile_rows=2, tile_cols=2, tile_overlap_fraction=0.2
        ),
    )
    with _OrientedObjectDetector.create_from_options(options) as detector:
      result = detector.detect(image)

    self.assertGreater(
        len(result.detections),
        0,
        'Expected at least one detection on cats_and_dogs.jpg with tiling',
    )
```

- [ ] **Step 8: Syntax-check the test (runs here)**

Run: `python3 -m py_compile mediapipe/tasks/python/test/vision/oriented_object_detector_test.py`
Expected: exit 0.

- [ ] **Step 9: Commit**

```bash
git add mediapipe/tasks/python/vision/oriented_object_detector.py \
        mediapipe/tasks/python/test/vision/oriented_object_detector_test.py
git commit -m "feat(tiling): OBB Python TilingOptions dataclasses + marshalling

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final Verification (after all tasks)

Runs here:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/tasks/c/vision/oriented_object_detector:tiling_options_converter_test \
  //mediapipe/tasks/c/vision/oriented_object_detector:tiling_options_abi_test \
  //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_test
python3 -m py_compile mediapipe/tasks/python/vision/oriented_object_detector.py
python3 -m py_compile mediapipe/tasks/python/test/vision/oriented_object_detector_test.py
```
Expected: all 3 cc_tests PASS (the OBB detector test's `TiledImageMode` RUNS and finds a ship — not skipped); both `py_compile` exit 0.

Cannot run here: the Python tests in `oriented_object_detector_test.py` (toolchain broken) — written + statically reviewed; the layout test asserts the same numbers the C++ pin verifies.

## Acceptance criteria (from spec)

1. OBB `tiling_options_converter_test` (6/6), `tiling_options_abi_test`, and `oriented_object_detector_test` (incl. the running `TiledImageMode`) all pass here.
2. The C struct, the Python ctypes struct, and OBB's C++ `TilingOptions` are byte-compatible; the Python layout test asserts the same numbers the C++ pin verifies.
3. The Python dataclasses + `_build_oriented_tiling_options_c` cover all 6 fields + `explicit_tiles`, with the keep-alive held through Create.
4. No YOLO / cc / proto change; no regression in existing OBB C-API tests.
