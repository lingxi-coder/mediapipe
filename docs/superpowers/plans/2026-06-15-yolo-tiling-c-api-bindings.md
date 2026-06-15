# YOLO Tiling C-API Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the C++ `YoloObjectDetectorOptions::TilingOptions` (full field parity) through the YOLO C API so C callers can configure static tiling.

**Architecture:** Add plain-C `MpTileRect` + `MpTilingOptions` structs to the YOLO C header (with a `tiling` field on `MpYoloObjectDetectorOptions`), translate them into the existing C++ `TilingOptions` via a new, unit-testable `CppConvertToTilingOptions` converter (mirroring the `classifier_options_converter` precedent), and call it from the existing `CppConvertToDetectorOptions`. The cc/proto layers already map `TilingOptions` and are unchanged.

**Tech Stack:** C/C++, Bazel (`--define MEDIAPIPE_DISABLE_GPU=1`), GoogleTest. Spec: `docs/superpowers/specs/2026-06-15-yolo-tiling-c-api-bindings-design.md`.

**Build constraint:** Only desktop C++ builds/runs here. The YOLO C-API integration test SKIPs without the gitignored `yolov8n.tflite` (no `yolo_test_models` target in `testdata/vision/BUILD`), so the **converter unit test (Task 1) is the verification gate**; the model-gated e2e (Task 2) builds and SKIPs cleanly.

**Standing constraints:** branch `dev` (no new branch); commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`; code comments in English.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h` | Modify | Add `MpTileRect`, `MpTilingOptions`; add `tiling` field to `MpYoloObjectDetectorOptions`. |
| `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.h` | Create | Declare `CppConvertToTilingOptions`. |
| `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.cc` | Create | Define it: 8 scalar copies + `explicit_tiles` ptr+count→vector loop. |
| `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter_test.cc` | Create | **Verification gate** — model-free converter unit test. |
| `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.cc` | Modify | Include converter header; one call line in `CppConvertToDetectorOptions`. |
| `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector_test.cc` | Modify | Add model-gated `TiledImageMode` e2e. |
| `mediapipe/tasks/c/vision/yolo_object_detector/BUILD` | Modify | Add converter files to both libs; add `tiling_options_converter_test`. |

---

## Task 1: C tiling structs + converter + unit test (the gate)

**Files:**
- Create: `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter_test.cc`
- Modify: `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h`
- Create: `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.h`
- Create: `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.cc`
- Modify: `mediapipe/tasks/c/vision/yolo_object_detector/BUILD`

- [ ] **Step 1: Write the failing converter unit test**

Create `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter_test.cc`:

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
      {0.25f, 0.25f, 0.5f, 0.5f},
      {0.75f, 0.75f, 0.4f, 0.3f},
  };
  MpTilingOptions in = {};
  in.explicit_tiles = tiles;
  in.explicit_tiles_count = 2;

  CppTilingOptions out;
  CppConvertToTilingOptions(in, &out);

  ASSERT_EQ(out.explicit_tiles.size(), 2u);
  EXPECT_FLOAT_EQ(out.explicit_tiles[0].x_center, 0.25f);
  EXPECT_FLOAT_EQ(out.explicit_tiles[0].y_center, 0.25f);
  EXPECT_FLOAT_EQ(out.explicit_tiles[0].width, 0.5f);
  EXPECT_FLOAT_EQ(out.explicit_tiles[0].height, 0.5f);
  EXPECT_FLOAT_EQ(out.explicit_tiles[1].x_center, 0.75f);
  EXPECT_FLOAT_EQ(out.explicit_tiles[1].y_center, 0.75f);
  EXPECT_FLOAT_EQ(out.explicit_tiles[1].width, 0.4f);
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

}  // namespace
}  // namespace mediapipe::tasks::c::vision::yolo_object_detector
```

- [ ] **Step 2: Add the test target to BUILD and confirm it fails to compile**

In `mediapipe/tasks/c/vision/yolo_object_detector/BUILD`, add this `cc_test` after the existing `yolo_object_detector_test` target (before the `yolo_object_detector_c_lib` target):

```python
cc_test(
    name = "tiling_options_converter_test",
    srcs = ["tiling_options_converter_test.cc"],
    deps = [
        ":yolo_object_detector_lib",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/vision/yolo_object_detector",
        "@com_google_googletest//:gtest_main",
    ],
)
```

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:tiling_options_converter_test`
Expected: FAIL — compile error, `tiling_options_converter.h` not found / `MpTilingOptions` / `CppConvertToTilingOptions` undefined. (This is the red state for the new symbols.)

- [ ] **Step 3: Add the C structs to the public C header**

In `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h`, insert the two structs immediately **before** `struct MpYoloObjectDetectorOptions {` (i.e. right after the comment block that ends at line 47, before line 48):

```c
// A frame-normalized tile given by its CENTER point and size (NOT corner-based
// like RectF). Mirrors YoloObjectDetectorOptions::TilingOptions::TileRect.
struct MpTileRect {
  float x_center;
  float y_center;
  float width;
  float height;
};

// Static tiling configuration. Mirrors
// YoloObjectDetectorOptions::TilingOptions field-for-field.
//
// Tiling is ENABLED when tile_rows * tile_cols > 1 or explicit_tiles_count > 0.
// A zero-initialized MpTilingOptions (e.g. from `MpYoloObjectDetectorOptions
// options = {}`) therefore means tiling DISABLED -- byte-identical to a C caller
// that never set tiling at all. To tile with a grid, set BOTH tile_rows and
// tile_cols (each >= 1); a zero in either disables tiling.
struct MpTilingOptions {
  // Grid tiling: rows x cols of equal tiles. Mutually exclusive with
  // explicit_tiles.
  int tile_rows;
  int tile_cols;
  // Fractional overlap added around each grid tile. Default 0.0.
  float tile_overlap_fraction;

  // Explicit (non-grid) tiles. Caller owns this array; it is COPIED during
  // Create and need not outlive the MpYoloObjectDetectorCreate call. Mirrors
  // the category_allowlist (pointer + count) ownership convention.
  const struct MpTileRect* explicit_tiles;
  uint32_t explicit_tiles_count;

  // Per-tile (in-decoder) NMS IoU threshold; <= 0 disables.
  float tile_local_nms_iou_threshold;
  // Per-tile cap after tile-local NMS; <= 0 disables.
  int max_detections_after_tile_nms;

  // VIDEO/LIVE_STREAM only: gate per-frame tiled inference with a motion
  // scheduler. Setting this in IMAGE mode is rejected by the C++ Create()
  // (surfaced as a non-kMpOk MpStatus); the binding only passes it through.
  bool enable_motion_scheduling;
  // Per DETECT-frame cap on inferred tiles (motion-prioritized). 0 = all.
  int max_scheduled_tiles;
};
```

Then, inside `struct MpYoloObjectDetectorOptions { ... }`, add the `tiling` field immediately **after** the `int num_classes;` field (line 79) and before the `result_callback_fn` typedef block:

```c
  // Static tiling configuration. Zero-initialized => tiling disabled.
  struct MpTilingOptions tiling;
```

(`uint32_t` and `bool` are already usable: the header includes `<cstdint>` and is consumed as C++.)

- [ ] **Step 4: Create the converter header**

Create `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.h`:

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

#ifndef MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_TILING_OPTIONS_CONVERTER_H_
#define MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_TILING_OPTIONS_CONVERTER_H_

#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

namespace mediapipe::tasks::c::vision::yolo_object_detector {

// Copies a plain-C MpTilingOptions into the C++ TilingOptions sub-struct.
// Scalars are copied 1:1; explicit_tiles (a caller-owned MpTileRect array) is
// copied element-by-element into a fresh std::vector (the caller retains
// ownership of the input array). Declared here so unit tests can verify the
// C->C++ mapping without constructing a live graph.
void CppConvertToTilingOptions(
    const MpTilingOptions& in,
    ::mediapipe::tasks::vision::yolo_object_detector::
        YoloObjectDetectorOptions::TilingOptions* out);

}  // namespace mediapipe::tasks::c::vision::yolo_object_detector

#endif  // MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_TILING_OPTIONS_CONVERTER_H_
```

- [ ] **Step 5: Create the converter implementation**

Create `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.cc`:

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

#include "mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.h"

#include <cstdint>

#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

namespace mediapipe::tasks::c::vision::yolo_object_detector {

namespace YoloNs = ::mediapipe::tasks::vision::yolo_object_detector;

void CppConvertToTilingOptions(
    const MpTilingOptions& in,
    YoloNs::YoloObjectDetectorOptions::TilingOptions* out) {
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
  out->enable_motion_scheduling = in.enable_motion_scheduling;
  out->max_scheduled_tiles = in.max_scheduled_tiles;
}

}  // namespace mediapipe::tasks::c::vision::yolo_object_detector
```

- [ ] **Step 6: Add the converter files to both libs in BUILD**

In `mediapipe/tasks/c/vision/yolo_object_detector/BUILD`, update `yolo_object_detector_lib` and `yolo_object_detector_c_lib` to include the new converter source + header. Replace the `yolo_object_detector_lib` target:

```python
cc_library(
    name = "yolo_object_detector_lib",
    srcs = [
        "tiling_options_converter.cc",
        "yolo_object_detector.cc",
    ],
    hdrs = [
        "tiling_options_converter.h",
        "yolo_object_detector.h",
    ],
    visibility = ["//visibility:public"],
    deps = YOLO_OBJECT_DETECTOR_DEPS,
)
```

…and replace the `yolo_object_detector_c_lib` target:

```python
cc_library(
    name = "yolo_object_detector_c_lib",
    srcs = [
        "tiling_options_converter.cc",
        "yolo_object_detector.cc",
    ],
    hdrs = [
        "tiling_options_converter.h",
        "yolo_object_detector.h",
    ],
    visibility = ["//mediapipe/tasks/c:__subpackages__"],
    deps = YOLO_OBJECT_DETECTOR_DEPS,
    alwayslink = 1,
)
```

(`YOLO_OBJECT_DETECTOR_DEPS` already includes `//mediapipe/tasks/cc/vision/yolo_object_detector`, which provides the C++ `TilingOptions` type the converter needs — no new dep required.)

- [ ] **Step 7: Run the converter test to verify it passes**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:tiling_options_converter_test`
Expected: PASS (5/5 cases).

- [ ] **Step 8: Verify both libs still build**

Run: `bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_lib //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_c_lib`
Expected: build succeeds (exit 0).

- [ ] **Step 9: Commit**

```bash
git add mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h \
        mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.h \
        mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.cc \
        mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter_test.cc \
        mediapipe/tasks/c/vision/yolo_object_detector/BUILD
git commit -m "feat(tiling): YOLO C-API MpTilingOptions struct + converter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Wire converter into options conversion + model-gated e2e

**Files:**
- Modify: `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.cc:74-93` (`CppConvertToDetectorOptions`)
- Modify: `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector_test.cc`

- [ ] **Step 1: Wire the tiling converter into `CppConvertToDetectorOptions`**

In `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.cc`, add the converter include alongside the existing includes (keep includes alphabetically grouped with the other `tasks/c/vision/yolo_object_detector` include just after line 36's block; place it immediately before the `mediapipe/tasks/cc/...` includes):

```cpp
#include "mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.h"
```

Then in `CppConvertToDetectorOptions`, add the tiling conversion as the **last** line of the function body, immediately after `out->num_classes = in.num_classes;` (line 92):

```cpp
  CppConvertToTilingOptions(in.tiling, &out->tiling);
```

`CppConvertToTilingOptions` is already in the same namespace (`mediapipe::tasks::c::vision::yolo_object_detector`), so no qualification is needed.

- [ ] **Step 2: Verify the C-API lib builds with the wiring**

Run: `bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_lib`
Expected: build succeeds (exit 0).

- [ ] **Step 3: Add the model-gated e2e tiling test**

In `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector_test.cc`, add this `TEST` immediately after the existing `ImageMode` test (after its closing brace at line 125, before the closing `}  // namespace`):

```cpp
// Configures a 2x2 tiling grid through the C API and runs detection. Like
// ImageMode, this SKIPs cleanly when the yolov8n.tflite fixture is absent
// (no yolo_test_models data dep is wired). It exercises the full C->C++->proto
// tiling path end to end when the model is present. In tiled mode the graph
// has no NORM_RECT input, so image_processing_options must be null.
TEST(YoloObjectDetectorCApiTest, TiledImageMode) {
  const std::string model_path = GetFullPath(kYoloModel);

  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "YOLO model fixture not available at " << model_path
                 << "; tiled integration assertions gated until yolov8n.tflite "
                    "is added to mediapipe/tasks/testdata/vision/.";
  }

  MpYoloObjectDetectorOptions options = {};
  options.base_options.model_asset_path = model_path.c_str();
  options.running_mode = MpRunningMode::MP_RUNNING_MODE_IMAGE;
  options.max_results = 10;
  options.score_threshold = 0.25f;
  options.iou_threshold = 0.45f;
  options.num_classes = 80;
  options.layout = 2;  // CHANNELS_LAST
  options.tiling.tile_rows = 2;
  options.tiling.tile_cols = 2;
  options.tiling.tile_overlap_fraction = 0.2f;

  MpYoloObjectDetectorPtr detector = nullptr;
  ASSERT_EQ(
      MpYoloObjectDetectorCreate(&options, &detector, /*error_msg=*/nullptr),
      kMpOk);
  EXPECT_NE(detector, nullptr);
  ScopedMpYoloObjectDetector scoped_detector;
  scoped_detector.ptr = detector;

  MpImagePtr raw_image = nullptr;
  ASSERT_EQ(
      MpImageCreateFromFile(GetFullPath(kImageFile).c_str(), &raw_image,
                            /*error_msg=*/nullptr),
      kMpOk);
  ScopedMpImage image(raw_image);

  MpYoloObjectDetectorResult result;
  ASSERT_EQ(MpYoloObjectDetectorDetectImage(detector, image.get(),
                                            /*options=*/nullptr, &result,
                                            /*error_msg=*/nullptr),
            kMpOk);

  EXPECT_GT(result.detections_count, 0u);
  MpYoloObjectDetectorCloseResult(&result);
}
```

- [ ] **Step 4: Run the integration test (builds + SKIPs cleanly without the fixture)**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_test`
Expected: PASS — both `ImageMode` and `TiledImageMode` run and report SKIPPED (no `yolov8n.tflite` fixture). The point of this step is that the new test **compiles and links** against the tiling field and SKIPs without failing.

- [ ] **Step 5: Re-run the converter test to confirm no regression**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:tiling_options_converter_test`
Expected: PASS (5/5).

- [ ] **Step 6: Commit**

```bash
git add mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.cc \
        mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector_test.cc
git commit -m "feat(tiling): wire YOLO C-API tiling converter + model-gated e2e

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final Verification (after all tasks)

Run the whole package's tests to confirm no regression:

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/tasks/c/vision/yolo_object_detector:tiling_options_converter_test \
  //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_test
```
Expected: `tiling_options_converter_test` PASSES (5/5); `yolo_object_detector_test` PASSES (tests SKIP without the model fixture).

Sanity-check the untouched OBB C-API path still builds (no shared code was changed, but confirms the tree is healthy):

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_lib
```
Expected: build succeeds (exit 0).

## Acceptance criteria (from spec)

1. Both YOLO C-API libs build under `--define MEDIAPIPE_DISABLE_GPU=1`.
2. `tiling_options_converter_test` passes (full round-trip incl. multi-element `explicit_tiles`, zero-init disabled, null-safe, reuse-clears).
3. `yolo_object_detector_test` passes (`TiledImageMode` SKIPs cleanly; `ImageMode` unchanged).
4. No cc/proto changes; no regression in existing C-API tests.
