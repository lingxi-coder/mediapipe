# YOLO Tiling Python Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the C++/C `TilingOptions` (full field parity) through the YOLO **Python** Tasks wrapper, so a Python caller can configure tiling via `YoloObjectDetectorOptions.tiling`.

**Architecture:** Add `MpTileRectC`/`MpTilingOptionsC` ctypes structs (byte-matching the Phase-1 C header) + a `tiling` field on the existing `MpYoloObjectDetectorOptionsC` ctypes struct; add public `TileRect`/`TilingOptions` dataclasses + a `tiling` field on `YoloObjectDetectorOptions`; marshal them in `create_from_options`. A compile-time C++ ABI-pin test anchors the ground-truth struct layout. The C/cc/proto layers are unchanged (Phase 1 already maps tiling).

**Tech Stack:** Python 3 + ctypes, C++/Bazel (`--define MEDIAPIPE_DISABLE_GPU=1`), GoogleTest, absltest. Spec: `docs/superpowers/specs/2026-06-15-yolo-tiling-python-bindings-design.md`.

**CRITICAL build constraint:** The Python Tasks vision API does **not** build or run on this machine (needs compiled `libmediapipe` + a Python toolchain that is broken here). So **do NOT run bazel for any Python target** — it will fail for environmental reasons unrelated to correctness. The runnable-here verifications are: (a) the **C++ ABI-pin test** (Task 1, `bazel test`), and (b) **`python3 -m py_compile`** (syntax) plus a **standalone `python3` ctypes layout check** (Task 2) that needs no mediapipe import. The full Python tests are written + statically reviewed and run once the toolchain is available.

**Standing constraints:** branch `dev` (no new branch); commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`; code comments in English.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_abi_test.cc` | Create | Compile-time `static_assert` ABI-pin for the C structs (runs here). |
| `mediapipe/tasks/c/vision/yolo_object_detector/BUILD` | Modify | Add the `tiling_options_abi_test` cc_test. |
| `mediapipe/tasks/python/vision/yolo_object_detector.py` | Modify | `MpTileRectC`/`MpTilingOptionsC` + the `tiling` ctypes field; `TileRect`/`TilingOptions` dataclasses + the `tiling` dataclass field; `create_from_options` marshalling + keep-alive. |
| `mediapipe/tasks/python/test/vision/yolo_object_detector_test.py` | Modify | Layout test (Task 2) + construct/defaults/detect tests (Task 3). |

No `tasks/python/vision/BUILD` or `tasks/python/test/vision/BUILD` change (inline structs/dataclasses; tests in the existing file).

---

## Task 1: C++ ABI-pin test (the layout anchor — runs here)

This pins the **existing** Phase-1 C struct layout; it changes no production code. It verifies the exact `sizeof`/`offsetof` numbers that the Python ctypes structs (Task 2) must match. Do this first so the ground-truth numbers are compiler-verified before Python mirrors them.

**Files:**
- Create: `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_abi_test.cc`
- Modify: `mediapipe/tasks/c/vision/yolo_object_detector/BUILD`

- [ ] **Step 1: Write the ABI-pin test**

Create `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_abi_test.cc`:

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

// `tiling` occupies a contiguous block between num_classes and result_callback.
static_assert(offsetof(MpYoloObjectDetectorOptions, tiling) >
                  offsetof(MpYoloObjectDetectorOptions, num_classes),
              "tiling must follow num_classes");
static_assert(offsetof(MpYoloObjectDetectorOptions, result_callback) ==
                  offsetof(MpYoloObjectDetectorOptions, tiling) +
                      sizeof(MpTilingOptions),
              "result_callback must immediately follow tiling");

namespace {

// The real guarantees are the compile-time static_asserts above; this gives the
// cc_test target a runtime case to execute.
TEST(TilingOptionsAbiTest, LayoutPinned) { SUCCEED(); }

}  // namespace
```

- [ ] **Step 2: Add the cc_test to BUILD**

In `mediapipe/tasks/c/vision/yolo_object_detector/BUILD`, add this `cc_test` (place it after the existing `tiling_options_converter_test` target):

```python
cc_test(
    name = "tiling_options_abi_test",
    srcs = ["tiling_options_abi_test.cc"],
    deps = [
        ":yolo_object_detector_lib",
        "//mediapipe/framework/port:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
```

- [ ] **Step 3: Build + run the pin test**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:tiling_options_abi_test`
Expected: PASS. The target **compiling** proves every `static_assert` holds.

**If any `static_assert` fails to compile:** do NOT silently change the magic number to make it pass — a failure may mean the real C layout differs from what the Python binding will assume. Instead, temporarily add a runtime `std::cout`/`EXPECT_EQ` printing the actual `sizeof`/`offsetof`, report the real numbers as DONE_WITH_CONCERNS, and stop — the controller will reconcile the C++ pin and the Python layout test together so they stay equal.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_abi_test.cc \
        mediapipe/tasks/c/vision/yolo_object_detector/BUILD
git commit -m "test(tiling): C-API ABI-pin for the YOLO tiling structs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Python ctypes structs + the `tiling` field + layout test

Mirrors the C structs in ctypes. **Do NOT run bazel for Python here.** Verify with `py_compile` + a standalone ctypes check (both run on this machine).

**Files:**
- Modify: `mediapipe/tasks/python/vision/yolo_object_detector.py`
- Modify: `mediapipe/tasks/python/test/vision/yolo_object_detector_test.py`

- [ ] **Step 1: Add the ctypes structs**

In `mediapipe/tasks/python/vision/yolo_object_detector.py`, immediately **after** the `_C_TYPES_RESULT_CALLBACK = ctypes.CFUNCTYPE(...)` definition and **before** `class MpYoloObjectDetectorOptionsC(ctypes.Structure):`, add:

```python
class MpTileRectC(ctypes.Structure):
  """Byte-matches struct MpTileRect in the YOLO C header.

  Field order/types MUST stay in sync with
  mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h (pinned
  by tiling_options_abi_test.cc).
  """

  _fields_ = [
      ('x_center', ctypes.c_float),
      ('y_center', ctypes.c_float),
      ('width', ctypes.c_float),
      ('height', ctypes.c_float),
  ]


class MpTilingOptionsC(ctypes.Structure):
  """Byte-matches struct MpTilingOptions in the YOLO C header.

  Field order/types MUST stay in sync with the C header (pinned by
  tiling_options_abi_test.cc). enable_motion_scheduling is c_bool (1 byte) to
  match the C `bool`; using c_int here would shift max_scheduled_tiles and
  corrupt every options struct.
  """

  _fields_ = [
      ('tile_rows', ctypes.c_int),
      ('tile_cols', ctypes.c_int),
      ('tile_overlap_fraction', ctypes.c_float),
      ('explicit_tiles', ctypes.POINTER(MpTileRectC)),
      ('explicit_tiles_count', ctypes.c_uint32),
      ('tile_local_nms_iou_threshold', ctypes.c_float),
      ('max_detections_after_tile_nms', ctypes.c_int),
      ('enable_motion_scheduling', ctypes.c_bool),
      ('max_scheduled_tiles', ctypes.c_int),
  ]
```

- [ ] **Step 2: Add the `tiling` field to `MpYoloObjectDetectorOptionsC`**

In the same file, in `MpYoloObjectDetectorOptionsC._fields_`, insert the `tiling` entry **between** `('num_classes', ctypes.c_int),` and `('result_callback', _C_TYPES_RESULT_CALLBACK),`:

```python
      ('num_classes', ctypes.c_int),
      ('tiling', MpTilingOptionsC),
      ('result_callback', _C_TYPES_RESULT_CALLBACK),
```

- [ ] **Step 3: Syntax-check the module (runs here)**

Run: `python3 -m py_compile mediapipe/tasks/python/vision/yolo_object_detector.py`
Expected: exit 0, no output. (`py_compile` parses/compiles to bytecode without importing mediapipe, so it works despite the broken Python toolchain. A non-zero exit means a syntax error to fix.)

- [ ] **Step 4: Standalone ctypes layout check (runs here, no mediapipe import)**

This confirms the ctypes field types yield the exact offsets the C++ pin (Task 1) verified, on this machine. Run:

```bash
python3 - <<'PY'
import ctypes


class MpTileRectC(ctypes.Structure):
    _fields_ = [('x_center', ctypes.c_float), ('y_center', ctypes.c_float),
                ('width', ctypes.c_float), ('height', ctypes.c_float)]


class MpTilingOptionsC(ctypes.Structure):
    _fields_ = [('tile_rows', ctypes.c_int), ('tile_cols', ctypes.c_int),
                ('tile_overlap_fraction', ctypes.c_float),
                ('explicit_tiles', ctypes.POINTER(MpTileRectC)),
                ('explicit_tiles_count', ctypes.c_uint32),
                ('tile_local_nms_iou_threshold', ctypes.c_float),
                ('max_detections_after_tile_nms', ctypes.c_int),
                ('enable_motion_scheduling', ctypes.c_bool),
                ('max_scheduled_tiles', ctypes.c_int)]


assert ctypes.sizeof(MpTileRectC) == 16, ctypes.sizeof(MpTileRectC)
assert ctypes.sizeof(MpTilingOptionsC) == 48, ctypes.sizeof(MpTilingOptionsC)
assert MpTilingOptionsC.explicit_tiles.offset == 16
assert MpTilingOptionsC.explicit_tiles_count.offset == 24
assert MpTilingOptionsC.tile_local_nms_iou_threshold.offset == 28
assert MpTilingOptionsC.max_detections_after_tile_nms.offset == 32
assert MpTilingOptionsC.enable_motion_scheduling.offset == 36
assert MpTilingOptionsC.max_scheduled_tiles.offset == 40
print('OK: ctypes tiling layout matches the C++-pinned numbers')
PY
```
Expected: prints `OK: ctypes tiling layout matches the C++-pinned numbers`. (This is a throwaway scratch check — it is NOT committed; it re-declares the same `_fields_` to confirm ctypes computes the pinned offsets on this platform. If any assert fails, the ctypes types in Step 1 are wrong — fix them.)

- [ ] **Step 5: Add the ctypes layout test (written here; runs when the toolchain is available)**

In `mediapipe/tasks/python/test/vision/yolo_object_detector_test.py`, add this test method to the `YoloObjectDetectorTest` class (place it before `test_detect_image`):

```python
  def test_ctypes_tiling_layout_matches_c_abi(self):
    """ctypes tiling structs byte-match the C header (see tiling_options_abi_test.cc)."""
    import ctypes  # pylint: disable=g-import-not-at-top

    self.assertEqual(ctypes.sizeof(yolo_object_detector.MpTileRectC), 16)
    self.assertEqual(ctypes.sizeof(yolo_object_detector.MpTilingOptionsC), 48)

    tiling_c = yolo_object_detector.MpTilingOptionsC
    self.assertEqual(tiling_c.explicit_tiles.offset, 16)
    self.assertEqual(tiling_c.explicit_tiles_count.offset, 24)
    self.assertEqual(tiling_c.tile_local_nms_iou_threshold.offset, 28)
    self.assertEqual(tiling_c.max_detections_after_tile_nms.offset, 32)
    self.assertEqual(tiling_c.enable_motion_scheduling.offset, 36)
    self.assertEqual(tiling_c.max_scheduled_tiles.offset, 40)

    options_c = yolo_object_detector.MpYoloObjectDetectorOptionsC
    # `tiling` sits contiguously between num_classes and result_callback.
    self.assertGreater(options_c.tiling.offset, options_c.num_classes.offset)
    self.assertEqual(
        options_c.result_callback.offset,
        options_c.tiling.offset + ctypes.sizeof(yolo_object_detector.MpTilingOptionsC),
    )
```

- [ ] **Step 6: Syntax-check the test module (runs here)**

Run: `python3 -m py_compile mediapipe/tasks/python/test/vision/yolo_object_detector_test.py`
Expected: exit 0, no output.

- [ ] **Step 7: Commit**

```bash
git add mediapipe/tasks/python/vision/yolo_object_detector.py \
        mediapipe/tasks/python/test/vision/yolo_object_detector_test.py
git commit -m "feat(tiling): YOLO Python ctypes tiling structs + layout test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Python dataclasses + marshalling + construct/detect tests

Adds the public API on top of Task 2's ctypes structs. **Do NOT run bazel for Python.** Verify with `py_compile` + self-review.

**Files:**
- Modify: `mediapipe/tasks/python/vision/yolo_object_detector.py`
- Modify: `mediapipe/tasks/python/test/vision/yolo_object_detector_test.py`

- [ ] **Step 1: Add the public dataclasses**

In `mediapipe/tasks/python/vision/yolo_object_detector.py`, immediately **before** `@dataclasses.dataclass\nclass YoloObjectDetectorOptions:`, add:

```python
@dataclasses.dataclass
class TileRect:
  """A frame-normalized tile given by its center point and size.

  Attributes:
    x_center: Tile center x, normalized to [0, 1].
    y_center: Tile center y, normalized to [0, 1].
    width: Tile width, normalized to [0, 1].
    height: Tile height, normalized to [0, 1].
  """

  x_center: float = 0.0
  y_center: float = 0.0
  width: float = 0.0
  height: float = 0.0


@dataclasses.dataclass
class TilingOptions:
  """Static tiling configuration for the YOLO object detector.

  Mirrors the C++ YoloObjectDetectorOptions.TilingOptions. Tiling is enabled
  when tile_rows * tile_cols > 1 or explicit_tiles is non-empty. The defaults
  (1x1, no explicit tiles) mean tiling disabled. To tile with a grid set BOTH
  tile_rows and tile_cols (each >= 1); a zero in either disables tiling.

  Attributes:
    tile_rows: Number of grid rows. Mutually exclusive with explicit_tiles.
    tile_cols: Number of grid columns. Mutually exclusive with explicit_tiles.
    tile_overlap_fraction: Fractional overlap added around each grid tile.
    explicit_tiles: Explicit (non-grid) tiles. Mutually exclusive with the grid
      params.
    tile_local_nms_iou_threshold: Per-tile (in-decoder) NMS IoU threshold;
      <= 0 disables.
    max_detections_after_tile_nms: Per-tile cap after tile-local NMS;
      <= 0 disables.
    enable_motion_scheduling: VIDEO/LIVE_STREAM only; gate per-frame tiled
      inference with a motion scheduler. Rejected in IMAGE mode by the
      underlying task.
    max_scheduled_tiles: Per DETECT-frame cap on inferred tiles
      (motion-prioritized). 0 = all.
  """

  tile_rows: int = 1
  tile_cols: int = 1
  tile_overlap_fraction: float = 0.0
  explicit_tiles: Optional[List[TileRect]] = None
  tile_local_nms_iou_threshold: float = 0.0
  max_detections_after_tile_nms: int = 0
  enable_motion_scheduling: bool = False
  max_scheduled_tiles: int = 0
```

- [ ] **Step 2: Add the `tiling` field to `YoloObjectDetectorOptions`**

In the same file, in the `YoloObjectDetectorOptions` dataclass, insert the `tiling` field **between** `num_classes: int = 0` and `result_callback: Optional[...] = None`:

```python
  num_classes: int = 0
  tiling: TilingOptions = dataclasses.field(default_factory=TilingOptions)
  result_callback: Optional[
      Callable[
          [detections_module.DetectionResult, image_module.Image, int], None
      ]
  ] = None
```

Also add a one-line entry to the dataclass docstring's Attributes section, after the `num_classes:` line:

```python
    tiling: Static tiling configuration. Defaults to disabled (1x1).
```

- [ ] **Step 3: Marshal tiling in `create_from_options`**

In the same file, in `create_from_options`, immediately **after** the `denylist_c = mediapipe_c_bindings_c_module.convert_strings_to_ctypes_array(...)` block and **before** `ctypes_options = MpYoloObjectDetectorOptionsC(`, add:

```python
    explicit_tiles = options.tiling.explicit_tiles or []
    # NOTE: tiles_array must stay referenced (as a local) through the
    # MpYoloObjectDetectorCreate call below. explicit_tiles is a raw pointer
    # into this array; ctypes does not keep it alive once MpTilingOptionsC is
    # copied by value into the parent options struct. The C converter copies the
    # tiles into a std::vector during Create, so this transient caller-owned
    # array only needs to outlive the Create call (same contract as
    # category_allowlist).
    tiles_array = (MpTileRectC * len(explicit_tiles))(
        *[
            MpTileRectC(t.x_center, t.y_center, t.width, t.height)
            for t in explicit_tiles
        ]
    )
    tiling_c = MpTilingOptionsC(
        tile_rows=options.tiling.tile_rows,
        tile_cols=options.tiling.tile_cols,
        tile_overlap_fraction=options.tiling.tile_overlap_fraction,
        explicit_tiles=(
            ctypes.cast(tiles_array, ctypes.POINTER(MpTileRectC))
            if explicit_tiles
            else None
        ),
        explicit_tiles_count=len(explicit_tiles),
        tile_local_nms_iou_threshold=options.tiling.tile_local_nms_iou_threshold,
        max_detections_after_tile_nms=options.tiling.max_detections_after_tile_nms,
        enable_motion_scheduling=options.tiling.enable_motion_scheduling,
        max_scheduled_tiles=options.tiling.max_scheduled_tiles,
    )
```

Then add `tiling=tiling_c,` to the `MpYoloObjectDetectorOptionsC(...)` constructor call, **between** `num_classes=options.num_classes,` and `result_callback=c_callback,`:

```python
        num_classes=options.num_classes,
        tiling=tiling_c,
        result_callback=c_callback,
    )
```

- [ ] **Step 4: Syntax-check the module (runs here)**

Run: `python3 -m py_compile mediapipe/tasks/python/vision/yolo_object_detector.py`
Expected: exit 0, no output.

- [ ] **Step 5: Add the construct + defaults tests**

In `mediapipe/tasks/python/test/vision/yolo_object_detector_test.py`, add these two test methods to `YoloObjectDetectorTest` (place them after `test_options_construct_without_model`):

```python
  def test_tiling_defaults_disabled(self):
    """A default YoloObjectDetectorOptions has disabled (1x1) tiling; no model."""
    options = _YoloObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path='/dummy/model.tflite')
    )
    self.assertEqual(options.tiling.tile_rows, 1)
    self.assertEqual(options.tiling.tile_cols, 1)
    self.assertIsNone(options.tiling.explicit_tiles)
    self.assertFalse(options.tiling.enable_motion_scheduling)

  def test_options_with_tiling_construct_without_model(self):
    """Constructs TilingOptions (grid + explicit_tiles + caps); no model needed."""
    tiling = yolo_object_detector.TilingOptions(
        tile_rows=2,
        tile_cols=2,
        tile_overlap_fraction=0.2,
        explicit_tiles=[
            yolo_object_detector.TileRect(0.25, 0.25, 0.5, 0.5),
            yolo_object_detector.TileRect(0.75, 0.6, 0.45, 0.3),
        ],
        tile_local_nms_iou_threshold=0.5,
        max_detections_after_tile_nms=50,
    )
    options = _YoloObjectDetectorOptions(
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
    self.assertEqual(options.tiling.max_detections_after_tile_nms, 50)
    self.assertFalse(options.tiling.enable_motion_scheduling)
```

- [ ] **Step 6: Add the model-gated tiled detect test**

In the same test file, add this test method to `YoloObjectDetectorTest` (place it after the existing `test_detect_image`):

```python
  @unittest.skipUnless(_MODEL_PRESENT, 'yolov8n.tflite fixture not present; skipping inference test')
  def test_detect_image_tiled(self):
    """Runs inference with a 2x2 tiling grid and validates the result."""
    model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _MODEL_FILE)
    )
    image_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _IMAGE_FILE)
    )
    image = _Image.create_from_file(image_path)

    options = _YoloObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RUNNING_MODE.IMAGE,
        score_threshold=0.25,
        layout=_Layout.CHANNELS_LAST,
        num_classes=80,
        tiling=yolo_object_detector.TilingOptions(
            tile_rows=2, tile_cols=2, tile_overlap_fraction=0.2
        ),
    )
    with _YoloObjectDetector.create_from_options(options) as detector:
      result = detector.detect(image)

    self.assertIsInstance(result, _DetectionResult)
    self.assertGreater(
        len(result.detections),
        0,
        'Expected at least one detection on cats_and_dogs.jpg with tiling',
    )
```

- [ ] **Step 7: Syntax-check the test module (runs here)**

Run: `python3 -m py_compile mediapipe/tasks/python/test/vision/yolo_object_detector_test.py`
Expected: exit 0, no output.

- [ ] **Step 8: Commit**

```bash
git add mediapipe/tasks/python/vision/yolo_object_detector.py \
        mediapipe/tasks/python/test/vision/yolo_object_detector_test.py
git commit -m "feat(tiling): YOLO Python TilingOptions dataclasses + marshalling

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final Verification (after all tasks)

Runs here:

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/tasks/c/vision/yolo_object_detector:tiling_options_abi_test \
  //mediapipe/tasks/c/vision/yolo_object_detector:tiling_options_converter_test
python3 -m py_compile mediapipe/tasks/python/vision/yolo_object_detector.py
python3 -m py_compile mediapipe/tasks/python/test/vision/yolo_object_detector_test.py
```
Expected: both cc_tests PASS; both `py_compile` exit 0.

Cannot run here (toolchain broken): the Python tests in `yolo_object_detector_test.py`. They are written + statically reviewed and expected to pass once the Python toolchain is available; the layout test asserts the same numbers the C++ pin verifies at compile time.

## Acceptance criteria (from spec)

1. `tiling_options_abi_test` passes here (compiles → static_asserts hold); Phase-1 C-API tests still pass.
2. The Python ctypes structs byte-match the C header field-for-field; the standalone ctypes check (Task 2, Step 4) confirms `sizeof`==48/16 and the pinned offsets on this machine; the Python layout test asserts the same numbers.
3. The dataclasses + `create_from_options` marshalling cover all 8 tiling fields + `explicit_tiles`, with the `tiles_array` keep-alive held through Create.
4. The Python tests are written + statically reviewed (cannot run here).
