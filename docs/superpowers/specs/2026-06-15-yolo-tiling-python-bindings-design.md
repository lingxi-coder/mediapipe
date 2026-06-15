# YOLO Tiling Python Bindings — Design

**Date:** 2026-06-15
**Status:** Approved (design); implementation pending
**Sub-project:** Plan B (C/Python language bindings for tiling) — **Phase 2: Python, YOLO only**

## Summary

Phase 1 exposed `TilingOptions` through the YOLO **C API** (`MpYoloObjectDetectorOptions.tiling`, an `MpTilingOptions` struct + `CppConvertToTilingOptions`). The Python Tasks wrapper (`mediapipe/tasks/python/vision/yolo_object_detector.py`) binds the C API via **ctypes** (a dataclass `Options` → a ctypes `Structure` that byte-matches the C struct → a `ctypes.CDLL` call into `libmediapipe`). That ctypes `Structure` does **not** yet carry the `tiling` field, so a Python caller cannot configure tiling.

This phase threads tiling through the Python layer with **full field parity** to the Phase-1 C struct: new `MpTileRectC`/`MpTilingOptionsC` ctypes structs + a `tiling` field on `MpYoloObjectDetectorOptionsC`, and public `TileRect`/`TilingOptions` dataclasses + a `tiling` field on `YoloObjectDetectorOptions`, marshalled in `create_from_options`.

## Goals

- Let a Python caller configure every field of the C++/C `TilingOptions` via `YoloObjectDetectorOptions.tiling`.
- Keep the Python ctypes `Structure` byte-identical to the C struct (the correctness crux of a ctypes binding).
- Provide the strongest verification achievable on this machine, where Python cannot be built or run.

## Non-Goals

- **OrientedObjectDetector (OBB) Python tiling** — deferred; needs OBB C-API tiling first (Phase 1 was YOLO-only), and OBB's `TilingOptions` lacks the two motion-scheduling fields.
- **Any C / cc / proto change** — Phase 1 already maps `MpTilingOptions` → C++ `TilingOptions` → proto; this phase only adds the Python layer above it (plus a C++ ABI-pin *test*, which changes no production code).
- **Partial-grid validation** — a `tile_rows=2, tile_cols=0` config silently disables tiling (documented footgun from Phase 1). Adding a `Create()` rejection is a separate, deferred follow-up that would touch the cc layer.

## Build & verification constraints

- The Python Tasks vision API does **not** build or run on this machine: it needs the compiled `libmediapipe.dylib`/`.so` (loaded via `ctypes.CDLL`) plus the Python framework toolchain, which is broken here (per project memory). So **no Python test runs here** and there is **no TDD** for the Python code.
- Desktop C++ (`bazel ... --define MEDIAPIPE_DISABLE_GPU=1`) does build and run here.
- Therefore verification is **dual**: a compile-time C++ ABI-pin test runs here and anchors the ground-truth struct layout; the Python binding + tests are written and statically reviewed, and run once the Python toolchain is available. The Python layout test mirrors the exact numbers the C++ pin verifies, forming a cross-language contract.

## Architecture

### Component 1 — ctypes structs (inline in `yolo_object_detector.py`)

`MpYoloObjectDetectorOptionsC` is defined inline in this module (unlike shared component structs in `*_c.py`), so the tiling ctypes structs go inline directly above it. They must byte-match the C header field-for-field:

```python
class MpTileRectC(ctypes.Structure):
  """Byte-matches struct MpTileRect in the YOLO C header."""
  _fields_ = [
      ('x_center', ctypes.c_float),
      ('y_center', ctypes.c_float),
      ('width', ctypes.c_float),
      ('height', ctypes.c_float),
  ]


class MpTilingOptionsC(ctypes.Structure):
  """Byte-matches struct MpTilingOptions in the YOLO C header."""
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

And one field inserted into `MpYoloObjectDetectorOptionsC._fields_`, **between `('num_classes', ctypes.c_int)` and `('result_callback', _C_TYPES_RESULT_CALLBACK)`**:

```python
      ('tiling', MpTilingOptionsC),
```

**Type-mapping decisions:** `enable_motion_scheduling` is `ctypes.c_bool` (1 byte — matches the C `bool`, NOT `c_int`); `explicit_tiles_count` is `ctypes.c_uint32` (matches the existing `category_allowlist_count` mapping); `explicit_tiles` is `POINTER(MpTileRectC)`. ctypes uses native alignment, so a faithful field-by-field mirror produces the same offsets as the C compiler.

### Component 2 — public dataclasses

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
  """Static tiling configuration. Mirrors the C++ YoloObjectDetectorOptions.TilingOptions.

  Tiling is enabled when tile_rows * tile_cols > 1 or explicit_tiles is non-empty.
  The defaults (1x1, no explicit tiles) mean tiling disabled. Set BOTH tile_rows
  and tile_cols (each >= 1) to tile with a grid; a zero in either disables tiling.
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

`YoloObjectDetectorOptions` gets a new field (always present — mirrors the C++ struct, avoids a None branch):

```python
  tiling: TilingOptions = dataclasses.field(default_factory=TilingOptions)
```

The dataclass defaults `tile_rows=1, tile_cols=1` mirror the C++ struct exactly, so a default `TilingOptions()` is disabled (1×1). Marshalled into the ctypes struct, those flow as `tile_rows=1, tile_cols=1`, which the C++ `TilingEnabled` (`tile_rows*tile_cols > 1`) treats as disabled — identical to a caller that sets nothing.

### Component 3 — `create_from_options` marshalling (+ keep-alive)

In `create_from_options`, build the explicit-tiles array and the tiling ctypes struct, then pass `tiling=tiling_c`:

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

`tiling_c` is then passed as the `tiling=` argument to the `MpYoloObjectDetectorOptionsC(...)` constructor. Both `tiles_array` and `tiling_c` remain in scope (locals) through the `lib.MpYoloObjectDetectorCreate(...)` call, satisfying the keep-alive requirement.

### Data flow

```
YoloObjectDetectorOptions.tiling (TilingOptions dataclass)
  -> create_from_options builds tiles_array + MpTilingOptionsC (tiling_c)
  -> MpYoloObjectDetectorOptionsC(tiling=tiling_c)  [ctypes, byte-matches C struct]
  -> lib.MpYoloObjectDetectorCreate (ctypes.CDLL -> libmediapipe)
  -> [C layer, Phase 1] CppConvertToTilingOptions -> C++ TilingOptions -> proto -> graph
```

## Verification

### Runs here — C++ ABI-pin test (the layout anchor)

New `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_abi_test.cc`: compile-time `static_assert`s over the C header, pinning the layout the Python ctypes structs must match. Because they are `static_assert`s, the assertions are verified by simply compiling the target on this machine.

```cpp
#include <cstddef>
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"

// 64-bit targets (arm64 / x86-64 / Windows LLP64): int=4, float=4, bool=1,
// pointer=8 with native alignment. These offsets are what the Python ctypes
// structs in yolo_object_detector.py (MpTileRectC / MpTilingOptionsC) must match.
static_assert(sizeof(MpTileRect) == 4 * sizeof(float), "MpTileRect must be 4 packed floats");
static_assert(offsetof(MpTileRect, x_center) == 0, "");
static_assert(offsetof(MpTileRect, y_center) == 4, "");
static_assert(offsetof(MpTileRect, width) == 8, "");
static_assert(offsetof(MpTileRect, height) == 12, "");

static_assert(sizeof(MpTilingOptions) == 48, "MpTilingOptions layout pinned for ctypes");
static_assert(offsetof(MpTilingOptions, tile_rows) == 0, "");
static_assert(offsetof(MpTilingOptions, tile_cols) == 4, "");
static_assert(offsetof(MpTilingOptions, tile_overlap_fraction) == 8, "");
static_assert(offsetof(MpTilingOptions, explicit_tiles) == 16, "");
static_assert(offsetof(MpTilingOptions, explicit_tiles_count) == 24, "");
static_assert(offsetof(MpTilingOptions, tile_local_nms_iou_threshold) == 28, "");
static_assert(offsetof(MpTilingOptions, max_detections_after_tile_nms) == 32, "");
static_assert(offsetof(MpTilingOptions, enable_motion_scheduling) == 36, "");
static_assert(offsetof(MpTilingOptions, max_scheduled_tiles) == 40, "");

// tiling occupies a contiguous block between num_classes and result_callback.
static_assert(offsetof(MpYoloObjectDetectorOptions, tiling) >
                  offsetof(MpYoloObjectDetectorOptions, num_classes),
              "tiling must follow num_classes");
static_assert(offsetof(MpYoloObjectDetectorOptions, result_callback) ==
                  offsetof(MpYoloObjectDetectorOptions, tiling) +
                      sizeof(MpTilingOptions),
              "result_callback must immediately follow tiling");

// A trivial runtime test so the cc_test target has a test to execute; the real
// guarantees are the static_asserts above (checked at compile time).
TEST(TilingOptionsAbiTest, LayoutPinned) { SUCCEED(); }
```

### Runs when the Python toolchain is available — Python tests

In `mediapipe/tasks/python/test/vision/yolo_object_detector_test.py`:

1. **`test_ctypes_tiling_layout_matches_c_abi`** — pure ctypes layout math (no `libmediapipe`); asserts the same numbers the C++ pin verifies:
   - `ctypes.sizeof(yolo_object_detector.MpTileRectC) == 16`
   - `ctypes.sizeof(yolo_object_detector.MpTilingOptionsC) == 48`
   - `MpTilingOptionsC.explicit_tiles.offset == 16`, `.explicit_tiles_count.offset == 24`, `.enable_motion_scheduling.offset == 36`, `.max_scheduled_tiles.offset == 40`
   - parent: `MpYoloObjectDetectorOptionsC.tiling.offset > MpYoloObjectDetectorOptionsC.num_classes.offset` and `MpYoloObjectDetectorOptionsC.result_callback.offset == MpYoloObjectDetectorOptionsC.tiling.offset + ctypes.sizeof(MpTilingOptionsC)`

2. **`test_options_with_tiling_construct_without_model`** — construct `TilingOptions(tile_rows=2, tile_cols=2, tile_overlap_fraction=0.2, explicit_tiles=[TileRect(0.25, 0.25, 0.5, 0.5)], max_detections_after_tile_nms=50, enable_motion_scheduling=False)`, attach to `YoloObjectDetectorOptions`, and assert the dataclass stored each field (incl. the `explicit_tiles` list contents). No model required.

3. **`test_detect_image_tiled`** (`@unittest.skipUnless(_MODEL_PRESENT, ...)`) — mirror the existing `test_detect_image`, adding `tiling=yolo_object_detector.TilingOptions(tile_rows=2, tile_cols=2, tile_overlap_fraction=0.2)`; assert ≥1 detection. SKIPs without the fixture.

## File structure

| File | Change | Responsibility |
|---|---|---|
| `mediapipe/tasks/python/vision/yolo_object_detector.py` | Modify | `MpTileRectC`/`MpTilingOptionsC`; the `tiling` ctypes field; `TileRect`/`TilingOptions` dataclasses; the `tiling` dataclass field; the `create_from_options` marshalling + keep-alive. |
| `mediapipe/tasks/python/test/vision/yolo_object_detector_test.py` | Modify | The 3 tests above. |
| `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_abi_test.cc` | Create | Compile-time ABI-pin (runs here). |
| `mediapipe/tasks/c/vision/yolo_object_detector/BUILD` | Modify | Add the `tiling_options_abi_test` cc_test (deps: `:yolo_object_detector_lib` for the header, gtest_main). |

No `tasks/python/vision/BUILD` or `tasks/python/test/vision/BUILD` change: the structs/dataclasses are inline in the existing module and the tests are in the existing test file (same deps).

## Acceptance criteria

1. `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:tiling_options_abi_test` **passes here** (i.e. the target compiles, proving the static_asserts hold), and the Phase-1 C-API tests still pass.
2. The Python ctypes structs (`MpTileRectC`, `MpTilingOptionsC`, the new `tiling` field) byte-match the C header field-for-field, and the Python layout test asserts the same numbers the C++ pin verifies.
3. The Python dataclasses + `create_from_options` marshalling are complete (all 8 tiling fields + `explicit_tiles`), with the `tiles_array` keep-alive held through the Create call.
4. The three Python tests are written and statically reviewed; they are expected to pass once the Python toolchain is available (they cannot run on this machine).

## Follow-ups (recorded, not in this phase)

- OBB Python tiling (after OBB C-API tiling).
- Optional `Create()` partial-grid validation (one of tile_rows/tile_cols = 0 while the other > 1).
- Vendor `yolov8n.tflite` into the Python test data so `test_detect_image_tiled` runs end to end where the toolchain works.
