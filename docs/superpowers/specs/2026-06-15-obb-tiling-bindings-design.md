# OBB Tiling Bindings (C-API + Python) — Design

**Date:** 2026-06-15
**Status:** Approved (design); implementation pending
**Sub-project:** Plan B (C/Python language bindings for tiling) — **Phase 3: OrientedObjectDetector (OBB), C-API + Python**

## Summary

Tiling already ships for the OBB detector in C++ (`OrientedObjectDetectorOptions::TilingOptions`, mapped to the proto), but it is unreachable from the C API or Python — both wrappers omit the `tiling` field, so an OBB caller always gets disabled tiling. This phase threads tiling through **both** the OBB C API and the OBB Python wrapper, with **full parity** to OBB's C++ `TilingOptions`.

It is a faithful mirror of the completed YOLO work (Phase 1 = YOLO C-API, Phase 2 = YOLO Python), adapted for OBB's differences:
- OBB's `TilingOptions` has **6 fields** (no `enable_motion_scheduling` / `max_scheduled_tiles` — those are YOLO-only).
- The OBB tiling structs are **separate** from YOLO's, with distinct `MpOriented*` names (the YOLO `MpTileRect`/`MpTilingOptions` are global `extern "C"` symbols with a different 8-field layout, so reuse would risk an ODR clash; separate structs also match the codebase's per-detector duplication and touch zero shipped YOLO code).
- The OBB C-API test **already has a live model fixture** (`yolo_obb_test_model`), so unlike YOLO Phase 1, a **real tiled-detection e2e runs on this machine** (not just a SKIP).

This phase incorporates all hardening lessons from YOLO Phases 1–2 up front: converter folded into the libs (no dep cycle), a C++ ABI-pin with absolute parent anchors, a testable marshalling helper, accurate FFI lifetime comments, and full test assertions.

## Goals

- Let a C and a Python caller configure every field of OBB's `TilingOptions` via `MpOrientedObjectDetectorOptions.tiling` / `OrientedObjectDetectorOptions.tiling`.
- Keep the C struct, the Python ctypes struct, and the C++ struct byte-compatible.
- Maximize verification on this desktop-C++-only machine: the C-API layer is fully verifiable (converter unit test + a real tiled e2e on the live fixture); the Python layer is anchored by a C++ ABI-pin + a standalone ctypes check.

## Non-Goals

- **Any YOLO change.** The YOLO structs/converter/bindings are untouched; OBB gets its own `MpOriented*` structs.
- **Any C++ / cc / proto change.** OBB's C++ `TilingOptions` and its options→proto mapping already exist; this phase only adds the C-API + Python layers above them (plus C-side *tests*).
- **Motion-scheduling fields.** OBB's `TilingOptions` does not have them; they are not exposed.
- **Partial-grid validation.** A `tile_rows=2, tile_cols=0` config silently disables tiling (documented footgun, same as YOLO); a `Create()` rejection is a deferred follow-up.

## Build & verification constraints

- Desktop C++ (`bazel ... --define MEDIAPIPE_DISABLE_GPU=1`) builds and runs here. The OBB C-API lib + test build here, and the OBB C-API test has its model fixture vendored (`//mediapipe/tasks/testdata/vision:yolo_obb_test_model`), so its assertions actually run.
- The Python Tasks vision API does **not** build/run here (needs compiled `libmediapipe` + the broken Python toolchain). So Python tiling is static-review-only; runnable-here Python checks are `python3 -m py_compile` and standalone `python3` ctypes scripts.
- Verification is therefore split: the C-API layer is fully verified here (incl. a real e2e); the Python layer's byte-match is anchored by a compile-time C++ ABI-pin + a standalone ctypes check, and the Python tests run once the toolchain is available.

## Architecture

### Component 1 — OBB C structs (in `tasks/c/vision/oriented_object_detector/oriented_object_detector.h`)

Added before `struct MpOrientedObjectDetectorOptions`:

```c
// A frame-normalized tile given by its CENTER point and size. Mirrors
// OrientedObjectDetectorOptions::TilingOptions::TileRect. Named MpOrientedTileRect
// (not MpTileRect) to avoid colliding with the YOLO detector's global extern "C"
// MpTileRect, which is a distinct type.
struct MpOrientedTileRect {
  float x_center;
  float y_center;
  float width;
  float height;
};

// Static tiling configuration. Mirrors OrientedObjectDetectorOptions::TilingOptions
// field-for-field (6 fields; OBB has no motion-scheduling knobs).
//
// Tiling is ENABLED when tile_rows * tile_cols > 1 or explicit_tiles_count > 0.
// A zero-initialized MpOrientedTilingOptions means tiling DISABLED. To tile with a
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

…and a field appended to `struct MpOrientedObjectDetectorOptions`, **after `int num_classes;` and before the `result_callback_fn` typedef block**:

```c
  // Static tiling configuration. Zero-initialized => tiling disabled.
  struct MpOrientedTilingOptions tiling;
```

### Component 2 — OBB C converter (`tasks/c/vision/oriented_object_detector/tiling_options_converter.{h,cc}`)

Namespace `mediapipe::tasks::c::vision::oriented_object_detector`:

```cpp
// tiling_options_converter.h
void CppConvertToTilingOptions(
    const MpOrientedTilingOptions& in,
    ::mediapipe::tasks::vision::oriented_object_detector::
        OrientedObjectDetectorOptions::TilingOptions* out);
```

```cpp
// tiling_options_converter.cc body
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
```

`explicit_tiles` is an input array → the converter copies element-by-element into a fresh `std::vector` each call; the caller owns/frees the C array (no `new[]`/`delete[]`).

### Component 3 — wire into the OBB C `CppConvertToDetectorOptions`

In `oriented_object_detector.cc`, `#include` the converter header and append one line after `out->num_classes = in.num_classes;`:

```cpp
  CppConvertToTilingOptions(in.tiling, &out->tiling);
```

### Component 4 — OBB Python ctypes (`tasks/python/vision/oriented_object_detector.py`)

Inline, next to `MpOrientedObjectDetectorOptionsC`:

```python
class MpOrientedTileRectC(ctypes.Structure):
  _fields_ = [
      ('x_center', ctypes.c_float), ('y_center', ctypes.c_float),
      ('width', ctypes.c_float), ('height', ctypes.c_float),
  ]


class MpOrientedTilingOptionsC(ctypes.Structure):
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

…and `('tiling', MpOrientedTilingOptionsC)` inserted into `MpOrientedObjectDetectorOptionsC._fields_` between `('num_classes', ctypes.c_int)` and `('result_callback', _C_TYPES_RESULT_CALLBACK)`.

### Component 5 — OBB Python dataclasses + marshalling

```python
@dataclasses.dataclass
class TileRect:
  x_center: float = 0.0
  y_center: float = 0.0
  width: float = 0.0
  height: float = 0.0


@dataclasses.dataclass
class TilingOptions:
  """Static tiling configuration for the OBB detector (6 fields; no motion scheduling).

  Defaults (1x1, no explicit tiles) mean disabled. Set BOTH tile_rows and
  tile_cols (each >= 1) to tile with a grid; a zero in either disables tiling.
  """
  tile_rows: int = 1
  tile_cols: int = 1
  tile_overlap_fraction: float = 0.0
  explicit_tiles: Optional[List[TileRect]] = None
  tile_local_nms_iou_threshold: float = 0.0
  max_detections_after_tile_nms: int = 0
```

(Public dataclass names `TileRect`/`TilingOptions` are module-scoped — same public names as YOLO's, but in `oriented_object_detector.py`, so no collision; OBB's has 6 fields.) `OrientedObjectDetectorOptions` gets `tiling: TilingOptions = dataclasses.field(default_factory=TilingOptions)` (between `num_classes` and `result_callback`).

A module-level helper builds the ctypes struct and returns it with the backing array (so the marshalling is testable without the native lib, and the keep-alive is explicit):

```python
def _build_oriented_tiling_options_c(
    tiling: TilingOptions,
) -> tuple['MpOrientedTilingOptionsC', object]:
  """Builds the ctypes MpOrientedTilingOptionsC from a TilingOptions dataclass.

  Returns the populated struct AND the backing explicit_tiles array. The caller
  MUST keep the array referenced until the C call that consumes the parent options
  struct returns (explicit_tiles is a raw pointer into it). ctypes also records the
  array in the parent struct's _objects via the by-value copy; the C converter
  copies the tiles into a std::vector synchronously during Create, so outliving the
  Create call is sufficient.
  """
  explicit_tiles = tiling.explicit_tiles or []
  tiles_array = (MpOrientedTileRectC * len(explicit_tiles))(
      *[MpOrientedTileRectC(t.x_center, t.y_center, t.width, t.height)
        for t in explicit_tiles])
  tiling_c = MpOrientedTilingOptionsC(
      tile_rows=tiling.tile_rows,
      tile_cols=tiling.tile_cols,
      tile_overlap_fraction=tiling.tile_overlap_fraction,
      explicit_tiles=(ctypes.cast(tiles_array, ctypes.POINTER(MpOrientedTileRectC))
                      if explicit_tiles else None),
      explicit_tiles_count=len(explicit_tiles),
      tile_local_nms_iou_threshold=tiling.tile_local_nms_iou_threshold,
      max_detections_after_tile_nms=tiling.max_detections_after_tile_nms,
  )
  return tiling_c, tiles_array
```

`create_from_options` calls it: `tiling_c, tiles_keepalive = _build_oriented_tiling_options_c(options.tiling)`, holds `tiles_keepalive` live through `MpOrientedObjectDetectorCreate`, and passes `tiling=tiling_c`.

### Data flow

```
OrientedObjectDetectorOptions.tiling (TilingOptions)
  -> _build_oriented_tiling_options_c -> MpOrientedTilingOptionsC (byte-matches C struct)
  -> MpOrientedObjectDetectorOptionsC(tiling=...) -> ctypes.CDLL -> libmediapipe
  -> [C] CppConvertToTilingOptions -> OrientedObjectDetectorOptions::TilingOptions
  -> [C++] options->proto mapping (already exists) -> graph
```

### Backward compatibility

A zero-initialized `MpOrientedTilingOptions` (or a default `TilingOptions()` → 1×1) yields `tile_rows*tile_cols ≤ 1` and empty `explicit_tiles` → `TilingEnabled` false → tiling disabled, identical to today. No regression.

## Computed layout (the C++ ABI-pin verifies these by compiling)

64-bit targets (int=4, float=4, pointer=8, native alignment):
- `sizeof(MpOrientedTileRect)` = 16; offsets 0/4/8/12.
- `sizeof(MpOrientedTilingOptions)` = **40**; offsets tile_rows=0, tile_cols=4, tile_overlap_fraction=8, explicit_tiles=16, explicit_tiles_count=24, tile_local_nms_iou_threshold=28, max_detections_after_tile_nms=32.
- Parent `MpOrientedObjectDetectorOptions`: `offsetof(tiling)` = **144**, `offsetof(result_callback)` = **184**, `sizeof` = **192**. (Estimates derived assuming `sizeof(MpBaseOptions)`=72; verified by compiling the pin — if any differs, the compiler reports the true value and the same number is used in the C++ pin and the Python layout test.)

## Verification

### Runs here — C-API

1. **`tiling_options_converter_test.cc`** (model-free gate): full round-trip of all 6 fields + a multi-element `explicit_tiles` array (with distinct x/y/w/h so a field swap is caught); zero-init → disabled; null/zero-count safe; clear-on-reuse (incl. empty input clears stale).
2. **Real tiled e2e** in `oriented_object_detector_test.cc`: mirror the existing OBB `ImageMode` test (same model/image, `yolo_obb_test_model` fixture) with `options.tiling.tile_rows = 2; tile_cols = 2; tile_overlap_fraction = 0.2;`; assert ≥1 detection. This RUNS here (live fixture) — a genuine end-to-end exercise of the C→C++→proto→graph tiling path. (Smoke-level: it verifies the tiled path runs and returns valid detections, not tiling efficacy.)
3. **`tiling_options_abi_test.cc`** (compile-time `static_assert`s): pins `sizeof`/`offsetof` of `MpOrientedTileRect`/`MpOrientedTilingOptions` and the parent (absolute anchors 144/184/192 + the relative `result_callback == tiling + sizeof(tiling)`). Anchors the layout the Python ctypes must match.

### Runs when the Python toolchain is available

In `tasks/python/test/vision/oriented_object_detector_test.py`:
- `test_ctypes_tiling_layout_matches_c_abi` — asserts `sizeof(MpOrientedTileRectC)==16`, `sizeof(MpOrientedTilingOptionsC)==40`, the field offsets (incl. front 0/4/8 and MpOrientedTileRectC 0/4/8/12), and the parent absolute anchors (`tiling.offset==144`, `result_callback.offset==184`, `sizeof==192`) — the same numbers the C++ pin verifies.
- `test_tiling_defaults_disabled` — a default `OrientedObjectDetectorOptions` has 1×1 tiling, `explicit_tiles` None, and all other tiling fields 0.
- `test_options_with_tiling_construct_without_model` — construct a grid + `explicit_tiles` + caps; assert every field stored.
- `test_build_oriented_tiling_options_c_marshalling` + `_empty` — call the helper (no native lib); assert the populated ctypes struct, the `explicit_tiles` round-trip through the pointer (keep-alive held), and the empty → null-pointer/zero-count case.
- `test_detect_image_tiled` (`skipUnless(_MODEL_PRESENT)`) — smoke-checks the tiled detect path end to end; SKIPs without the fixture.

Static checks here: `python3 -m py_compile` on the module + test, and a standalone `python3` ctypes script confirming `sizeof`/offsets (16/40 + the field offsets) on this platform.

## File structure

| File | Change | Responsibility |
|---|---|---|
| `tasks/c/vision/oriented_object_detector/oriented_object_detector.h` | Modify | `MpOrientedTileRect`/`MpOrientedTilingOptions` + the `tiling` field. |
| `tasks/c/vision/oriented_object_detector/tiling_options_converter.{h,cc}` | Create | `CppConvertToTilingOptions` (6 fields + explicit_tiles). |
| `tasks/c/vision/oriented_object_detector/oriented_object_detector.cc` | Modify | Include the converter; one call line in `CppConvertToDetectorOptions`. |
| `tasks/c/vision/oriented_object_detector/tiling_options_converter_test.cc` | Create | Model-free converter gate. |
| `tasks/c/vision/oriented_object_detector/tiling_options_abi_test.cc` | Create | Compile-time ABI-pin (runs here). |
| `tasks/c/vision/oriented_object_detector/oriented_object_detector_test.cc` | Modify | Add the real tiled e2e. |
| `tasks/c/vision/oriented_object_detector/BUILD` | Modify | Add converter to both libs; add the converter test + the abi test. |
| `tasks/python/vision/oriented_object_detector.py` | Modify | ctypes structs + `tiling` field; `TileRect`/`TilingOptions` dataclasses + `tiling` field; `_build_oriented_tiling_options_c` + marshalling. |
| `tasks/python/test/vision/oriented_object_detector_test.py` | Modify | layout/construct/defaults/marshalling/detect tests. |

No `tasks/python/vision/BUILD` change (inline structs/dataclasses; tests in the existing test file).

## Acceptance criteria

1. `bazel test --define MEDIAPIPE_DISABLE_GPU=1` passes here for the OBB `tiling_options_converter_test`, `tiling_options_abi_test`, and `oriented_object_detector_test` (the **tiled e2e runs and asserts**, not skips, thanks to the live fixture).
2. The C struct, the Python ctypes struct, and OBB's C++ `TilingOptions` are byte-compatible; the Python layout test asserts the same numbers the C++ pin verifies.
3. The Python dataclasses + `_build_oriented_tiling_options_c` cover all 6 fields + `explicit_tiles`, with the keep-alive held through Create.
4. No YOLO / cc / proto change; no regression in existing OBB C-API tests.

## Follow-ups (recorded, not in this phase)

- Vendor `yolov8n.tflite` into the YOLO + Python test data to un-gate the YOLO tiled e2e tests on this machine (OBB already has its fixture).
- Optional `Create()` partial-grid validation for both detectors.
