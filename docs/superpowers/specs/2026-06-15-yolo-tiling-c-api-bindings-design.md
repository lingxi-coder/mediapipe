# YOLO Tiling C-API Bindings — Design

**Date:** 2026-06-15
**Status:** Approved (design); implementation pending
**Sub-project:** Plan B (C/Python language bindings for tiling) — **Phase 1: C API, YOLO only**

## Summary

The C++ `YoloObjectDetector` exposes a `TilingOptions` sub-struct (static tiling: grid + explicit tiles + tile-local NMS + motion scheduling), and the cc layer already maps it into the options proto. But the **C API** wrapper
(`mediapipe/tasks/c/vision/yolo_object_detector/`) never touches tiling — a C caller (and therefore the Python ctypes layer above it) can only ever get the default, disabled tiling.

This sub-project threads `TilingOptions` through the YOLO **C API** so C callers can configure tiling, with **full field parity** to the C++ struct. It deliberately stops at the C layer; the Python ctypes/dataclass layer is a separate follow-up (it cannot be built or run on this machine), and the Oriented (OBB) detector is out of scope for this phase.

## Goals

- Expose every field of the C++ `YoloObjectDetectorOptions::TilingOptions` through the YOLO C API.
- Preserve backward compatibility: existing C callers (zero-initialized options, no tiling) behave exactly as before.
- Provide a **model-free, unit-testable** marshalling seam so the binding is verifiable on this desktop-C++-only machine (TDD-able without the gitignored model fixture).

## Non-Goals (explicitly out of scope)

- **Python bindings** — a separate follow-up spec. The Python Tasks toolchain (compiled `libmediapipe` + framework bindings) does not build on this machine, so Python tiling cannot be TDD'd here.
- **Oriented (OBB) detector** — deferred. (Its C++ `TilingOptions` lacks `enable_motion_scheduling`/`max_scheduled_tiles`; a future shared `MpTilingOptions` would let its converter ignore those two fields.)
- **Proto / cc-layer changes** — the C++ `YoloObjectDetectorOptions::TilingOptions` and its options→proto mapping already exist and are unchanged.
- **A standalone reusable C struct** — `MpTilingOptions` lives in the YOLO C header for now; hoisting it to a shared header is a refactor for when OBB is added.

## Build & verification constraints

- Only desktop C++ (`bazel ... --define MEDIAPIPE_DISABLE_GPU=1`) builds and runs on this machine. The two YOLO C-API libs already build here.
- The YOLO C-API **integration test SKIPs** without `yolov8n.tflite` wired into its BUILD (the fixture is gitignored). So an end-to-end tiling test **cannot be the verification gate**.
- The verification gate is therefore the **converter unit test**, which needs no model and always runs here.

## Architecture

### Approach

Mirror the existing nested-option-group precedent in this same C-API tree —
`MpClassifierOptions` + `classifier_options_converter.cc`
(`mediapipe/tasks/c/components/processors/`) — with a **dedicated, unit-testable converter** rather than inlining the marshalling into `CppConvertToDetectorOptions`. The converter is the smallest unit that fully owns the C→C++ tiling translation, so it can be tested in isolation with no model.

### Component 1 — Plain-C structs (public C header)

Added to `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h`, before `struct MpYoloObjectDetectorOptions`:

```c
// A frame-normalized tile given by its CENTER point and size (NOT corner-based).
// Mirrors YoloObjectDetectorOptions::TilingOptions::TileRect.
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
// options = {}`) therefore means tiling DISABLED — byte-identical to a C caller
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

And one field appended to `struct MpYoloObjectDetectorOptions`:

```c
  // Static tiling configuration. Zero-initialized => tiling disabled.
  struct MpTilingOptions tiling;
```

**Design decisions:**
- `bool enable_motion_scheduling` (not `int`) — consistent with the existing `bool class_agnostic_nms` in the OBB C options struct; both sides of the ABI are compiled as C++ here.
- `explicit_tiles` as `const MpTileRect* + uint32_t count` — the exact `category_allowlist` idiom already in `MpYoloObjectDetectorOptions`. Caller-owned, converter-copied.

### Component 2 — The converter (new unit-testable seam)

New files `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_converter.{h,cc}`, in namespace `mediapipe::tasks::c::vision::yolo_object_detector`:

```cpp
// tiling_options_converter.h
void CppConvertToTilingOptions(
    const MpTilingOptions& in,
    ::mediapipe::tasks::vision::yolo_object_detector::
        YoloObjectDetectorOptions::TilingOptions* out);
```

```cpp
// tiling_options_converter.cc — body
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
```

Ownership: `explicit_tiles` is an **input options** array → the converter copies element-by-element into a fresh `std::vector` each call; the caller owns/frees the C array. This is the input-direction convention (same as `category_allowlist`), NOT the result-converter `new[]`/`delete[]` pattern. `MpTileRect` is 4 POD floats, so no per-element free regardless.

### Component 3 — Wire into the options conversion

In `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.cc`,
`#include "...tiling_options_converter.h"` and append one line to the existing
`CppConvertToDetectorOptions(const MpYoloObjectDetectorOptions& in, YoloObjectDetectorOptions* out)`:

```cpp
  CppConvertToTilingOptions(in.tiling, &out->tiling);
```

No other changes to `CppConvertToDetectorOptions` or `CppYoloObjectDetectorCreate`.

### Data flow

```
C caller fills MpYoloObjectDetectorOptions.tiling (MpTilingOptions)
  -> CppConvertToDetectorOptions
       -> CppConvertToTilingOptions  (this sub-project: C struct -> C++ TilingOptions)
  -> YoloObjectDetectorOptions::TilingOptions
  -> ConvertYoloObjectDetectorOptionsToProto         (already exists, unchanged)
  -> proto::YoloObjectDetectorOptions.tiling
  -> YOLO graph (TilingEnabled gates the tiled vs plain path)
```

### Backward compatibility

Existing C callers zero-initialize `MpYoloObjectDetectorOptions` and never set tiling, so `tiling` is all-zero. The converter writes `tile_rows=0, tile_cols=0, explicit_tiles` empty into the C++ struct. `TilingEnabled` evaluates `0 * 0 > 1 || empty → false`, i.e. tiling disabled — identical to today (where the C++ defaults `tile_rows=1, tile_cols=1` also yield "1 tile = disabled"). The non-tiled path never reads `tile_rows` for grid math, so writing 0 instead of the C++ default 1 is inert. **No regression.**

## File structure

| File | Change | Responsibility |
|---|---|---|
| `.../c/vision/yolo_object_detector/yolo_object_detector.h` | Modify | Add `MpTileRect`, `MpTilingOptions`; add `tiling` field to `MpYoloObjectDetectorOptions`. |
| `.../c/vision/yolo_object_detector/tiling_options_converter.h` | Create | Declare `CppConvertToTilingOptions`. |
| `.../c/vision/yolo_object_detector/tiling_options_converter.cc` | Create | Define it (8 scalar copies + `explicit_tiles` loop). |
| `.../c/vision/yolo_object_detector/yolo_object_detector.cc` | Modify | Include the converter header; one call line in `CppConvertToDetectorOptions`. |
| `.../c/vision/yolo_object_detector/tiling_options_converter_test.cc` | Create | **Verification gate** — model-free unit test. |
| `.../c/vision/yolo_object_detector/yolo_object_detector_test.cc` | Modify | Add a model-gated `TiledImageMode` e2e (best-effort; SKIPs without fixture). |
| `.../c/vision/yolo_object_detector/BUILD` | Modify | Add the two converter files to `yolo_object_detector_lib` + `yolo_object_detector_c_lib` srcs/hdrs; add the `tiling_options_converter_test` cc_test. |

**BUILD note (no dependency cycle):** the converter `.cc/.h` are folded into the existing `yolo_object_detector_lib` (and `_c_lib`) rather than a separate `cc_library`. The converter needs `MpTilingOptions` (from `yolo_object_detector.h`, same lib) and `YoloObjectDetectorOptions::TilingOptions` (from `//mediapipe/tasks/cc/vision/yolo_object_detector`, already a lib dep). Keeping them in the same library avoids a `lib → converter → lib` cycle and needs no header split. The unit test depends on `:yolo_object_detector_lib` (for the converter header) and `//mediapipe/tasks/cc/vision/yolo_object_detector` (to assert against the C++ type).

## Testing strategy

### Gate: `tiling_options_converter_test.cc` (no model — always runs here)

- **Full round-trip:** populate every `MpTilingOptions` scalar with a distinct non-default value plus a **multi-element** `explicit_tiles` array; call `CppConvertToTilingOptions`; assert all 8 scalars match and `explicit_tiles` matches element-by-element (size + each of x_center/y_center/width/height).
- **Zero-init → disabled:** a `{}`-initialized `MpTilingOptions` yields `tile_rows==0, tile_cols==0`, empty `explicit_tiles`.
- **Empty explicit_tiles is safe:** `explicit_tiles_count==0` with `explicit_tiles==nullptr` produces an empty vector and does not dereference the null pointer.
- **Reuse safety:** converting into an `out` that already holds `explicit_tiles` clears it first (no accumulation across calls).

### Best-effort: `TiledImageMode` e2e in `yolo_object_detector_test.cc` (model-gated)

Mirrors the existing `ImageMode` test but sets `options.tiling.tile_rows = 2; options.tiling.tile_cols = 2;` (and an overlap), then runs `MpYoloObjectDetectorDetectImage`. Uses the same `GTEST_SKIP()`-when-fixture-absent guard as the existing test, so it builds and runs cleanly without the model. If `yolov8n.tflite` is later wired into `testdata/vision/BUILD` and this package's BUILD `data`, it provides real end-to-end confidence that tiling flows through to detection. **Not the gate.**

## Acceptance criteria

1. `bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_lib //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_c_lib` succeeds.
2. `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:tiling_options_converter_test` passes (all cases above).
3. `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_test` passes (the new `TiledImageMode` SKIPs cleanly absent the fixture; existing `ImageMode` unchanged).
4. No changes to the cc/proto layers; no regression in existing C-API tests.

## Follow-ups (recorded, not in this phase)

- Python ctypes/dataclass tiling bindings for YOLO (mirror this C struct; static-review-only here).
- OBB C-API tiling bindings (+ decide shared vs per-detector `MpTilingOptions`).
- Vendor `yolov8n.tflite` into the YOLO C-API test BUILD to un-gate the e2e tiling assertions on this machine.
