# C-API + Python TrackingOptions Bindings — Design

**Date:** 2026-06-16
**Branch:** `dev` (MediaPipe fork; unmerged, no upstream PR)
**Status:** Approved design — ready for implementation plan

## Goal

Expose the YOLO object detector's `TrackingOptions` (tracker selection + motion-only
BoTSORT knobs) through the **C-API** and **Python**, so callers in those layers can
select the tracker — completing the bindings arc for the tracking feature, exactly
as the tiling bindings were phased. YOLO only (tracking is a YOLO-only feature); full
parity (all 7 fields). The C++/proto/graph layers are unchanged — this is a binding
layer only.

## Background

The tracking feature (shipped on `dev`) added a `tracking` option to the C++
`YoloObjectDetectorOptions` (proto `TrackingOptions` + public struct + converter +
Create-time validation). The C-API (`mediapipe/tasks/c/vision/yolo_object_detector/`)
and Python (`mediapipe/tasks/python/vision/yolo_object_detector.py`) wrappers already
expose `tiling` via the same pattern this spec mirrors:

- C-API: a flat `MpTilingOptions` struct + a `tracking`-style field on
  `MpYoloObjectDetectorOptions` + `CppConvertToTilingOptions` converter + a
  `tiling_options_abi_test.cc` of compile-time `static_assert`s pinning the layout.
- Python: a ctypes `MpTilingOptionsC` byte-matching the C struct + a `TilingOptions`
  dataclass + a `_build_tiling_options_c` marshalling helper.

The cc-layer Create() validation (BOTSORT requires stream-mode + tiling + no
motion-scheduling + num_classes∈[1,256]) lives below the binding and therefore
applies to C-API and Python callers automatically — it is reused, not re-implemented.

## Architecture

### Phase 1 — C-API

**`MpTrackingOptions` struct** in
`mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h`, mirroring
`YoloObjectDetectorOptions::TrackingOptions` field-for-field:

```c
// Tracker selection for the tiled VIDEO/LIVE_STREAM path. Mirrors
// YoloObjectDetectorOptions::TrackingOptions field-for-field.
//
// tracker_type: 0 = unspecified (-> BOX_TRACKER), 1 = BOX_TRACKER (optical-flow,
// default), 2 = BOTSORT (tracking-by-detection, motion-only). Numerically equal to
// the C++/proto enum. A zero-initialized MpTrackingOptions therefore means
// BOX_TRACKER -- byte-identical to a C caller who never set tracking at all.
//
// The threshold/buffer knobs are honored only for BOTSORT. A plain C struct cannot
// distinguish "unset" from 0, so a BOTSORT caller MUST set the knobs explicitly
// (a zero-init struct yields 0.0 thresholds); the C++ struct defaults
// (0.6/0.1/0.7/30/0.7) are not reachable through a zero-init C struct. BOX_TRACKER
// ignores the knobs. (Same convention as score_threshold.)
struct MpTrackingOptions {
  int tracker_type;
  float track_high_threshold;
  float track_low_threshold;
  float new_track_threshold;
  int track_buffer;
  float match_threshold;
  bool enable_gmc;
};
```

**`tracking` field** added to `MpYoloObjectDetectorOptions`, placed **after `tiling`
and before `result_callback`** (keeps the option-data fields contiguous, the callback
last):

```c
  struct MpTilingOptions tiling;
  struct MpTrackingOptions tracking;   // NEW
  result_callback_fn result_callback;
```

**`CppConvertToTrackingOptions`** converter in new files
`tracking_options_converter.{h,cc}`:

```cpp
void CppConvertToTrackingOptions(
    const MpTrackingOptions& in,
    ::mediapipe::tasks::vision::yolo_object_detector::
        YoloObjectDetectorOptions::TrackingOptions* out);
```

It copies all 7 fields verbatim (`tracker_type` via
`static_cast<...TrackingOptions::TrackerType>(in.tracker_type)`). Called in
`MpYoloObjectDetectorCreate` immediately after `CppConvertToTilingOptions(in.tiling, ...)`:
`CppConvertToTrackingOptions(in.tracking, &out->tracking);`. The converter is folded
into **both** the converter cc_library and the `alwayslink` `_c_lib` (mirroring the
tiling converter) to avoid a dependency cycle.

### Phase 1 — ABI impact (the one non-additive part)

Inserting `tracking` (sizeof 28) between `tiling@136` and `result_callback` **shifts**
the trailing offsets. New layout (x86-64 / arm64, 8-byte pointer):

- `MpTrackingOptions`: **sizeof 28**, alignment 4. Offsets:
  `tracker_type@0, track_high_threshold@4, track_low_threshold@8,
  new_track_threshold@12, track_buffer@16, match_threshold@20, enable_gmc@24`
  (bool width 1 @24; 3 bytes tail padding to 28).
- Parent `MpYoloObjectDetectorOptions`: `tiling@136` (size 48, ends @184) →
  `tracking@184` (size 28, ends @212) → `result_callback@216` (8-byte-aligned, so
  4 bytes padding after tracking) → **sizeof 224**.

New `tracking_options_abi_test.cc` pins `MpTrackingOptions` (sizeof 28; the 7 offsets;
bool width 1) and the parent (`tracking@184`, `result_callback@216`, `sizeof 224`),
plus a relative pin `offsetof(tracking) == offsetof(tiling) + sizeof(MpTilingOptions)`.

The **existing** `tiling_options_abi_test.cc` has pins that now move:
`result_callback@184`→**216**, `sizeof@192`→**224**, and the relative assert
`result_callback == tiling + sizeof(MpTilingOptions)` is no longer true (tracking now
sits between them) — it is **replaced** with `tracking == tiling + sizeof(MpTilingOptions)`.
These updates are required (not optional) and are the lockstep the Python ctypes match.

### Phase 2 — Python

In `mediapipe/tasks/python/vision/yolo_object_detector.py`:

- **`MpTrackingOptionsC`** ctypes `Structure` byte-matching `MpTrackingOptions`:
  `tracker_type` `c_int`, four `c_float` thresholds, `track_buffer` `c_int`,
  `match_threshold` `c_float`, `enable_gmc` `c_bool`. Add `('tracking',
  MpTrackingOptionsC)` to `MpYoloObjectDetectorOptionsC._fields_` **after** `tiling`
  (matching the C struct order).
- **`TrackerType`** `enum.IntEnum` (`BOX_TRACKER = 1`, `BOTSORT = 2`) for an ergonomic
  public API.
- **`TrackingOptions`** dataclass mirroring the C++ struct, with the REAL defaults
  (`tracker_type=TrackerType.BOX_TRACKER, track_high_threshold=0.6,
  track_low_threshold=0.1, new_track_threshold=0.7, track_buffer=30,
  match_threshold=0.7, enable_gmc=False`). Python users thus get correct defaults
  (unlike a zero-init C struct). Add `tracking: TrackingOptions =
  dataclasses.field(default_factory=TrackingOptions)` to `YoloObjectDetectorOptions`.
- **`_build_tracking_options_c(tracking) -> MpTrackingOptionsC`** marshalling helper
  (`tracker_type=int(tracking.tracker_type)`, the rest verbatim). Wire it into
  `Create` alongside `_build_tiling_options_c`, setting the `tracking=` field of the
  options ctypes struct. No backing array needed (no pointer fields), so it returns
  just the struct (unlike tiling, which also returns a keep-alive array).

## Error handling

- A C/Python caller selecting BOTSORT in an invalid configuration (IMAGE mode, no
  tiling, motion-scheduling on, >256 classes) is rejected by the cc-layer Create()
  validation, surfaced as a non-`kMpOk` `MpStatus` (C) / raised error (Python). No
  new validation in the binding.
- A zero-init C struct → `tracker_type=0` → BOX_TRACKER (backward compatible).

## Testing

**C-API (runs here):**
- `tracking_options_converter_test.cc` — the gate. ~6 cases: each field round-trips;
  `tracker_type` 0/1/2 → correct proto enum; enable_gmc true/false.
- `tracking_options_abi_test.cc` — compile-time `static_assert` pins (the authoritative
  layout guarantee).
- A **model-free wiring test** in `yolo_object_detector_test.cc` (C-API) asserting
  `MpYoloObjectDetectorCreate` returns a non-`kMpOk` status when `tracking.tracker_type`
  is BOTSORT in IMAGE mode (the cc validation fires before model load), mirroring the
  tiling `MotionSchedulingInImageModeRejected` test. Proves the field is wired through
  Create.
- Update the existing `tiling_options_abi_test.cc` pins (result_callback@216,
  sizeof@224) — these compile-time asserts are the proof the layout shift is correct.

**Python (static-only here — toolchain broken; see build-verifiability-constraints):**
- The C++ ABI-pin is the authoritative cross-check that the ctypes layout is right.
- A standalone ctypes layout check (run on this machine) confirming
  `sizeof(MpTrackingOptionsC) == 28` and the parent `sizeof == 224` /
  `tracking` offset 184, matching the C++ pins.
- `py_compile` on the wrapper.
- Full Python construct/defaults/marshalling/detect tests are authored but run only
  where the Python toolchain works.

## Scope boundaries (YAGNI)

**In scope:** YOLO detector only · C-API + Python in one spec, phased (Phase 1 C-API,
Phase 2 Python) · full parity (7 fields) · reuse the cc-layer validation.

**Out of scope:** OBB (no tracking feature) · any C++/proto/graph change · surfacing
track IDs through the bindings (the C++ output contract doesn't expose them) ·
ReID/appearance knobs (motion-only).

## Verifiability

The C-API converter test, ABI test, and model-free wiring test build+run here under
`--define MEDIAPIPE_DISABLE_GPU=1`. Python is static-review-only on this machine
(verified via the C++ ABI-pin + a standalone ctypes check + py_compile), consistent
with how the YOLO Python tiling bindings were verified.
