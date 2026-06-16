# OBB Tracking C-API + Python Bindings — Design

**Date:** 2026-06-16
**Branch:** `dev` (MediaPipe fork; unmerged, no upstream PR)
**Status:** Approved design — ready for implementation plan
**Scope:** Sub-project 2 of 2 (bindings). Sub-project 1 (C++ core OBB BoTSORT tracking) is shipped.

## Goal

Expose, through the C-API and Python, both sides of the OBB tracking feature:
- **Input:** the OBB `TrackingOptions` (tracker selection + BoTSORT knobs), so C/Python
  callers can enable BoTSORT on the OBB tiled-livestream path.
- **Output:** the oriented result's `track_id`, so C/Python callers can read the
  stable per-object id.

This completes the OBB tracking arc (C++ → C-API → Python). The C++/proto/graph layers
are unchanged — this is a binding layer only. It is the union of two already-shipped
patterns ([[tracking-bindings-shipped]] for the YOLO input options; the YOLO track-id
exposure for the output id), applied to OBB's *separate* `MpOriented*` / oriented-result
types.

## Background

The OBB C-API already exposes `tiling` via `MpOrientedTilingOptions` + a converter +
ABI pins (the OBB tiling bindings). The OBB Python wrapper
(`mediapipe/tasks/python/vision/oriented_object_detector.py`) mirrors it with ctypes +
dataclasses. The OBB result is a *separate* oriented type: C `MpOrientedDetection`
(`mediapipe/tasks/c/components/containers/oriented_detection_result.h`: categories, cx,
cy, width, height, rotation) and Python `OrientedDetection`
(`mediapipe/tasks/python/components/containers/oriented_detections.py`). The C++ core
(sub-project 1) added an OBB `TrackingOptions` (proto + struct + converter + Create
validation: BOX_TRACKER rejected, BOTSORT requires stream+tiling+num_classes∈[1,256])
and `OrientedObjectDetection.track_id` (`std::optional<std::string>`).

Current OBB options parent layout (verified): `tiling @144` (sizeof
`MpOrientedTilingOptions` = 40, ends @184), `result_callback @184`, `sizeof @192`.

## Architecture

### Phase 1 — C-API input: `MpOrientedTrackingOptions`

`struct MpOrientedTrackingOptions` in the OBB C header, mirroring `MpTrackingOptions`
field-for-field:
```c
struct MpOrientedTrackingOptions {
  int tracker_type;            // 0=unspecified->no tracking; 1=BOX_TRACKER (rejected
                               // by Create for OBB); 2=BOTSORT. Numerically equal to
                               // the C++/proto enum.
  float track_high_threshold;
  float track_low_threshold;
  float new_track_threshold;
  int track_buffer;
  float match_threshold;
  bool enable_gmc;
};
```
(sizeof 28; offsets 0/4/8/12/16/20/24; enable_gmc bool width 1.) Doc comment notes:
zero-init → no tracking (backward compatible); a BOTSORT caller must set the knobs
explicitly (a zero-init struct yields 0.0 thresholds); the confirmation-threshold-≤-
score_threshold foot-gun. `tracking` field added to `MpOrientedObjectDetectorOptions`
**after `tiling`, before `result_callback`**.

`CppConvertToTrackingOptions` in new `tracking_options_converter.{h,cc}` (OBB C dir):
copies all 7 fields verbatim (`tracker_type` via `static_cast` to
`OrientedObjectDetectorOptions::TrackingOptions::TrackerType`). Folded into BOTH OBB C
libs (incl. the alwayslink `_c_lib`), called in `MpOrientedObjectDetectorCreate` right
after `CppConvertToTilingOptions`.

**ABI shift (lockstep):** inserting `tracking` (sizeof 28) between `tiling` (ends @184)
and `result_callback` moves the trailing offsets — `tracking @184`,
`result_callback @216` (8-byte-aligned, 4 pad bytes), `sizeof @224` (identical numbers
to the YOLO tracking bindings, since the OBB `tiling` block also ends at 184). New
`tracking_options_abi_test.cc` pins `MpOrientedTrackingOptions` (sizeof 28 + offsets +
bool width) and the parent (`tracking@184`, `result_callback@216`, `sizeof@224`); the
existing `tiling_options_abi_test.cc` parent pins are updated (`result_callback
184→216`, `sizeof 192→224`; `tiling@144` unchanged; the relative
`result_callback == tiling + sizeof(tiling)` assert replaced with
`tracking == tiling + sizeof(tiling)`).

**Validation reuse:** the cc Create() gates fire for C callers as a non-`kMpOk`
`MpStatus`. A model-free wiring test asserts `MpOrientedObjectDetectorCreate` rejects
`tracking.tracker_type = 2` (BOTSORT) in IMAGE mode (the validation fires before model
load), asserting the BOTSORT-specific message substring — mirroring the YOLO C-API
wiring test.

### Phase 2 — C-API output: `MpOrientedDetection.track_id`

Add `const char* track_id;` to `struct MpOrientedDetection`
(`c/components/containers/oriented_detection_result.h`), after `rotation` (nullptr when
absent). `CppConvertToOrientedDetection` sets
`out->track_id = in.track_id.has_value() ? strdup(in.track_id->c_str()) : nullptr;`;
`CppCloseOrientedDetection` frees it (`free(const_cast<char*>(in->track_id));
in->track_id = nullptr;`). Mirrors the `category_name` / YOLO `MpDetection.track_id`
ownership idiom. Converter test: track_id round-trips + freed to nullptr; absent →
nullptr.

### Phase 3 — Python input

In `oriented_object_detector.py`: `MpOrientedTrackingOptionsC` ctypes struct
(byte-matching the C struct) + `('tracking', MpOrientedTrackingOptionsC)` on the options
ctypes struct (after `tiling`); a `TrackerType` `enum.IntEnum`
(`UNSPECIFIED = 0, BOTSORT = 2` — **BOX_TRACKER omitted**, since the OBB API should not
surface a value that always errors at Create); a `TrackingOptions` dataclass with the
upstream defaults and **default `tracker_type = TrackerType.UNSPECIFIED`** (no tracking);
`tracking: TrackingOptions = field(default_factory=TrackingOptions)` on the options
dataclass; `_build_oriented_tracking_options_c` marshalling helper; wired into `Create`.

### Phase 4 — Python output

In `oriented_detections_c.py`: add `('track_id', ctypes.c_char_p)` to
`MpOrientedDetectionC` (after `rotation`, matching the C struct order). In
`oriented_detections.py`: add `track_id: Optional[str] = None` to the `OrientedDetection`
dataclass; parse it in `from_ctypes` (`c_obj.track_id.decode('utf-8') if c_obj.track_id
else None`); add it to `__eq__`.

## Error handling / edge cases

- Zero-init C options / default Python options → `tracker_type` 0 → no tracking
  (backward compatible; OBB default unchanged).
- A C/Python BOTSORT misconfiguration (IMAGE / no tiling / >256 classes) is rejected by
  the reused cc Create() validation, surfaced as a non-`kMpOk` status / raised error.
- `MpOrientedDetection.track_id` is nullptr / Python `None` when untracked.
- Memory: the strdup'd `track_id` is freed exactly once in `CppCloseOrientedDetection`
  (per element via `CppCloseOrientedDetectionResult`); `free(nullptr)` is a no-op.

## Testing

- **C-API (runs here):** `tracking_options_converter_test` (the gate; all-fields +
  enum mapping); `tracking_options_abi_test` (compile-time pins); the updated
  `tiling_options_abi_test` pins; an oriented-detection converter test (track_id
  strdup/free + nullptr); a model-free `BotsortInImageModeRejectedThroughBinding` wiring
  test asserting the BOTSORT-specific reject message.
- **Python (static-only here — toolchain broken; see [[build-verifiability-constraints]]):**
  the C++ ABI pins are authoritative; a standalone pure-ctypes check (sizeof 28 +
  offsets for `MpOrientedTrackingOptionsC`; and `track_id` decode) confirms the mirror;
  `py_compile` on the wrapper + containers. Full Python functional tests are authored
  but run only where the toolchain works.

## Scope boundaries (YAGNI)

**In scope:** OBB only · C-API + Python · input `TrackingOptions` + output `track_id` ·
reuse the cc Create() validation. **Out of scope:** any C++/proto/graph change; YOLO
(already done); BOX_TRACKER in the OBB Python enum (rejected upstream); rotation-aware
tracking.

## Verifiability

The C-API converter/ABI/wiring tests build+run here under
`--define MEDIAPIPE_DISABLE_GPU=1`. Python is static-verified (C++ ABI pin + standalone
ctypes + py_compile), consistent with the YOLO Python bindings.
