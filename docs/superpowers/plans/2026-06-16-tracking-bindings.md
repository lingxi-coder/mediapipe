# C-API + Python TrackingOptions Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the YOLO detector's `TrackingOptions` (tracker selection + motion-only BoTSORT knobs) through the C-API and Python, mirroring the shipped tiling-bindings pattern.

**Architecture:** A flat `MpTrackingOptions` C struct + `tracking` field on `MpYoloObjectDetectorOptions` + `CppConvertToTrackingOptions` converter (folded into both libs); compile-time `static_assert` ABI pins (and updates to the existing tiling pins, since the new field shifts trailing offsets); a ctypes `MpTrackingOptionsC` + `TrackingOptions` dataclass + `TrackerType` IntEnum + `_build_tracking_options_c` in the Python wrapper. The cc-layer Create() validation is reused (not re-implemented). C++/proto/graph layers are unchanged.

**Tech Stack:** C++20, Bazel, plain-C ABI structs, ctypes, googletest. Build/test only via `bazel ... --define MEDIAPIPE_DISABLE_GPU=1` on this machine (authoritative; IGNORE clangd "file not found" noise). Python toolchain is broken here → Python is verified by the C++ ABI pin + a standalone `python3` ctypes check + `py_compile`.

**Reference spec:** `docs/superpowers/specs/2026-06-16-tracking-bindings-design.md`

**Standing constraints (every task):** stay on branch `dev`; never branch/merge; all comments English; TDD; commit after each task; commit-message trailer MUST be exactly:
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h` | `MpTrackingOptions` struct + `tracking` field | 1 |
| `.../tracking_options_abi_test.cc` (new) | Pin `MpTrackingOptions` + shifted parent layout | 1 |
| `.../tiling_options_abi_test.cc` (modify) | Update result_callback/sizeof pins (shifted) | 1 |
| `.../tracking_options_converter.{h,cc}` (new) | `CppConvertToTrackingOptions` | 2 |
| `.../tracking_options_converter_test.cc` (new) | Converter unit test | 2 |
| `.../yolo_object_detector.cc` (modify) | Call converter in Create | 2 |
| `.../BUILD` (modify) | Add converter to both libs + new test targets | 2 |
| `.../yolo_object_detector_test.cc` (modify) | Model-free BOTSORT+IMAGE rejection test | 3 |
| `mediapipe/tasks/python/vision/yolo_object_detector.py` | ctypes + IntEnum + dataclass + marshalling | 4 |
| `mediapipe/tasks/python/test/vision/yolo_object_detector_test.py` (modify) | ctypes layout test + functional tests | 4 |

---

## Task 1: C `MpTrackingOptions` struct + ABI pins

**Files:**
- Modify: `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h`
- Create: `mediapipe/tasks/c/vision/yolo_object_detector/tracking_options_abi_test.cc`
- Modify: `mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_abi_test.cc`
- Modify: `mediapipe/tasks/c/vision/yolo_object_detector/BUILD`

- [ ] **Step 1: Add the `MpTrackingOptions` struct + `tracking` field to the header**

In `yolo_object_detector.h`, add this struct immediately AFTER the `struct MpTilingOptions { ... };` definition:

```c
// Tracker selection for the tiled VIDEO/LIVE_STREAM path. Mirrors
// YoloObjectDetectorOptions::TrackingOptions field-for-field.
//
// tracker_type: 0 = unspecified (-> BOX_TRACKER), 1 = BOX_TRACKER (optical-flow,
// default), 2 = BOTSORT (tracking-by-detection, motion-only). Numerically equal
// to the C++/proto enum. A zero-initialized MpTrackingOptions therefore means
// BOX_TRACKER -- byte-identical to a C caller who never set tracking at all.
//
// The threshold/buffer knobs are honored only for BOTSORT. A plain C struct
// cannot distinguish "unset" from 0, so a BOTSORT caller MUST set the knobs
// explicitly (a zero-init struct yields 0.0 thresholds); the C++ struct defaults
// (0.6/0.1/0.7/30/0.7) are not reachable through a zero-init C struct.
// BOX_TRACKER ignores the knobs. (Same convention as score_threshold.)
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

Then in `struct MpYoloObjectDetectorOptions`, add the `tracking` field immediately AFTER the `struct MpTilingOptions tiling;` field and BEFORE the `result_callback_fn result_callback;` typedef/field:

```c
  // Static tiling configuration. Zero-initialized => tiling disabled.
  struct MpTilingOptions tiling;

  // Tracker selection. Zero-initialized => BOX_TRACKER (the existing default).
  struct MpTrackingOptions tracking;
```

- [ ] **Step 2: Write the new ABI test (the layout gate)**

Create `tracking_options_abi_test.cc` (copy the Apache header from `tiling_options_abi_test.cc`):

```cpp
// Pins the byte layout of the YOLO C-API MpTrackingOptions struct and its
// placement in MpYoloObjectDetectorOptions. The Python ctypes MpTrackingOptionsC
// (in mediapipe/tasks/python/vision/yolo_object_detector.py) MUST match these
// offsets; a reorder fails compilation. 64-bit targets (int=4, float=4, bool=1,
// pointer=8, native alignment).

#include <cstddef>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"

static_assert(sizeof(MpTrackingOptions) == 28,
              "MpTrackingOptions layout pinned for the Python ctypes mirror");
static_assert(offsetof(MpTrackingOptions, tracker_type) == 0, "");
static_assert(offsetof(MpTrackingOptions, track_high_threshold) == 4, "");
static_assert(offsetof(MpTrackingOptions, track_low_threshold) == 8, "");
static_assert(offsetof(MpTrackingOptions, new_track_threshold) == 12, "");
static_assert(offsetof(MpTrackingOptions, track_buffer) == 16, "");
static_assert(offsetof(MpTrackingOptions, match_threshold) == 20, "");
static_assert(offsetof(MpTrackingOptions, enable_gmc) == 24, "");
// Pin the bool field's WIDTH (1 byte): a c_bool->c_int swap on the Python side
// would NOT change any offset (tail padding absorbs it), so an offset-only check
// cannot catch it; this size pin can.
static_assert(sizeof(MpTrackingOptions::enable_gmc) == 1,
              "enable_gmc must stay 1 byte (matches Python c_bool)");

// `tracking` sits immediately after `tiling`; `result_callback` follows tracking
// (8-byte aligned -> 4 bytes padding after tracking ends at 212).
static_assert(offsetof(MpYoloObjectDetectorOptions, tracking) ==
                  offsetof(MpYoloObjectDetectorOptions, tiling) +
                      sizeof(MpTilingOptions),
              "tracking must immediately follow tiling");
static_assert(offsetof(MpYoloObjectDetectorOptions, tracking) == 184,
              "tracking offset pinned for the Python ctypes mirror");
static_assert(offsetof(MpYoloObjectDetectorOptions, result_callback) == 216, "");
static_assert(sizeof(MpYoloObjectDetectorOptions) == 224, "");

namespace {
TEST(TrackingOptionsAbiTest, LayoutPinned) { SUCCEED(); }
}  // namespace
```

- [ ] **Step 3: Update the existing tiling ABI test (offsets shifted by the insert)**

In `tiling_options_abi_test.cc`, the `tracking` insert moved `result_callback` and `sizeof`, and `result_callback` no longer immediately follows `tiling`. Make these three edits:

REPLACE this assert:
```cpp
static_assert(offsetof(MpYoloObjectDetectorOptions, result_callback) ==
                  offsetof(MpYoloObjectDetectorOptions, tiling) +
                      sizeof(MpTilingOptions),
              "result_callback must immediately follow tiling");
```
WITH (tracking now sits between tiling and result_callback):
```cpp
static_assert(offsetof(MpYoloObjectDetectorOptions, tracking) ==
                  offsetof(MpYoloObjectDetectorOptions, tiling) +
                      sizeof(MpTilingOptions),
              "tracking must immediately follow tiling");
```
CHANGE the result_callback absolute pin from `== 184` to `== 216`:
```cpp
static_assert(offsetof(MpYoloObjectDetectorOptions, result_callback) == 216, "");
```
CHANGE the sizeof pin from `== 192` to `== 224`:
```cpp
static_assert(sizeof(MpYoloObjectDetectorOptions) == 224, "");
```
(Leave `tiling == 136` unchanged — tiling's offset did not move.)

- [ ] **Step 4: Add the ABI test target to BUILD**

In `mediapipe/tasks/c/vision/yolo_object_detector/BUILD`, after the `tiling_options_abi_test` cc_test, add:
```python
cc_test(
    name = "tracking_options_abi_test",
    srcs = ["tracking_options_abi_test.cc"],
    deps = [
        ":yolo_object_detector_lib",
        "//mediapipe/framework/port:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
```

- [ ] **Step 5: Build both ABI tests (compile = pins hold)**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/tasks/c/vision/yolo_object_detector:tracking_options_abi_test \
  //mediapipe/tasks/c/vision/yolo_object_detector:tiling_options_abi_test \
  --test_output=errors
```
Expected: both PASS. If a static_assert FAILS, the actual offset is in the error — that is the source of truth; fix the pin to the real value ONLY if your reasoning was off (e.g. alignment), and note it. (On any standard 64-bit target the values above are correct.)

- [ ] **Step 6: Commit**

```bash
git add mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h \
        mediapipe/tasks/c/vision/yolo_object_detector/tracking_options_abi_test.cc \
        mediapipe/tasks/c/vision/yolo_object_detector/tiling_options_abi_test.cc \
        mediapipe/tasks/c/vision/yolo_object_detector/BUILD
git commit -m "feat(tracking): MpTrackingOptions C struct + ABI pins

Adds MpTrackingOptions (mirrors YoloObjectDetectorOptions::TrackingOptions) and a
tracking field on MpYoloObjectDetectorOptions (after tiling). New ABI test pins
sizeof 28 + offsets; the tracking insert shifts result_callback 184->216 and
sizeof 192->224, so the tiling ABI test pins are updated in lockstep.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: C `CppConvertToTrackingOptions` converter + wire into Create

**Files:**
- Create: `mediapipe/tasks/c/vision/yolo_object_detector/tracking_options_converter.{h,cc}`
- Create: `mediapipe/tasks/c/vision/yolo_object_detector/tracking_options_converter_test.cc`
- Modify: `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.cc`
- Modify: `mediapipe/tasks/c/vision/yolo_object_detector/BUILD`

- [ ] **Step 1: Write the failing converter test**

Create `tracking_options_converter_test.cc` (Apache header), mirroring `tiling_options_converter_test.cc`'s structure:

```cpp
#include "mediapipe/tasks/c/vision/yolo_object_detector/tracking_options_converter.h"

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

namespace mediapipe::tasks::c::vision::yolo_object_detector {
namespace {

using CppTrackingOptions = ::mediapipe::tasks::vision::yolo_object_detector::
    YoloObjectDetectorOptions::TrackingOptions;

TEST(TrackingOptionsConverterTest, CopiesAllScalarFields) {
  MpTrackingOptions in = {};
  in.tracker_type = 2;  // BOTSORT
  in.track_high_threshold = 0.55f;
  in.track_low_threshold = 0.15f;
  in.new_track_threshold = 0.65f;
  in.track_buffer = 25;
  in.match_threshold = 0.75f;
  in.enable_gmc = true;

  CppTrackingOptions out;
  CppConvertToTrackingOptions(in, &out);

  EXPECT_EQ(out.tracker_type, CppTrackingOptions::kBotsort);
  EXPECT_FLOAT_EQ(out.track_high_threshold, 0.55f);
  EXPECT_FLOAT_EQ(out.track_low_threshold, 0.15f);
  EXPECT_FLOAT_EQ(out.new_track_threshold, 0.65f);
  EXPECT_EQ(out.track_buffer, 25);
  EXPECT_FLOAT_EQ(out.match_threshold, 0.75f);
  EXPECT_TRUE(out.enable_gmc);
}

TEST(TrackingOptionsConverterTest, MapsTrackerTypeEnumValues) {
  CppTrackingOptions out;

  MpTrackingOptions box = {};
  box.tracker_type = 1;  // BOX_TRACKER
  CppConvertToTrackingOptions(box, &out);
  EXPECT_EQ(out.tracker_type, CppTrackingOptions::kBoxTracker);

  MpTrackingOptions unspecified = {};
  unspecified.tracker_type = 0;  // -> treated as BOX_TRACKER downstream
  CppConvertToTrackingOptions(unspecified, &out);
  EXPECT_EQ(static_cast<int>(out.tracker_type), 0);
}

}  // namespace
}  // namespace mediapipe::tasks::c::vision::yolo_object_detector
```

- [ ] **Step 2: Write the converter header**

Create `tracking_options_converter.h` (Apache header):

```cpp
#ifndef MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_TRACKING_OPTIONS_CONVERTER_H_
#define MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_TRACKING_OPTIONS_CONVERTER_H_

#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

namespace mediapipe::tasks::c::vision::yolo_object_detector {

// Copies a plain-C MpTrackingOptions into the C++ TrackingOptions sub-struct.
// All 7 fields are copied 1:1 (tracker_type via static_cast to the C++ enum).
// Declared here so unit tests can verify the C->C++ mapping without a live graph.
void CppConvertToTrackingOptions(
    const MpTrackingOptions& in,
    ::mediapipe::tasks::vision::yolo_object_detector::
        YoloObjectDetectorOptions::TrackingOptions* out);

}  // namespace mediapipe::tasks::c::vision::yolo_object_detector

#endif  // MEDIAPIPE_TASKS_C_VISION_YOLO_OBJECT_DETECTOR_TRACKING_OPTIONS_CONVERTER_H_
```

- [ ] **Step 3: Write the converter implementation**

Create `tracking_options_converter.cc` (Apache header):

```cpp
#include "mediapipe/tasks/c/vision/yolo_object_detector/tracking_options_converter.h"

#include "mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

namespace mediapipe::tasks::c::vision::yolo_object_detector {

namespace YoloNs = ::mediapipe::tasks::vision::yolo_object_detector;

void CppConvertToTrackingOptions(
    const MpTrackingOptions& in,
    YoloNs::YoloObjectDetectorOptions::TrackingOptions* out) {
  // tracker_type is copied verbatim: 0/1/2 are numerically equal to the C++
  // enum (kBoxTracker=1, kBotsort=2; 0 means unspecified -> BOX_TRACKER, which
  // the cc layer treats identically to the default). The knobs are copied 1:1;
  // they are honored only for BOTSORT (validated/consumed in the cc layer).
  out->tracker_type =
      static_cast<YoloNs::YoloObjectDetectorOptions::TrackingOptions::
                      TrackerType>(in.tracker_type);
  out->track_high_threshold = in.track_high_threshold;
  out->track_low_threshold = in.track_low_threshold;
  out->new_track_threshold = in.new_track_threshold;
  out->track_buffer = in.track_buffer;
  out->match_threshold = in.match_threshold;
  out->enable_gmc = in.enable_gmc;
}

}  // namespace mediapipe::tasks::c::vision::yolo_object_detector
```

- [ ] **Step 4: Call the converter in Create**

In `yolo_object_detector.cc`, add the include near the existing tiling converter include:
```cpp
#include "mediapipe/tasks/c/vision/yolo_object_detector/tracking_options_converter.h"
```
And immediately AFTER the existing line `CppConvertToTilingOptions(in.tiling, &out->tiling);`, add:
```cpp
  CppConvertToTrackingOptions(in.tracking, &out->tracking);
```

- [ ] **Step 5: Wire the converter into BUILD (both libs + new test target)**

In `BUILD`, add `"tracking_options_converter.cc"` to `srcs` and `"tracking_options_converter.h"` to `hdrs` of BOTH `yolo_object_detector_lib` AND `yolo_object_detector_c_lib` (the alwayslink lib). Then add the converter test target after `tiling_options_converter_test`:
```python
cc_test(
    name = "tracking_options_converter_test",
    srcs = ["tracking_options_converter_test.cc"],
    deps = [
        ":yolo_object_detector_lib",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/vision/yolo_object_detector",
        "@com_google_googletest//:gtest_main",
    ],
)
```

- [ ] **Step 6: Run the converter test + build the C lib**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:tracking_options_converter_test --test_output=errors
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_c_lib
```
Expected: test PASSES (2 cases); lib builds. (Before Steps 2-4 the test fails to compile — that's the RED.)

- [ ] **Step 7: Commit**

```bash
git add mediapipe/tasks/c/vision/yolo_object_detector/tracking_options_converter.h \
        mediapipe/tasks/c/vision/yolo_object_detector/tracking_options_converter.cc \
        mediapipe/tasks/c/vision/yolo_object_detector/tracking_options_converter_test.cc \
        mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.cc \
        mediapipe/tasks/c/vision/yolo_object_detector/BUILD
git commit -m "feat(tracking): CppConvertToTrackingOptions + wire into C-API Create

Copies MpTrackingOptions into the C++ TrackingOptions sub-struct (tracker_type
via static_cast, knobs 1:1) and calls it in MpYoloObjectDetectorCreate after the
tiling converter. Folded into both libs (incl. alwayslink _c_lib) to avoid a dep
cycle, mirroring the tiling converter. Converter unit test included.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: C-API model-free wiring test (BOTSORT + IMAGE rejected)

**Files:**
- Modify: `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector_test.cc`

- [ ] **Step 1: Add the model-free rejection test**

First READ the existing `MotionSchedulingInImageModeRejectedThroughBinding` test (around line 187) to match its exact idiom (zero-init options, set the field, call `MpYoloObjectDetectorCreate` with an `error_msg`, assert non-`kMpOk`, free the error). Add this test right after it:

```cpp
// BOTSORT selection is rejected in IMAGE mode by the C++ Create(), WITHOUT a
// model fixture: the tracking validation fires before model load. This proves
// the new `tracking` field is wired through the binding into Create.
TEST(YoloObjectDetectorCApiTest, BotsortInImageModeRejectedThroughBinding) {
  MpYoloObjectDetectorOptions options = {};
  options.running_mode = MpRunningMode::MP_RUNNING_MODE_IMAGE;
  options.num_classes = 80;
  options.tracking.tracker_type = 2;  // BOTSORT

  MpYoloObjectDetectorPtr detector = nullptr;
  char* error_msg = nullptr;
  const MpStatus status =
      MpYoloObjectDetectorCreate(&options, &detector, &error_msg);

  EXPECT_NE(status, kMpOk);
  if (error_msg) free(error_msg);
}
```
Match the EXACT free/cleanup idiom of the sibling test (it may use `free(error_msg)` or a helper — copy what it does). Confirm `MpYoloObjectDetectorPtr`/`MpStatus`/`kMpOk` are the names used in the file.

- [ ] **Step 2: Run it**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_test --test_output=errors --test_filter='*BotsortInImageMode*'
```
Expected: PASS (runs model-free, like the sibling). Then run the whole C-API test target to confirm no regression:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_test --test_output=errors
```
Expected: PASS (model-gated cases SKIP if the fixture is absent).

- [ ] **Step 3: Commit**

```bash
git add mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector_test.cc
git commit -m "test(tracking): C-API rejects BOTSORT in IMAGE mode (model-free)

Proves the tracking field is wired through MpYoloObjectDetectorCreate: a BOTSORT
selection in IMAGE mode is rejected by the cc validation before model load,
mirroring the motion-scheduling binding test.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Python ctypes + dataclass + marshalling

**Files:**
- Modify: `mediapipe/tasks/python/vision/yolo_object_detector.py`
- Modify: `mediapipe/tasks/python/test/vision/yolo_object_detector_test.py`

- [ ] **Step 1: Add the `MpTrackingOptionsC` ctypes struct + parent field**

In `yolo_object_detector.py`, after the `MpTilingOptionsC` class, add:

```python
class MpTrackingOptionsC(ctypes.Structure):
  """Byte-matches struct MpTrackingOptions in the YOLO C header.

  Layout pinned by tracking_options_abi_test.cc (sizeof 28; offsets
  0/4/8/12/16/20/24; enable_gmc is c_bool = 1 byte). tracker_type is c_int with
  values 0=unspecified->BOX_TRACKER, 1=BOX_TRACKER, 2=BOTSORT.
  """

  _fields_ = [
      ('tracker_type', ctypes.c_int),
      ('track_high_threshold', ctypes.c_float),
      ('track_low_threshold', ctypes.c_float),
      ('new_track_threshold', ctypes.c_float),
      ('track_buffer', ctypes.c_int),
      ('match_threshold', ctypes.c_float),
      ('enable_gmc', ctypes.c_bool),
  ]
```

Then in `MpYoloObjectDetectorOptionsC._fields_`, add `('tracking', MpTrackingOptionsC)` immediately AFTER the `('tiling', MpTilingOptionsC),` line and before `('result_callback', _C_TYPES_RESULT_CALLBACK)`:

```python
      ('tiling', MpTilingOptionsC),
      ('tracking', MpTrackingOptionsC),
      ('result_callback', _C_TYPES_RESULT_CALLBACK),
```

- [ ] **Step 2: Add the `TrackerType` IntEnum + `TrackingOptions` dataclass + options field**

After the existing `class Layout(enum.IntEnum):` (and near the `TilingOptions` dataclass), add the enum:

```python
class TrackerType(enum.IntEnum):
  """Tracker selection for the tiled VIDEO/LIVE_STREAM path.

  Values are numerically equal to the C++/proto TrackerType enum.
  """

  BOX_TRACKER = 1  # optical-flow propagation (default)
  BOTSORT = 2  # tracking-by-detection, motion-only
```

And add the dataclass (place it next to `TilingOptions`):

```python
@dataclasses.dataclass
class TrackingOptions:
  """Tracker selection for the YOLO object detector (tiled stream path).

  Mirrors the C++ YoloObjectDetectorOptions.TrackingOptions. Honored only when
  tiling is enabled and running mode is not IMAGE; otherwise ignored. BOTSORT is
  motion-only (no ReID). The knobs are used only for BOTSORT.
  """

  tracker_type: TrackerType = TrackerType.BOX_TRACKER
  track_high_threshold: float = 0.6
  track_low_threshold: float = 0.1
  new_track_threshold: float = 0.7
  track_buffer: int = 30
  match_threshold: float = 0.7
  enable_gmc: bool = False
```

In the `YoloObjectDetectorOptions` dataclass, add the field immediately AFTER `tiling: TilingOptions = dataclasses.field(default_factory=TilingOptions)`:

```python
  tracking: TrackingOptions = dataclasses.field(default_factory=TrackingOptions)
```

- [ ] **Step 3: Add the marshalling helper + wire into Create**

After `_build_tiling_options_c`, add:

```python
def _build_tracking_options_c(
    tracking: TrackingOptions,
) -> 'MpTrackingOptionsC':
  """Builds the ctypes MpTrackingOptionsC from a TrackingOptions dataclass.

  No pointer fields, so (unlike tiling) there is no backing array to keep alive.
  """
  return MpTrackingOptionsC(
      tracker_type=int(tracking.tracker_type),
      track_high_threshold=tracking.track_high_threshold,
      track_low_threshold=tracking.track_low_threshold,
      new_track_threshold=tracking.new_track_threshold,
      track_buffer=tracking.track_buffer,
      match_threshold=tracking.match_threshold,
      enable_gmc=tracking.enable_gmc,
  )
```

In `Create`, after the `tiling_c, tiles_keepalive = _build_tiling_options_c(options.tiling)` line, add:
```python
    tracking_c = _build_tracking_options_c(options.tracking)
```
And in the `MpYoloObjectDetectorOptionsC(...)` constructor call, add `tracking=tracking_c,` immediately after the `tiling=tiling_c,` argument.

- [ ] **Step 4: Add the ctypes layout test (runs where the toolchain works)**

In `mediapipe/tasks/python/test/vision/yolo_object_detector_test.py`, find the existing tiling ctypes-layout test (search for `test_ctypes_tiling_layout` or a `sizeof`/`192` assertion) and add an analogous test right after it:

```python
  def test_ctypes_tracking_layout_matches_c_abi(self):
    # Mirrors tracking_options_abi_test.cc.
    self.assertEqual(ctypes.sizeof(_yolo.MpTrackingOptionsC), 28)
    fields = {f[0]: getattr(_yolo.MpTrackingOptionsC, f[0]).offset
              for f in _yolo.MpTrackingOptionsC._fields_}
    self.assertEqual(fields['tracker_type'], 0)
    self.assertEqual(fields['track_high_threshold'], 4)
    self.assertEqual(fields['track_low_threshold'], 8)
    self.assertEqual(fields['new_track_threshold'], 12)
    self.assertEqual(fields['track_buffer'], 16)
    self.assertEqual(fields['match_threshold'], 20)
    self.assertEqual(fields['enable_gmc'], 24)
    # Parent placement shifted by the tracking insert.
    self.assertEqual(
        _yolo.MpYoloObjectDetectorOptionsC.tracking.offset, 184)
    self.assertEqual(
        _yolo.MpYoloObjectDetectorOptionsC.result_callback.offset, 216)
    self.assertEqual(ctypes.sizeof(_yolo.MpYoloObjectDetectorOptionsC), 224)
```
Match the module alias the test file already uses for the wrapper (shown as `_yolo` here — use the file's actual import alias). Also add a small functional test if the file has a pattern for it: construct `YoloObjectDetectorOptions(..., tracking=TrackingOptions(tracker_type=TrackerType.BOTSORT))` and assert the dataclass round-trips and `_build_tracking_options_c` produces a struct with `tracker_type == 2` (this needs no model). Match the file's existing test style.

- [ ] **Step 5: Verify on this machine (Python toolchain is broken → static checks)**

The C++ ABI pin (Task 1) is the authoritative layout guarantee. Cross-check the ctypes mirror with a standalone pure-ctypes run (no mediapipe import, so it works here):
```bash
python3 - <<'EOF'
import ctypes
class MpTrackingOptionsC(ctypes.Structure):
    _fields_ = [
        ('tracker_type', ctypes.c_int),
        ('track_high_threshold', ctypes.c_float),
        ('track_low_threshold', ctypes.c_float),
        ('new_track_threshold', ctypes.c_float),
        ('track_buffer', ctypes.c_int),
        ('match_threshold', ctypes.c_float),
        ('enable_gmc', ctypes.c_bool),
    ]
assert ctypes.sizeof(MpTrackingOptionsC) == 28, ctypes.sizeof(MpTrackingOptionsC)
exp = {'tracker_type':0,'track_high_threshold':4,'track_low_threshold':8,
       'new_track_threshold':12,'track_buffer':16,'match_threshold':20,'enable_gmc':24}
for n,o in exp.items():
    assert getattr(MpTrackingOptionsC, n).offset == o, (n, getattr(MpTrackingOptionsC,n).offset)
print('MpTrackingOptionsC OK: sizeof=28, offsets pinned, enable_gmc@24')
EOF
```
Expected: prints the OK line. This `_fields_` MUST be character-identical to the wrapper's `MpTrackingOptionsC` — diff them by eye.

Then byte-check the wrapper compiles:
```bash
python3 -m py_compile mediapipe/tasks/python/vision/yolo_object_detector.py && echo "py_compile OK"
python3 -m py_compile mediapipe/tasks/python/test/vision/yolo_object_detector_test.py && echo "test py_compile OK"
```
Expected: both print OK. (The full Python functional + ctypes-layout tests run only where the Python/bazel toolchain works; note that in the report — do NOT attempt `bazel test` of the Python target here, it is known-broken.)

- [ ] **Step 6: Commit**

```bash
git add mediapipe/tasks/python/vision/yolo_object_detector.py \
        mediapipe/tasks/python/test/vision/yolo_object_detector_test.py
git commit -m "feat(tracking): Python TrackingOptions bindings (ctypes + dataclass)

Adds MpTrackingOptionsC (byte-matching the C struct), a TrackerType IntEnum, a
TrackingOptions dataclass with upstream defaults, _build_tracking_options_c, and
wires tracking into Create. ctypes layout test mirrors the C++ ABI pin; verified
here via a standalone ctypes check (sizeof 28, offsets) + py_compile (Python
toolchain otherwise unavailable on this machine).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- C `MpTrackingOptions` struct + `tracking` field → Task 1 ✓
- ABI pins (new + updated tiling pins for the offset shift) → Task 1 ✓
- `CppConvertToTrackingOptions` + folded into both libs + called in Create → Task 2 ✓
- Converter unit test → Task 2 ✓
- C-API model-free wiring test (BOTSORT+IMAGE rejected, reusing cc validation) → Task 3 ✓
- Python ctypes struct + parent field + TrackerType IntEnum + TrackingOptions dataclass + `_build_tracking_options_c` + Create wiring → Task 4 ✓
- Python ctypes layout test + standalone check + py_compile → Task 4 ✓
- YOLO only, no C++/proto/graph change, validation reused → all tasks respect this ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code; bazel/python commands have expected output.

**Type consistency:** `MpTrackingOptions` C fields (tracker_type/track_high_threshold/track_low_threshold/new_track_threshold/track_buffer/match_threshold/enable_gmc) are identical across the header (T1), ABI test (T1), converter (T2), converter test (T2), ctypes struct (T4), and the standalone check (T4). Enum values 1/2 match `CppTrackingOptions::kBoxTracker/kBotsort` (converter test) and `TrackerType.BOX_TRACKER/BOTSORT` (Python). ABI numbers consistent everywhere: MpTrackingOptions sizeof 28, offsets 0/4/8/12/16/20/24; parent tracking@184, result_callback@216, sizeof 224.

**Known verification-time notes flagged in-plan:** the exact module import alias in the Python test file (T4 S4); the sibling test's error-free cleanup idiom (T3 S1); Python full tests run only where the toolchain works (T4 S5).
