# OBB Tracking C-API + Python Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the OBB `TrackingOptions` (input) and the oriented result's `track_id` (output) through the C-API and Python, completing the OBB tracking arc.

**Architecture:** Mirror the shipped YOLO tracking bindings + track-id bindings, applied to OBB's separate `MpOriented*` / oriented-result types: a flat `MpOrientedTrackingOptions` struct + converter + ABI pins; `MpOrientedDetection.track_id` (strdup/free); ctypes mirrors + a `TrackerType` IntEnum + dataclass fields. The cc-layer Create() validation is reused. C++/proto/graph unchanged.

**Tech Stack:** plain-C ABI structs, Bazel, ctypes, googletest. Build/test only via `bazel ... --define MEDIAPIPE_DISABLE_GPU=1` (authoritative; IGNORE clangd noise). Python toolchain broken here → Python verified by the C++ ABI pin + a standalone `python3` ctypes check + `py_compile`.

**Reference spec:** `docs/superpowers/specs/2026-06-16-obb-tracking-bindings-design.md`

**Standing constraints (every task):** stay on branch `dev`; never branch/merge; all comments English; TDD; commit after each task; trailer MUST be exactly:
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h` | `MpOrientedTrackingOptions` + `tracking` field | 1 |
| `.../oriented_object_detector/tracking_options_abi_test.cc` (new) | Pin struct + shifted parent layout | 1 |
| `.../oriented_object_detector/tiling_options_abi_test.cc` (modify) | Update shifted parent pins | 1 |
| `.../oriented_object_detector/tracking_options_converter.{h,cc}` (new) | `CppConvertToTrackingOptions` | 2 |
| `.../oriented_object_detector/tracking_options_converter_test.cc` (new) | Converter unit test | 2 |
| `.../oriented_object_detector/oriented_object_detector.cc` (modify) | Call converter in Create | 2 |
| `.../oriented_object_detector/BUILD` (modify) | Converter in both libs + test targets | 1,2 |
| `.../oriented_object_detector/oriented_object_detector_test.cc` (modify) | Model-free wiring test | 3 |
| `mediapipe/tasks/c/components/containers/oriented_detection_result.h` + `oriented_detection_result_converter.cc` (+ test) | `MpOrientedDetection.track_id` strdup/free | 4 |
| `mediapipe/tasks/python/vision/oriented_object_detector.py` | ctypes + IntEnum + dataclass + marshalling | 5 |
| `mediapipe/tasks/python/components/containers/oriented_detections_c.py` + `oriented_detections.py` (+ test) | ctypes track_id + dataclass | 6 |

---

## Task 1: C `MpOrientedTrackingOptions` struct + ABI pins

**Files:**
- Modify: `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h`
- Create: `.../oriented_object_detector/tracking_options_abi_test.cc`
- Modify: `.../oriented_object_detector/tiling_options_abi_test.cc`
- Modify: `.../oriented_object_detector/BUILD`

The OBB options parent layout (verified): `tiling @144` (sizeof `MpOrientedTilingOptions` = 40, ends @184), `result_callback @184`, `sizeof @192`. Inserting `tracking` (sizeof 28) shifts: `tracking @184`, `result_callback @216` (8-byte aligned → 4 pad bytes), `sizeof @224`. Distinct name `MpOrientedTrackingOptions` (NOT `MpTrackingOptions`) to avoid an extern-"C" ODR collision with YOLO's struct, matching how `MpOrientedTilingOptions` is named.

- [ ] **Step 1: Add the struct + field to the header**

In `oriented_object_detector.h`, add immediately AFTER `struct MpOrientedTilingOptions { ... };`:
```c
// Tracker selection for the OBB tiled VIDEO/LIVE_STREAM path. Mirrors
// OrientedObjectDetectorOptions::TrackingOptions field-for-field.
//
// tracker_type: 0 = unspecified (-> no tracking, the OBB default), 1 =
// BOX_TRACKER (NOT supported for OBB -- rejected by Create()), 2 = BOTSORT
// (tracking-by-detection, motion-only). Numerically equal to the C++/proto enum.
// A zero-initialized MpOrientedTrackingOptions therefore means NO tracking --
// byte-identical to a C caller who never set tracking at all.
//
// The threshold/buffer knobs are honored only for BOTSORT. A plain C struct
// cannot distinguish "unset" from 0, so a BOTSORT caller MUST set them
// explicitly. BoTSORT only emits a track_id for CONFIRMED tracks, so
// track_high_threshold / new_track_threshold must be at/below the detector's
// score_threshold or no track_id is produced.
struct MpOrientedTrackingOptions {
  int tracker_type;
  float track_high_threshold;
  float track_low_threshold;
  float new_track_threshold;
  int track_buffer;
  float match_threshold;
  bool enable_gmc;
};
```
In `struct MpOrientedObjectDetectorOptions`, add the `tracking` field immediately AFTER `struct MpOrientedTilingOptions tiling;` and BEFORE the `result_callback` typedef/field:
```c
  // Tracker selection. Zero-initialized => no tracking (the OBB default).
  struct MpOrientedTrackingOptions tracking;
```

- [ ] **Step 2: Write the new ABI test**

Create `tracking_options_abi_test.cc` (copy the Apache header + the explanatory comment style from `tiling_options_abi_test.cc`):
```cpp
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
static_assert(sizeof(MpOrientedTrackingOptions::enable_gmc) == 1,
              "enable_gmc must stay 1 byte (matches Python c_bool)");
static_assert(offsetof(MpOrientedObjectDetectorOptions, tracking) ==
                  offsetof(MpOrientedObjectDetectorOptions, tiling) +
                      sizeof(MpOrientedTilingOptions),
              "tracking must immediately follow tiling");
static_assert(offsetof(MpOrientedObjectDetectorOptions, tracking) == 184,
              "tracking offset pinned for the Python ctypes mirror");
// result_callback is 8-byte aligned; tracking ends at 212 (184+28), so 4 pad
// bytes precede result_callback at 216.
static_assert(offsetof(MpOrientedObjectDetectorOptions, result_callback) == 216, "");
static_assert(sizeof(MpOrientedObjectDetectorOptions) == 224, "");

namespace {
TEST(OrientedTrackingOptionsAbiTest, LayoutPinned) { SUCCEED(); }
}  // namespace
```
Add a top comment block (mirror the tiling ABI test) noting the Python ctypes mirror in `oriented_object_detector.py` must match, and that the parent pins are intentionally duplicated in the sibling tiling ABI test.

- [ ] **Step 3: Update the existing tiling ABI test (shifted offsets)**

In `tiling_options_abi_test.cc`:
- REPLACE the assert `offsetof(..., result_callback) == offsetof(..., tiling) + sizeof(MpOrientedTilingOptions)` (msg "result_callback must immediately follow tiling") WITH the tracking analog: `offsetof(..., tracking) == offsetof(..., tiling) + sizeof(MpOrientedTilingOptions)` (msg "tracking must immediately follow tiling").
- CHANGE `offsetof(MpOrientedObjectDetectorOptions, result_callback) == 184` to `== 216`.
- CHANGE `sizeof(MpOrientedObjectDetectorOptions) == 192` to `== 224`.
- LEAVE `tiling == 144` unchanged.

- [ ] **Step 4: Add the ABI test target to BUILD**

In `mediapipe/tasks/c/vision/oriented_object_detector/BUILD`, after the `tiling_options_abi_test` cc_test, add a `tracking_options_abi_test` cc_test with the SAME deps as `tiling_options_abi_test`.

- [ ] **Step 5: Build both ABI tests**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/tasks/c/vision/oriented_object_detector:tracking_options_abi_test \
  //mediapipe/tasks/c/vision/oriented_object_detector:tiling_options_abi_test \
  --test_output=errors
```
Expected: both PASS. If a static_assert prints a different actual offset, that's ground truth — only adjust a pin if your arithmetic was off (it's correct for 64-bit), and report.

- [ ] **Step 6: Commit**

```bash
git add mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h \
        mediapipe/tasks/c/vision/oriented_object_detector/tracking_options_abi_test.cc \
        mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_abi_test.cc \
        mediapipe/tasks/c/vision/oriented_object_detector/BUILD
git commit -m "feat(obb-tracking): MpOrientedTrackingOptions C struct + ABI pins

Adds MpOrientedTrackingOptions (mirrors OrientedObjectDetectorOptions::
TrackingOptions) and a tracking field on MpOrientedObjectDetectorOptions (after
tiling). New ABI test pins sizeof 28 + offsets; the insert shifts result_callback
184->216 and sizeof 192->224, so the tiling ABI pins are updated in lockstep.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: C `CppConvertToTrackingOptions` + wire into Create

**Files:**
- Create: `.../oriented_object_detector/tracking_options_converter.{h,cc}`
- Create: `.../oriented_object_detector/tracking_options_converter_test.cc`
- Modify: `.../oriented_object_detector/oriented_object_detector.cc`
- Modify: `.../oriented_object_detector/BUILD`

The C++ enum is `::mediapipe::tasks::vision::oriented_object_detector::OrientedObjectDetectorOptions::TrackingOptions::{kTrackerUnspecified, kBoxTracker, kBotsort}`. READ the OBB `tiling_options_converter.{h,cc}` for the exact namespace/alias/structure to mirror.

- [ ] **Step 1: Write the failing converter test**

Create `tracking_options_converter_test.cc` (Apache header), mirroring `tiling_options_converter_test.cc`:
```cpp
#include "mediapipe/tasks/c/vision/oriented_object_detector/tracking_options_converter.h"

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"
#include "mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h"

namespace mediapipe::tasks::c::vision::oriented_object_detector {
namespace {

using CppTrackingOptions = ::mediapipe::tasks::vision::oriented_object_detector::
    OrientedObjectDetectorOptions::TrackingOptions;

TEST(OrientedTrackingOptionsConverterTest, CopiesAllScalarFields) {
  MpOrientedTrackingOptions in = {};
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

TEST(OrientedTrackingOptionsConverterTest, MapsTrackerTypeEnumValues) {
  CppTrackingOptions out;
  MpOrientedTrackingOptions box = {};
  box.tracker_type = 1;  // BOX_TRACKER
  CppConvertToTrackingOptions(box, &out);
  EXPECT_EQ(out.tracker_type, CppTrackingOptions::kBoxTracker);

  MpOrientedTrackingOptions uns = {};
  uns.tracker_type = 0;  // unspecified -> no tracking
  CppConvertToTrackingOptions(uns, &out);
  EXPECT_EQ(static_cast<int>(out.tracker_type), 0);
}

}  // namespace
}  // namespace mediapipe::tasks::c::vision::oriented_object_detector
```

- [ ] **Step 2: Converter header**

Create `tracking_options_converter.h` (Apache header, mirror tiling_options_converter.h's guard/namespace/doc):
```cpp
#ifndef MEDIAPIPE_TASKS_C_VISION_ORIENTED_OBJECT_DETECTOR_TRACKING_OPTIONS_CONVERTER_H_
#define MEDIAPIPE_TASKS_C_VISION_ORIENTED_OBJECT_DETECTOR_TRACKING_OPTIONS_CONVERTER_H_

#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"
#include "mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h"

namespace mediapipe::tasks::c::vision::oriented_object_detector {

// Copies a plain-C MpOrientedTrackingOptions into the C++ TrackingOptions
// sub-struct. All 7 fields are copied 1:1 (tracker_type via static_cast).
void CppConvertToTrackingOptions(
    const MpOrientedTrackingOptions& in,
    ::mediapipe::tasks::vision::oriented_object_detector::
        OrientedObjectDetectorOptions::TrackingOptions* out);

}  // namespace mediapipe::tasks::c::vision::oriented_object_detector

#endif  // MEDIAPIPE_TASKS_C_VISION_ORIENTED_OBJECT_DETECTOR_TRACKING_OPTIONS_CONVERTER_H_
```

- [ ] **Step 3: Converter implementation**

Create `tracking_options_converter.cc` (Apache header):
```cpp
#include "mediapipe/tasks/c/vision/oriented_object_detector/tracking_options_converter.h"

#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"
#include "mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h"

namespace mediapipe::tasks::c::vision::oriented_object_detector {

namespace ObbNs = ::mediapipe::tasks::vision::oriented_object_detector;

void CppConvertToTrackingOptions(
    const MpOrientedTrackingOptions& in,
    ObbNs::OrientedObjectDetectorOptions::TrackingOptions* out) {
  // tracker_type copied verbatim: 0/1/2 == unspecified/kBoxTracker/kBotsort.
  // Knobs copied 1:1; honored only for BOTSORT. BOX_TRACKER is rejected at the
  // cc Create(), not here.
  out->tracker_type =
      static_cast<ObbNs::OrientedObjectDetectorOptions::TrackingOptions::
                      TrackerType>(in.tracker_type);
  out->track_high_threshold = in.track_high_threshold;
  out->track_low_threshold = in.track_low_threshold;
  out->new_track_threshold = in.new_track_threshold;
  out->track_buffer = in.track_buffer;
  out->match_threshold = in.match_threshold;
  out->enable_gmc = in.enable_gmc;
}

}  // namespace mediapipe::tasks::c::vision::oriented_object_detector
```

- [ ] **Step 4: Call in Create**

In `oriented_object_detector.cc`, add the include near the tiling converter include:
```cpp
#include "mediapipe/tasks/c/vision/oriented_object_detector/tracking_options_converter.h"
```
After the existing `CppConvertToTilingOptions(in.tiling, &out->tiling);`, add:
```cpp
  CppConvertToTrackingOptions(in.tracking, &out->tracking);
```

- [ ] **Step 5: BUILD — both libs + test target**

In `BUILD`, add `"tracking_options_converter.cc"`/`.h` to `srcs`/`hdrs` of BOTH `oriented_object_detector_lib` AND `oriented_object_detector_c_lib` (alwayslink). Add a `tracking_options_converter_test` cc_test with the SAME deps as `tiling_options_converter_test`.

- [ ] **Step 6: Run + build**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:tracking_options_converter_test --test_output=errors
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_c_lib
```
Expected: test PASSES (2 cases); lib builds.

- [ ] **Step 7: Commit**

```bash
git add mediapipe/tasks/c/vision/oriented_object_detector/tracking_options_converter.h \
        mediapipe/tasks/c/vision/oriented_object_detector/tracking_options_converter.cc \
        mediapipe/tasks/c/vision/oriented_object_detector/tracking_options_converter_test.cc \
        mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.cc \
        mediapipe/tasks/c/vision/oriented_object_detector/BUILD
git commit -m "feat(obb-tracking): CppConvertToTrackingOptions + wire into OBB C-API Create

Copies MpOrientedTrackingOptions into the C++ TrackingOptions sub-struct and
calls it in MpOrientedObjectDetectorCreate after the tiling converter. Folded
into both libs (incl. alwayslink _c_lib). Converter unit test included.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: C-API model-free wiring test (BOTSORT + IMAGE rejected)

**Files:**
- Modify: `.../oriented_object_detector/oriented_object_detector_test.cc`

- [ ] **Step 1: Add the test**

READ the OBB C-API test file for an existing model-free rejection test (the OBB C-API has tiling rejection tests like `RejectsExplicitTilesWithGrid` — mirror its idiom: zero-init options, call `MpOrientedObjectDetectorCreate(&options,&detector,&error_msg)`, assert non-`kMpOk`, free error via the file's mechanism — `MpErrorFree` or `free`; copy what the sibling uses). The cc Create() rejects BOTSORT in IMAGE mode with a message containing `tracking.tracker_type=BOTSORT`. Add:
```cpp
// BOTSORT in IMAGE mode is rejected by the C++ Create() WITHOUT a model fixture,
// proving the tracking field is wired through the binding.
TEST(OrientedObjectDetectorCApiTest, BotsortInImageModeRejectedThroughBinding) {
  MpOrientedObjectDetectorOptions options = {};
  options.running_mode = MpRunningMode::MP_RUNNING_MODE_IMAGE;
  options.num_classes = 15;
  options.tracking.tracker_type = 2;  // BOTSORT

  MpOrientedObjectDetectorPtr detector = nullptr;
  char* error_msg = nullptr;
  const MpStatus status =
      MpOrientedObjectDetectorCreate(&options, &detector, &error_msg);

  EXPECT_NE(status, kMpOk);
  ASSERT_NE(error_msg, nullptr);
  EXPECT_NE(std::string(error_msg).find("tracking.tracker_type=BOTSORT"),
            std::string::npos);
  <free error_msg per the file's idiom, e.g. MpErrorFree(error_msg);>
}
```
Match the EXACT detector-ptr type, status type, and error-free idiom the sibling rejection tests use (copy them). Ensure `<string>` is included (it likely is). The substring `tracking.tracker_type=BOTSORT` is unique to the OBB tracking validation messages (so the test isn't satisfied by the model-loader's generic error).

- [ ] **Step 2: Run it (model-free) + full target**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_test --test_output=all --test_filter='*BotsortInImageMode*' --nocache_test_results
```
Expected: RAN (not skipped) + PASS. Then the whole target → PASS.

- [ ] **Step 3: Commit**

```bash
git add mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector_test.cc
git commit -m "test(obb-tracking): C-API rejects BOTSORT in IMAGE mode (model-free)

Proves the tracking field is wired through MpOrientedObjectDetectorCreate: a
BOTSORT selection in IMAGE mode is rejected by the cc validation before model
load (asserts the BOTSORT-specific message).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: C-API output — `MpOrientedDetection.track_id`

**Files:**
- Modify: `mediapipe/tasks/c/components/containers/oriented_detection_result.h`
- Modify: `mediapipe/tasks/c/components/containers/oriented_detection_result_converter.cc`
- Test: `mediapipe/tasks/c/components/containers/oriented_detection_result_converter_test.cc` (create if absent)

The C++ container `OrientedObjectDetection` already has `std::optional<std::string> track_id` (sub-project 1). String ownership mirrors `category_converter.cc` (`strdup`/`free`).

- [ ] **Step 1: Write the failing converter test**

Check if `oriented_detection_result_converter_test.cc` exists (`ls` the dir). If absent, create it (Apache header):
```cpp
#include "mediapipe/tasks/c/components/containers/oriented_detection_result_converter.h"

#include <optional>
#include <string>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/oriented_detection_result.h"
#include "mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h"

namespace mediapipe::tasks::c::components::containers {
namespace {

namespace cc = ::mediapipe::tasks::components::containers;

TEST(OrientedDetectionResultConverterTest, ConvertsTrackId) {
  cc::OrientedObjectDetection in;
  in.categories.push_back({/*index=*/1, /*score=*/0.9f, "", ""});
  in.cx = 1.0f; in.cy = 2.0f; in.width = 3.0f; in.height = 4.0f;
  in.rotation = 0.3f;
  in.track_id = std::string("42");

  MpOrientedDetection out;
  CppConvertToOrientedDetection(in, &out);
  ASSERT_NE(out.track_id, nullptr);
  EXPECT_STREQ(out.track_id, "42");
  CppCloseOrientedDetection(&out);
  EXPECT_EQ(out.track_id, nullptr);
}

TEST(OrientedDetectionResultConverterTest, NoTrackIdIsNull) {
  cc::OrientedObjectDetection in;
  in.categories.push_back({/*index=*/1, /*score=*/0.9f, "", ""});
  MpOrientedDetection out;
  CppConvertToOrientedDetection(in, &out);
  EXPECT_EQ(out.track_id, nullptr);
  CppCloseOrientedDetection(&out);
}

}  // namespace
}  // namespace mediapipe::tasks::c::components::containers
```
Confirm the cc `Category` aggregate-init order + the cc namespace alias + the `ConvertToOrientedObjectDetectionResult` header path by reading the files; adjust. Add a `cc_test` `oriented_detection_result_converter_test` to the dir's BUILD if creating the file (mirror the sibling `detection_result_converter_test` deps).

- [ ] **Step 2: Run RED**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/components/containers:oriented_detection_result_converter_test --test_output=errors
```
Expected: FAIL to compile (`MpOrientedDetection` has no `track_id`).

- [ ] **Step 3: Add the field + convert + free**

In `oriented_detection_result.h`, add to `struct MpOrientedDetection` (after `rotation`):
```c
  // Optional persistent track ID (e.g. from BoTSORT), as a NUL-terminated
  // string. `nullptr` when the detection is not part of a track. Owned by the
  // result; freed by the detector's CloseResult.
  const char* track_id;
```
In `oriented_detection_result_converter.cc`: add `#include <cstdlib>` if absent (match `category_converter.cc`). In `CppConvertToOrientedDetection`, after `out->rotation = in.rotation;`, add:
```cpp
  out->track_id =
      in.track_id.has_value() ? strdup(in.track_id->c_str()) : nullptr;
```
In `CppCloseOrientedDetection`, after freeing categories, add:
```cpp
  free(const_cast<char*>(in->track_id));
  in->track_id = nullptr;
```

- [ ] **Step 4: Run GREEN + downstream**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/components/containers:oriented_detection_result_converter_test --test_output=errors
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_c_lib
```
Expected: test PASSES (2 cases); the OBB C lib builds.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/tasks/c/components/containers/oriented_detection_result.h \
        mediapipe/tasks/c/components/containers/oriented_detection_result_converter.cc \
        mediapipe/tasks/c/components/containers/oriented_detection_result_converter_test.cc \
        mediapipe/tasks/c/components/containers/BUILD
git commit -m "feat(obb-tracking): C-API MpOrientedDetection.track_id (strdup/free)

MpOrientedDetection gains an optional const char* track_id; the converter
strdup's the container's optional track_id (nullptr when absent) and
CppCloseOrientedDetection frees it. Mirrors the category_name idiom.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Python input — `TrackingOptions`

**Files:**
- Modify: `mediapipe/tasks/python/vision/oriented_object_detector.py`
- Modify: `mediapipe/tasks/python/test/vision/oriented_object_detector_test.py`

- [ ] **Step 1: ctypes struct + parent field**

In `oriented_object_detector.py`, after `MpOrientedTilingOptionsC`, add:
```python
class MpOrientedTrackingOptionsC(ctypes.Structure):
  """Byte-matches struct MpOrientedTrackingOptions in the OBB C header.

  Layout pinned by tracking_options_abi_test.cc (sizeof 28; offsets
  0/4/8/12/16/20/24; enable_gmc c_bool = 1 byte). tracker_type is c_int with
  0=unspecified->no tracking, 2=BOTSORT (1=BOX_TRACKER is rejected at Create).
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
In `MpOrientedObjectDetectorOptionsC._fields_`, add `('tracking', MpOrientedTrackingOptionsC),` immediately AFTER `('tiling', MpOrientedTilingOptionsC),` and before `('result_callback', ...)`.

- [ ] **Step 2: TrackerType IntEnum + TrackingOptions dataclass + options field**

After `class Layout(enum.IntEnum):`, add:
```python
class TrackerType(enum.IntEnum):
  """Tracker selection for the OBB tiled VIDEO/LIVE_STREAM path.

  OBB supports BOTSORT only; UNSPECIFIED means no tracking (the default).
  Values are numerically equal to the C++/proto TrackerType enum.
  """

  UNSPECIFIED = 0  # no tracking (default)
  BOTSORT = 2  # tracking-by-detection, motion-only
```
Next to the `TilingOptions` dataclass, add:
```python
@dataclasses.dataclass
class TrackingOptions:
  """Tracker selection for the OBB object detector (tiled stream path).

  OBB supports BOTSORT only; default is no tracking. Honored only when tiling is
  enabled and running mode is not IMAGE. NOTE: BoTSORT only emits a track_id for
  confirmed tracks, so set track_high_threshold / new_track_threshold at or below
  your score_threshold.
  """

  tracker_type: TrackerType = TrackerType.UNSPECIFIED
  track_high_threshold: float = 0.6
  track_low_threshold: float = 0.1
  new_track_threshold: float = 0.7
  track_buffer: int = 30
  match_threshold: float = 0.7
  enable_gmc: bool = False
```
In the `OrientedObjectDetectorOptions` dataclass, add after the `tiling: TilingOptions = dataclasses.field(default_factory=TilingOptions)` line:
```python
  tracking: TrackingOptions = dataclasses.field(default_factory=TrackingOptions)
```

- [ ] **Step 3: Marshalling helper + Create wiring**

After `_build_oriented_tiling_options_c`, add:
```python
def _build_oriented_tracking_options_c(
    tracking: TrackingOptions,
) -> 'MpOrientedTrackingOptionsC':
  """Builds the ctypes MpOrientedTrackingOptionsC from a TrackingOptions
  dataclass. No pointer fields, so there is no backing array to keep alive."""
  return MpOrientedTrackingOptionsC(
      tracker_type=int(tracking.tracker_type),
      track_high_threshold=tracking.track_high_threshold,
      track_low_threshold=tracking.track_low_threshold,
      new_track_threshold=tracking.new_track_threshold,
      track_buffer=tracking.track_buffer,
      match_threshold=tracking.match_threshold,
      enable_gmc=tracking.enable_gmc,
  )
```
In `Create`, after `tiling_c, tiles_keepalive = _build_oriented_tiling_options_c(options.tiling)`, add `tracking_c = _build_oriented_tracking_options_c(options.tracking)`, and in the `MpOrientedObjectDetectorOptionsC(...)` constructor add `tracking=tracking_c,` immediately after `tiling=tiling_c,`.

- [ ] **Step 4: ctypes layout test**

In `mediapipe/tasks/python/test/vision/oriented_object_detector_test.py`, find the existing tiling ctypes-layout test (search `sizeof`/`MpOrientedTilingOptionsC`/`192`) and add an analog (use the file's actual wrapper import alias for `<W>`):
```python
  def test_ctypes_tracking_layout_matches_c_abi(self):
    self.assertEqual(ctypes.sizeof(<W>.MpOrientedTrackingOptionsC), 28)
    fields = {f[0]: getattr(<W>.MpOrientedTrackingOptionsC, f[0]).offset
              for f in <W>.MpOrientedTrackingOptionsC._fields_}
    self.assertEqual(fields['tracker_type'], 0)
    self.assertEqual(fields['track_high_threshold'], 4)
    self.assertEqual(fields['track_low_threshold'], 8)
    self.assertEqual(fields['new_track_threshold'], 12)
    self.assertEqual(fields['track_buffer'], 16)
    self.assertEqual(fields['match_threshold'], 20)
    self.assertEqual(fields['enable_gmc'], 24)
    self.assertEqual(<W>.MpOrientedObjectDetectorOptionsC.tracking.offset, 184)
    self.assertEqual(
        <W>.MpOrientedObjectDetectorOptionsC.result_callback.offset, 216)
    self.assertEqual(ctypes.sizeof(<W>.MpOrientedObjectDetectorOptionsC), 224)
```
Also add a small functional test: `TrackingOptions(tracker_type=TrackerType.BOTSORT)` → `_build_oriented_tracking_options_c` → assert the struct's `tracker_type == 2` + knobs (model/.so-free). Match the file's import names for `TrackingOptions`/`TrackerType`.

- [ ] **Step 5: Verify here (Python toolchain broken)**

```bash
python3 - <<'EOF'
import ctypes
class MpOrientedTrackingOptionsC(ctypes.Structure):
    _fields_ = [
        ('tracker_type', ctypes.c_int),
        ('track_high_threshold', ctypes.c_float),
        ('track_low_threshold', ctypes.c_float),
        ('new_track_threshold', ctypes.c_float),
        ('track_buffer', ctypes.c_int),
        ('match_threshold', ctypes.c_float),
        ('enable_gmc', ctypes.c_bool),
    ]
assert ctypes.sizeof(MpOrientedTrackingOptionsC) == 28, ctypes.sizeof(MpOrientedTrackingOptionsC)
exp = {'tracker_type':0,'track_high_threshold':4,'track_low_threshold':8,
       'new_track_threshold':12,'track_buffer':16,'match_threshold':20,'enable_gmc':24}
for n,o in exp.items():
    assert getattr(MpOrientedTrackingOptionsC, n).offset == o, (n, getattr(MpOrientedTrackingOptionsC,n).offset)
print('MpOrientedTrackingOptionsC OK: sizeof=28, offsets pinned')
EOF
python3 -m py_compile mediapipe/tasks/python/vision/oriented_object_detector.py && echo "wrapper py_compile OK"
python3 -m py_compile mediapipe/tasks/python/test/vision/oriented_object_detector_test.py && echo "test py_compile OK"
```
Expected: all print OK. The standalone `_fields_` MUST be character-identical to the wrapper's. Do NOT `bazel test` the Python target (toolchain broken); note this.

- [ ] **Step 6: Commit**

```bash
git add mediapipe/tasks/python/vision/oriented_object_detector.py \
        mediapipe/tasks/python/test/vision/oriented_object_detector_test.py
git commit -m "feat(obb-tracking): Python OBB TrackingOptions bindings

Adds MpOrientedTrackingOptionsC (byte-matching the C struct), a TrackerType
IntEnum {UNSPECIFIED, BOTSORT}, a TrackingOptions dataclass (default no tracking),
_build_oriented_tracking_options_c, and wires tracking into Create. ctypes layout
test mirrors the C++ ABI pin; verified via a standalone ctypes check + py_compile.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Python output — `OrientedDetection.track_id`

**Files:**
- Modify: `mediapipe/tasks/python/components/containers/oriented_detections_c.py`
- Modify: `mediapipe/tasks/python/components/containers/oriented_detections.py`
- Modify: `mediapipe/tasks/python/test/components/containers/` (the oriented detections test)

- [ ] **Step 1: ctypes field**

In `oriented_detections_c.py`, add to `MpOrientedDetectionC._fields_` at the END (after `rotation`, matching the C struct order):
```python
      ('track_id', ctypes.c_char_p),
```

- [ ] **Step 2: dataclass field + from_ctypes + __eq__**

In `oriented_detections.py`:
- Add to the `OrientedDetection` dataclass (after `rotation`): `track_id: Optional[str] = None` + an Attributes docstring line ("Optional persistent track ID string (e.g. from BoTSORT); None when not tracked.").
- In `from_ctypes`, add before the return: `track_id = c_obj.track_id.decode('utf-8') if c_obj.track_id else None`, and pass `track_id=track_id,` in the `OrientedDetection(...)` constructor.
- In `__eq__`, add `and self.track_id == other.track_id`.
Ensure `Optional` is imported (it is used by other containers — confirm in this file's imports; add `from typing import Optional` if missing).

- [ ] **Step 3: from_ctypes test**

Find the oriented-detections from_ctypes test (search `mediapipe/tasks/python/test/components/containers/` for `MpOrientedDetectionC`/`from_ctypes`). Add a test building an `MpOrientedDetectionC` with `track_id=b'42'` → assert `.track_id == '42'`, and `track_id=None` → `None`. Mirror the file's existing construction.

- [ ] **Step 4: Verify (static)**

```bash
python3 -m py_compile mediapipe/tasks/python/components/containers/oriented_detections_c.py && echo "c py_compile OK"
python3 -m py_compile mediapipe/tasks/python/components/containers/oriented_detections.py && echo "wrapper py_compile OK"
python3 - <<'EOF'
import ctypes
class _D(ctypes.Structure):
    _fields_ = [('track_id', ctypes.c_char_p)]
assert (_D(track_id=b'42').track_id.decode('utf-8')) == '42'
assert (_D(track_id=None).track_id or None) is None
print('oriented track_id decode OK')
EOF
```
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/tasks/python/components/containers/oriented_detections_c.py \
        mediapipe/tasks/python/components/containers/oriented_detections.py \
        mediapipe/tasks/python/test/components/containers/
git commit -m "feat(obb-tracking): Python OrientedDetection.track_id

MpOrientedDetectionC gains a c_char_p track_id; the OrientedDetection dataclass
exposes track_id: Optional[str] parsed in from_ctypes (None when null) and
compared in __eq__. Verified via py_compile + a standalone ctypes decode check.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- C `MpOrientedTrackingOptions` + field + ABI pins (+ shifted tiling pins) → T1 ✓
- C converter + wire into Create + folded into both libs → T2 ✓
- C model-free wiring test (BOTSORT+IMAGE rejected, reusing cc validation) → T3 ✓
- C `MpOrientedDetection.track_id` strdup/free → T4 ✓
- Python input (ctypes + IntEnum {UNSPECIFIED,BOTSORT} + dataclass + marshalling + Create + layout test) → T5 ✓
- Python output (ctypes track_id + dataclass + from_ctypes + eq) → T6 ✓
- Validation reuse / no C++/proto/graph change → all tasks respect this ✓

**Placeholder scan:** No TBD/TODO; complete code for the new struct/converter/ABI test/ctypes; tests have real assertions; commands have expected output. `<W>` (Task 5) and the "free per the file's idiom" (Task 3) are file-local facts the implementer confirms — flagged, not logic gaps.

**Type consistency:** `MpOrientedTrackingOptions` fields identical across header (T1), ABI test (T1), converter (T2), converter test (T2), ctypes (T5), standalone check (T5). Enum values 1/2 == `kBoxTracker`/`kBotsort` (converter) and `TrackerType.BOTSORT=2` (Python). `track_id` is C `const char*` (T4) ↔ ctypes `c_char_p` (T6) ↔ `Optional[str]` (T6). ABI numbers consistent: sizeof 28, offsets 0/4/8/12/16/20/24, parent tracking@184/result_callback@216/sizeof@224.

**Known verification-time notes:** the OBB C-API sibling rejection test's error-free idiom (T3); the cc `Category` aggregate-init order + whether the oriented C converter test file exists (T4); the Python wrapper import alias `<W>` + the oriented-detections from_ctypes test location (T5/T6).
