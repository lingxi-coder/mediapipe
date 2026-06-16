# Expose BoTSORT Track IDs Through the Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface BoTSORT's persistent per-object track ID on output `Detection`s at every layer (C++ task → C-API → Python) via the framework proto's existing `optional string track_id`, with output geometry/recall unchanged.

**Architecture:** The `BotsortTrackingCalculator` writes `track_id`; `TiledFrameSuppressionCalculator` (Approach A) transfers a tracker detection's `track_id` onto the IoU-matched surviving fresh detection (fresh-wins geometry unchanged; no-op for BoxTracker). The shared `Detection` container / C `MpDetection` / Python dataclass each gain an additive optional `track_id` read from the proto. Field survival through the post-merge calculators is already verified.

**Tech Stack:** C++20, Bazel, proto2, ctypes. Build/test only via `bazel ... --define MEDIAPIPE_DISABLE_GPU=1` on this machine (authoritative; IGNORE clangd noise). Python toolchain broken here → Python verified by py_compile + a conversion check; full Python tests run where the toolchain works.

**Reference spec:** `docs/superpowers/specs/2026-06-16-botsort-track-id-exposure-design.md`

**Standing constraints (every task):** stay on branch `dev`; never branch/merge; all comments English; TDD; commit after each task; commit-message trailer MUST be exactly:
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `mediapipe/calculators/tensor/botsort_tracking_calculator.cc` (+ test) | Write `track_id` on output Detections | 1 |
| `mediapipe/calculators/tensor/tiled_frame_suppression_calculator.cc` (+ test) | Approach A: transfer id to surviving fresh det | 2 |
| `mediapipe/tasks/cc/components/containers/detection_result.{h,cc}` (+ test) | Shared container `track_id` + conversion | 3 |
| `mediapipe/tasks/c/components/containers/detection_result.h` + `detection_result_converter.cc` (+ test) | C `MpDetection.track_id` + strdup/free | 4 |
| `mediapipe/tasks/python/components/containers/detections_c.py` + `detections.py` (+ test) | ctypes field + dataclass + from_ctypes | 5 |
| `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc` | e2e: stable track_ids across frames | 6 |

---

## Task 1: `BotsortTrackingCalculator` writes `track_id`

**Files:**
- Modify: `mediapipe/calculators/tensor/botsort_tracking_calculator.cc`
- Modify: `mediapipe/calculators/tensor/botsort_tracking_calculator_test.cc`

- [ ] **Step 1: Flip the failing test assertion**

In `botsort_tracking_calculator_test.cc`, the test `PreservesLabelAndScoreAndCarriesNoTrackId` (line 107) currently asserts `EXPECT_FALSE(last[0].has_track_id())` (line 142). Replace those parity assertions and rename the test:

Replace:
```cpp
  // Parity output: no public track id surfaced.
  EXPECT_FALSE(last[0].has_track_id());
  EXPECT_FALSE(last[0].has_detection_id());
```
with:
```cpp
  // BoTSORT now surfaces its persistent track id (as the proto string field).
  // detection_id remains unused.
  EXPECT_TRUE(last[0].has_track_id());
  EXPECT_FALSE(last[0].track_id().empty());
  EXPECT_FALSE(last[0].has_detection_id());
```
Rename the test `PreservesLabelAndScoreAndCarriesNoTrackId` →
`PreservesLabelAndScoreAndSurfacesTrackId` (update the comment on line 106 to match: "...and surfaces the BoTSORT track id").

- [ ] **Step 2: Run it to confirm RED**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:botsort_tracking_calculator_test --test_output=errors --test_filter='*SurfacesTrackId*'
```
Expected: FAIL (`has_track_id()` is currently false — the calculator doesn't write it yet).

- [ ] **Step 3: Write the track_id**

In `botsort_tracking_calculator.cc`, in `Process`, the output Detection is built (around line 113-123):
```cpp
      Detection det;
      det.add_score(t->get_score());
      det.add_label_id(static_cast<int>(t->get_class_id()));
      auto* loc = det.mutable_location_data();
      ...
      out.push_back(std::move(det));
```
Add the track_id immediately after `det.add_label_id(...)`:
```cpp
      // Surface BoTSORT's persistent track id (an int) as the proto's string
      // track_id field ("part of a track"). Consumed downstream by
      // TiledFrameSuppression (id propagation) and the result containers.
      det.set_track_id(std::to_string(t->track_id));
```
`t` is a `std::shared_ptr<Track>`; `track_id` is the public `int` member (confirmed in `third_party/botsort/include/track.h`). `<string>` is needed for `std::to_string` — add `#include <string>` to the includes if not already present.

Also update the class doc comment near line 47 that says it does NOT write track_id (the "parity contract" line) to state it now writes the track id.

- [ ] **Step 4: Run it to confirm GREEN + full target**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:botsort_tracking_calculator_test --test_output=errors
```
Expected: PASS (all cases, incl. the renamed one + the empty-frame test).

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/botsort_tracking_calculator.cc \
        mediapipe/calculators/tensor/botsort_tracking_calculator_test.cc
git commit -m "feat(tracking): BotsortTrackingCalculator writes track_id

Reverses the parity-output decision: BoTSORT's persistent track id is now set on
each output Detection as the proto string track_id. Test assertion flipped to
expect it.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `TiledFrameSuppressionCalculator` ID propagation (Approach A)

**Files:**
- Modify: `mediapipe/calculators/tensor/tiled_frame_suppression_calculator.cc`
- Modify: `mediapipe/calculators/tensor/tiled_frame_suppression_calculator_test.cc`

- [ ] **Step 1: Write the failing tests**

First READ `tiled_frame_suppression_calculator_test.cc` to match its harness (runner construction, how it builds Detections with bbox + label, the input/output `DETECTIONS` + `TRACKER_DETECTIONS` tags). Add a helper to build a Detection with a track_id if the file doesn't have one. Add three cases (adapt the Detection-builder + runner idiom to the file's style):

```cpp
// Approach A: when a tracker detection overlaps a surviving fresh detection,
// its track_id is transferred onto the fresh one (geometry unchanged).
TEST(TiledFrameSuppressionCalculatorTest, TransfersTrackIdToOverlappingFresh) {
  // fresh: one box, NO track_id. tracker: same box, track_id "7".
  // gap_fill_only=true. Expect: one output box (the fresh geometry) carrying
  // track_id "7".
  // (build fresh det at e.g. xmin .25 ymin .25 w .1 h .2, label 0, score 0.9;
  //  tracker det same box, label 0, score 0.9, track_id "7")
  // ... assert output size 1, output[0].track_id() == "7" ...
}

TEST(TiledFrameSuppressionCalculatorTest, GapFillTrackerKeepsOwnTrackId) {
  // fresh: box A (no id). tracker: box B far away, track_id "3" (no overlap).
  // Expect: 2 outputs; the gap-fill (box B) carries track_id "3"; box A has none.
}

TEST(TiledFrameSuppressionCalculatorTest, NoTrackIdNoOp) {
  // fresh: box A (no id). tracker: same box A, NO track_id (BoxTracker style).
  // Expect: output box A still has NO track_id (no spurious id written).
}
```
Fill in the runner config (calculator `TiledFrameSuppressionCalculator`, `input_stream: "DETECTIONS:..."`, `input_stream: "TRACKER_DETECTIONS:..."`, `output_stream: "DETECTIONS:..."`, `node_options { [type.googleapis.com/mediapipe.TiledFrameSuppressionCalculatorOptions] { iou_threshold: 0.5 class_agnostic: true tracker_is_gap_fill_only: true } }`) and the Detection builders, mirroring the existing tests in the file. For each case feed both streams at the same Timestamp, `Run()`, read the single output packet's `std::vector<Detection>`. Use `set_track_id("7")` etc. on the tracker Detection's proto.

- [ ] **Step 2: Run to confirm RED**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tiled_frame_suppression_calculator_test --test_output=errors --test_filter='*TrackId*:*GapFill*:*NoOp*'
```
Expected: `TransfersTrackIdToOverlappingFresh` and `GapFillTrackerKeepsOwnTrackId` FAIL (no id transferred yet); `NoTrackIdNoOp` may already pass.

- [ ] **Step 3: Implement Approach A in the gap-fill loop**

In `tiled_frame_suppression_calculator.cc`, replace the `tracker_is_gap_fill_only` loop (currently lines 70-85, the `const size_t fresh_count = ...` through the `if (!overlaps) combined.push_back(t);`) with the best-IoU-match + id-transfer version:

```cpp
        // Keep only tracker boxes that don't overlap any fresh box (gap-fill).
        // When a tracker box DOES overlap a fresh box, fresh-wins geometry is
        // unchanged, but we propagate the tracker's persistent track_id onto
        // the best-IoU fresh box (Approach A). This is a no-op when the tracker
        // carries no track_id (e.g. the optical-flow BoxTracker path).
        const size_t fresh_count = combined.size();
        for (const Detection& t : tr) {
          int best_idx = -1;
          float best_iou = options_.iou_threshold();
          for (size_t i = 0; i < fresh_count; ++i) {
            const Detection& f = combined[i];
            if (!options_.class_agnostic() && t.label_id_size() > 0 &&
                f.label_id_size() > 0 && t.label_id(0) != f.label_id(0)) {
              continue;
            }
            const float iou = DetectionRelativeIoU(t, f);
            if (iou >= best_iou) {
              best_iou = iou;
              best_idx = static_cast<int>(i);
            }
          }
          if (best_idx < 0) {
            combined.push_back(t);  // gap-fill: keeps its own track_id
          } else if (t.has_track_id() && !combined[best_idx].has_track_id()) {
            combined[best_idx].set_track_id(t.track_id());
          }
        }
```
This preserves the exact gap-fill set (a tracker box is a gap-fill iff it overlaps no fresh box at IoU >= threshold) and only adds id transfer on overlap. `GreedyDetectionNms` (line 90) returns the original Detection objects, so the transferred/own track_id survives the final NMS.

- [ ] **Step 4: Run to confirm GREEN + full target**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tiled_frame_suppression_calculator_test --test_output=errors
```
Expected: PASS (the 3 new cases + all pre-existing suppression tests — geometry behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/tiled_frame_suppression_calculator.cc \
        mediapipe/calculators/tensor/tiled_frame_suppression_calculator_test.cc
git commit -m "feat(tracking): propagate tracker track_id to surviving fresh detection

Approach A: in the gap-fill suppression loop, when a tracker detection overlaps
the best-IoU fresh detection, transfer its track_id onto the surviving fresh box
(fresh-wins geometry unchanged). Gap-fill tracker boxes keep their own id. No-op
when the tracker carries no track_id (BoxTracker path).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Shared C++ `Detection` container `track_id`

**Files:**
- Modify: `mediapipe/tasks/cc/components/containers/detection_result.h`
- Modify: `mediapipe/tasks/cc/components/containers/detection_result.cc`
- Test: `mediapipe/tasks/cc/components/containers/detection_result_test.cc` (create if absent)

- [ ] **Step 1: Write the failing conversion test**

Check whether `mediapipe/tasks/cc/components/containers/detection_result_test.cc` exists (`ls` the dir). If absent, create it with the Apache header. Add:

```cpp
#include "mediapipe/tasks/cc/components/containers/detection_result.h"

#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::tasks::components::containers {
namespace {

TEST(DetectionResultTest, ConvertToDetectionMapsTrackId) {
  mediapipe::Detection proto;
  proto.add_score(0.9f);
  proto.add_label_id(1);
  proto.set_track_id("42");
  Detection d = ConvertToDetection(proto);
  ASSERT_TRUE(d.track_id.has_value());
  EXPECT_EQ(*d.track_id, "42");
}

TEST(DetectionResultTest, ConvertToDetectionNoTrackId) {
  mediapipe::Detection proto;
  proto.add_score(0.9f);
  proto.add_label_id(1);
  Detection d = ConvertToDetection(proto);
  EXPECT_FALSE(d.track_id.has_value());
}

}  // namespace
}  // namespace mediapipe::tasks::components::containers
```
If creating the file, add a `cc_test` target named `detection_result_test` to `mediapipe/tasks/cc/components/containers/BUILD` (deps: `:detection_result`, `//mediapipe/framework/formats:detection_cc_proto`, `//mediapipe/framework/port:gtest_main`). Confirm `ConvertToDetection` is declared in the header (it is — `detection_result.h` declares it).

- [ ] **Step 2: Run to confirm RED**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/components/containers:detection_result_test --test_output=errors
```
Expected: FAIL to compile (`Detection` has no `track_id` member yet).

- [ ] **Step 3: Add the field + conversion**

In `detection_result.h`, add a `<string>` include if absent, and add the field to `struct Detection` (after `keypoints`):
```cpp
  // Optional persistent track ID (set by tracking-by-detection trackers, e.g.
  // BoTSORT). std::nullopt when the detection is not part of a track.
  std::optional<std::string> track_id = std::nullopt;
```
(`<optional>` is already used by `keypoints`.)

In `detection_result.cc`, in `ConvertToDetection`, before `return detection;`, add:
```cpp
  if (detection_proto.has_track_id()) {
    detection.track_id = detection_proto.track_id();
  }
```

- [ ] **Step 4: Run to confirm GREEN**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/components/containers:detection_result_test --test_output=errors
```
Expected: PASS (2 cases).

- [ ] **Step 5: Commit**

```bash
git add mediapipe/tasks/cc/components/containers/detection_result.h \
        mediapipe/tasks/cc/components/containers/detection_result.cc \
        mediapipe/tasks/cc/components/containers/detection_result_test.cc \
        mediapipe/tasks/cc/components/containers/BUILD
git commit -m "feat(tracking): optional track_id on the shared Detection container

ConvertToDetection now reads the proto's optional-string track_id into a new
std::optional<std::string> Detection.track_id (nullopt when absent). Additive;
empty for detectors that do not set track_id.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: C-API `MpDetection.track_id`

**Files:**
- Modify: `mediapipe/tasks/c/components/containers/detection_result.h`
- Modify: `mediapipe/tasks/c/components/containers/detection_result_converter.cc`
- Test: `mediapipe/tasks/c/components/containers/detection_result_converter_test.cc` (create if absent)

- [ ] **Step 1: Write the failing converter test**

Check whether `detection_result_converter_test.cc` exists. If absent, create it (Apache header). Add:

```cpp
#include "mediapipe/tasks/c/components/containers/detection_result_converter.h"

#include <cstring>
#include <optional>
#include <string>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/detection_result.h"
#include "mediapipe/tasks/cc/components/containers/detection_result.h"

namespace mediapipe::tasks::c::components::containers {
namespace {

namespace cc = ::mediapipe::tasks::components::containers;

TEST(DetectionResultConverterTest, ConvertsTrackId) {
  cc::Detection in;
  in.categories.push_back({/*index=*/1, /*score=*/0.9f, "", ""});
  in.track_id = std::string("42");

  MpDetection out;
  CppConvertToDetection(in, &out);
  ASSERT_NE(out.track_id, nullptr);
  EXPECT_STREQ(out.track_id, "42");
  CppCloseDetection(&out);  // must not crash / leak; frees track_id
  EXPECT_EQ(out.track_id, nullptr);
}

TEST(DetectionResultConverterTest, NoTrackIdIsNull) {
  cc::Detection in;
  in.categories.push_back({/*index=*/1, /*score=*/0.9f, "", ""});

  MpDetection out;
  CppConvertToDetection(in, &out);
  EXPECT_EQ(out.track_id, nullptr);
  CppCloseDetection(&out);
}

}  // namespace
}  // namespace mediapipe::tasks::c::components::containers
```
If creating the file, add a `cc_test` `detection_result_converter_test` to `mediapipe/tasks/c/components/containers/BUILD` (deps: the converter lib target, `//mediapipe/tasks/cc/components/containers:detection_result`, gtest_main). Confirm the `Category` aggregate-init order (`{index, score, category_name, display_name}`) matches the cc `Category` struct; adjust if the struct differs.

- [ ] **Step 2: Run to confirm RED**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/components/containers:detection_result_converter_test --test_output=errors
```
Expected: FAIL to compile (`MpDetection` has no `track_id`).

- [ ] **Step 3: Add the C field + convert + free**

In `mediapipe/tasks/c/components/containers/detection_result.h`, add to `struct MpDetection` (after `keypoints_count`):
```cpp
  // Optional persistent track ID (e.g. from BoTSORT), as a NUL-terminated
  // string. `nullptr` when the detection is not part of a track. Owned by the
  // result; freed by the detector's CloseResult.
  const char* track_id;
```
In `detection_result_converter.cc`, add `#include <cstring>` if needed (for `strdup` it's `<cstring>`/`<cstdlib>` — match the category converter's includes). In `CppConvertToDetection`, after the keypoints block and before the closing brace, add:
```cpp
  out->track_id =
      in.track_id.has_value() ? strdup(in.track_id->c_str()) : nullptr;
```
In `CppCloseDetection`, after freeing keypoints, add:
```cpp
  free(const_cast<char*>(in->track_id));
  in->track_id = nullptr;
```
(`free` on `nullptr` is a no-op, so the null case is safe. `strdup`/`free` mirror the `category_name` ownership idiom in `category_converter.cc`.)

- [ ] **Step 4: Run to confirm GREEN**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/components/containers:detection_result_converter_test --test_output=errors
```
Expected: PASS (2 cases). Also rebuild the YOLO C lib to confirm the shared struct change compiles through it:
```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_c_lib
```

- [ ] **Step 5: Commit**

```bash
git add mediapipe/tasks/c/components/containers/detection_result.h \
        mediapipe/tasks/c/components/containers/detection_result_converter.cc \
        mediapipe/tasks/c/components/containers/detection_result_converter_test.cc \
        mediapipe/tasks/c/components/containers/BUILD
git commit -m "feat(tracking): C-API MpDetection.track_id (strdup/free)

MpDetection gains an optional const char* track_id; the converter strdup's the
container's optional track_id (nullptr when absent) and CppCloseDetection frees
it. Mirrors the category_name string-ownership idiom.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Python `Detection.track_id`

**Files:**
- Modify: `mediapipe/tasks/python/components/containers/detections_c.py`
- Modify: `mediapipe/tasks/python/components/containers/detections.py`
- Modify: `mediapipe/tasks/python/test/vision/` (or the containers test) — add a from_ctypes test

- [ ] **Step 1: Add the ctypes field**

In `detections_c.py`, add to `MpDetectionC._fields_` (at the END, after `keypoints_count`, matching the C struct order):
```python
      ('track_id', ctypes.c_char_p),
```

- [ ] **Step 2: Add the dataclass field + from_ctypes parse + __eq__**

In `detections.py`, add to the `Detection` dataclass (after `keypoints`):
```python
  track_id: Optional[str] = None
```
Add the Attributes docstring line for `track_id` ("Optional persistent track ID string, e.g. from BoTSORT; None when not tracked.").
In `Detection.from_ctypes`, add the parse (mirroring the category_name decode idiom) and pass it to the constructor:
```python
    track_id = c_obj.track_id.decode('utf-8') if c_obj.track_id else None
    return Detection(
        bounding_box=py_bounding_box,
        categories=py_categories,
        keypoints=py_keypoints,
        track_id=track_id,
    )
```
In `Detection.__eq__`, add `and self.track_id == other.track_id` to the returned conjunction.

- [ ] **Step 3: Add a from_ctypes test**

Find the Python test that covers `Detection.from_ctypes` (search the containers/vision tests for `MpDetectionC` / `from_ctypes`). Add a test constructing an `MpDetectionC` with `track_id=b'42'` and asserting `Detection.from_ctypes(...).track_id == '42'`, plus a `track_id=None` → `None` case. Match the file's existing harness/imports. If no such test exists near the detections containers, add it to the containers test module.

- [ ] **Step 4: Verify on this machine (Python toolchain broken → static checks)**

```bash
python3 -m py_compile mediapipe/tasks/python/components/containers/detections_c.py && echo "c py_compile OK"
python3 -m py_compile mediapipe/tasks/python/components/containers/detections.py && echo "wrapper py_compile OK"
```
Then a standalone ctypes round-trip (pure ctypes, no mediapipe import — runs here):
```bash
python3 - <<'EOF'
import ctypes
# Minimal stand-in matching the c_char_p field we added.
class _D(ctypes.Structure):
    _fields_ = [('track_id', ctypes.c_char_p)]
d = _D(track_id=b'42')
assert (d.track_id.decode('utf-8') if d.track_id else None) == '42'
e = _D(track_id=None)
assert (e.track_id.decode('utf-8') if e.track_id else None) is None
print('track_id ctypes decode OK')
EOF
```
Expected: all three print OK. Do NOT `bazel test` the Python target (toolchain broken here); note that full Python tests run where the toolchain works.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/tasks/python/components/containers/detections_c.py \
        mediapipe/tasks/python/components/containers/detections.py \
        mediapipe/tasks/python/test/
git commit -m "feat(tracking): Python Detection.track_id

MpDetectionC gains a c_char_p track_id; the Detection dataclass exposes
track_id: Optional[str] parsed in from_ctypes (None when null) and compared in
__eq__. Verified via py_compile + a standalone ctypes decode check.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Real YOLO BOTSORT e2e — stable track IDs across frames

**Files:**
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc`

- [ ] **Step 1: Extend the existing BOTSORT e2e**

In the test `TiledVideoTracksBoatsWithBotsort` (added previously), after the existing per-frame loop that calls `DetectForVideo` and accumulates `last_result` / boat detection, add cross-frame track-id collection and assertions. Read the existing test first to match its variable names (the result type is `YoloObjectDetectorResult` = `DetectionResult`; each detection has `.track_id` after Task 3). Add, inside the frame loop, collection of the track_ids seen on the last two frames; after the loop assert:
```cpp
  // BoTSORT track ids survive the full pipeline to the public result, and at
  // least one object keeps a stable id across consecutive frames.
  ASSERT_GE(per_frame_track_ids.size(), 2u);  // collected >= 2 frames' id sets
  const auto& prev_ids = per_frame_track_ids[per_frame_track_ids.size() - 2];
  const auto& last_ids = per_frame_track_ids.back();
  EXPECT_FALSE(last_ids.empty())
      << "final frame produced no track ids";
  bool stable = false;
  for (const auto& id : last_ids) {
    if (prev_ids.count(id)) { stable = true; break; }
  }
  EXPECT_TRUE(stable)
      << "expected at least one track id stable across the last two frames";
```
where `per_frame_track_ids` is a `std::vector<std::set<std::string>>` you populate each frame from `result.detections[i].track_id` (only inserting when `.has_value()`). Place the `#include <set>` / `<string>` if needed. Keep the existing boat-content assertions intact.

- [ ] **Step 2: Run the e2e**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_output=all --test_filter='*BoatsWithBotsort*' --nocache_test_results
```
Expected: RUNS (fixture present) and PASSES — detections carry track ids and at least one id is stable across the last two frames. If the stability assertion is flaky on this scene, first confirm ids ARE present (the survival half); if present but not stable across the *last two* frames specifically, widen to "stable across some consecutive pair" rather than dropping the assertion — and report what you observed. Do NOT weaken to "ids present only" without reporting.

- [ ] **Step 3: Full regression of touched suites**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/calculators/tensor:botsort_tracking_calculator_test \
  //mediapipe/calculators/tensor:tiled_frame_suppression_calculator_test \
  //mediapipe/tasks/cc/components/containers:detection_result_test \
  //mediapipe/tasks/c/components/containers:detection_result_converter_test \
  //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test \
  //mediapipe/graphs/tiled_detection:tiled_tracking_graphs_test \
  --test_output=errors
```
Expected: all PASS (model-gated cases skip without the fixture; here the YOLO fixture is present so the e2e runs).

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc
git commit -m "test(tracking): e2e asserts BoTSORT track ids stable across frames

Extends the YOLO BOTSORT tiled-stream e2e to assert output detections carry
track ids that survive the full pipeline and stay stable across consecutive
frames.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Populate id: calculator (T1) + suppression Approach A (T2) ✓
- Shared C++ container track_id + conversion (T3) ✓
- C-API MpDetection.track_id + strdup/free (T4) ✓
- Python ctypes + dataclass + from_ctypes (T5) ✓
- Real e2e stable-across-frames (T6) ✓
- Field-survival through post-merge calculators: verified in spec (no task needed — they already preserve it) ✓
- Representation = proto string track_id throughout (T1 to_string → T3 optional<string> → T4 const char* → T5 Optional[str]) ✓
- Reversal of parity-output: T1 flips the calculator test assertion ✓

**Placeholder scan:** No TBD/TODO; each code step has complete code; the two test-harness steps (T2 builders, T5/T6 collection) give the exact assertion code + instruct mirroring the file's existing builder idiom (the one detail not literally inlined, because it must match each file's local helper). Commands have expected output.

**Type consistency:** `track_id` is: proto `string` (set via `set_track_id(std::to_string(t->track_id))` T1; propagated `set_track_id(t.track_id())` T2); container `std::optional<std::string>` (T3); C `const char*` (T4); Python `Optional[str]` / ctypes `c_char_p` (T5). `ConvertToDetection` name matches the header. `CppConvertToDetection`/`CppCloseDetection` names match the existing converter. `MpDetection`/`MpDetectionC` field added at struct end on both sides (C↔ctypes order consistent).

**Known verification-time notes flagged in-plan:** the `Category` aggregate-init field order in the C test (T4 S1); whether `detection_result_test.cc` / `detection_result_converter_test.cc` already exist (T3/T4 S1); the existing test harness's Detection-builder idiom (T2); the e2e's existing variable names + the stability-assertion fallback (T6 S2).
