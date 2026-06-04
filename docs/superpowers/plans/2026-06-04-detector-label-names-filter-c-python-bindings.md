# Detector Category Names + Allowlist/Denylist — Plan B (C API + Python bindings) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the OBB detector's new `display_names_locale` / `category_allowlist` / `category_denylist` options through the C API and Python bindings (mirroring YOLO), fix doc/comments that Plan A made stale, and prove the names + allow/deny path end-to-end where the toolchain allows.

**Architecture:** Plan A wired category-name population + allow/deny filtering into the YOLO and OBB **C++ graphs**. Plan B is pure binding plumbing on top: the C API copies the three option fields from its struct into the (already-extended, from Plan A Task 6) C++ options; the Python ctypes layer mirrors that struct and builds the string arrays. No new behavior — just surface.

**Tech Stack:** C ABI struct + ctypes; Bazel `cc_test` (C-API tests run here) and `py_test` (Python tests build-deferred).

---

## Verifiability split (READ THIS FIRST)

This environment builds desktop C++ only (`--define MEDIAPIPE_DISABLE_GPU=1`); the monolithic `libmediapipe.so` does **not** link here (`builtin_task_graphs` → `interactive_segmenter` missing BUILD).

- **C API — FULLY VERIFIABLE.** The C-API `cc_test` links the task's `_lib` static archive + gtest directly (NOT the shared `.so`). Confirmed: `//mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_test` builds and **runs** here today. Tasks 1–2 are verified by real build+test.
- **Python — BUILD-DEFERRED.** The Python task loads the monolith `.so` at runtime via `ctypes` (`load_shared_library`), which can't be built here, and the pip env is broken. Tasks 3–4 cannot be run here. Mitigation: each Python step is written to mirror the **verified** YOLO Python pattern byte-for-byte, and includes a runnable `python3 -m py_compile` syntax check (which does not import mediapipe) plus the exact `bazel test` command to run once the `.so` builds. Do **not** weaken or skip the Python code because it can't run — mirror YOLO exactly.

## Task order / file map

| Task | Files | Verifiable here? |
|---|---|---|
| 1: OBB C-API options + conversion + hardened C test | `tasks/c/vision/oriented_object_detector/{oriented_object_detector.h,.cc,_test.cc}` | ✅ build + run |
| 2: Fix stale "no-op" comments (Plan A made them false) | `tasks/c/vision/yolo_object_detector/yolo_object_detector.h`; `tasks/python/vision/yolo_object_detector.py` | ✅ compile (C header) / `py_compile` |
| 3: OBB Python dataclass + ctypes + conversion | `tasks/python/vision/oriented_object_detector.py` | ⛔ deferred (`py_compile` only) |
| 4: OBB Python test hardening + fallback note | `tasks/python/test/vision/oriented_object_detector_test.py`; `tasks/python/vision/oriented_object_detector.py` | ⛔ deferred (`py_compile` only) |

Reference (already has all three fields, verified in Plan A): the YOLO C API (`tasks/c/vision/yolo_object_detector/yolo_object_detector.{h,cc}`) and YOLO Python (`tasks/python/vision/yolo_object_detector.py`).

---

### Task 1: OBB C-API options + conversion + hardened C test

**Files:**
- Modify: `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h`
- Modify: `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.cc`
- Modify: `mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector_test.cc`

- [ ] **Step 1: Add the three fields to the C options struct.** In `oriented_object_detector.h`, inside `struct MpOrientedObjectDetectorOptions`, insert `display_names_locale` immediately after the `running_mode` line, and the allow/deny block immediately after the `score_threshold` line (mirroring the YOLO struct field positions). Result — the struct becomes:

```c
struct MpOrientedObjectDetectorOptions {
  struct MpBaseOptions base_options;
  MpRunningMode running_mode;

  // Locale for display names in TFLite metadata, if any. Defaults to English.
  const char* display_names_locale;

  // Max number of top-scored results. < 0 returns all; 0 is invalid.
  int max_results;

  // Score threshold overriding the model metadata value. Default 0.25.
  float score_threshold;

  // Allowlist of category names (mutually exclusive with denylist).
  const char** category_allowlist;
  uint32_t category_allowlist_count;

  // Denylist of category names (mutually exclusive with allowlist).
  const char** category_denylist;
  uint32_t category_denylist_count;

  // IoU threshold for rotated non-maximum suppression. Default 0.45.
  float iou_threshold;

  // If true, NMS is applied across all classes jointly.
  bool class_agnostic_nms;

  // Output tensor layout: CHANNELS_FIRST=1, CHANNELS_LAST=2.
  int layout;

  // Number of classes. If 0, derived from model metadata at graph build time.
  int num_classes;

  // Result callback for live-stream mode. Must be set iff running_mode is
  // MP_RUNNING_MODE_LIVE_STREAM. The arguments passed to the callback are valid
  // only for the duration of the callback invocation.
  typedef void (*result_callback_fn)(
      MpStatus status, const MpOrientedObjectDetectorResult* result,
      const MpImagePtr image, int64_t timestamp_ms);
  result_callback_fn result_callback;
};
```

The header already `#include <cstdint>` (for `uint32_t`). Keep `class_agnostic_nms`/`layout`/`num_classes`/`result_callback` in their existing order after `iou_threshold`.

- [ ] **Step 2: Copy the three fields in the C→C++ conversion.** In `oriented_object_detector.cc`, `CppConvertToDetectorOptions` currently reads:

```cpp
void CppConvertToDetectorOptions(const MpOrientedObjectDetectorOptions& in,
                                 ObbNs::OrientedObjectDetectorOptions* out) {
  out->max_results = in.max_results;
  out->score_threshold = in.score_threshold;
  out->iou_threshold = in.iou_threshold;
  out->class_agnostic_nms = in.class_agnostic_nms;
  out->layout =
      static_cast<ObbNs::OrientedObjectDetectorOptions::Layout>(in.layout);
  out->num_classes = in.num_classes;
}
```

Replace it with (mirrors YOLO's converter — `display_names_locale` defaults to `"en"` when null; allow/deny copied into `std::vector<std::string>`):

```cpp
void CppConvertToDetectorOptions(const MpOrientedObjectDetectorOptions& in,
                                 ObbNs::OrientedObjectDetectorOptions* out) {
  out->display_names_locale =
      in.display_names_locale ? std::string(in.display_names_locale) : "en";
  out->max_results = in.max_results;
  out->score_threshold = in.score_threshold;
  out->category_allowlist =
      std::vector<std::string>(in.category_allowlist_count);
  for (uint32_t i = 0; i < in.category_allowlist_count; ++i) {
    out->category_allowlist[i] = in.category_allowlist[i];
  }
  out->category_denylist = std::vector<std::string>(in.category_denylist_count);
  for (uint32_t i = 0; i < in.category_denylist_count; ++i) {
    out->category_denylist[i] = in.category_denylist[i];
  }
  out->iou_threshold = in.iou_threshold;
  out->class_agnostic_nms = in.class_agnostic_nms;
  out->layout =
      static_cast<ObbNs::OrientedObjectDetectorOptions::Layout>(in.layout);
  out->num_classes = in.num_classes;
}
```

Confirm the file already includes `<string>` and `<vector>` (it uses `std::string` paths already; add the includes if the compiler complains in Step 5).

- [ ] **Step 3: Harden the C-API test — fix the image + layout, assert names.** In `oriented_object_detector_test.cc`:
  - Change the test image constant (the OBB DOTA model reliably detects a ship on boats.jpg; cats_and_dogs.jpg has no DOTA ground truth):

```cpp
constexpr char kImageFile[] = "boats.jpg";
```

  - In `TEST(OrientedObjectDetectorCApiTest, ImageMode)`, fix the layout to match the model (`yolov8n-obb.tflite` emits `[1,20,8400]` = CHANNELS_FIRST; the current `layout = 2` misreads it and only "passes" because the assertions are weak):

```cpp
  options.layout = 1;  // CHANNELS_FIRST (matches yolov8n-obb.tflite [1,20,8400]).
```

  - Replace the per-detection assertion loop with one that also checks the category name is populated and that the ship (DOTA index 1) is named "ship". Add `#include <cstring>` to the test includes for `strcmp`/`EXPECT_STREQ`:

```cpp
  bool saw_ship = false;
  for (uint32_t i = 0; i < result.detections_count; ++i) {
    EXPECT_GT(result.detections[i].width, 0.0f);
    EXPECT_GT(result.detections[i].height, 0.0f);
    EXPECT_TRUE(std::isfinite(result.detections[i].rotation));
    ASSERT_EQ(result.detections[i].categories_count, 1u);
    const MpCategory& cat = result.detections[i].categories[0];
    EXPECT_GT(cat.score, 0.0f);
    // Plan A populates category_name in-graph from model metadata; the C
    // converter strdup's it into the result.
    ASSERT_NE(cat.category_name, nullptr);
    if (cat.index == 1) {
      EXPECT_STREQ(cat.category_name, "ship");
      saw_ship = true;
    }
  }
  EXPECT_TRUE(saw_ship) << "expected a 'ship' (DOTA class 1) on boats.jpg";
```

- [ ] **Step 4: Add the allow/deny C-API sub-test.** Immediately after the `ImageMode` test (before the closing `}  // namespace`), add a second test that exercises the new struct fields end-to-end. It builds two detectors (allowlist then denylist) and checks the C result:

```cpp
TEST(OrientedObjectDetectorCApiTest, CategoryAllowlistAndDenylistFilterByName) {
  const std::string model_path = GetFullPath(kObbModel);
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "OBB model fixture not available at " << model_path;
  }

  MpImagePtr raw_image = nullptr;
  ASSERT_EQ(MpImageCreateFromFile(GetFullPath(kImageFile).c_str(), &raw_image,
                                  /*error_msg=*/nullptr),
            kMpOk);
  ScopedMpImage image(raw_image);

  auto make_base_options = []() {
    MpOrientedObjectDetectorOptions o = {};
    o.running_mode = MpRunningMode::MP_RUNNING_MODE_IMAGE;
    o.max_results = 10;
    o.score_threshold = 0.25f;
    o.iou_threshold = 0.45f;
    o.num_classes = 15;
    o.layout = 1;  // CHANNELS_FIRST
    return o;
  };

  // Allowlist {"ship"}: only ships (index 1, name "ship") may survive.
  {
    const char* allow[] = {"ship"};
    MpOrientedObjectDetectorOptions options = make_base_options();
    options.base_options.model_asset_path = model_path.c_str();
    options.category_allowlist = allow;
    options.category_allowlist_count = 1;

    MpOrientedObjectDetectorPtr detector = nullptr;
    ASSERT_EQ(MpOrientedObjectDetectorCreate(&options, &detector,
                                             /*error_msg=*/nullptr),
              kMpOk);
    ScopedMpOrientedObjectDetector scoped_detector;
    scoped_detector.ptr = detector;

    MpOrientedObjectDetectorResult result;
    ASSERT_EQ(MpOrientedObjectDetectorDetectImage(detector, image.get(),
                                                  /*options=*/nullptr, &result,
                                                  /*error_msg=*/nullptr),
              kMpOk);
    EXPECT_GT(result.detections_count, 0u) << "allowlist {ship} dropped all";
    for (uint32_t i = 0; i < result.detections_count; ++i) {
      ASSERT_EQ(result.detections[i].categories_count, 1u);
      EXPECT_EQ(result.detections[i].categories[0].index, 1);
      ASSERT_NE(result.detections[i].categories[0].category_name, nullptr);
      EXPECT_STREQ(result.detections[i].categories[0].category_name, "ship");
    }
    MpOrientedObjectDetectorCloseResult(&result);
  }

  // Denylist {"ship"}: ships (index 1) must be excluded.
  {
    const char* deny[] = {"ship"};
    MpOrientedObjectDetectorOptions options = make_base_options();
    options.base_options.model_asset_path = model_path.c_str();
    options.category_denylist = deny;
    options.category_denylist_count = 1;

    MpOrientedObjectDetectorPtr detector = nullptr;
    ASSERT_EQ(MpOrientedObjectDetectorCreate(&options, &detector,
                                             /*error_msg=*/nullptr),
              kMpOk);
    ScopedMpOrientedObjectDetector scoped_detector;
    scoped_detector.ptr = detector;

    MpOrientedObjectDetectorResult result;
    ASSERT_EQ(MpOrientedObjectDetectorDetectImage(detector, image.get(),
                                                  /*options=*/nullptr, &result,
                                                  /*error_msg=*/nullptr),
              kMpOk);
    for (uint32_t i = 0; i < result.detections_count; ++i) {
      ASSERT_EQ(result.detections[i].categories_count, 1u);
      EXPECT_NE(result.detections[i].categories[0].index, 1);
    }
    MpOrientedObjectDetectorCloseResult(&result);
  }
}
```

- [ ] **Step 5: Add the mutual-exclusion C-API test.** The C++ graph rejects allowlist + denylist set together (Plan A Task 8); confirm that error propagates through the C API as a non-`kMpOk` status. Add after the allow/deny test:

```cpp
TEST(OrientedObjectDetectorCApiTest, RejectsAllowlistAndDenylistTogether) {
  const std::string model_path = GetFullPath(kObbModel);
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "OBB model fixture not available at " << model_path;
  }
  const char* allow[] = {"ship"};
  const char* deny[] = {"ship"};
  MpOrientedObjectDetectorOptions options = {};
  options.base_options.model_asset_path = model_path.c_str();
  options.running_mode = MpRunningMode::MP_RUNNING_MODE_IMAGE;
  options.max_results = 10;
  options.score_threshold = 0.25f;
  options.iou_threshold = 0.45f;
  options.num_classes = 15;
  options.layout = 1;
  options.category_allowlist = allow;
  options.category_allowlist_count = 1;
  options.category_denylist = deny;
  options.category_denylist_count = 1;

  MpOrientedObjectDetectorPtr detector = nullptr;
  EXPECT_NE(MpOrientedObjectDetectorCreate(&options, &detector,
                                           /*error_msg=*/nullptr),
            kMpOk);
  if (detector) MpOrientedObjectDetectorClose(detector, /*error_msg=*/nullptr);
}
```

- [ ] **Step 6: Build + run the C-API test.**

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_test --test_output=all`

Expected: target builds, all three tests **RUN** (the `yolov8n-obb.tflite` + `boats.jpg` fixtures are present — not GTEST_SKIP) and PASS. Read the real gtest summary (`[  PASSED  ] 3 tests.`). A piped `exit 0` is NOT proof. If `"ship"` is not observed, report the actual `category_name`/`index` values — do NOT weaken the assertion.

- [ ] **Step 7: Commit.**

```bash
git add mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h \
        mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.cc \
        mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector_test.cc
git commit -m "feat(tasks-obb-c): expose display_names_locale + category_allowlist/denylist (C API)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Fix stale "no-op" comments that Plan A invalidated

Plan A made YOLO's graph apply label mapping + allow/deny, so the binding comments claiming those fields are presently no-ops are now false. Correct them.

**Files:**
- Modify: `mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h`
- Modify: `mediapipe/tasks/python/vision/yolo_object_detector.py`

- [ ] **Step 1: Fix the YOLO C header comment.** In `yolo_object_detector.h`, replace the stale NOTE block above `struct MpYoloObjectDetectorOptions` (the paragraph beginning "NOTE: display_names_locale, category_allowlist, and category_denylist are passed through faithfully ... presently no-ops; only score_threshold / iou_threshold / max_results affect output today.") with:

```c
// The options for configuring a MediaPipe YOLO object detector task.
//
// display_names_locale, category_allowlist, and category_denylist are applied
// by the YOLO graph: category names are read from the model metadata's label
// file and allow/deny filter results by class name (resolved to indices).
```

- [ ] **Step 2: Fix the YOLO Python dataclass docstring.** In `yolo_object_detector.py`, replace the stale trailing NOTE in the `YoloObjectDetectorOptions` docstring (the paragraph "NOTE: display_names_locale / category_allowlist / category_denylist are accepted for parity with the C++ task, but the current YOLO graph does not apply label mapping or allowlist/denylist filtering. Category names are populated best-effort in Python from TFLite metadata (None if unavailable).") with:

```python
  Category names are populated by the YOLO graph from the model metadata's
  label file, and category_allowlist / category_denylist filter results by
  class name. The Python `_load_label_map` fallback below is a display-only
  safety net for models whose graph did not populate names; it never overrides
  a name the graph already provided and does not implement filtering.
```

- [ ] **Step 3: Verify both changes are comment-only and valid.**

Run: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_lib`
Expected: builds (comment-only change compiles).

Run: `python3 -m py_compile mediapipe/tasks/python/vision/yolo_object_detector.py && echo PYOK`
Expected: prints `PYOK` (no syntax error). Note: `py_compile` only parses; it does not import mediapipe.

- [ ] **Step 4: Commit.**

```bash
git add mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h \
        mediapipe/tasks/python/vision/yolo_object_detector.py
git commit -m "docs(tasks-yolo): correct stale 'no-op' comments — Plan A applies names + filter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: OBB Python dataclass + ctypes + conversion (BUILD-DEFERRED)

Mirror the YOLO Python option surface into the OBB Python task. Cannot run here (needs the monolith `.so`); mirror the verified YOLO file exactly and `py_compile`-check.

**Files:**
- Modify: `mediapipe/tasks/python/vision/oriented_object_detector.py`

- [ ] **Step 1: Add the three fields to the ctypes struct — byte-match the new C header.** In `MpOrientedObjectDetectorOptionsC._fields_`, the order MUST exactly match `struct MpOrientedObjectDetectorOptions` after Task 1 (display_names_locale after running_mode; allowlist/denylist + counts after score_threshold; class_agnostic_nms stays after iou_threshold). Replace the `_fields_` list with:

```python
  _fields_ = [
      ('base_options', base_options_c_module.MpBaseOptionsC),
      ('running_mode', ctypes.c_int),
      ('display_names_locale', ctypes.c_char_p),
      ('max_results', ctypes.c_int),
      ('score_threshold', ctypes.c_float),
      ('category_allowlist', ctypes.POINTER(ctypes.c_char_p)),
      ('category_allowlist_count', ctypes.c_uint32),
      ('category_denylist', ctypes.POINTER(ctypes.c_char_p)),
      ('category_denylist_count', ctypes.c_uint32),
      ('iou_threshold', ctypes.c_float),
      ('class_agnostic_nms', ctypes.c_bool),
      ('layout', ctypes.c_int),
      ('num_classes', ctypes.c_int),
      ('result_callback', _C_TYPES_RESULT_CALLBACK),
  ]
```

- [ ] **Step 2: Add the three fields to the dataclass + update the stale NOTE.** In `OrientedObjectDetectorOptions`, add the attribute docs (after `running_mode`, mirror YOLO wording) and the three fields (after `score_threshold`), and replace the trailing `NOTE: category names ... the OBB graph emits index+score only.` Final dataclass:

```python
@dataclasses.dataclass
class OrientedObjectDetectorOptions:
  """Options for the oriented (OBB) object detector task.

  Attributes:
    base_options: Base options for the oriented object detector task.
    running_mode: The running mode of the task. Default to the image mode.
      Oriented object detector task has three running modes: 1) The image mode
      for detecting objects on single image inputs. 2) The video mode for
      detecting objects on the decoded frames of a video. 3) The live stream
      mode for detecting objects on a live stream of input data, such as from
      camera.
    display_names_locale: The locale to use for display names specified through
      the TFLite Model Metadata.
    max_results: The maximum number of top-scored detection results to return.
    score_threshold: Overrides the ones provided in the model metadata. Results
      below this value are rejected. Default 0.25.
    category_allowlist: Allowlist of category names. If non-empty, detection
      results whose category name is not in this set will be filtered out.
      Duplicate or unknown category names are ignored. Mutually exclusive with
      `category_denylist`.
    category_denylist: Denylist of category names. If non-empty, detection
      results whose category name is in this set will be filtered out. Duplicate
      or unknown category names are ignored. Mutually exclusive with
      `category_allowlist`.
    iou_threshold: IoU threshold for rotated non-maximum suppression. Default
      0.45.
    class_agnostic_nms: If True, NMS is applied across all classes jointly.
    layout: The output tensor layout of the OBB detect head. Default
      CHANNELS_FIRST.
    num_classes: Number of classes. If 0, derived from model metadata.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.

  Category names are populated by the OBB graph from the model metadata's label
  file, and category_allowlist / category_denylist filter results by class name.
  The Python `_load_label_map` fallback below is a display-only safety net; it
  never overrides a name the graph already provided and does not implement
  filtering.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  display_names_locale: Optional[str] = None
  max_results: Optional[int] = -1
  score_threshold: Optional[float] = 0.25
  category_allowlist: Optional[List[str]] = None
  category_denylist: Optional[List[str]] = None
  iou_threshold: float = 0.45
  class_agnostic_nms: bool = False
  layout: Layout = Layout.CHANNELS_FIRST
  num_classes: int = 0
  result_callback: Optional[
      Callable[[OrientedObjectDetectorResult, image_module.Image, int], None]
  ] = None
```

- [ ] **Step 3: Build the allow/deny ctypes arrays + pass the three fields in `create_from_options`.** Replace the `ctypes_options = MpOrientedObjectDetectorOptionsC(...)` construction (and add the two `convert_strings_to_ctypes_array` calls just before it, mirroring YOLO) so the block reads:

```python
    allowlist_c = mediapipe_c_bindings_c_module.convert_strings_to_ctypes_array(
        options.category_allowlist
    )
    denylist_c = mediapipe_c_bindings_c_module.convert_strings_to_ctypes_array(
        options.category_denylist
    )
    ctypes_options = MpOrientedObjectDetectorOptionsC(
        base_options=options.base_options.to_ctypes(),
        running_mode=options.running_mode.ctype,
        display_names_locale=(
            options.display_names_locale.encode('utf-8')
            if options.display_names_locale
            else None
        ),
        max_results=options.max_results,
        score_threshold=options.score_threshold,
        category_allowlist=allowlist_c,
        category_allowlist_count=(
            len(options.category_allowlist) if options.category_allowlist else 0
        ),
        category_denylist=denylist_c,
        category_denylist_count=(
            len(options.category_denylist) if options.category_denylist else 0
        ),
        iou_threshold=options.iou_threshold,
        class_agnostic_nms=options.class_agnostic_nms,
        layout=int(options.layout),
        num_classes=options.num_classes,
        result_callback=c_callback,
    )
```

- [ ] **Step 4: Syntax-check (runnable here).**

Run: `python3 -m py_compile mediapipe/tasks/python/vision/oriented_object_detector.py && echo PYOK`
Expected: prints `PYOK`.

**DEFERRED full verification** (run once `libmediapipe.so` builds): `bazel test //mediapipe/tasks/python/test/vision:oriented_object_detector_test`. State in the commit body that runtime verification is deferred and the layout mirrors the verified YOLO ctypes struct.

- [ ] **Step 5: Commit.**

```bash
git add mediapipe/tasks/python/vision/oriented_object_detector.py
git commit -m "feat(tasks-obb-py): expose display_names_locale + category_allowlist/denylist (Python)

Mirrors the verified YOLO Python ctypes layout; runtime verification deferred
(needs libmediapipe.so, not buildable in this environment).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: OBB Python test hardening + fallback note (BUILD-DEFERRED)

Add Python-level assertions proving the graph path provides names + filtering without relying on the `_load_label_map` fallback. Cannot run here; mirror the YOLO Python test and `py_compile`-check.

**Files:**
- Modify: `mediapipe/tasks/python/test/vision/oriented_object_detector_test.py`

- [ ] **Step 1: Read the YOLO Python test for the established pattern.** Open `mediapipe/tasks/python/test/vision/yolo_object_detector_test.py` and find how it (a) gates on the model fixture, (b) builds `*Options`, (c) asserts `category_name`. Mirror its fixture-gating + option-construction idioms exactly (helper names, skip mechanism, import aliases).

- [ ] **Step 2: Add a names + allow/deny test for OBB.** In `oriented_object_detector_test.py`, add a test mirroring the YOLO one but for the OBB detector on `boats.jpg` (ship = DOTA index 1). Use the same fixture-gating the file already uses for the existing OBB tests. The assertions:

```python
  def test_category_names_and_allow_deny_filter(self):
    # Gate on the model fixture exactly as the other tests in this file do.
    if not os.path.exists(self.model_path):
      self.skipTest('yolov8n-obb.tflite fixture not available')

    base = _BaseOptions(model_asset_path=self.model_path)
    image = _Image.create_from_file(self.boats_image_path)

    # Names come from the graph (not the Python fallback).
    options = _OrientedObjectDetectorOptions(
        base_options=base, num_classes=15, score_threshold=0.25,
        max_results=10)
    with _OrientedObjectDetector.create_from_options(options) as detector:
      result = detector.detect(image)
      self.assertTrue(result.detections)
      saw_ship = False
      for det in result.detections:
        cat = det.categories[0]
        self.assertIsNotNone(cat.category_name)
        if cat.index == 1:
          self.assertEqual(cat.category_name, 'ship')
          saw_ship = True
      self.assertTrue(saw_ship)

    # Allowlist {"ship"} -> only ships.
    allow_options = _OrientedObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        num_classes=15, score_threshold=0.25, max_results=10,
        category_allowlist=['ship'])
    with _OrientedObjectDetector.create_from_options(allow_options) as detector:
      result = detector.detect(image)
      self.assertTrue(result.detections)
      for det in result.detections:
        self.assertEqual(det.categories[0].index, 1)
        self.assertEqual(det.categories[0].category_name, 'ship')

    # Denylist {"ship"} -> no ships.
    deny_options = _OrientedObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        num_classes=15, score_threshold=0.25, max_results=10,
        category_denylist=['ship'])
    with _OrientedObjectDetector.create_from_options(deny_options) as detector:
      result = detector.detect(image)
      for det in result.detections:
        self.assertNotEqual(det.categories[0].index, 1)
```

Adapt the symbol aliases (`_BaseOptions`, `_Image`, `_OrientedObjectDetector`, `_OrientedObjectDetectorOptions`), the `self.model_path` / `self.boats_image_path` attribute names, and `import os` to whatever the file already defines in Step 1. Do not introduce a second model/image path convention — reuse the file's existing setup.

- [ ] **Step 3: Syntax-check (runnable here).**

Run: `python3 -m py_compile mediapipe/tasks/python/test/vision/oriented_object_detector_test.py && echo PYOK`
Expected: prints `PYOK`.

**DEFERRED full verification** (run once `libmediapipe.so` builds): `bazel test //mediapipe/tasks/python/test/vision:oriented_object_detector_test --test_output=all`.

- [ ] **Step 4: Commit.**

```bash
git add mediapipe/tasks/python/test/vision/oriented_object_detector_test.py
git commit -m "test(tasks-obb-py): assert graph-provided names + allow/deny (deferred run)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review checklist
- C struct field ORDER (Task 1 Step 1) and the Python ctypes `_fields_` (Task 3 Step 1) are identical, member-for-member, including `class_agnostic_nms` after `iou_threshold` (the one OBB-specific field absent from YOLO). A mismatch silently corrupts every option.
- C converter (Task 1 Step 2) copies all three new fields; `display_names_locale` null → `"en"`.
- C test fixed `layout = 1` (CHANNELS_FIRST) wherever it builds OBB options — the model is `[1,20,8400]`; the old `layout = 2` was a latent bug masked by weak assertions.
- The two `category_name` ownership paths are unchanged: the C result strdup's it (`category_converter.cc`), Python `from_ctypes` copies it; no double-free introduced (we add only option-side fields, not result-side).
- Stale-comment fixes (Task 2) are comment-only; no behavior change.
- Python fallback (`_enrich_with_label_map`) is left display-only and still only sets `category_name` when it is `None` — it must never override a graph-provided name or implement allow/deny.

## Final verification
- [ ] `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/oriented_object_detector:oriented_object_detector_test --test_output=all` → both C-API tests run + pass.
- [ ] `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c/vision/yolo_object_detector:yolo_object_detector_lib` → builds.
- [ ] `python3 -m py_compile` clean for all three touched Python files.
- [ ] `git status` clean except gitignored model fixtures.
- [ ] DEFERRED (record, do not block): the two `py_test` targets, to be run when `libmediapipe.so` builds.

## Out of scope
iOS / Java / Web bindings; letterbox/aspect-preserving preprocessing; score calibration; any change to the YOLO Python ctypes/dataclass (YOLO already exposes the three fields and is verified) beyond the Task 2 comment fix.
