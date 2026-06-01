# YoloObjectDetector (Tasks API) Implementation Plan (Phase 2.1a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `YoloObjectDetector` C++ Task (`tasks/cc/vision/yolo_object_detector/`) exposing axis-aligned YOLO detection through the high-level Tasks API, parallel to the untouched `ObjectDetector`, returning the shared `DetectionResult` (pixel units).

**Architecture:** A thin Tasks-layer wrapper that mirrors `ObjectDetector` exactly, swapping only the postprocessing: where `ObjectDetectorGraph` uses `DetectionPostprocessingGraph` (SSD), `YoloObjectDetectorGraph` uses the Group-1 `YoloTensorsToDetectionsCalculator` + `NonMaxSuppressionCalculator`. Everything else (image preprocessing, `AddInference`, detection projection → transformation → dedup, `BaseVisionTaskApi` wrapper, options proto) is reused/copied from `ObjectDetector`.

**Tech Stack:** C++17, MediaPipe Tasks (`ModelTaskGraph`, `BaseVisionTaskApi`, builder `Graph`), protobuf2, Bazel.

**Spec:** `docs/superpowers/specs/2026-06-01-tasks-api-yolo-obb-design.md` (§4.1, §5, §6).

**Depends on:** Group 1 (`YoloTensorsToDetectionsCalculator`, the `METADATA` seam) — done on `dev`.

## TESTABILITY NOTE (read first)

The Tasks layer cannot be unit-tested without a model: building the graph needs `core::ModelResources` loaded from a `.tflite`. Therefore this plan is **author + build-verify**, with one **integration test gated** on a real `yolov8n.tflite` fixture (per the spec's gating decision). "Run" steps below are `bazel build` (verifiable now) except the final integration test, which is authored but **skipped until the fixture exists**. There is no locally-runnable red-green TDD loop here — that rigor lives in Plan 2.1b's result-container converter. Do not fabricate a passing integration test without the model.

---

## File structure

- Create: `mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.proto`
- Create: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc`
- Create: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h`
- Create: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.cc`
- Create: `mediapipe/tasks/cc/vision/yolo_object_detector/object_detector_result.h` *(alias only)* — or reuse the shared one directly.
- Create: `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`
- Create: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc` (gated)

Reference files to mirror (read them before writing): `mediapipe/tasks/cc/vision/object_detector/{object_detector.h,object_detector.cc,object_detector_graph.cc,BUILD,proto/object_detector_options.proto,object_detector_test.cc}`.

All commands run from repo root with `--define MEDIAPIPE_DISABLE_GPU=1`.

---

### Task 1: Options proto

**Files:**
- Create: `mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.proto`
- Create: `mediapipe/tasks/cc/vision/yolo_object_detector/proto/BUILD`

- [ ] **Step 1: Pick a unique extension id.** Run `grep -rhoE "ext = [0-9]+" mediapipe | sort -t= -k2 -n | tail -25`. Pick an unused number; this plan uses `471230010` (change if taken).

- [ ] **Step 2: Read the sibling** `mediapipe/tasks/cc/vision/object_detector/proto/object_detector_options.proto` to copy its structure exactly (it imports `core/proto/base_options.proto` and `framework/calculator.proto`).

- [ ] **Step 3: Write the proto.** Create `yolo_object_detector_options.proto` — identical to `ObjectDetectorOptions` plus a YOLO decode block:

```proto
// Copyright 2026 The MediaPipe Authors. (full Apache 2.0 header)
syntax = "proto2";

package mediapipe.tasks.vision.yolo_object_detector.proto;

import "mediapipe/framework/calculator.proto";
import "mediapipe/tasks/cc/core/proto/base_options.proto";

option java_package = "com.google.mediapipe.tasks.vision.yoloobjectdetector.proto";
option java_outer_classname = "YoloObjectDetectorOptionsProto";

message YoloObjectDetectorOptions {
  extend mediapipe.CalculatorOptions {
    optional YoloObjectDetectorOptions ext = 471230010;
  }
  optional core.proto.BaseOptions base_options = 1;
  optional string display_names_locale = 2 [default = "en"];
  optional int32 max_results = 3 [default = -1];
  optional float score_threshold = 4 [default = 0.25];
  repeated string category_allowlist = 5;
  repeated string category_denylist = 6;
  // YOLO NMS IoU threshold (axis-aligned).
  optional float iou_threshold = 7 [default = 0.45];
  // Output tensor layout of the YOLO detect head.
  enum Layout {
    LAYOUT_UNSPECIFIED = 0;
    CHANNELS_FIRST = 1;
    CHANNELS_LAST = 2;
  }
  optional Layout layout = 8 [default = CHANNELS_FIRST];
  // Number of classes. If unset (0), derived from model metadata at build time.
  optional int32 num_classes = 9 [default = 0];
}
```

- [ ] **Step 4: Write `proto/BUILD`** mirroring `object_detector/proto/BUILD` (uses `mediapipe_proto_library`):

```python
# Copyright 2026 ... (license)
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")
package(default_visibility = ["//mediapipe/tasks:internal"])
licenses(["notice"])

mediapipe_proto_library(
    name = "yolo_object_detector_options_proto",
    srcs = ["yolo_object_detector_options.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_proto",
    ],
)
```

- [ ] **Step 5: Build the proto.**

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector/proto:yolo_object_detector_options_cc_proto
```
Expected: success.

- [ ] **Step 6: Commit.**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/proto/
git commit -m "feat(tasks-yolo): YoloObjectDetector options proto"
```

---

### Task 2: `YoloObjectDetectorGraph`

**Files:**
- Create: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc`
- Create/extend: `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`

This graph mirrors `object_detector_graph.cc::BuildObjectDetectionTask` (which you should re-read in full) and changes **only** the postprocessing block.

- [ ] **Step 1: Read** `mediapipe/tasks/cc/vision/object_detector/object_detector_graph.cc` end-to-end. Note the includes, the tag constants (`kImageTag`, `kNormRectTag`, `kTensorTag`, `kDetectionsTag`, `kMatrixTag`, `kImageSizeTag`, `kPixelDetectionsTag`, `kProjectionMatrixTag`), the `ObjectDetectionOutputStreams` struct, `SanityCheckOptions`, the `GetConfig` entry point, `AddInference`, and the `REGISTER_MEDIAPIPE_GRAPH` line.

- [ ] **Step 2: Write `yolo_object_detector_graph.cc`** as a copy of `object_detector_graph.cc` with these exact changes:
  1. Namespace/option types → `YoloObjectDetectorOptions` (the proto from Task 1).
  2. Graph class name → `YoloObjectDetectorGraph`; registered name `::mediapipe::tasks::vision::yolo_object_detector::YoloObjectDetectorGraph` via `REGISTER_MEDIAPIPE_GRAPH`.
  3. **Replace the postprocessing block** (the `DetectionPostprocessingGraph` node, lines ~210–231 of the original) with:

```cpp
    // YOLO decode: raw tensors -> axis-aligned Detections (batched).
    auto& yolo_decode = graph.AddNode("YoloTensorsToDetectionsCalculator");
    {
      auto& opts = yolo_decode.GetOptions<
          mediapipe::YoloTensorsToDetectionsCalculatorOptions>();
      // num_classes: prefer the option; else derive from model metadata.
      int num_classes = task_options.num_classes();
      if (num_classes <= 0) {
        MP_ASSIGN_OR_RETURN(num_classes,
                            GetNumClassesFromMetadata(model_resources));
      }
      opts.set_num_classes(num_classes);
      opts.set_conf_threshold(task_options.score_threshold());
      switch (task_options.layout()) {
        case YoloObjectDetectorOptions::CHANNELS_LAST:
          opts.set_layout(mediapipe::YoloTensorsToDetectionsCalculatorOptions::
                              CHANNELS_LAST);
          break;
        default:
          opts.set_layout(mediapipe::YoloTensorsToDetectionsCalculatorOptions::
                              CHANNELS_FIRST);
      }
    }
    model_output_tensors >> yolo_decode.In(kTensorTag);
    // The YOLO decoder emits std::vector<std::vector<Detection>> (one inner
    // vector per batch row). For the single-image Task, batch N==1: take row 0.
    auto& batch_to_single =
        graph.AddNode("YoloBatchDetectionsToSingleCalculator");
    yolo_decode.Out(kDetectionsTag) >> batch_to_single.In(kDetectionsTag);

    // Axis-aligned NMS over the flattened detections.
    auto& nms = graph.AddNode("NonMaxSuppressionCalculator");
    {
      auto& nms_opts =
          nms.GetOptions<mediapipe::NonMaxSuppressionCalculatorOptions>();
      nms_opts.set_min_suppression_threshold(task_options.iou_threshold());
      nms_opts.set_max_num_detections(task_options.max_results());
      nms_opts.set_overlap_type(
          mediapipe::NonMaxSuppressionCalculatorOptions::INTERSECTION_OVER_UNION);
      nms_opts.set_return_empty_detections(true);
    }
    batch_to_single.Out(kDetectionsTag) >> nms.In("");
    auto detections = nms.Out("");
```
  4. Keep the rest (projection → transformation → dedup → outputs) **unchanged**.

> Notes for the implementer:
> - **Ordering:** `YoloBatchDetectionsToSingleCalculator` (flatten `vector<vector<Detection>>` → `vector<Detection>` for N==1) is created in **Task 3**. The graph `.cc` here *compiles* without it (nodes are added by string name, resolved at runtime), but the graph's BUILD `deps` (Step 3) reference its target. **Do Task 3 before Step 3/Step 4 of this task** (it is independent and locally testable), then add its dep and build the graph.
> - `GetNumClassesFromMetadata(model_resources)` is a small local helper: read the output tensor's class dimension from the model, or the TFLite label-map size. If deriving is non-trivial for a given export, require `num_classes` in options and `RET_CHECK` it (`> 0`). Confirm against a real yolov8n.tflite during the gated test.
> - Confirm the exact `NonMaxSuppressionCalculatorOptions` field names against `mediapipe/calculators/util/non_max_suppression_calculator.proto` (e.g. `min_suppression_threshold`, `max_num_detections`, `overlap_type`).

- [ ] **Step 3: Add the graph `cc_library`** to `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`, mirroring the `object_detector_graph` target's deps and adding: `//mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator`, `//mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_cc_proto`, `//mediapipe/calculators/util:non_max_suppression_calculator` (+ its cc_proto), `:yolo_object_detector_options_cc_proto`, and the shared tasks graph deps (`//mediapipe/tasks/cc/core:model_task_graph`, `//mediapipe/tasks/cc/components/processors:image_preprocessing_graph`, the projection/transformation/dedup calculators `//mediapipe/calculators/util:detection_projection_calculator`, `:detection_transformation_calculator`, `//mediapipe/tasks/cc/components/calculators:detections_deduplicate_calculator`). Copy the exact label set from `object_detector/BUILD`'s `object_detector_graph` target and add the YOLO ones.

- [ ] **Step 4: Build the graph.**

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_graph
```
Expected: success. Resolve any missing dep labels by grepping the sibling BUILD.

- [ ] **Step 5: Commit.**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc mediapipe/tasks/cc/vision/yolo_object_detector/BUILD
git commit -m "feat(tasks-yolo): YoloObjectDetectorGraph (YOLO decode + NMS)"
```

---

### Task 3: Batch→single flatten calculator (helper)

The YOLO decoder emits `std::vector<std::vector<Detection>>` (batched, §5.4 of Group 1). For the single-image Task (N==1), flatten to `std::vector<Detection>`. This is locally unit-testable (no model).

**Files:**
- Create: `mediapipe/calculators/tensor/yolo_batch_detections_to_single_calculator.cc`
- Create: `mediapipe/calculators/tensor/yolo_batch_detections_to_single_calculator_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

- [ ] **Step 1: Write the failing test** (`..._test.cc`): feed a `std::vector<std::vector<Detection>>` of `{{det_a, det_b}}` (one batch row, two detections) on tag `DETECTIONS`, expect a `std::vector<Detection>` of size 2 out; and `{{det_a},{det_b}}` (two rows) → RET_CHECK error (single-image Task expects N==1). Use the `CalculatorRunner` idiom from `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc`.

- [ ] **Step 2: Run → fail** (`No calculator "YoloBatchDetectionsToSingleCalculator"`), after adding the BUILD target.

- [ ] **Step 3: Write the calculator** (api2 `Node`): `Input<std::vector<std::vector<Detection>>> kIn{"DETECTIONS"}`, `Output<std::vector<Detection>> kOut{"DETECTIONS"}`; in `Process`, `RET_CHECK_LE(in.size(), 1)`, send `in.empty() ? {} : in[0]`. Full Apache header; `MEDIAPIPE_REGISTER_NODE`.

- [ ] **Step 4: Add BUILD lib + test; run.**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_batch_detections_to_single_calculator_test --test_output=all
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit.**

```bash
git add mediapipe/calculators/tensor/yolo_batch_detections_to_single_calculator*.* mediapipe/calculators/tensor/BUILD
git commit -m "feat(yolo): batch->single Detections flatten calculator"
```

---

### Task 4: `YoloObjectDetector` Task class

**Files:**
- Create: `yolo_object_detector.h`, `yolo_object_detector.cc`
- Modify: `BUILD`

This is wrapper boilerplate ~identical to `object_detector.{h,cc}`. **Copy them and rename**; do not hand-write from scratch.

- [ ] **Step 1: Read** `object_detector.h` and `object_detector.cc` fully.

- [ ] **Step 2: Create `yolo_object_detector.h`** by copying `object_detector.h` and changing: class `ObjectDetector`→`YoloObjectDetector`; `ObjectDetectorOptions`→`YoloObjectDetectorOptions`; the result alias stays `using YoloObjectDetectorResult = ::mediapipe::tasks::components::containers::DetectionResult;`; the graph type name string → `"mediapipe.tasks.vision.yolo_object_detector.YoloObjectDetectorGraph"`; keep `Create/Detect/DetectForVideo/DetectAsync` signatures (swap the result/option type names). Namespace → `mediapipe::tasks::vision::yolo_object_detector`.

- [ ] **Step 3: Create `yolo_object_detector.cc`** by copying `object_detector.cc` with the same renames: the options→proto conversion populates `YoloObjectDetectorOptions` proto fields (base_options, score_threshold, max_results, allow/deny, iou_threshold, layout, num_classes); the graph name string matches Step 2; result conversion reuses `ConvertToDetectionResult` (same as ObjectDetector). Keep the `BaseVisionTaskApi`/`TaskRunner` wiring identical.

- [ ] **Step 4: Add the `cc_library` targets** (`yolo_object_detector` + any `_graph` dep) to `BUILD`, mirroring `object_detector/BUILD`'s `object_detector` target deps, swapping in `:yolo_object_detector_graph` and `:yolo_object_detector_options_cc_proto`.

- [ ] **Step 5: Build.**

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector
```
Expected: success.

- [ ] **Step 6: Commit.**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/
git commit -m "feat(tasks-yolo): YoloObjectDetector Task class"
```

---

### Task 5: Gated integration test

**Files:**
- Create: `yolo_object_detector_test.cc`
- Modify: `BUILD`

- [ ] **Step 1: Locate / add a YOLO fixture.** Check whether a `yolov8n.tflite` (or similar small YOLO detect model) exists under `mediapipe/tasks/testdata/vision/`. If not, this test is **gated**: write it, add the `data` dep referencing the intended path, but it will not run until the fixture is supplied. Document the exact expected path (e.g. `mediapipe/tasks/testdata/vision/yolov8n.tflite`).

- [ ] **Step 2: Write the test** mirroring `object_detector_test.cc`: `Create` a `YoloObjectDetector` from the fixture, `Detect` on an existing test image (reuse one referenced in `object_detector_test.cc`'s data deps), assert non-empty detections with plausible categories/boxes in pixel units; a `DetectForVideo` case; a `DetectAsync` callback case.

- [ ] **Step 3: Add the `cc_test`** to `BUILD` (mirror `object_detector_test`'s deps + `data`), with the YOLO model `data` dep.

- [ ] **Step 4: Run if the fixture exists; otherwise record gated.**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_output=all
```
Expected: PASS **if** the fixture is present. If absent, the build of the test target should still succeed (compile-only); note in the commit that the run is gated on the model fixture. **Do not fake a pass.**

- [ ] **Step 5: Commit.**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc mediapipe/tasks/cc/vision/yolo_object_detector/BUILD
git commit -m "test(tasks-yolo): YoloObjectDetector integration test (gated on model fixture)"
```

---

## Done criteria (2.1a)

- `yolo_object_detector_options_cc_proto`, `:yolo_object_detector_graph`, and `:yolo_object_detector` all **build** (`--define MEDIAPIPE_DISABLE_GPU=1`).
- `yolo_batch_detections_to_single_calculator_test` is **green** (the one locally-runnable test).
- `YoloObjectDetector` exposes `Create/Detect/DetectForVideo/DetectAsync` returning the shared `DetectionResult`, wrapping the Group-1 YOLO decoder; `ObjectDetector` is untouched.
- Integration test authored; **runs once a `yolov8n.tflite` fixture is added** (gated).

## Handoff to Plan 2.1b

Plan 2.1b adds `OrientedObjectDetector` (OBB): a new `OrientedObjectDetectionResult` container + `ConvertToOrientedObjectDetectionResult` (locally unit-testable, no model), an `OrientedObjectDetectorGraph` (YOLO-OBB decode + `RotatedNonMaxSuppressionCalculator` + oriented projection), and the Task class — reusing this plan's structure.
