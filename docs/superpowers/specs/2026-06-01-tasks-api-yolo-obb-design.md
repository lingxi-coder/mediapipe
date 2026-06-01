# Tasks API — YOLO + OBB Detectors (Phase 2.1, cc layer)

- **Date:** 2026-06-01
- **Status:** Approved (design); pending implementation plan.
- **Roadmap:** Phase 2 (Tasks API exposure) in `2026-06-01-roadmap.md`. This spec is **sub-project 2.1 — the C++ (cc) layer only**; Python/Web/Java/iOS bindings are later follow-ons (2.2–2.5) that wrap what is built here.
- **Build model:** Fork / custom build of MediaPipe.

## 1. Background & goal

Group 1 (Phase 1) built the YOLO/OBB detection pipeline at the **calculator/graph** layer: batch-native YOLO decode (`YoloTensorsToDetectionsCalculator`), OBB decode (`YoloObbTensorsToOrientedDetectionsCalculator`) + rotated NMS (`RotatedNonMaxSuppressionCalculator`), tiling, and the M4 `METADATA` seam on `InferenceCalculator`. None of it is reachable through the high-level **Tasks API** — app developers can't yet write `detector.Detect(image)` for a YOLO or OBB model.

**Goal of 2.1:** add the C++ Tasks-layer detectors that expose **single-image / video / live-stream YOLO detection** (axis-aligned and oriented), wrapping the Group-1 calculators. This is the foundation that the language bindings later wrap.

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Binding scope | **cc Task layer only** (2.1); Python/Web/Java/iOS are separate follow-ons | cc is the foundation every binding wraps; it must exist and be correct first. Locally testable. |
| Tiling in the API | **Not exposed.** Tasks API is detection-only (`Detect(image)`) | Tiling needs an externally-supplied per-frame tile *stream*, which doesn't fit the single-image `Detect(image)` shape. Tiling stays a calculator/graph feature for advanced users. |
| Relationship to `ObjectDetector` | **New parallel Task(s); `ObjectDetector` untouched** | Matches Group 1's rebaseable-fork principle (parallel new calculators, SSD path untouched). No regression risk to the battle-tested `ObjectDetector`. |
| Task / result structure | **Two Tasks:** `YoloObjectDetector` (axis-aligned → shared `DetectionResult`) + `OrientedObjectDetector` (OBB → new oriented result type) | One clean result type per Task — idiomatic for strongly-typed bindings later. Mirrors Group 1's separate `Detection` vs `OrientedDetection` streams. |
| 2.1 scope | **Both Tasks**, sequenced A (axis-aligned) then B (OBB) | They share the graph-wrapping pattern; OBB adds the new container type on top. |
| Coordinates | **Pixel units** in results | Matches the existing Tasks API convention (`ObjectDetector` returns pixel-unit boxes); convert from normalized using input image size. |
| Metadata | M4 `METADATA` side packet consumed **internally** to size preprocessing; **no public accessor** in 2.1 | YAGNI; a getter can be added later if a binding needs it. |
| Model-fixture tests | Integration `Detect()` tests are **gated on provided `yolov8n.tflite` / `yolov8n-obb.tflite` fixtures**; option/result-conversion logic tested without a model | Group 1 used hand-built golden tensors + tiny `add.bin`; real Task tests need real exported models, a new asset dependency. |

## 3. Current-state findings (codebase, verified)

- `ObjectDetector` is layered as: `tasks/cc/vision/object_detector/` (C++ Task + `object_detector_graph.cc` + `proto/object_detector_options.proto`), then `tasks/c/`, `tasks/python/`, `tasks/java/`, `tasks/web/`, `tasks/ios/`. 2.1 touches only the cc layer.
- C++ public shape (`object_detector.h`): `struct ObjectDetectorOptions { base_options; display_names_locale; max_results; score_threshold; category_allowlist/denylist; multiclass_nms; min_suppression_threshold; running_mode; result_callback }`; `static Create(...)`; `Detect(image, opts)`, `DetectForVideo(image, ts, opts)`, `DetectAsync(image, ts, opts)`; class derives `BaseVisionTaskApi`.
- `ObjectDetectorResult = components::containers::DetectionResult` (axis-aligned, pixel units). There is **no** oriented container at the Tasks layer.
- `object_detector_graph.cc` is a `ModelTaskGraph` subclass that wires `image_preprocessing_graph` → SSD inference/postprocessing. The new graphs follow the same skeleton but swap in the Group-1 YOLO/OBB calculators.
- Group-1 calculators available (done, on `dev`): `YoloTensorsToDetectionsCalculator`, `YoloObbTensorsToOrientedDetectionsCalculator`, `RotatedNonMaxSuppressionCalculator`, `OrientedDetection` proto, the `METADATA` side packet on `InferenceCalculator`. Axis-aligned NMS reuses `non_max_suppression_calculator`.

## 4. Architecture

Two Tasks, each = {options proto, C++ Task class, `ModelTaskGraph` subclass, result conversion}. Both reuse `image_preprocessing_graph`, `InferenceCalculator` (+`METADATA`), and `BaseVisionTaskApi`.

```
Image ─► ImagePreprocessingGraph ─► InferenceCalculator(+METADATA side pkt)
                                          │ raw output tensors [N,…]
                  ┌───────────────────────┴───────────────────────┐
   (YoloObjectDetector)                              (OrientedObjectDetector)
   YoloTensorsToDetectionsCalculator                 YoloObbTensorsToOrientedDetectionsCalculator
        ▼                                                  ▼
   NonMaxSuppression (reuse)                          RotatedNonMaxSuppressionCalculator
        ▼                                                  ▼
   project → pixel coords                             project → pixel coords
        ▼                                                  ▼
   DetectionResult (shared)                           OrientedObjectDetectionResult (new)
```

### 4.1 Task A — `YoloObjectDetector` (axis-aligned)
- **Path:** `tasks/cc/vision/yolo_object_detector/` (`yolo_object_detector.{h,cc}`, `yolo_object_detector_graph.cc`, `proto/yolo_object_detector_options.proto`, `BUILD`, tests).
- **Options** (`YoloObjectDetectorOptions`): `base_options`, `score_threshold`, `iou_threshold`, `max_results`, `category_allowlist/denylist`, `display_names_locale`, `running_mode`, `result_callback`. (Layout/`num_classes` derived from `METADATA` when available; an explicit `layout` option is the fallback.)
- **Result:** `using YoloObjectDetectorResult = components::containers::DetectionResult;` — drop-in compatible with `ObjectDetector` consumers.
- **API:** `Create / Detect / DetectForVideo / DetectAsync`, deriving `BaseVisionTaskApi` — identical shape to `ObjectDetector`.
- **Graph (`YoloObjectDetectorGraph`):** `image_preprocessing_graph` → `InferenceCalculator` (wire `METADATA`) → `YoloTensorsToDetectionsCalculator` → `NonMaxSuppressionCalculator` → detections-to-pixel projection → `DETECTIONS` out.

### 4.2 Task B — `OrientedObjectDetector` (OBB)
- **Path:** `tasks/cc/vision/oriented_object_detector/` (parallel file set).
- **New result container:** `components::containers::OrientedObjectDetectionResult` = `std::vector<OrientedObjectDetection>`, where `OrientedObjectDetection { float cx, cy, width, height; float rotation; std::vector<Category> categories; }` in **pixel units** (rotation in radians, CCW). Converts from the framework `OrientedDetection` proto produced by the calculator. This is a Tasks-layer C++ struct (like `Detection`/`DetectionResult`), distinct from the framework proto.
- **Options:** same set as Task A.
- **Graph (`OrientedObjectDetectorGraph`):** preprocessing → inference(+`METADATA`) → `YoloObbTensorsToOrientedDetectionsCalculator` → `RotatedNonMaxSuppressionCalculator` → oriented-to-pixel projection → `ORIENTED_DETECTIONS` out.

### 4.3 Shared concerns
- **Metadata:** the `METADATA` side packet is wired into the graph and consumed internally (input H/W/C for preprocessing, class count). Not surfaced publicly in 2.1.
- **Pixel projection:** decoded boxes are normalized; convert to pixel units using the original image dimensions (accounting for the preprocessing letterbox/scale), matching `ObjectDetector` output semantics. For OBB, scale center+size and keep the angle.
- **Category mapping:** reuse the existing label-map / category-name handling from the components layer (`display_names_locale`, allow/deny lists) where it applies to the YOLO class ids.

## 5. Error handling

- Wrong model kind (OBB tensor in `YoloObjectDetector`, or vice-versa) → clear `absl::Status` error, surfaced via the Group-1 decoder shape guards (channel count `4+nc` vs `4+nc+1`).
- Missing/incompatible `METADATA` → fail at `Create()`/graph init with an actionable message (the M4 seam already gates emission on successful model init).
- Unsupported model dtype (non-float32) → clear error (mirrors the Group-1 decoder's float32-first guard).

## 6. Testing strategy

**Locally runnable now (no model):**
- Option-proto round-trip and conversion unit tests (options struct ↔ proto).
- Result-conversion unit tests: framework `Detection`/`OrientedDetection` proto → Tasks `DetectionResult`/`OrientedObjectDetectionResult`, including normalized→pixel math and angle preservation.

**Gated on provided fixtures (`yolov8n.tflite`, `yolov8n-obb.tflite`):**
- `YoloObjectDetector` integration: `Create` + `Detect` on a sample image → expected categories/boxes (mirrors `object_detector_test.cc`); `DetectForVideo`; `DetectAsync` callback.
- `OrientedObjectDetector` integration: same, asserting oriented results (center/size/angle) in pixel units.
- These tests are authored but skip/gate when the fixtures are absent; they run once the models are added to testdata (fetched from GCS or supplied).

## 7. File structure (for the implementation plan)

- Create `tasks/cc/vision/yolo_object_detector/`: `yolo_object_detector.{h,cc}`, `yolo_object_detector_graph.cc`, `proto/yolo_object_detector_options.proto`, `BUILD`, `yolo_object_detector_test.cc`.
- Create `tasks/cc/vision/oriented_object_detector/`: parallel file set + `oriented_object_detector_graph.cc` + `oriented_object_detector_test.cc`.
- Create `tasks/cc/components/containers/oriented_object_detection_result.{h,cc}` (+ BUILD) for the new oriented result container + its proto-conversion helpers.
- Reuse (no change): `image_preprocessing_graph`, `BaseVisionTaskApi`, `non_max_suppression_calculator`, the Group-1 calculators, `components/containers/detection_result`.

## 8. Decomposition & sequencing (within 2.1)

1. New `OrientedObjectDetectionResult` container + conversion (CPU-unit-testable).
2. **Task A** `YoloObjectDetector` (options proto → graph → Task class → conversion → unit tests; integration test gated on fixture).
3. **Task B** `OrientedObjectDetector` (same, using the new container + rotated NMS).
4. Wire integration tests; gate on fixtures.

## 9. Risks & open items

1. **Model fixtures** — real `yolov8n.tflite` / `yolov8n-obb.tflite` are needed for integration tests; treated as a gated dependency (Decision §2). The non-integration logic is fully testable now.
2. **`num_classes` / layout source** — ideally from `METADATA`; if a given export doesn't carry it, fall back to an explicit option. Confirm during implementation against a real YOLO tflite's metadata.
3. **Pixel projection correctness** — letterbox/aspect handling in preprocessing must be inverted correctly for pixel-unit boxes; verify against `ObjectDetector`'s projection path (reuse its helpers where possible).
4. **Graph registration / naming** — new `ModelTaskGraph` subgraph names must be unique; follow the `mediapipe.tasks.vision.*` naming.

## 10. Non-goals

Tiling via the Task; language bindings (Python/Web/Java/iOS — 2.2–2.5); a public metadata accessor; segmentation/pose heads; modifying `ObjectDetector`.
