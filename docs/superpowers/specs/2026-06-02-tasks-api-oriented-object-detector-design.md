# Tasks API — OrientedObjectDetector (OBB) Design (Phase 2.1b, cc layer)

- **Date:** 2026-06-02
- **Status:** Approved (design); pending implementation plan.
- **Roadmap:** Phase 2 (Tasks API exposure) in `2026-06-01-roadmap.md`; sub-project **2.1b**. Builds on **2.1a** (`YoloObjectDetector`, done) and shares the cc-layer decisions from `2026-06-01-tasks-api-yolo-obb-design.md` (§4.2). cc layer only; Python/Web/Java/iOS are later follow-ons.
- **Build model:** Fork / custom build of MediaPipe.

## 1. Background & goal

2.1a added `YoloObjectDetector` (axis-aligned YOLO → shared `DetectionResult`, pixel units). 2.1b adds its oriented sibling: **`OrientedObjectDetector`**, exposing single-image / video / live-stream OBB detection through the Tasks API, wrapping Group-1's OBB decoder (`YoloObbTensorsToOrientedDetectionsCalculator`) + rotated NMS (`RotatedNonMaxSuppressionCalculator`), and returning a **new oriented result type in pixel units**.

The one piece 2.1a did not face: **oriented projection back to original-image coordinates**. 2.1a reused `DetectionProjectionCalculator` (4×4 `PROJECTION_MATRIX`) + `DetectionTransformationCalculator` (normalized→pixel). Verified: neither works for OBB — both operate on the `Detection` proto, and `DetectionTransformationCalculator` `RET_CHECK`s the format is `RELATIVE_BOUNDING_BOX`/`BOUNDING_BOX`. `OrientedDetection` is a separate proto (cx/cy/w/h/rotation) whose geometry is defined as normalized. So 2.1b needs a new in-graph oriented projection calculator, while pixel conversion happens only at the Tasks result-container boundary.

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Inherited from 2.1 | New parallel Task (`ObjectDetector`/`YoloObjectDetector` untouched); two-Tasks split; pixel-unit results; metadata consumed internally; cc layer only; tiling not exposed | Carries over from `2026-06-01-tasks-api-yolo-obb-design.md`. |
| Oriented projection | **`OrientedDetectionProjectionCalculator`**: `ORIENTED_DETECTIONS` (model-input-normalized) + `PROJECTION_MATRIX` → `ORIENTED_DETECTIONS` (original-image-normalized) | Preserves the `OrientedDetection` proto contract. No pixel-unit `OrientedDetection` packets. |
| Result container | **New `components/containers/oriented_object_detection_result.{h,cc}`**, pixel units | Mirrors `detection_result.h`; result containers live in `components/containers`. The locally-TDD-able core. |
| NMS vs projection order | **Rotated NMS first (model-input-normalized), then projection** — Group-1 `RotatedNonMaxSuppressionCalculator` unchanged | Under keep-aspect-ratio letterbox preprocessing the model-input space is square (1:1), so rotated IoU there is faithful (uniform scale ⇒ IoU-invariant). This keeps the Group-1 NMS untouched and avoids needing `IMAGE_SIZE` inside NMS. (Non-letterbox/stretch caveat: §9.) |
| Batch flatten | New `YoloObbBatchDetectionsToSingleCalculator` (oriented analog of 2.1a's flatten) | The decoder emits batched `vector<vector<OrientedDetection>>`; rotated NMS takes a flat `vector<OrientedDetection>`; N≤1 for the single-image Task. |
| Label names and category filters | Defer label-name/display-name enrichment plus allowlist/denylist until a shared YOLO label-enrichment stage exists | The current YOLO/OBB decoders emit `label_id` + `score`, not label strings. 2.1b returns category index/score only. |

## 3. Current-state findings (verified)

- `DetectionTransformationCalculator` only accepts `RELATIVE_BOUNDING_BOX`/`BOUNDING_BOX` (`detection_transformation_calculator.cc`), and `DetectionProjectionCalculator` operates on `Detection` + a `std::array<float,16>` `PROJECTION_MATRIX`. Neither handles `OrientedDetection`.
- The only existing `OrientedDetection` handling in `calculators/util/` is `rotated_non_max_suppression_calculator` (Group 1). No oriented projection or normalized→pixel transformation exists.
- Group-1 docs describe the desired corner-transform/refit math, but the current `MergeTileDetectionsAccumulator` implementation only handles axis-aligned tile scale/offset and preserves rotation. Treat the new oriented projection calculator as new geometry work with its own tests.
- `components/containers/category.h` provides `Category { int index; float score; optional category_name; optional display_name; }` — reused in the new result struct.
- 2.1a artifacts to mirror (done, on `dev`): `YoloObjectDetector{,Graph,Options}`, `YoloBatchDetectionsToSingleCalculator`, and the gated-test pattern.

## 4. Components

### 4.1 `OrientedObjectDetectionResult` container — `components/containers/oriented_object_detection_result.{h,cc}`
```cpp
namespace mediapipe::tasks::components::containers {
struct OrientedObjectDetection {
  std::vector<Category> categories;
  float cx = 0, cy = 0;       // box center, PIXEL units
  float width = 0, height = 0; // PIXEL units
  float rotation = 0;          // radians, counter-clockwise
};
struct OrientedObjectDetectionResult {
  std::vector<OrientedObjectDetection> detections;
};
// Converts original-image-normalized OrientedDetection protos to the pixel-unit
// Tasks result container.
OrientedObjectDetectionResult ConvertToOrientedObjectDetectionResult(
    std::vector<mediapipe::OrientedDetection> detections_proto,
    std::pair<int, int> image_size);
}  // namespace
```
The input `OrientedDetection` protos are still original-image-normalized. The converter multiplies center/width by image width and center/height by image height and writes pixel-unit floats into the Tasks container. Category mapping initially returns `index` + `score`; `category_name` and `display_name` remain empty until a shared YOLO label-enrichment/filter stage is added. **Locally unit-testable** (proto→struct, no model).

### 4.2 `OrientedDetectionProjectionCalculator` — `calculators/util/`
- Inputs: `ORIENTED_DETECTIONS` (`std::vector<OrientedDetection>`, normalized in model-input space), `PROJECTION_MATRIX` (`std::array<float,16>`, from preprocessing).
- Output: `ORIENTED_DETECTIONS` (`std::vector<OrientedDetection>`) still normalized, but now in original-image coordinate space.
- Logic: for each detection, transform the four oriented corners through the 4×4 affine using the same matrix semantics as `DetectionProjectionCalculator`, then fit/recompute the oriented rectangle (`cx/cy/width/height/rotation`) in original-image-normalized coordinates. Do not multiply by image size here; that would violate the `OrientedDetection` proto contract.
- **Locally unit-testable** with golden geometry (known box + known matrix → expected normalized cx/cy/w/h/rotation), no model.

### 4.3 `YoloObbBatchDetectionsToSingleCalculator` — `calculators/tensor/`
- Input `ORIENTED_DETECTIONS` (`std::vector<std::vector<OrientedDetection>>`), output `ORIENTED_DETECTIONS` (`std::vector<OrientedDetection>`); `RET_CHECK_LE(N, 1)`; emits row 0 (or empty). Oriented analog of 2.1a's `YoloBatchDetectionsToSingleCalculator`. **Locally unit-testable.**

### 4.4 `OrientedObjectDetectorGraph` — `tasks/cc/vision/oriented_object_detector/`
Mirrors `YoloObjectDetectorGraph`'s skeleton (preprocessing → `AddInference` (+`METADATA`)), postprocessing chain:
```
model_output_tensors
  → YoloObbTensorsToOrientedDetectionsCalculator   (decode, batched)
  → YoloObbBatchDetectionsToSingleCalculator       (flatten, N==1)
  → RotatedNonMaxSuppressionCalculator             (model-input-normalized; Group-1 calc, UNCHANGED)
  → OrientedDetectionProjectionCalculator          (+PROJECTION_MATRIX → original-image-normalized)
  → ORIENTED_DETECTIONS out                        (normalized proto)

preprocessing.IMAGE → IMAGE out
preprocessing.IMAGE_SIZE → IMAGE_SIZE out          (consumed by the Task for pixel conversion)
```
Decoder options (`num_classes`, `conf_threshold`, `layout`) and NMS options (`iou_threshold`, `max_detections`, `class_agnostic_nms`) come from the task options. `num_classes` from options (metadata-derived is a future enhancement, as in 2.1a). NMS runs first, in the (square, under letterbox) model-input-normalized space where rotated IoU is faithful, using the Group-1 `RotatedNonMaxSuppressionCalculator` **unchanged**. The graph outputs `IMAGE` and `IMAGE_SIZE` alongside detections so `DetectAsync` can return the image and the Task class can convert the normalized detections to pixel-unit result containers (the only place pixels appear).

### 4.5 `oriented_object_detector_options.proto` + `OrientedObjectDetector` Task class — `tasks/cc/vision/oriented_object_detector/`
- Options proto fields for 2.1b: `base_options`, `max_results`, `score_threshold`, `iou_threshold`, `class_agnostic_nms`, `layout`, `num_classes`. Unique extension id. `display_names_locale`, `category_allowlist`, and `category_denylist` are deferred until label-name enrichment exists for YOLO-family Tasks.
- Task class mirrors `YoloObjectDetector` (`Create/Detect/DetectForVideo/DetectAsync`, `BaseVisionTaskApi`), but: result alias `using OrientedObjectDetectorResult = components::containers::OrientedObjectDetectionResult;`, result conversion via `ConvertToOrientedObjectDetectionResult(detections, image_size)`, output stream type `std::vector<OrientedDetection>` (normalized) plus `IMAGE` and `IMAGE_SIZE`, graph type-name `"mediapipe.tasks.vision.oriented_object_detector.OrientedObjectDetectorGraph"`.

## 5. Error handling

- Wrong model kind (axis-aligned model fed to OBB Task) → clear error via the OBB decoder's channel guard (`4 + num_classes + 1`).
- Missing/incompatible model metadata → fail at `Create()` (same as 2.1a / `ObjectDetectorGraph`).
- `num_classes` unset → `RET_CHECK_GT(num_classes, 0)` in the graph (metadata-derivation deferred).
- Non-float32 model output → clear error (Group-1 decoder guard).
- Missing `IMAGE_SIZE` packet at the Task boundary → clear internal graph error; pixel-unit result conversion requires it.
- `category_allowlist`, `category_denylist`, and display-name localization are not accepted in 2.1b options. They require a later label-enrichment/filter stage because the current OBB decoder emits only `label_id` and `score`.

## 6. Testing strategy

**Locally runnable now (no model):**
- `ConvertToOrientedObjectDetectionResult`: normalized proto + image size → pixel struct, including category index/score mapping; verify `category_name`/`display_name` are empty until label enrichment exists.
- `YoloObbBatchDetectionsToSingleCalculator`: single-row flatten, empty, multi-row `RET_CHECK` (3 cases, mirror 2.1a's flatten test).
- `OrientedDetectionProjectionCalculator`: golden geometry — e.g. a centered box `(0.5,0.5,0.4,0.2,θ)` with scale, letterbox, rotation/translation, and identity matrices → expected normalized cx/cy/w/h/rotation in original-image space.
- NMS-before-projection ordering: two overlapping rotated boxes in model-input-normalized space are correctly suppressed by the (unchanged) rotated NMS, and projection then preserves the surviving box's geometry.

**Gated on a `yolov8n-obb.tflite` fixture (skips honestly when absent, like 2.1a):**
- `OrientedObjectDetector` integration: `Create` + `Detect` on a sample image → non-empty oriented result containers with plausible centers/sizes/angles in pixel units; `DetectForVideo`; `DetectAsync` returns both result and image, using `IMAGE_SIZE` for conversion.

## 7. File structure (for the plan)

- Create `mediapipe/tasks/cc/components/containers/oriented_object_detection_result.{h,cc}` (+ BUILD entry, + test).
- Create `mediapipe/calculators/util/oriented_detection_projection_calculator.cc` (+ BUILD, + test).
- Create `mediapipe/calculators/tensor/yolo_obb_batch_detections_to_single_calculator.cc` (+ BUILD, + test).
- Create `mediapipe/tasks/cc/vision/oriented_object_detector/`: `proto/oriented_object_detector_options.proto` (+BUILD), `oriented_object_detector_graph.cc`, `oriented_object_detector.{h,cc}`, `BUILD`, `oriented_object_detector_test.cc` (gated).
- Reuse (NO change): Group-1 OBB decoder **and `RotatedNonMaxSuppressionCalculator`** (NMS runs in model-input space, unchanged), `image_preprocessing_graph`, `BaseVisionTaskApi`, `components/containers/category`, the 2.1a graph/Task pattern. The new graph surfaces `IMAGE_SIZE` to the Task class for pixel conversion.

## 8. Decomposition & sequencing (within 2.1b)

1. `OrientedObjectDetectionResult` container + converter (TDD).
2. `YoloObbBatchDetectionsToSingleCalculator` (TDD).
3. `OrientedDetectionProjectionCalculator` (TDD — the core new logic).
4. `oriented_object_detector_options.proto`.
5. `OrientedObjectDetectorGraph` (build-verify; wires 1–4 + Group-1 OBB decoder + unchanged rotated NMS; outputs normalized `ORIENTED_DETECTIONS`, `IMAGE`, and `IMAGE_SIZE`).
6. `OrientedObjectDetector` Task class (build-verify; converts normalized detections + `IMAGE_SIZE` to pixel result container).
7. Gated integration test.

Steps 1–3 are locally TDD-able (more verifiable than 2.1a); 5–6 are build-verify; 7 is gated.

## 9. Risks & open items

1. **OBB model fixture** — `yolov8n-obb.tflite` needed for the integration test; gated dependency (like 2.1a's `yolov8n.tflite`). The container/flatten/projection logic is fully testable without it.
2. **Oriented projection under letterbox** — the corner-transform + refit must correctly handle aspect-preserving padding and any input `NormalizedRect` rotation; covered by golden-geometry tests (identity, scale, letterbox, rotated cases).
3. **NMS coordinate space (and the stretch caveat)** — NMS runs first in model-input-normalized space, which is square (faithful rotated IoU) **under keep-aspect-ratio letterbox preprocessing** (the standard). If preprocessing is configured to *stretch* (`keep_aspect_ratio=false`), that space is anisotropic and rotated IoU there is approximate — and the projected OBB is itself lossy (rotated rect → parallelogram → refit). If stretch becomes a required mode, revisit by adding an optional `IMAGE_SIZE`-scaled NMS path (deferred; not built here to keep the Group-1 calculator unchanged).
4. **Angle convention** — `OrientedDetection.rotation` is radians CCW (Group-1); the result container preserves it verbatim. Document clearly so bindings/renderers agree.
5. **`num_classes` source** — options-provided for now; metadata-derivation deferred (same as 2.1a).
6. **Label names and category filters** — deferred. 2.1b returns category index/score only. Add display-name/allowlist/denylist only with a shared YOLO label-enrichment/filter stage.

## 10. Non-goals

Tiling via the Task; language bindings; a public metadata accessor; segmentation/pose; modifying `ObjectDetector`/`YoloObjectDetector`.
