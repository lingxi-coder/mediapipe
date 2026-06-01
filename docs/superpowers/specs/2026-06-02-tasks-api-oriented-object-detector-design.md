# Tasks API — OrientedObjectDetector (OBB) Design (Phase 2.1b, cc layer)

- **Date:** 2026-06-02
- **Status:** Approved (design); pending implementation plan.
- **Roadmap:** Phase 2 (Tasks API exposure) in `2026-06-01-roadmap.md`; sub-project **2.1b**. Builds on **2.1a** (`YoloObjectDetector`, done) and shares the cc-layer decisions from `2026-06-01-tasks-api-yolo-obb-design.md` (§4.2). cc layer only; Python/Web/Java/iOS are later follow-ons.
- **Build model:** Fork / custom build of MediaPipe.

## 1. Background & goal

2.1a added `YoloObjectDetector` (axis-aligned YOLO → shared `DetectionResult`, pixel units). 2.1b adds its oriented sibling: **`OrientedObjectDetector`**, exposing single-image / video / live-stream OBB detection through the Tasks API, wrapping Group-1's OBB decoder (`YoloObbTensorsToOrientedDetectionsCalculator`) + rotated NMS (`RotatedNonMaxSuppressionCalculator`), and returning a **new oriented result type in pixel units**.

The one piece 2.1a did not face: **oriented projection to pixel coordinates**. 2.1a reused `DetectionProjectionCalculator` (4×4 `PROJECTION_MATRIX`) + `DetectionTransformationCalculator` (normalized→pixel). Verified: neither works for OBB — both operate on the `Detection` proto, and `DetectionTransformationCalculator` `RET_CHECK`s the format is `RELATIVE_BOUNDING_BOX`/`BOUNDING_BOX`. `OrientedDetection` is a separate proto (cx/cy/w/h/rotation). So 2.1b needs **new in-graph oriented projection**.

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Inherited from 2.1 | New parallel Task (`ObjectDetector`/`YoloObjectDetector` untouched); two-Tasks split; pixel-unit results; metadata consumed internally; cc layer only; tiling not exposed | Carries over from `2026-06-01-tasks-api-yolo-obb-design.md`. |
| Oriented projection | **One combined `OrientedDetectionProjectionCalculator`**: `ORIENTED_DETECTIONS` (normalized) + `PROJECTION_MATRIX` + `IMAGE_SIZE` → pixel-unit `ORIENTED_DETECTIONS` | No pre-existing oriented projection/transformation pair to reuse; a single calc (affine-project + normalized→pixel) is simplest and has no other consumers (YAGNI). Reuses Group-1 corner-transform math. |
| Result container | **New `components/containers/oriented_object_detection_result.{h,cc}`**, pixel units | Mirrors `detection_result.h`; result containers live in `components/containers`. The locally-TDD-able core. |
| NMS vs projection order | **Rotated NMS in model-input-normalized space, then projection** | Symmetric with 2.1a (NMS runs before projection); rotated-IoU is scale-invariant. |
| Batch flatten | New `YoloObbBatchDetectionsToSingleCalculator` (oriented analog of 2.1a's flatten) | The decoder emits batched `vector<vector<OrientedDetection>>`; rotated NMS takes a flat `vector<OrientedDetection>`; N≤1 for the single-image Task. |

## 3. Current-state findings (verified)

- `DetectionTransformationCalculator` only accepts `RELATIVE_BOUNDING_BOX`/`BOUNDING_BOX` (`detection_transformation_calculator.cc`), and `DetectionProjectionCalculator` operates on `Detection` + a `std::array<float,16>` `PROJECTION_MATRIX`. Neither handles `OrientedDetection`.
- The only existing `OrientedDetection` handling in `calculators/util/` is `rotated_non_max_suppression_calculator` (Group 1). No oriented projection or normalized→pixel transformation exists.
- Group-1 `MergeTileDetectionsAccumulator` already projects an `OrientedDetection` through an affine (transform corners, recompute center/size/rotation) — prior art for the new calculator's math.
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
// Converts pixel-unit OrientedDetection protos to the container.
OrientedObjectDetectionResult ConvertToOrientedObjectDetectionResult(
    std::vector<mediapipe::OrientedDetection> detections_proto);
}  // namespace
```
Category mapping (label id → name/display_name via label map) reuses the existing components category handling. **Locally unit-testable** (proto→struct, no model).

### 4.2 `OrientedDetectionProjectionCalculator` — `calculators/util/`
- Inputs: `ORIENTED_DETECTIONS` (`std::vector<OrientedDetection>`, normalized in model-input space), `PROJECTION_MATRIX` (`std::array<float,16>`, from preprocessing), `IMAGE_SIZE` (`std::pair<int,int>`).
- Output: `ORIENTED_DETECTIONS` (`std::vector<OrientedDetection>`) in **pixel** units.
- Logic: for each detection, transform the box center and the four oriented corners through the 4×4 affine (same approach as `MergeTileDetectionsAccumulator`), recompute cx/cy/w/h/rotation in the projected (original-image-normalized) space, then multiply by `IMAGE_SIZE` to get pixels. (For the common no-rotation letterbox case the matrix is scale+offset and the angle is unchanged; the corner-transform path handles the general case.)
- **Locally unit-testable** with golden geometry (known box + known matrix + image size → expected pixel cx/cy/w/h/rotation), no model.

### 4.3 `YoloObbBatchDetectionsToSingleCalculator` — `calculators/tensor/`
- Input `ORIENTED_DETECTIONS` (`std::vector<std::vector<OrientedDetection>>`), output `ORIENTED_DETECTIONS` (`std::vector<OrientedDetection>`); `RET_CHECK_LE(N, 1)`; emits row 0 (or empty). Oriented analog of 2.1a's `YoloBatchDetectionsToSingleCalculator`. **Locally unit-testable.**

### 4.4 `OrientedObjectDetectorGraph` — `tasks/cc/vision/oriented_object_detector/`
Mirrors `YoloObjectDetectorGraph`'s skeleton (preprocessing → `AddInference` (+`METADATA`)), postprocessing chain:
```
model_output_tensors
  → YoloObbTensorsToOrientedDetectionsCalculator   (decode, batched)
  → YoloObbBatchDetectionsToSingleCalculator       (flatten, N==1)
  → RotatedNonMaxSuppressionCalculator             (rotated NMS, normalized)
  → OrientedDetectionProjectionCalculator          (+PROJECTION_MATRIX,+IMAGE_SIZE → pixel)
  → ORIENTED_DETECTIONS out
```
Decoder options (num_classes, conf_threshold, layout) and NMS options (iou_threshold, max_detections, class_agnostic) come from the task options. `num_classes` from options (metadata-derived is a future enhancement, as in 2.1a).

### 4.5 `oriented_object_detector_options.proto` + `OrientedObjectDetector` Task class — `tasks/cc/vision/oriented_object_detector/`
- Options proto: same fields as `YoloObjectDetectorOptions` (base_options, display_names_locale, max_results, score_threshold, category_allowlist/denylist, iou_threshold, layout, num_classes). Unique extension id.
- Task class mirrors `YoloObjectDetector` (`Create/Detect/DetectForVideo/DetectAsync`, `BaseVisionTaskApi`), but: result alias `using OrientedObjectDetectorResult = components::containers::OrientedObjectDetectionResult;`, result conversion via `ConvertToOrientedObjectDetectionResult`, output stream type `std::vector<OrientedDetection>`, graph type-name `"mediapipe.tasks.vision.oriented_object_detector.OrientedObjectDetectorGraph"`.

## 5. Error handling

- Wrong model kind (axis-aligned model fed to OBB Task) → clear error via the OBB decoder's channel guard (`4 + num_classes + 1`).
- Missing/incompatible model metadata → fail at `Create()` (same as 2.1a / `ObjectDetectorGraph`).
- `num_classes` unset → `RET_CHECK_GT(num_classes, 0)` in the graph (metadata-derivation deferred).
- Non-float32 model output → clear error (Group-1 decoder guard).

## 6. Testing strategy

**Locally runnable now (no model):**
- `ConvertToOrientedObjectDetectionResult`: proto→struct, including category mapping; verify pixel values pass through unchanged.
- `YoloObbBatchDetectionsToSingleCalculator`: single-row flatten, empty, multi-row `RET_CHECK` (3 cases, mirror 2.1a's flatten test).
- `OrientedDetectionProjectionCalculator`: golden geometry — e.g. a centered box `(0.5,0.5,0.4,0.2,θ)` with a scale/letterbox matrix + `IMAGE_SIZE` → expected pixel cx/cy/w/h and preserved/rotated angle; an off-center box; an identity-matrix case.

**Gated on a `yolov8n-obb.tflite` fixture (skips honestly when absent, like 2.1a):**
- `OrientedObjectDetector` integration: `Create` + `Detect` on a sample image → non-empty oriented detections with plausible centers/sizes/angles in pixel units; `DetectForVideo`; `DetectAsync`.

## 7. File structure (for the plan)

- Create `mediapipe/tasks/cc/components/containers/oriented_object_detection_result.{h,cc}` (+ BUILD entry, + test).
- Create `mediapipe/calculators/util/oriented_detection_projection_calculator.cc` (+ BUILD, + test).
- Create `mediapipe/calculators/tensor/yolo_obb_batch_detections_to_single_calculator.cc` (+ BUILD, + test).
- Create `mediapipe/tasks/cc/vision/oriented_object_detector/`: `proto/oriented_object_detector_options.proto` (+BUILD), `oriented_object_detector_graph.cc`, `oriented_object_detector.{h,cc}`, `BUILD`, `oriented_object_detector_test.cc` (gated).
- Reuse (no change): Group-1 OBB decoder + rotated NMS, `image_preprocessing_graph`, `BaseVisionTaskApi`, `components/containers/category`, the 2.1a graph/Task pattern.

## 8. Decomposition & sequencing (within 2.1b)

1. `OrientedObjectDetectionResult` container + converter (TDD).
2. `YoloObbBatchDetectionsToSingleCalculator` (TDD).
3. `OrientedDetectionProjectionCalculator` (TDD — the core new logic).
4. `oriented_object_detector_options.proto`.
5. `OrientedObjectDetectorGraph` (build-verify; wires 1–4 + Group-1 calcs).
6. `OrientedObjectDetector` Task class (build-verify).
7. Gated integration test.

Steps 1–3 are locally TDD-able (more verifiable than 2.1a); 5–6 are build-verify; 7 is gated.

## 9. Risks & open items

1. **OBB model fixture** — `yolov8n-obb.tflite` needed for the integration test; gated dependency (like 2.1a's `yolov8n.tflite`). The container/flatten/projection logic is fully testable without it.
2. **Oriented projection under letterbox** — the corner-transform + refit must correctly handle aspect-preserving padding and any input `NormalizedRect` rotation; covered by golden-geometry tests (identity, scale, letterbox, rotated cases).
3. **Angle convention** — `OrientedDetection.rotation` is radians CCW (Group-1); the result container preserves it verbatim. Document clearly so bindings/renderers agree.
4. **`num_classes` source** — options-provided for now; metadata-derivation deferred (same as 2.1a).

## 10. Non-goals

Tiling via the Task; language bindings; a public metadata accessor; segmentation/pose; modifying `ObjectDetector`/`YoloObjectDetector`.
