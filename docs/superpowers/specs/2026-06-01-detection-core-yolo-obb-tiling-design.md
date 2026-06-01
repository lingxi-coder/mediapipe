# Detection Core — YOLO + OBB + Multi-batch + Tiling + Metadata (Group 1)

- **Date:** 2026-06-01
- **Status:** Approved (design); pending implementation plan
- **Scope:** Group 1 (M1–M4) only. M5–M7 are tracked in the roadmap appendix and get their own specs.
- **Build model:** Fork / custom build of MediaPipe (not an upstream PR — MediaPipe does not accept new-feature PRs).

---

## 1. Background

The parent request adds seven features to MediaPipe:

1. Multiple model formats (ONNX, PyTorch `.pt`, TensorRT, CoreML), each gated to its compatible platform.
2. YOLO support.
3. Multi-batch (input/output `N ≥ 1`) for ObjectDetection and OBB.
4. Frame tiling: split a frame into tiles from externally-supplied tile info, run inference in batches of size `N`, merge results back to one frame.
5. Inference metadata output.
6. Zero-copy performance optimization (LiteRT-Next GPU — https://ai.google.dev/edge/litert/next/gpu).
7. BoTSORT multi-object tracking.

These span four subsystems and are too large for one spec. They were decomposed into three groups, ordered by dependency:

- **Group 1 — Detection core (this spec):** M1 multi-batch, M2 YOLO detect + OBB, M3 tiling, M4 inference metadata. Buildable on the existing TFLite/LiteRT inference path; lowest risk; delivers a working YOLO-OBB tiled batched detector.
- **Group 2 — Inference backends:** M5 pluggable backend + ONNX/PT/TensorRT/CoreML, M6 zero-copy GPU.
- **Group 3 — Tracking:** M7 BoTSORT.

Dependencies: M1→M3, M2↔M3, M5→M6; M2 introduces the OBB type consumed by M1/M3/M7; M7 is otherwise independent.

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Target platforms | Desktop/Server, NVIDIA Jetson, Android, iOS/macOS (all four) | Drives the full backend matrix in M5 (Group 2); Group 1 stays on TFLite/LiteRT and is platform-neutral. |
| First milestone | Group 1 (M1–M4) | Lowest risk; runs on TFLite-exported YOLO models before the heavy backend work. |
| Target layer | Calculators/graphs first, Tasks API after | Reusable, independently testable; Tasks API + bindings as a follow-up sub-milestone. |
| YOLO heads | Detection + OBB only | The explicit ask. Segmentation/Pose excluded (YAGNI). |
| OBB representation | New `OrientedDetection` proto on its own stream; `Detection` untouched | Keeps the fork rebaseable on upstream; avoids threading rotation through battle-tested shared code. Cost: OBB gets its own NMS + render path. |
| Implementation approach | A — parallel new calculators | New decoders are batch-native, so M1 batching comes "for free" without touching the existing SSD `tensors_to_detections` (which hard-asserts `batch==1`). |
| Tile spec source | Per-frame input stream | Supports dynamic/adaptive/ROI tiling; a static grid is a constant stream. |
| Tiling output | Merged detections in full-frame coordinates (global NMS) | Rendering delegated to the existing `annotation_overlay`. Clean separation. |

## 3. Current-state findings (codebase)

- Inference is TFLite/LiteRT-only: `InferenceCalculator` (`mediapipe/calculators/tensor/`) wraps TFLite across CPU/GL/Metal/XNNPACK backends. No abstraction for other formats.
- Object-detection postprocessing is SSD-anchor based (`detection_postprocessing_graph.cc`, `ssd_anchors_calculator`, `tflite_tensors_to_detections_calculator`). No YOLO anywhere.
- `tensors_to_detections_calculator.cc:365` hard-asserts the batch dim is 1 (`RET_CHECK_EQ(raw_box_tensor->shape().dims[0], 1)`). This is the batch bottleneck; Approach A sidesteps it with new batch-native decoders.
- `LocationData` (`mediapipe/framework/formats/location_data.proto`) supports only `BOUNDING_BOX`, `RELATIVE_BOUNDING_BOX`, `MASK`, `RelativeKeypoint` — no oriented box. `non_max_suppression_calculator` is axis-aligned IoU only.
- Reusable primitives exist: `image_cropping_calculator`, `scale_image_calculator`, `detection_projection_calculator` (DETECTIONS + PROJECTION_MATRIX → DETECTIONS), `annotation_overlay_calculator`, `detections_to_render_data_calculator`, `BeginLoop`/`EndLoop` calculators.
- No tensor-batching/concat calculator exists → a new `StackTensorsForBatchCalculator` is required.
- `InferenceCalculator` emits no timing/metadata today → M4 is genuinely new.
- `Detection.track_id` already exists → BoTSORT (M7) has a home for track IDs; `OrientedDetection` will mirror it.

## 4. Architecture & data flow

```
Frame ─┐
Tiles ─┤ (per-frame stream of tile rects)
       ▼
  TilingCalculator ──► vector<Image> tiles + vector<NormalizedRect> geometry   [NEW, M3]
       ▼ (BeginLoop over tiles)
  ImageToTensor (reuse) ──► tensor_i
       ▼ (EndLoop → vector<Tensor>)
  StackTensorsForBatchCalculator ──► [N,H,W,C]                                 [NEW, M1]
       ▼
  InferenceCalculator (reuse) ──► raw out [N, …]
       │
       └─► InferenceMetadataCalculator ──► INFERENCE_METADATA                  [NEW, M4]
       ▼
  YoloTensorsToDetectionsCalculator        (axis-aligned)                      [NEW, M2]
  YoloObbTensorsToOrientedDetectionsCalc.  (oriented)                          [NEW, M2]
       ▼  vector<vector<Detection>> / vector<vector<OrientedDetection>>
         (outer index = tile/batch item)
       ▼
  NonMaxSuppression (reuse)  /  RotatedNonMaxSuppressionCalculator             [NEW, M2]
       ▼
  MergeTileDetectionsCalculator ──► full-frame DETECTIONS / ORIENTED_DETECTIONS [NEW, M3]
       ▼ (optional)
  annotation_overlay (reuse) ──► rendered Image
```

When tile count `T > N` (batch capacity), the tiling stage emits `ceil(T/N)` batches and the merge stage accumulates across them before global NMS.

A non-tiled detector is the same graph without the tiling/merge stages (single image → image_to_tensor → [1 or N,…] → decode → NMS → detections).

## 5. New data types

### 5.1 `OrientedDetection` — `mediapipe/framework/formats/oriented_detection.proto`
```proto
syntax = "proto2";
package mediapipe;

// An oriented (rotated) bounding box detection. Coordinates are normalized by
// image dimensions. Mirrors the identity fields of Detection so trackers
// (BoTSORT, M7) can operate on oriented detections.
message OrientedDetection {
  optional float cx = 1;        // box center x, normalized [0,1]
  optional float cy = 2;        // box center y, normalized [0,1]
  optional float width = 3;     // normalized [0,1]
  optional float height = 4;    // normalized [0,1]
  optional float rotation = 5;  // radians, counter-clockwise

  repeated string label = 6;
  repeated int32 label_id = 7 [packed = true];
  repeated float score = 8 [packed = true];
  repeated string display_name = 9;

  optional int64 detection_id = 10;
  optional string track_id = 11;       // populated by M7
  optional int64 timestamp_usec = 12;
}

message OrientedDetectionList {
  repeated OrientedDetection detection = 1;
}
```

### 5.2 Batched-output convention
Decoders emit `std::vector<std::vector<Detection>>` / `std::vector<std::vector<OrientedDetection>>`; the outer index is the batch/tile item. This avoids a batch-index field on the proto and keeps single-image use (`N==1`) as a one-element outer vector.

### 5.3 `InferenceMetadata` — `mediapipe/framework/formats/inference_metadata.proto` (M4)
```proto
syntax = "proto2";
package mediapipe;

message TensorSpec {
  optional string name = 1;
  repeated int32 shape = 2 [packed = true];
  optional string dtype = 3;            // "float32", "uint8", …
  optional float quant_scale = 4;
  optional int32 quant_zero_point = 5;
}

message InferenceMetadata {
  optional string model_id = 1;
  repeated TensorSpec input = 2;
  repeated TensorSpec output = 3;
  optional string backend = 4;          // "cpu","gpu","xnnpack","nnapi","metal"
  optional int32 batch_size = 5;
  optional int32 class_count = 6;
  optional int64 inference_latency_us = 7;   // per-frame; see §9
}
```

## 6. M1 — Multi-batch (N ≥ 1)

- **`StackTensorsForBatchCalculator`** — input: `std::vector<Tensor>` (one per image/tile, each `[1,H,W,C]` or `[H,W,C]`); output: one `[N,H,W,C]` tensor. Validates uniform shape/dtype.
- The new YOLO decoders read `dim[0]=N` and emit one inner vector per item — batch is native; the existing SSD calculator is never modified.
- **Hard constraint:** the model must be exported with the target batch dim (fixed `N`) or a dynamic batch dim. Add a guard that compares the model input shape against requested `N` and fails with a clear, actionable error.
- Interaction with GPU delegate and dynamic batch is a known risk (§9).

## 7. M2 — YOLO decode + NMS

- **`YoloTensorsToDetectionsCalculator`** — anchor-free decode for Ultralytics **v8/v11** detect heads.
  - Options: `layout` (`CHANNELS_FIRST [N,4+nc,A]` | `CHANNELS_LAST [N,A,4+nc]`), `num_classes`, `conf_threshold` (default 0.25), `iou_threshold` (default 0.45), `max_detections` (default 300), `class_agnostic_nms`.
  - Output: batched `Detection` with `RELATIVE_BOUNDING_BOX`.
- **`YoloObbTensorsToOrientedDetectionsCalculator`** — same decode plus a per-box angle channel → batched `OrientedDetection`. Shares the common options; adds angle handling.
- **`RotatedNonMaxSuppressionCalculator`** — rotated-IoU NMS over `OrientedDetection`. Axis-aligned detection reuses the existing `non_max_suppression_calculator`.
- v5 (anchor-based) decode is a later add-on, not in this milestone.

## 8. M3 — Tiling (SAHI-style)

- **`TilingCalculator`** — inputs `IMAGE` (Image) + `TILES` (`std::vector<NormalizedRect>`, per-frame); outputs `vector<Image>` cropped tiles (composing `image_cropping_calculator`) + the tile geometry (passed through for merge). Tiles may overlap.
- Preprocessing/batching uses `BeginLoop`/`EndLoop` over the tile vector → `image_to_tensor` per tile → `StackTensorsForBatchCalculator` → inference.
- **`MergeTileDetectionsCalculator`** — inputs the per-tile detections (`vector<vector<Detection>>` / `vector<vector<OrientedDetection>>`) + tile geometry; projects each tile's boxes back to full-frame coordinates (axis-aligned via a per-tile 4×4 projection matrix, reusing `detection_projection_calculator` semantics; OBB via a rotation-aware transform applied to center + angle), then runs cross-tile **global NMS** → a single full-frame result.
- Output: merged detections in full-frame coords; rendering optional via `annotation_overlay`.
- **Forward link to M6:** a fused `TilesToTensorsCalculator` (crop+resize+stack in one GPU-friendly pass) is a natural perf optimization once zero-copy lands; out of scope here.

## 9. M4 — Inference metadata

- **`InferenceMetadataCalculator`** emits `InferenceMetadata`.
- Static fields (model id, input/output specs, backend, batch size, class count) are computed at `Open` from model resources and can be emitted as a side packet.
- Per-frame latency is the one open sub-decision: a thin stopwatch-bracket calculator (no core changes — preferred, honors isolation-first) vs a small `InferenceCalculator` hook (more accurate, more invasive). Default: stopwatch-bracket; revisit if accuracy is insufficient.

## 10. Tasks API follow-up ("Both")

After the calculators land, extend `tasks/cc/vision/object_detector`:
- New options: `model_kind: {SSD | YOLO | YOLO_OBB}`, `batch`, `tiles`.
- New result type for oriented detections.
- Thread through Python / Java / iOS / Web bindings.
This is a separate sub-plan, sequenced after Group 1 calculators are verified.

## 11. Testing strategy

- Per-calculator unit tests:
  - YOLO detect/OBB decode against golden tensors → expected boxes (incl. layout variants).
  - `RotatedNonMaxSuppressionCalculator`: overlapping/rotated cases, class-agnostic vs per-class.
  - `StackTensorsForBatchCalculator`: shape/dtype validation, N stacking.
  - `MergeTileDetectionsCalculator`: known-overlap tiles, projection correctness, global NMS dedup.
  - `InferenceMetadataCalculator`: static fields populated; latency present.
- Testdata: a small exported `yolov8n` and `yolov8n-obb` `.tflite` (with batch and dynamic-batch variants).
- Graph-level integration test: tiled, batched YOLO (detect + OBB) on a sample image, asserting merged full-frame detections.

## 12. Suggested build order (within Group 1)

1. `OrientedDetection` proto + batched-output conventions.
2. `YoloTensorsToDetectionsCalculator` (batch-native) + reuse NMS → working batched YOLO detection.
3. `YoloObbTensorsToOrientedDetectionsCalculator` + `RotatedNonMaxSuppressionCalculator`.
4. `StackTensorsForBatchCalculator` + `TilingCalculator` + `MergeTileDetectionsCalculator` (M1 + M3).
5. `InferenceMetadataCalculator` (M4 — small, can land in parallel).
6. Tasks API wrappers + bindings.

## 13. Risks & open items

1. **TFLite fixed-batch export for N>1** and its interaction with the GPU delegate — some delegates dislike dynamic batch. Validate early with a real batched export.
2. **YOLO output layout variance** across export tools/versions — mitigated by the configurable `layout` option; consider light autodetection from tensor rank/shape.
3. **Rotated-IoU NMS cost** — rotated-rect intersection is more expensive; benchmark, cap `max_detections`.
4. **M4 latency sourcing** — stopwatch-bracket vs core hook (see §9).
5. **OBB projection through tiles** — rotation-aware transform must be verified against rotated ground truth, including tiles whose crop is itself unrotated (the common case) vs future rotated tiles.

---

## Appendix A — Full roadmap (context only; not this spec's scope)

| Group | Milestone | Feature | Depends on |
|---|---|---|---|
| 1 | M1 | Multi-batch (N≥1), ObjectDetection + OBB | — |
| 1 | M2 | YOLO detect + OBB decode + rotated NMS | — |
| 1 | M3 | Tiling (split → batch → merge) | M1, M2 |
| 1 | M4 | Inference metadata output | — |
| 2 | M5 | Pluggable inference backend + ONNX/PT/TensorRT/CoreML (platform-gated) | — |
| 2 | M6 | Zero-copy GPU buffers (LiteRT-Next) | M5 |
| 3 | M7 | BoTSORT multi-object tracking | consumes M1–M3 output |

Platform/format matrix for M5 (future spec): ONNX → all (ONNX Runtime / Mobile); PyTorch → desktop/server (libtorch); TensorRT → NVIDIA/Jetson; CoreML → Apple. Zero-copy GPU API per platform: Metal (Apple), GLES/Vulkan (Android), CUDA (NVIDIA).
