# Detection Core — YOLO + OBB + Multi-batch + Tiling + Metadata (Group 1)

- **Date:** 2026-06-01
- **Status:** Approved (design); pending implementation plan
- **Scope:** Group 1 core = **M1–M4 on the CPU TFLite/LiteRT path** (multi-batch, YOLO detect + OBB, external tiling, inference metadata). OpenGL zero-copy, caching, and video-mode scheduling are **deferred follow-on milestones** (§14) with their contracts kept stable here. M5–M7 are in the roadmap appendix.
- **Build model:** Fork / custom build of MediaPipe (not an upstream PR — MediaPipe does not accept new-feature PRs).

> **Revision note (post-review):** An earlier draft pulled OpenGL zero-copy, a multi-layer cache subsystem, and a video/FlowPackager scheduler into Group 1. Those are valuable and their designs are preserved in §14, but they are **out of Group 1 core** to keep milestone one the lowest-risk, CPU-correct path. The streaming-batch model, timestamp discipline, single-global-NMS invariant, fixed/dynamic-batch handling, and tile-validation contract from that draft are **kept** — they are correctness improvements, not scope additions.

---

## 1. Background

The parent request adds seven features to MediaPipe:

1. Multiple model formats (ONNX, PyTorch `.pt`, TensorRT, CoreML), each gated to its compatible platform.
2. YOLO support.
3. Multi-batch (`N ≥ 1`) for ObjectDetection and OBB.
4. Frame tiling: split a frame into tiles from externally-supplied tile info, run inference in batches of size `N`, merge results back to one frame.
5. Inference metadata output.
6. Zero-copy performance optimization (LiteRT-Next GPU — https://ai.google.dev/edge/litert/next/gpu).
7. BoTSORT multi-object tracking.

These span four subsystems and are too large for one spec. They were decomposed by dependency:

- **Group 1 — Detection core (this spec):** M1 multi-batch, M2 YOLO detect + OBB, M3 external tiling, M4 inference metadata — all on the **CPU** TFLite/LiteRT path. Buildable on TFLite-exported YOLO models; lowest risk; delivers a working YOLO-OBB tiled batched detector.
- **Deferred follow-ons (designed here, implemented later — §14):** **M3.5** tile/input/output caching (perf, default-off), **M4.5** video-mode FlowPackager scheduler, and the **OpenGL zero-copy** input path (folded into **M6**).
- **Group 2 — Inference backends:** M5 pluggable backend + ONNX/PT/TensorRT/CoreML, M6 zero-copy GPU (OpenGL-on-TFLite slice + backend-wide LiteRT-Next).
- **Group 3 — Tracking:** M7 BoTSORT.

Dependencies: M1→M3, M2↔M3, M5→M6; M2 introduces the OBB type consumed by M1/M3/M7; M7 is otherwise independent. The Group-1 calculator contracts are designed so M3.5/M4.5/M6 attach without breaking them.

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Target platforms | Desktop/Server, NVIDIA Jetson, Android, iOS/macOS (all four) | Drives the full backend matrix in M5 (Group 2); Group 1 stays on TFLite/LiteRT CPU and is platform-neutral. |
| First milestone | Group 1 core (M1–M4, CPU) | Lowest risk; fastest path to a working detector before perf layers and backends. |
| Target layer | Calculators/graphs first, Tasks API after | Reusable, independently testable; Tasks API + bindings as a follow-up sub-milestone. |
| YOLO heads | Detection + OBB only | The explicit ask. Segmentation/Pose excluded (YAGNI). |
| OBB representation | New `OrientedDetection` proto on its own stream; `Detection` untouched | Keeps the fork rebaseable on upstream; avoids threading rotation through battle-tested shared code. Cost: OBB gets its own NMS + render path. |
| Implementation approach | A — parallel new calculators | New decoders are batch-native, so M1 batching comes "for free" without touching the existing SSD `tensors_to_detections` (which hard-asserts `batch==1`). |
| Tile spec source | Per-frame stream of externally supplied tiles | Any upstream algorithm can generate tiles: overlap grid, ROI proposal, tracker, saliency, custom service, or a static grid. Group 1 validates and consumes tiles; it does not own the tiling algorithm. |
| Tiling output | Merged detections in full-frame-normalized coordinates with one final global NMS after tile projection | Keeps calculator outputs consistent with existing relative detection streams. Tasks API follow-up handles pixel-unit results. |
| Batching boundary | Batch tiles from one frame only; never merge multiple frames into one model input. | Preserves per-frame latency, metadata, and global-NMS semantics. Underfilled batches allowed for `T < batch_capacity`. |
| Tiling performance model | Stream each filled tile batch directly to inference; do not materialize all tile images or all tile batches. | For `T > batch_capacity`, inference starts after the first `batch_capacity` tiles are tensorized instead of waiting for every tile. This is a CPU-path design property, independent of GPU zero-copy. |

**Deferred by design (see §14, contracts kept stable):** GPU/OpenGL zero-copy input path → **M6**; tile/input/output caches → **M3.5**; video-mode FlowPackager scheduling → **M4.5**. Group 1 is CPU-correct first; these attach as default-off perf/feature layers, each gated on the CPU baseline passing tests.

## 3. Current-state findings (codebase)

Group-1-relevant:
- Inference is TFLite/LiteRT-only: `InferenceCalculator` (`mediapipe/calculators/tensor/`) wraps TFLite across CPU/GL/Metal/XNNPACK backends. No abstraction for other formats.
- Object-detection postprocessing is SSD-anchor based (`detection_postprocessing_graph.cc`, `ssd_anchors_calculator`, `tflite_tensors_to_detections_calculator`). No YOLO anywhere.
- `tensors_to_detections_calculator.cc:365` hard-asserts the batch dim is 1 (`RET_CHECK_EQ(raw_box_tensor->shape().dims[0], 1)`). This is the batch bottleneck; Approach A sidesteps it with new batch-native decoders.
- `LocationData` (`mediapipe/framework/formats/location_data.proto`) supports only `BOUNDING_BOX`, `RELATIVE_BOUNDING_BOX`, `MASK`, `RelativeKeypoint` — no oriented box. `non_max_suppression_calculator` is axis-aligned IoU only.
- Reusable primitives exist: `image_cropping_calculator`, `scale_image_calculator`, `detection_projection_calculator` (DETECTIONS + PROJECTION_MATRIX → DETECTIONS), `annotation_overlay_calculator`, `detections_to_render_data_calculator`. The production tiled path does **not** use `BeginLoop`/`EndLoop` for tile batches because it streams filled batches directly to inference.
- No streaming tile-aware tensor-batching calculator exists → a new `StreamingTilesToTensorBatchCalculator` is required. Its **CPU implementation is Group 1** and can share preprocessing logic with `ImageToTensorCalculator`.
- `InferenceCalculator` emits no timing/metadata today → M4 is genuinely new.
- `Detection.track_id` already exists → BoTSORT (M7) has a home for track IDs; `OrientedDetection` will mirror it.

Relevant to deferred milestones (verified present, used in §14):
- `Tensor` already exposes CPU storage plus OpenGL views (`OpenGlTexture2dView`, `OpenGlBufferView`, the latter with a GlSync fence); the header notes `OpenGlTexture2dView` is BHWC-only. MediaPipe has `GpuBuffer`/OpenGL buffer pools. → reused by the **M6** OpenGL zero-copy path; do not introduce a parallel GPU memory abstraction.
- `FlowPackagerCalculator` packages `RegionFlowFeatureList` + optional `CameraMotion` into per-frame `TrackingData`, can emit `TrackingDataChunk`, supports a `CACHE_DIR` side packet, and emits `COMPLETE` on `PreStream`. → the existing video-mode motion-metadata surface used by **M4.5**.

## 4. Architecture & data flow (CPU)

### 4.1 Shape notation

All image input tensors use **BHWC** layout unless stated otherwise.

- Tensor shapes use the form **`[N,H,W,C]`**.
- **`N`** — actual batch dimension of the tensor passed to `InferenceCalculator` for the current tile batch.
- **`H`, `W`, `C`** — model input height, width, channels.
- **`batch_capacity`** — configured/model batch capacity. For fixed-batch models, the exported model input batch dim. For dynamic-batch models, the max tile rows per emitted batch.
- **`valid_count`** — valid tile rows in the current tensor. `0 < valid_count <= N`. For fixed-batch models, `N = batch_capacity` and rows `[valid_count,N)` are padded and ignored downstream. For dynamic-batch models, `N = valid_count`.
- **`T`** — tile count for the frame.
- **`A`** — YOLO candidate count before score filtering/top-K.

```
Frame ─┐
Tiles ─┤ (per-frame stream of tile rects)
       ▼
  TileSpecToTilePlanCalculator ──► TilePlan ──────────────────────────┐        [NEW, M3]
       │                                                              │
       ▼                                                              │
  StreamingTilesToTensorBatchCalculator (CPU) ──► Tensor batch [N,H,W,C]        [NEW, M1/M3]
                                         └► TensorBatchInfo
                                         └► vector<Matrix> tile_to_tensor_matrices
       ▼ (one packet per tile batch; source frame timestamp carried in TensorBatchInfo)
  InferenceCalculator (reuse) ──► raw outputs [N, …]
       │
       └─► batch_latency_us from stopwatch bracket
       ▼
  YoloTensorsToDetectionsCalculator        (axis-aligned)                      [NEW, M2]
  YoloObbTensorsToOrientedDetectionsCalc.  (oriented)                          [NEW, M2]
       ▼
  BatchDetections / BatchOrientedDetections + TensorBatchInfo
       ▼
  MergeTileDetectionsAccumulator ◄────────────────────────────────────┘       [NEW, M3]
       │  project rows, accumulate frame candidates, emit at source timestamp
       ├─► InferenceMetadataCalculator ──► INFERENCE_METADATA                  [NEW, M4]
       ▼
  NonMaxSuppression (reuse) / RotatedNonMaxSuppressionCalculator               [NEW, M2]
       ▼
  full-frame-normalized DETECTIONS / ORIENTED_DETECTIONS
       ▼ (optional)
  annotation_overlay (reuse) ──► rendered Image
```

**Frame-level invariant:** every frame emits exactly one detection result after **one** final global NMS — empty frames, underfilled batches, single-tile frames, and multi-batch tiled frames included. Decode may threshold/top-K candidates for performance, but NMS is frame-global and runs only after all valid tile rows for that source frame are projected to full-frame-normalized coordinates.

**Tile-count cases:**
- `T == 0`: no inference; emit empty detections and metadata with `inference_batch_count=0`.
- `T < batch_capacity`: dynamic-batch models emit one underfilled tensor with `N=valid_count=T`; fixed-batch models emit one padded tensor with `N=batch_capacity`, and merge drops rows `>= valid_count`. Never borrow tiles from the next frame.
- `T == batch_capacity`: one full tensor with `N=valid_count=batch_capacity`.
- `T > batch_capacity`: emit each full tensor as soon as `batch_capacity` rows are tensorized. The last tensor uses `N=valid_count` (dynamic) or `N=batch_capacity` with padding (fixed).

**Timestamp discipline:** batch packets use monotonically increasing batch timestamps because one frame can produce multiple batch packets. `TensorBatchInfo` carries `source_frame_timestamp`, `batch_index`, `total_batches`. The accumulator groups by `source_frame_timestamp`, flushes when all expected batches arrive, and emits the final result at the original frame timestamp. It never groups tiles from different source frames into one batch.

A non-tiled detector is a `TilePlan` with one identity full-frame tile, so the same batch/merge/NMS invariant applies.

## 5. New data types

### 5.1 `OrientedDetection` — `mediapipe/framework/formats/oriented_detection.proto`
```proto
syntax = "proto2";
package mediapipe;

// An oriented (rotated) bounding box detection. Calculator streams use
// normalized coordinates in the coordinate space of the current image. After
// tile merge, that space is the original full frame. Tasks API wrappers may
// convert to pixel units to match existing ObjectDetector results.
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

### 5.2 Tile / `TileSpec` definition

A **tile** is a logical crop request over one source frame. It is not an image buffer, not a tensor, and not tied to any specific tiling algorithm.

External `TILES` input is a per-frame ordered list of tile specs. Minimum representation is `std::vector<NormalizedRect>`; optional priority/metadata can be carried by a wrapper proto if a graph needs it.

Tile coordinate contract:
- Coordinates are in the original source frame's normalized space.
- `NormalizedRect.x_center` / `y_center` = tile center in `[0,1]` relative to full-frame width/height.
- `NormalizedRect.width` / `height` = tile size normalized by full-frame width/height; must be positive.
- `NormalizedRect.rotation` optional; absent = `0`; if present, radians CCW around the tile center.
- Tiles may overlap. The same object may appear in multiple tiles; dedup is handled only by the final frame-level global NMS.
- Input order is stable. `tile_index` is assigned from this order after validation; optional priority is used only by overflow policy (`DROP_LOW_PRIORITY`).
- Default validation rejects non-finite values, non-positive size, and tiles that do not intersect the frame. Silent clipping is not the default.

### 5.3 Internal tiling/batching records (calculator packet types, not public Tasks API)

- **`TileGeometry`** — validated internal form of one `TileSpec`: `tile_index`, source `NormalizedRect`, optional priority/metadata, logical tile size before model resize, and the affine matrix mapping tile-local coords back to full-frame-normalized space.
- **`TilePlan`** — per-frame list of `TileGeometry` plus `tile_count`, `max_tiles_per_frame`, and overflow policy. Default overflow policy = fail with an actionable error; no silent clipping.
- **`TensorBatchInfo`** — `source_frame_timestamp`, `batch_timestamp`, `batch_index`, `total_batches`, `batch_capacity`, `batch_size` (`N`), `valid_count`, `padded_count`, and `tile_indices` (one per valid row). The last batch can be padded for fixed-batch models; merge discards decoded rows whose index is `>= valid_count`. `tile_to_tensor_matrices` travel with the batch packet, indexed by the same valid-row order.
- **`BatchDetections` / `BatchOrientedDetections`** — one inference batch's decoded detections: `std::vector<std::vector<Detection>>` / `std::vector<std::vector<OrientedDetection>>`, outer index = row within that inference batch.

### 5.4 Batched-output convention

YOLO decoders emit one `BatchDetections`/`BatchOrientedDetections` packet per inference batch (one inner vector per tensor row) and pass `TensorBatchInfo` through unchanged. `MergeTileDetectionsAccumulator` maps each valid `row_index` to `tile_indices[row_index]`, projects candidates, and stores them under `source_frame_timestamp` until `total_batches` for that frame arrive. This avoids a batch-index field on public protos and keeps single-image use (`N==1`) as a one-row batch.

### 5.5 `InferenceMetadata` — `mediapipe/framework/formats/inference_metadata.proto` (M4)
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
  optional string backend = 4;          // declared/configured backend
  optional int32 batch_capacity = 5;
  optional int32 class_count = 6;
  optional int64 inference_latency_us = 7;   // per-frame sum across inference batches; see §9
  optional int32 inference_batch_count = 8;
  optional int32 valid_input_count = 9;      // images or tiles processed for this frame
}
```

## 6. M1 — Multi-batch (`N ≥ 1`)

- **`StreamingTilesToTensorBatchCalculator` (CPU, Group 1)** — input: source `IMAGE`, `TilePlan`, preprocessing options, configured/model `batch_capacity`, and fixed-vs-dynamic batch mode. Output: one `Tensor` batch packet, one `TensorBatchInfo`, and one tile-to-tensor matrix vector per emitted tile batch.
- It crops, resizes, normalizes, and writes directly into batch tensor rows. It must **not** emit `std::vector<Image>` cropped tiles, one tensor packet per tile, or a `std::vector<Tensor>` holding all batches for the frame in the production path.
- For `T > batch_capacity`, it emits a batch packet immediately when `batch_capacity` rows are filled, then continues tensorizing later tiles for the same frame — letting inference overlap with remaining preprocessing.
- Each output tensor has shape `[N,H,W,C]`; it validates uniform target shape/dtype and records `batch_size` (`N`), `valid_count`, `batch_capacity` in `TensorBatchInfo`.
- New YOLO decoders read `dim[0]` from each inference batch and emit one inner vector per tensor row. Padded rows are dropped later by merge using `TensorBatchInfo`; the existing SSD calculator is never modified.
- Cross-frame batching is out of scope. If `T < batch_capacity`, run an underfilled/padded single-frame batch rather than combining with another frame.
- **Hard constraint:** the model must be exported with fixed batch `batch_capacity` or a dynamic batch dim accepting the emitted `N`. Add a guard comparing model input shape against emitted `N`, failing with a clear, actionable error.
- A bounded input-tensor cache and the OpenGL write path are **deferred** (§14, M3.5/M6); the CPU calculator allocates/reuses CPU-backed `Tensor`s simply for Group 1.
- Interaction with GPU delegate and dynamic batch is a known risk (§13).

## 7. M2 — YOLO decode + NMS

- **`YoloTensorsToDetectionsCalculator`** — anchor-free decode for Ultralytics **v8/v11** detect heads.
  - Options: `layout` (`CHANNELS_FIRST [N,4+num_classes,A]` | `CHANNELS_LAST [N,A,4+num_classes]`), `num_classes`, `conf_threshold` (default 0.25), `max_detections_before_nms` (default 300), `class_agnostic_nms`, coordinate convention (`xywh_normalized` first; pixel-space export unsupported until added).
  - Supported dtypes: `float32` initially. Quantized `uint8/int8` outputs require explicit dequantization support before acceptance.
  - Output: batched `Detection` with `RELATIVE_BOUNDING_BOX`.
- **`YoloObbTensorsToOrientedDetectionsCalculator`** — same decode plus a per-box angle channel → batched `OrientedDetection`. Shares common options; adds angle handling.
- **`RotatedNonMaxSuppressionCalculator`** — rotated-IoU NMS over flattened full-frame `OrientedDetection`.
- Axis-aligned detection reuses the existing `non_max_suppression_calculator`, but only after batched outputs are flattened and (for tiled graphs) projected to full-frame-normalized coordinates. **No final NMS runs per tile before projection**; an optional per-tile top-K prefilter is allowed only as a perf optimization and must not replace global NMS.
- v5 (anchor-based) decode is a later add-on, not in this milestone.

## 8. M3 — External tiling (CPU)

- **`TileSpecToTilePlanCalculator`** — inputs `IMAGE_SIZE`/`IMAGE` metadata plus externally supplied `TILES` (`std::vector<TileSpec>`, min `std::vector<NormalizedRect>`); validates the §5.2 contract, normalizes tile metadata, applies overflow policy, emits `TilePlan`. Empty tile lists emit an empty result for that frame.
- The tile-generation algorithm is outside this calculator. Upstream may provide overlap grids, ROI proposals, tracker-guided tiles, saliency tiles, manual regions, or any custom source.
- **`StreamingTilesToTensorBatchCalculator`** consumes the source `IMAGE` + `TilePlan` directly, filling one inference batch tensor row-by-row and emitting it as soon as it is full. Memory is bounded by the current output batch plus accumulator state; it does not wait for all `T` tiles before the first inference.
- Performance guardrails: `max_tiles_per_frame` and `max_batches_per_frame` are explicit options. Default overflow = `FAIL`; optional `DROP_LOW_PRIORITY` only when tile priority is present. Per-tile score thresholding and top-K caps may run before merge to reduce NMS cost, but the only NMS remains the final frame-level global NMS.
- **`MergeTileDetectionsAccumulator`** — inputs `BatchDetections`/`BatchOrientedDetections`, `TensorBatchInfo`, `TilePlan`, per-tile tensor matrices. It maps each valid decoded row to its `tile_index`, projects boxes to full-frame-normalized coords, drops padded rows, and accumulates frame candidates until all batches for `source_frame_timestamp` arrive.
  - Bounded: `max_in_flight_frames` defaults to `1`; memory remains bounded by `max_in_flight_frames * max_candidates_per_frame`.
  - Merge inverts each tile's tile-to-tensor matrix, then composes tensor-to-tile with `TileGeometry`, accounting for crop offset, resize scale, keep-aspect-ratio padding, and any `NormalizedRect` rotation.
  - Axis-aligned projection can **share helper code** with `detection_projection_calculator` but cannot call that calculator directly (this path is batched/tile-indexed). OBB projection is separate: transform the four oriented-box corners through the composed affine matrix, fit the rotated rectangle, and recompute center/size/rotation in full-frame-normalized coords.
- Output after final NMS: one merged detection vector in full-frame-normalized coords per frame; rendering optional via `annotation_overlay`.
- **Contract stability for deferred work:** keep the calculator contract stable so the M3.5 caches and the M6 OpenGL zero-copy input path attach without interface changes.

## 9. M4 — Inference metadata

- **`InferenceMetadataCalculator`** emits `InferenceMetadata`.
- Required inputs/side packets: model identifier/path or resource, declared backend/delegate options, input/output tensor specs, requested batch size, class count. Without an `InferenceCalculator` hook, `backend` means the *declared/configured* backend, not the guaranteed fallback-selected one.
- Static fields (model id, input/output specs, declared backend, batch size, class count) computed at `Open`; emitted as a side packet or repeated per frame.
- Per-frame latency default: a thin stopwatch-bracket calculator around each `InferenceCalculator` invocation (no core changes; honors isolation-first). The merge accumulator (or a paired metadata accumulator) sums batch latencies into one frame-level `InferenceMetadata` packet when `total_batches` for the source frame arrive. If precise backend/delegate timing is required, add a small `InferenceCalculator` hook in a later revision.

## 10. Tasks API follow-up ("Both")

After the calculators land, extend `tasks/cc/vision/object_detector`:
- New options: `model_kind: {SSD | YOLO | YOLO_OBB}`, `batch`, `tiles`.
- New result type for oriented detections.
- Thread through Python / Java / iOS / Web bindings.

Separate sub-plan, sequenced after Group 1 calculators are verified. Not a Group 1 acceptance requirement.

## 11. Testing strategy (Group 1 core)

- Per-calculator unit tests:
  - YOLO detect/OBB decode against golden tensors → expected boxes (incl. layout variants).
  - `RotatedNonMaxSuppressionCalculator`: overlapping/rotated cases, class-agnostic vs per-class.
  - `StreamingTilesToTensorBatchCalculator` (CPU): `T=0`, `T<batch_capacity`, `T=batch_capacity`, `T>batch_capacity`, fixed-batch padding, dynamic last batch, overflow policies, no cross-frame batching, no cropped-tile materialization, and first batch emitted before later tiles are tensorized.
  - `MergeTileDetectionsAccumulator`: known-overlap tiles, projection correctness, padded-row drop, frame-candidate flattening, out-of-order batch arrival, bounded in-flight state.
  - Final NMS graph tests: every frame (single-tile and multi-batch tiled) runs exactly one global NMS after projection.
  - `InferenceMetadataCalculator`: static fields populated; latency present; multiple inference batches aggregate to one frame-level packet.
- Edge-case unit tests:
  - Tile definition/validation: positive size, finite coords, frame intersection, rotation handling, stable input-order `tile_index`, optional priority metadata.
  - External tile sources: overlap grids, tracker-guided ROI tiles, manual tiles, custom unordered inputs all normalize to the same `TilePlan` contract.
  - Projection: overlapping tiles containing the same object, keep-aspect-ratio padding, rotated `NormalizedRect`, OBB angle preservation.
  - Two consecutive `T<batch_capacity` frames are never combined into one batch.
  - Decoder input validation: unsupported quantized output tensors fail clearly until dequant support is added.
- Testdata: small exported `yolov8n` and `yolov8n-obb` `.tflite` (with batch and dynamic-batch variants).
- Graph-level integration test: tiled, batched YOLO (detect + OBB) on a sample image, asserting merged full-frame detections.

## 12. Suggested build order (CPU baseline first)

1. `OrientedDetection` proto + batched-output conventions.
2. `TileSpecToTilePlanCalculator` (validation + `TilePlan`).
3. `StreamingTilesToTensorBatchCalculator` **(CPU)** + batch metadata/edge-case tests.
4. `YoloTensorsToDetectionsCalculator` (batch-native) + reuse NMS → working batched YOLO detection.
5. `YoloObbTensorsToOrientedDetectionsCalculator` + `RotatedNonMaxSuppressionCalculator`.
6. `MergeTileDetectionsAccumulator` (M3): projection, bounded frame accumulation, final global-NMS ordering.
7. `InferenceMetadataCalculator` (M4 — small, can land in parallel once metadata side packets are defined).
8. Graph-level examples/tests for tiled batched YOLO and YOLO-OBB → **Group 1 done.**

Then, as separate gated milestones (§14), each default-off and only after the CPU baseline passes: **M3.5** caches → **M6** OpenGL zero-copy input path → **M4.5** video scheduler. Tasks API wrappers (§10) remain a follow-up sub-plan.

## 13. Risks & open items (Group 1)

1. **TFLite fixed-batch export for N>1** and its GPU-delegate interaction — some delegates dislike dynamic batch. Validate early with a real batched export.
2. **YOLO output layout & dtype variance** across export tools — mitigated by configurable `layout` + initial `float32`-only validation; add dequant/autodetection only after golden tests exist.
3. **Rotated-IoU NMS cost** — rotated-rect intersection is expensive; benchmark, cap `max_detections`.
4. **M4 latency sourcing** — stopwatch-bracket vs core hook (see §9).
5. **OBB projection through tiles** — rotation-aware transform must be verified against rotated ground truth, incl. crop, resize, padding, and future rotated tiles.
6. **Tile count explosion** — external tile streams can exceed the latency budget. Enforce `max_tiles_per_frame`/`max_batches_per_frame`; make overflow explicit.
7. **Streaming timestamp discipline** — batch packets use batch timestamps while outputs use source frame timestamps. Accumulators must enforce monotonic output, bounded in-flight frames, and clear errors for missing/duplicate batch indices.

---

## 14. Deferred follow-on milestones (designed here, implemented after Group 1)

These were promoted out of Group 1 to keep milestone one CPU-correct and low-risk. Designs are preserved so the Group-1 contracts can be built to accommodate them. Each is default-off and gated on the CPU baseline passing tests.

### 14.1 M3.5 — Tile / input / output caching (perf)
Bounded caches, all bounded by explicit options (`max_in_flight_frames`, `max_cached_tile_plans`, `max_tile_surfaces`, `max_input_tensors`, `max_output_buffers`):
1. **Tile geometry/matrix cache** — `TilePlan`, normalized rects, priority order, tile-to-tensor matrices, keyed by image size, tile-rect hash, preprocessing options, model input size, tensor layout, keep-aspect-ratio. Geometry only, no pixels.
2. **Tile surface cache** — reusable tile render targets/surfaces for implementations that need intermediate surfaces (keyed by tile size, format, GL context).
3. **Input tensor cache** — batch input tensors keyed by shape, dtype, batch mode, layout, backend, memory type (`CPU`, `OpenGlTexture2D`, `OpenGlBuffer`). Primary zero-copy cache in GPU mode.
4. **Output/decoder buffer cache** — raw-output read views, decoder scratch, candidate vectors, NMS work buffers keyed by output specs, class count, max candidates, dtype, backend, memory type.
5. **Model/delegate cache** — wire existing `InferenceCalculator` knobs (`cached_kernel_path`, NNAPI `cache_dir`, model resources). Startup/compilation only; no per-frame correctness effect.

**Cache correctness rules:** cache hits must not bypass the one-global-NMS-per-frame invariant; must not mix tiles/detections from different source frames into one batch; invalidate on model id, decoder/preprocessing/tile-policy options, image size, tensor layout, backend/delegate, memory type, class labels, thresholds, or OBB geometry semantics changes; GPU caches are GL-context-scoped and released only after downstream packets release them. Caches default off until covered by tests.

### 14.2 M6 — OpenGL zero-copy input path (folded into Group 2's M6)
When input is `IMAGE_GPU` or an `Image` backed by `GpuBuffer`, tile crop/resize/normalize runs in `GlCalculatorHelper` and writes into cached `Tensor::OpenGlTexture2dView` / `Tensor::OpenGlBufferView` storage compatible with the selected inference path. The tiled path must not call `GetCpuReadView`/readback or materialize `ImageFrame` unless explicitly configured for CPU fallback. Use double/triple-buffered input tensors + GL fences/release callbacks; if the source is CPU-only and GPU mode is requested, upload once then keep all tile/input work on GPU. M6 later generalizes this beyond the OpenGL/TFLite path to LiteRT-Next and platform-specific backends (Metal/Vulkan/CUDA).

**Risks:** OpenGL zero-copy compatibility (tensor texture layout, GL context ownership, delegate expectations, padded rows must line up) — fall back to CPU or GPU-with-copy explicitly rather than implicit readbacks.

### 14.3 M4.5 — Video-mode FlowPackager scheduler
Image mode and video mode share the same decode/merge/NMS calculators; only scheduling differs. Video mode optionally adds a motion/cache branch:

```
Video frame ─► MotionAnalysis ─► FlowPackagerCalculator ─► TRACKING:TrackingData
                                      ├─► TRACKING_CHUNK / CACHE_DIR / COMPLETE (optional)
                                      ▼
Prior final detections cache ─► VideoTileSchedulerCalculator ─► TilePlan
```
`FlowPackagerCalculator` is not a detector and does not replace YOLO; Group-1-family video mode uses its `TrackingData` only as a scheduling/cache signal (prioritize tiles, shrink the tile list, decide a frame can use propagated cached candidates). It does **not** assign long-lived track IDs — BoTSORT remains M7.

**Scheduling policy:** no cross-frame input batches even in video mode; refresh full tiled detection on first frame / cache miss / flow discontinuity / scene cut / confidence drop / at least every `detect_every_n_frames`; track-only frames emit an empty inference `TilePlan` + cache-propagated candidates that still pass through the final frame-level global NMS at the current timestamp; tile priority from `TrackingData`/prior detections gives `DROP_LOW_PRIORITY` deterministic semantics.

**Boundary risk:** cross-frame candidate propagation overlaps M7 (BoTSORT). Keep M4.5 strictly scheduling/caching; if flow confidence is poor or cache chunks are missing, fall back to full tiled detection. Revisit whether candidate propagation should instead live in M7 before implementing.

`FlowPackagerCalculator`'s `CACHE_DIR` is a video-metadata cache (offline/random-access tracking metadata), distinct from the M3.5 tile/input/output caches; in live mode consume `TRACKING` directly and treat `COMPLETE` as offline-cache-finalization only.

### 14.4 Deferred tests
Cache hit/invalidation (tile-plan/matrix, surface, input-tensor, output-buffer); OpenGL zero-copy (GPU-backed input takes GL path, no CPU readback/`ImageFrame`, textures released only after downstream use); video integration (synthetic/fixture video with FlowPackager `TrackingData`, repeated tile plans, skipped-inference frames, cache invalidation, final per-frame global NMS).

---

## Appendix A — Full roadmap (context only)

| Group | Milestone | Feature | Depends on |
|---|---|---|---|
| 1 | M1 | Multi-batch (`N≥1`), ObjectDetection + OBB (CPU) | — |
| 1 | M2 | YOLO detect + OBB decode + rotated NMS | — |
| 1 | M3 | External tiling (externally supplied tiles → batch → merge), CPU | M1, M2 |
| 1 | M4 | Inference metadata output | — |
| 1.5 | M3.5 | Tile/input/output caching (perf, default-off) | M1–M3 |
| 1.5 | M4.5 | Video-mode FlowPackager scheduler | M1–M3 |
| 2 | M5 | Pluggable inference backend + ONNX/PT/TensorRT/CoreML (platform-gated) | — |
| 2 | M6 | Zero-copy GPU: OpenGL-on-TFLite input path + backend-wide LiteRT-Next/platform backends | M3 (contract), M5 |
| 3 | M7 | BoTSORT multi-object tracking | consumes M1–M3 output |

Platform/format matrix for M5 (future spec): ONNX → all (ONNX Runtime / Mobile); PyTorch → desktop/server (libtorch); TensorRT → NVIDIA/Jetson; CoreML → Apple. Zero-copy GPU API per platform: Metal (Apple), GLES/Vulkan (Android), CUDA (NVIDIA).
