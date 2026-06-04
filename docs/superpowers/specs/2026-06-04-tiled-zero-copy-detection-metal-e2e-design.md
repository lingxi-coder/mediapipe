# Spec — End-to-end Metal zero-copy tiled-detection integration test (Phase 5)

Date: 2026-06-04
Status: Design (pre-plan)
Branch: `dev` (long-lived integration branch; commit only when asked)
Roadmap: `docs/superpowers/specs/2026-06-01-roadmap.md` Phase 5 (zero-copy preprocessing).
Builds on: the Metal zero-copy writer/calculator (`metal-zerocopy-phase5-progress`), the M6 zero-copy design (`2026-06-01-m6-zero-copy-design.md` §4–6), and the Phase-1 tiling/merge calculators.

## Problem

The Metal zero-copy tiled-preprocessing path is implemented and GPU-verified **in isolation**, but it has never been wired into a full detection pipeline. The pieces exist and are individually tested, yet they are split across three tests and never chained from a GPU frame to merged global detections:

- **Front** (`video_tile_scheduler_pipeline_test.cc`): frame → `VideoTileScheduler` → `TileSpecToTilePlan` → `StreamingTilesToTensorBatch`. Stops at the tensor batch.
- **Inference seam** (`streaming_to_inference_metal_zero_copy_test.cc`): zero-copy tensors → `InferenceCalculator` (Metal), but against a dummy `1x256x256x3_softmax.tflite` — no real detector, no decode.
- **Back** (`tiled_obb_pipeline_test.cc`): already-decoded tile detections + `BATCH_INFO` → `MergeTileDetectionsAccumulator` → `RotatedNonMaxSuppression`.

No test proves the zero-copy GPU front actually produces correct detections through the real detection back-half. This spec closes that gap with a single end-to-end integration test on Metal.

## Goal & success criteria

1. One apple-gated integration test assembles the **full tiled OBB detection pipeline** — GPU frame → tile → zero-copy tensor batch → Metal inference → OBB decode → tile→global merge → global rotated NMS → `std::vector<OrientedDetection>` — and runs it on Apple M2 Max.
2. **Equivalence proof:** the same pipeline is run with the **CPU front** (`IMAGE` + CPU inference) and the **GPU zero-copy front** (`IMAGE_GPU` + Metal inference) on the same frame and tile plan; the final global detections must match within tolerance (count + per-detection center/size/rotation/score/label). This proves the zero-copy path is equivalent to the already-verified CPU-tiled path.
3. **No silent fallback:** the test asserts the GPU front actually ran on GPU (`ready_on_gpu && !ready_on_cpu` from the calculator's `STATS`), upholding the M6 "no silent fallback" invariant.
4. **Sanity anchor:** at least one "ship" (DOTA class index 1) is detected, confirming the run is a real detection, not an empty-but-equal pair.
5. Assembly only — no new calculators and no production-code change. The deliverable is a test file + a BUILD target (+ a small GpuBuffer test helper if not already shared).
6. CPU build (`--define MEDIAPIPE_DISABLE_GPU=1`) stays green: the new test is apple/Metal-gated and excluded from the CPU build, like the sibling metal tests.

## Background facts (verified)

- **`StreamingTilesToTensorBatchCalculator` contract** (`mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc`): `Input<ImageFrame>::Optional IMAGE`, `Input<GpuBuffer>::Optional IMAGE_GPU`, `Input<TilePlan> TILE_PLAN`, `SideInput<InferenceMetadata> METADATA` → `Output<std::vector<Tensor>> TENSORS`, `Output<TensorBatchInfo> BATCH_INFO`, `Output<...> STATS`. The Metal zero-copy branch is taken when `options.enable_gpu_zero_copy() && IMAGE_GPU` is connected and non-empty; otherwise the CPU `IMAGE` path runs. `BATCH_INFO` (the `TileBatchGeometry` + tile transforms) is derived from `TILE_PLAN` + frame size and is therefore **identical for the CPU and GPU fronts** — only `TENSORS` differ (and must be ≈equal).
- **Metal zero-copy emits a logical `[N,H,W,3]` Metal-backed tensor**; the unmodified `InferenceCalculatorMetal` runs its normal BHWC→BPHWC4 conversion (correct for all N). "Zero-copy" = no CPU round-trip, not "no input conversion". The calculator's `STATS` carries `ready_on_gpu` / `ready_on_cpu` counters (asserted in `streaming_tiles_to_tensor_batch_calculator_metal_test.cc`).
- **OBB decoder `YoloObbTensorsToOrientedDetectionsCalculator`**: `Input TENSORS` → `Output<std::vector<std::vector<OrientedDetection>>> ORIENTED_DETECTIONS` (outer = batch row / tile, inner = per-tile detections). Normalized-by-default; `CHANNELS_FIRST` for `yolov8n-obb.tflite` (`[1,20,8400]` = 4 box + 15 DOTA classes + 1 angle); `num_classes = 15`.
- **`MergeTileDetectionsAccumulatorCalculator`**: `Input<std::vector<std::vector<OrientedDetection>>> ORIENTED_DETECTIONS` + `Input<TensorBatchInfo> BATCH_INFO` → `Output<std::vector<OrientedDetection>> ORIENTED_DETECTIONS`. It projects each tile-local box to global frame coordinates using the per-tile transform from `BATCH_INFO`'s `TileBatchGeometry.tile_geometries`, then flattens. **The decoder's output type matches this input type exactly — no adapter needed.** This calculator is OBB-only (there is no axis-aligned `Detection` accumulator), which is why this integration uses OBB.
- **`RotatedNonMaxSuppressionCalculator`**: global rotated NMS over the merged `OrientedDetection`s. (Both the merge and this NMS are already chained + tested in `tiled_obb_pipeline_test.cc`.)
- **Fixtures** (gitignored, glob-gated via `//mediapipe/tasks/testdata/vision:yolo_obb_test_model`, `allow_empty = True`): `yolov8n-obb.tflite`, `yolov8n_obb_labels.txt`, `boats.jpg`. Ship = DOTA index 1. The test `GTEST_SKIP`s cleanly when absent.
- **Metal runs on this Mac (M2 Max)**: GPU comes up as "2.1 Metal - 89.4" (the `gpu_buffer_format.h` CVPixelBuffer fix, commit `9c13613`). Every Metal task is build-AND-run verifiable in the bazel sandbox here. The GLES path is build-deferred; this test is Metal-only.
- **`InferenceMetadata` side packet** carries target `H, W, C, batch_capacity, dynamic`. For this test it is derived from the OBB model's input shape; `batch_capacity` must be ≥ the tile count.

## Approach (assembly + CPU-equivalence; no new calculators)

### The assembled pipeline (one graph template, two fronts)

```
                ┌─ [GPU] IMAGE_GPU:GpuBuffer → StreamingTilesToTensorBatch(enable_gpu_zero_copy=true)
TILE_PLAN ──────┤                                                                                       → TENSORS, BATCH_INFO
(+ METADATA)    └─ [CPU] IMAGE:ImageFrame   → StreamingTilesToTensorBatch (CPU front, zero-copy off)
                         │
                         ▼
        InferenceCalculator  (Metal delegate for the GPU front · CPU for the CPU front)   → raw OBB tensors [N,20,8400]
                         ▼
        YoloObbTensorsToOrientedDetectionsCalculator   → std::vector<std::vector<OrientedDetection>>  (tile-local)
                         ▼
        MergeTileDetectionsAccumulatorCalculator (+ BATCH_INFO)   → projects tile→global, flattens
                         ▼
        RotatedNonMaxSuppressionCalculator (global)   → std::vector<OrientedDetection>
```

The back-half (decode → merge → global NMS) is byte-identical between the two fronts; only the preprocessing front and the inference backend differ.

### Test procedure
1. Decode `boats.jpg` to an `ImageFrame`; wrap a copy as a Metal-backed `GpuBuffer` (reuse the existing metal-test image→GpuBuffer helper; add a small shared helper only if none exists).
2. Build a **deterministic 2-tile `TilePlan`** covering the frame (e.g. left/right halves with the model's input aspect), plus a matching `InferenceMetadata` (`H,W,C` from the model, `batch_capacity = 2`). The fixed plan keeps the test focused on the zero-copy front + back-half rather than the scheduler. (`TileSpecToTilePlan` is an allowed alternative if a fixed plan proves awkward; either way both fronts consume the identical plan.)
3. Run the **CPU pipeline** → `global_dets_cpu`.
4. Run the **GPU pipeline** (`enable_gpu_zero_copy = true`, `IMAGE_GPU` fed) under a `GpuResources`/Metal context → `global_dets_gpu`, and capture the front calculator's `STATS`.
5. Assert (success criteria 2–4 below).

### Assertions
- **GPU-resident:** `STATS.ready_on_gpu > 0 && STATS.ready_on_cpu == 0` for the GPU run (no silent CPU fallback).
- **Equivalence:** `global_dets_gpu.size() == global_dets_cpu.size()`; sorting both by score, each paired detection agrees within tolerance on `cx, cy, width, height` (a few px), `rotation` (small radians), `score` (small epsilon), and exact `label_id`. Tolerances account for GPU vs CPU float and the BHWC→BPHWC4 conversion; they are not zero.
- **Sanity:** at least one detection has `label_id == 1` ("ship").

### Why CPU-equivalence (vs an external oracle)
Comparing GPU↔CPU within the same graph makes the proof self-contained and robust: it does not depend on hand-derived oracle box coordinates, it directly tests the property we care about (the zero-copy front is equivalent to the verified CPU front), and it reuses the already-verified CPU-tiled back-half as the reference.

## Error handling / fallback / gating

- **Fixture-gated:** if `yolov8n-obb.tflite` or `boats.jpg` is absent, `GTEST_SKIP()` with a clear message (mirror the sibling task/metal tests). The target must still build.
- **Metal-gated:** the test is compiled only on apple with Metal enabled (same BUILD copts/guards as `streaming_tiles_to_tensor_batch_calculator_metal_test`), and excluded from the CPU `--define MEDIAPIPE_DISABLE_GPU=1` build.
- **No silent fallback:** if the GPU front fell back to CPU, assertion (3) fails loudly rather than the test passing on CPU numbers.

## Testing (CPU + Metal)

- New: `mediapipe/calculators/tensor/tiled_zero_copy_detection_metal_test.cc` (apple/ObjC++, Metal-gated, fixture-gated) implementing the procedure above; new apple-gated `cc_test` target in `mediapipe/calculators/tensor/BUILD` (mirror the metal test target's copts + `data` deps incl. `//mediapipe/tasks/testdata/vision:yolo_obb_test_model`).
- The CPU build stays green (target is apple-only); the existing Group-1 / metal / merge tests are untouched.
- Verified on Apple M2 Max: the test runs (fixtures present) and passes — GPU front ran on GPU, GPU≈CPU detections, ship detected.

## Out of scope

- GLES end-to-end (build-deferred; the shared front means this test still de-risks it). Axis-aligned YOLO tiled detection (needs a `Detection` merge accumulator — separate work). Wiring tiled zero-copy into the public YOLO/OBB Tasks API. The bounded fence-gated input-tensor pool (design §5.1 — separate Phase-5 follow-up). Any production-code change to the calculators.

## Verification environment

Metal-verifiable here on Apple M2 Max: build/run the new apple-gated target with the **default apple configuration (Metal GPU enabled)** — i.e. the same way the sibling `streaming_tiles_to_tensor_batch_calculator_metal_test` is invoked, NOT with `--define MEDIAPIPE_DISABLE_GPU=1` — with the gitignored OBB model fixture present. The plan pins the exact `bazel test` incantation. The CPU build (`--define MEDIAPIPE_DISABLE_GPU=1`) excludes the apple-only target and stays green.
