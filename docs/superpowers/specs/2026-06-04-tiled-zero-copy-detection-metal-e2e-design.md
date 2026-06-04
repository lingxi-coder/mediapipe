# Spec - End-to-end true Metal zero-copy tiled OBB detection (Phase 5)

Date: 2026-06-04
Status: Design (pre-plan)
Branch: `dev` (long-lived integration branch; commit only when asked)
Roadmap: `docs/superpowers/specs/2026-06-01-roadmap.md` Phase 5 (zero-copy preprocessing).
Builds on: the Metal tiled writer/calculator, the M6 zero-copy design
(`2026-06-01-m6-zero-copy-design.md` sections 4-6), and the Phase-1
tiling/merge calculators.

## Problem

The Metal tiled-preprocessing path is currently verified only as a GPU-resident
logical tensor path: it can crop/resize/normalize on Metal, emit a Metal-backed
logical `[N,H,W,C]` tensor, and let the unmodified `InferenceCalculatorMetal`
perform its normal BHWC->BPHWC4 input conversion. That proves "no CPU round-trip",
but it does **not** satisfy the stricter zero-copy requirement for this project:
the TFLite Metal delegate must consume the **same `MTLBuffer`** produced by tiled
preprocessing.

The real OBB fixture also has fixed batch 1:

- input: `[1,640,640,3]`
- output: `[1,20,8400]`
- `shape_signature`: `[1,640,640,3]`

Therefore a tiled test with two tiles must emit two `N=1` inference batches, not
one `[2,640,640,3]` batch. The current design must prove the real fixed-batch
multi-tile path: GPU frame -> two tile batches -> direct Metal delegate input ->
OBB decode -> tile-to-global merge -> one global rotated NMS result.

## Goal & success criteria

1. One apple-gated integration test assembles the **full tiled OBB detection
   pipeline** on the real `yolov8n-obb.tflite` fixture:
   `IMAGE_GPU` -> tile -> direct-delegate Metal tensor batch -> Metal inference
   -> OBB decode -> tile-to-global merge -> global rotated NMS ->
   `std::vector<OrientedDetection>`.
2. True zero-copy is proven, not inferred: for every inference batch, the
   `MTLBuffer` written by tiled preprocessing is the same buffer bound to the
   TFLite Metal delegate input, and the Metal input converter
   (`converter_to_BPHWC4_`) is not used for that input.
3. Multi-tile fixed-batch behavior is covered: the deterministic two-tile plan
   runs as two `N=1` batches (`batch_capacity=1`, `total_batches=2`) and the merge
   accumulator emits exactly one source-frame result.
4. Correctness is proven by comparison against a Metal-backend control on the same
   frame and tile plan: CPU tiled front + normal Metal inference vs GPU tiled front
   + direct Metal inference must match within tolerance on count, center, size,
   rotation, score, and `label_id`. A CPU-inference run may remain as an additional
   sanity reference, but it is not the primary equivalence oracle.
5. No silent fallback: the direct GPU front tensors are Metal-resident and not
   CPU-materialized before inference, and `CACHE_STATS.gpu_to_cpu_fallbacks == 0`.
6. At least one detection has DOTA class id 1 (`ship`) on `boats.jpg`, proving the
   test is a real detector run and not an empty equality check.
7. CPU build stays green: the new Metal test and any direct-Metal implementation
   are apple/Metal-gated and excluded from
   `--define MEDIAPIPE_DISABLE_GPU=1` builds.

## Background facts (verified)

- `StreamingTilesToTensorBatchCalculator` contract:
  `IMAGE` (`ImageFrame`, optional), `IMAGE_GPU` (`GpuBuffer`, optional),
  `TILE_PLAN`, side input `METADATA` (`InferenceMetadata`) -> `TENSORS`,
  `BATCH_INFO`, optional `CACHE_STATS`. The output tag is `CACHE_STATS`, not
  `STATS`; stats are emitted only when `emit_cache_stats=true` and the output is
  connected.
- `ready_on_gpu()`, `ready_on_cpu()`, and `ready_as_metal_buffer()` are `Tensor`
  methods, not fields in `TilingCacheStats`. The test must observe the front
  `TENSORS` stream before explicit readback to assert residency.
- The shipped Metal front emits a logical `[N,H,W,3]` Metal-backed tensor and the
  unmodified `InferenceCalculatorMetal` converts that to delegate layout. That is
  useful as a control path but is not true zero-copy under this spec.
- Direct delegate input with a contiguous packet buffer is valid for this fixture
  because every inference batch has `N=1`. The earlier direct-input approach was
  reverted for `N>1` because the delegate's batched layout is batch-innermost
  (`SHWBC4`/BPHWC4), while a normal contiguous packet tensor is batch-outermost.
  This test deliberately keeps `batch_capacity=1` and covers multiple tiles via
  multiple batches.
- `InferenceMetadata.batch_capacity` is the model input batch dim. It must be
  derived from the loaded model metadata and must be 1 for `yolov8n-obb.tflite`;
  it must not be set to the tile count.
- `YoloObbTensorsToOrientedDetectionsCalculator` consumes raw OBB model output
  `TENSORS` and emits
  `std::vector<std::vector<OrientedDetection>>`, one outer row per batch row.
  For `yolov8n-obb.tflite`, use `CHANNELS_FIRST`, `num_classes=15`,
  `conf_threshold=0.25`, and the default pre-NMS cap unless the implementation
  needs a tighter deterministic cap.
- `MergeTileDetectionsAccumulatorCalculator` already supports multiple
  synthetic batch timestamps for one source frame. It waits until all
  `TensorBatchInfo.total_batches` arrive, projects tile-local OBBs to global
  frame coordinates, flattens them, and emits once at the source frame timestamp.
- `RotatedNonMaxSuppressionCalculator` runs the final global NMS over the merged
  frame detections.
- Fixtures are gitignored and glob-gated in
  `//mediapipe/tasks/testdata/vision:yolo_obb_test_model`, but that filegroup is
  currently visible only to `//mediapipe/tasks/...` and `//mediapipe/python/...`.
  A test under `//mediapipe/calculators/tensor` needs the filegroup visibility
  widened, or the test must live under a visible package.

## Approach

### Direct Metal input seam

This is not assembly-only. If direct delegate input support is not present, add
the minimal production seam required to prove true zero-copy:

- `StreamingTilesToTensorBatchCalculator` Metal path gets an explicit direct-input
  mode for Metal delegate layout. It is valid only when:
  - `enable_gpu_zero_copy=true`
  - `IMAGE_GPU` is connected and non-empty
  - model input channels are `C<=4`
  - emitted batch size is `N=1`
- In that mode, the Metal writer emits one physical delegate-layout input buffer
  per tile batch: logical model input `[1,H,W,C]`, physical storage
  `[1,H,W,RoundUp(C,4)]`, with padded channels zero-filled. Do not feed this
  physical tensor to generic CPU calculators.
- `InferenceCalculatorMetal` gets an explicit direct external-input mode. For a
  direct input tensor, it binds the packet tensor's `MTLBuffer` to the model input
  with `TFLGpuDelegateBindMetalBufferToTensor()` for that invocation and skips the
  normal input `TFLBufferConvert`.
- The direct mode validates model/input compatibility up front and fails loudly
  for unsupported batch sizes or channel counts. Unsupported cases must not fall
  back to the normal converter when direct zero-copy was requested.

The test must include a diagnostic proof of direct binding. Acceptable evidence:
a debug-gated stat/output from `InferenceCalculatorMetal` (a real, minimal output
field — not a test-only backdoor) that reports, per batch, the bound input
`MTLBuffer` identity and whether the input converter was skipped. The E2E test
compares that identity with the `MTLBuffer` observed from the front `TENSORS`
packet.

Implementation note (PHWC4 parity): the direct PHWC4 writer must reproduce the
Metal input converter's normalization and PHWC4 packing exactly (same range
mapping; padded channels zero-filled), since variant 2 skips
`converter_to_BPHWC4_` while variant 1 runs it. The variant-2-vs-variant-1
equivalence assertion is the guard for this parity.

### Graph variants

Run the same frame and deterministic tile plan through these variants:

1. **Metal control:** `IMAGE` -> CPU tiled front -> normal `InferenceCalculator`
   with Metal delegate -> OBB decode -> merge -> global rotated NMS.
2. **True zero-copy:** `IMAGE_GPU` -> Metal tiled front in direct-input mode ->
   `InferenceCalculatorMetal` direct external input -> OBB decode -> merge ->
   global rotated NMS.
3. **Optional CPU sanity:** `IMAGE` -> CPU tiled front -> CPU inference -> OBB
   decode -> merge -> global rotated NMS. This can help debugging but must not be
   the primary oracle for direct zero-copy correctness because it changes the
   inference backend.

The primary comparison is variant 2 vs variant 1, so both paths use the same
Metal delegate and differ only in front-end preprocessing plus direct-input
binding.

### Fixed-batch multi-tile procedure

1. Gate on `yolov8n-obb.tflite` and `boats.jpg`. If absent, `GTEST_SKIP()` with a
   clear message.
2. Load the model metadata from `InferenceCalculator`'s `METADATA` side packet or
   the same helper used by existing inference metadata tests. Assert:
   `input_height=640`, `input_width=640`, `input_channels=3`,
   `batch_capacity=1`, `is_dynamic_batch=false`.
3. Decode `boats.jpg` into an RGB `ImageFrame`. Build a matching RGBA
   `GpuBuffer` with the same RGB pixels and opaque alpha.
4. Build a deterministic two-tile `TilePlan` over the same source frame. Prefer
   overlapping full-height tiles, e.g. left `x=[0.0,0.6]` and right
   `x=[0.4,1.0]`, so global NMS is exercised when duplicate detections occur.
5. Feed both graph variants with the same source timestamp and same tile plan.
   Because model batch capacity is 1, each variant must emit two tensor packets
   and two `BATCH_INFO` packets at synthetic batch timestamps, both with
   `valid_count=1`, `batch_capacity=1`, `total_batches=2`. Configure the direct
   front with `emit_cache_stats=true` and connect `CACHE_STATS`.
6. Decode, merge, and run one global rotated NMS per source frame.

### Assertions

- **Direct buffer identity:** for each emitted batch, the front tensor's
  `MTLBuffer` identity equals the delegate-bound input buffer identity reported by
  direct `InferenceCalculatorMetal`.
- **Input converter skipped:** direct-mode diagnostics report zero input
  `TFLBufferConvert` uses and one direct input bind per emitted batch.
- **GPU residency:** before any test-side CPU readback, each direct front tensor is
  `ready_as_metal_buffer() == true`, `ready_on_gpu() == true`, and
  `ready_on_cpu() == false`.
- **No fallback:** connected `CACHE_STATS` is emitted with
  `gpu_to_cpu_fallbacks == 0`; if direct zero-copy cannot run, the test fails
  rather than silently using the CPU or normal Metal conversion path.
- **Multi-batch merge:** the final output stream emits exactly one packet at the
  original source timestamp, not one packet per tile batch.
- **Equivalence:** sort paired outputs deterministically (score descending, then
  `label_id`, then center) and compare variant 2 to variant 1 within tolerances:
  centers/sizes in normalized frame units or pixels according to the current
  calculator output contract, small rotation epsilon, score epsilon, exact
  `label_id`. Count equality is asserted, but to avoid flakiness when a single
  borderline detection near `conf_threshold` survives in only one path, pick
  `conf_threshold` + tile geometry so the expected detections are unambiguous
  (well above threshold), or assert exact equality only on the confident subset
  (e.g. `score >= conf_threshold + margin`). Document whichever guard the test
  uses.
- **Sanity:** at least one final detection has `label_id == 1` (`ship`).

## Error handling / fallback / gating

- Fixture-gated: missing `yolov8n-obb.tflite` or `boats.jpg` skips the runtime
  assertions cleanly, while the target still builds.
- Metal-gated: direct Metal code and the new E2E test compile only on Apple with
  Metal enabled. The CPU build with `MEDIAPIPE_DISABLE_GPU=1` must exclude them.
- Visibility: add an explicit `visibility = ["//mediapipe/calculators/tensor:__pkg__", ...]`
  attribute to the `yolo_obb_test_model` filegroup (overriding the package
  `default_visibility = ["//mediapipe/tasks:internal"]`) so the new test keeps its
  siblings under `//mediapipe/calculators/tensor`. (Alternative, only if widening
  is undesirable: place the E2E target under a package already covered by
  `//mediapipe/tasks:internal`.)
- Unsupported direct cases (`N>1`, `C>4`, non-Metal build, missing Metal buffer)
  return an actionable error when direct zero-copy is requested. They must not
  fall back to normal conversion.

## Testing

- New or updated production tests for the direct Metal input seam:
  - direct binding smoke: a packet `MTLBuffer` is bound to the delegate input
    and the input converter is skipped.
  - unsupported `N>1` direct mode fails clearly.
- New E2E test:
  `mediapipe/calculators/tensor/tiled_zero_copy_detection_metal_test.cc`
  (apple/ObjC++, Metal-gated, fixture-gated), plus the required BUILD target and
  fixture visibility/data wiring.
- Existing tests remain:
  - `streaming_tiles_to_tensor_batch_calculator_metal_test` for front-only
    GPU-resident logical tensor behavior.
  - `streaming_to_inference_metal_zero_copy_test` can remain as the normal
    logical-tensor Metal handoff test, but it no longer proves true zero-copy.
  - `tiled_obb_pipeline_test` for merge/global NMS behavior.
- Verification command is a normal Apple Metal build/test, not
  `--define MEDIAPIPE_DISABLE_GPU=1`. The CPU regression build still runs with
  `--define MEDIAPIPE_DISABLE_GPU=1` and must stay green.

## Out of scope

- Direct true zero-copy for `N>1` in a single delegate invocation. Multi-tile is
  covered here through multiple `N=1` batches because the current OBB fixture is
  fixed batch 1. A future `N>1` direct path needs an SHWBC4/BPHWC4-aware writer.
- Direct output zero-copy; this test may keep the existing Metal output converter
  and read final tensors for decode.
- GLES end-to-end, axis-aligned YOLO tiled detection, and public YOLO/OBB Tasks
  API wiring.

## Verification environment

Metal-verifiable here on Apple M2 Max with the gitignored OBB fixture present:
build/run the new apple-gated target with the default Apple GPU configuration
(no `--define MEDIAPIPE_DISABLE_GPU=1`). Also run the CPU regression build with
`--define MEDIAPIPE_DISABLE_GPU=1` to prove the Metal-only target and direct seam
are excluded cleanly.
