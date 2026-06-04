# True Metal Zero-Copy Tiled OBB Detection — End-to-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove true Metal zero-copy end-to-end — the same `MTLBuffer` produced by tiled preprocessing is bound directly to the TFLite Metal delegate (converter skipped) — through a real `yolov8n-obb.tflite` tiled detection pipeline on Apple M2 Max.

**Architecture:** Revive the direct-external-input mode of `InferenceCalculatorMetal` (removed in `2fa11df`, recoverable from `ae0676c`), gated to N=1. Add a physical-PHWC4 output mode to the existing Metal compute writer (small delta — no need to revive the render-pipeline writer). Wire both behind opt-in options; the existing logical-tensor Metal path stays the default and keeps its tests. A new apple-gated E2E test runs two variants (CPU front + normal Metal inference vs GPU direct front + direct Metal inference) and asserts equivalence + true-zero-copy diagnostics.

**Tech Stack:** Bazel; ObjC++ Metal (`MEDIAPIPE_METAL_ENABLED`); TFLite Metal delegate (`TFLGpuDelegateBindMetalBufferToTensor`); `Tensor` (`GetMtlBufferView`/`ready_as_metal_buffer`); api2 calculators.

**Spec:** `docs/superpowers/specs/2026-06-04-tiled-zero-copy-detection-metal-e2e-design.md` (true-direct-bind, N=1, Codex-optimized + reviewed).

---

## Why N=1 direct-bind is safe (read before starting)

`2fa11df` removed the direct-bind path because at N>1 the delegate's input is **batch-innermost** `SHWBC4` (`((S*H+Y)*W+X)*B + b`) while a contiguous packet tensor is batch-outermost — they only match at **N=1** (where `B=1` collapses `SHWBC4` to plain contiguous `PHWC4`). The real `yolov8n-obb.tflite` is fixed batch 1 (`[1,640,640,3]`), so every inference batch is N=1 and multiple tiles run as multiple N=1 batches (`total_batches = T`). Every direct-mode code path in this plan therefore **RET_CHECKs N==1 and fails loudly otherwise** — this guard is permanent, not a temporary shim.

## File map

| File | Change |
|---|---|
| `mediapipe/tasks/testdata/vision/BUILD` | widen `yolo_obb_test_model` filegroup visibility (Task 1) |
| `mediapipe/calculators/tensor/inference_calculator.proto` | re-add `metal_external_input_zero_copy = 14` (Task 2) |
| `mediapipe/calculators/tensor/inference_calculator_metal.cc` | re-add direct-bind mode + **N=1 guard** + `ZERO_COPY_DEBUG` output (Task 2) |
| `mediapipe/calculators/tensor/inference_calculator_metal_zero_copy_test.cc` | re-add `DirectBindMatchesNormalPath` + add `RejectsBatchGreaterThanOne` (Task 2) |
| `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal.{h,cc}` | add physical-PHWC4 output mode to `TiledBatchMetalWriter` (Task 3) |
| `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal_test.cc` | add physical-PHWC4 writer test (Task 3) |
| `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.proto` | add `metal_direct_delegate_input = 11` (Task 4) |
| `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc` | physical-direct branch in `ProcessMetal` + N=1/C≤4 guard (Task 4) |
| `mediapipe/calculators/tensor/tiled_zero_copy_detection_metal_test.cc` | NEW E2E test (Task 5) |
| `mediapipe/calculators/tensor/BUILD` | test/target wiring (Tasks 2,3,5) |

All Metal code is guarded `#if MEDIAPIPE_METAL_ENABLED` and the new test target is apple-gated, so the CPU `--define MEDIAPIPE_DISABLE_GPU=1` build is unaffected (verified in Task 6).

---

### Task 1: Widen test-fixture visibility

**Files:** Modify `mediapipe/tasks/testdata/vision/BUILD`

- [ ] **Step 1: Add an explicit `visibility` to the `yolo_obb_test_model` filegroup.** The package `default_visibility` is `["//mediapipe/tasks:internal"]`, which excludes `//mediapipe/calculators/tensor`. Find the `filegroup(name = "yolo_obb_test_model", ...)` and add a `visibility` attribute (this overrides the package default for just this target):

```python
filegroup(
    name = "yolo_obb_test_model",
    srcs = glob(
        [
            "yolov8n-obb.tflite",
            "yolov8n_obb_labels.txt",
            "boats.jpg",
        ],
        allow_empty = True,
    ),
    visibility = [
        "//mediapipe/tasks:internal",
        "//mediapipe/calculators/tensor:__pkg__",
    ],
)
```

- [ ] **Step 2: Verify the label resolves from `calculators/tensor`.**

Run: `bazel query 'visible(//mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator, //mediapipe/tasks/testdata/vision:yolo_obb_test_model)'`
Expected: prints `//mediapipe/tasks/testdata/vision:yolo_obb_test_model` (i.e. it is visible). If `visible(...)` is unavailable, instead run `bazel query //mediapipe/tasks/testdata/vision:yolo_obb_test_model` and confirm no error.

- [ ] **Step 3: Commit.**

```bash
git add mediapipe/tasks/testdata/vision/BUILD
git commit -m "build(testdata): expose yolo_obb_test_model to calculators/tensor for the zero-copy e2e test"
```

---

### Task 2: Revive `InferenceCalculatorMetal` direct external-input mode (gated N=1)

**Files:**
- Modify: `mediapipe/calculators/tensor/inference_calculator.proto`
- Modify: `mediapipe/calculators/tensor/inference_calculator_metal.cc`
- Create: `mediapipe/calculators/tensor/inference_calculator_metal_zero_copy_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

The complete removed implementation is in commit `ae0676c` (reverted by `2fa11df`). Recover it, then add the N=1 guard and the debug output.

- [ ] **Step 1: Read the removed code.**

Run: `git show ae0676c` and `git show 2fa11df -- mediapipe/calculators/tensor/inference_calculator_metal.cc mediapipe/calculators/tensor/inference_calculator.proto`
Note exactly what `ae0676c` added and `2fa11df` removed: the proto field, the `external_input_zero_copy_` member + `Open()` read + the `Process()` bind branch, and the test file.

- [ ] **Step 2: Re-add the proto field.** In `inference_calculator.proto`, inside `message Gpu` (the last field today is `optional WebNn webnn = 13;`), add field 14 exactly as `ae0676c` had it:

```proto
      // Fork extension (Phase 5, Apple/Metal only). When true, the Metal
      // delegate binds each input packet's MTLBuffer directly as its external
      // input and SKIPS the built-in BHWC->BPHWC4 input conversion — true
      // zero-copy delegate input. The input packet tensor must already be
      // physical PHWC4 float32 ([N,H,W,RoundUp(C,4)]) AND batch N==1 (the
      // delegate's batched layout is batch-innermost; only N==1 matches a
      // contiguous packet). Default off => byte-identical normal path.
      optional bool metal_external_input_zero_copy = 14 [default = false];
```

Run: `bazel build -c opt //mediapipe/calculators/tensor:inference_calculator_cc_proto` → builds.

- [ ] **Step 3: Re-add the bind logic in `inference_calculator_metal.cc`, WITH the new N==1 guard.** Re-apply the `ae0676c` changes: the `bool external_input_zero_copy_ = false;` member, the `Open()` read + `RET_CHECK(!allow_precision_loss_)`, and the `Process()` branch. In the `Process()` branch, BEFORE the existing byte-size `RET_CHECK_EQ`, add the permanent N==1 guard. The branch reads (adapt variable names to the recovered code):

```cpp
    if (external_input_zero_copy_) {
      // True zero-copy: bind the packet's PHWC4 MTLBuffer directly as the
      // delegate input and skip BHWC->BPHWC4 conversion.
      // PERMANENT GUARD: only correct at N==1 — the delegate input is
      // batch-innermost (SHWBC4); a contiguous packet matches it only when
      // batch == 1. Fail loudly for anything else (do NOT fall back).
      const auto& shape = tensor_span[i].shape();
      RET_CHECK(!shape.dims.empty() && shape.dims[0] == 1)
          << "metal_external_input_zero_copy requires input batch N==1, got N="
          << (shape.dims.empty() ? -1 : shape.dims[0])
          << " (input #" << i << "). N>1 would corrupt the batch-innermost "
             "delegate layout.";
      RET_CHECK_EQ(tensor_span[i].bytes(), gpu_buffers_in_[i]->bytes())
          << "metal_external_input_zero_copy input #" << i
          << " byte size mismatch with delegate input.";
      auto input_view = tensor_span[i].GetMtlBufferReadView(command_buffer);
      RET_CHECK_EQ(TFLGpuDelegateBindMetalBufferToTensor(
                       delegate_, interpreter_->inputs()[i], input_view.buffer()),
                   true);
      // record diagnostics (Step 4)
      last_bound_input_addr_ = reinterpret_cast<int64_t>(
          (__bridge void*)input_view.buffer());
      ++direct_binds_;
    } else {
      // ... existing normal converter_to_BPHWC4_ path (unchanged) ...
    }
```

Use the EXACT view/accessor names from the recovered `ae0676c` code (e.g. `GetMtlBufferReadView` vs `GetMtlBufferWriteView`, and how `tensor_span`/`gpu_buffers_in_` are spelled today). The only NEW lines vs `ae0676c` are the N==1 `RET_CHECK` and the two diagnostic lines.

- [ ] **Step 4: Add a minimal debug output.** Add an optional output stream so the E2E test can prove direct binding without a backdoor. In `GetContract`, declare an optional output `ZERO_COPY_DEBUG` of `int64_t`; in `Process()` after running inference, if connected, emit `last_bound_input_addr_` when `external_input_zero_copy_` (else emit `0`). Reset `last_bound_input_addr_ = 0; ` at the top of each `Process()`. (Apple/Metal-gated; unconnected => no effect; `int64_t` avoids a new proto type.) Members:

```cpp
  int64_t last_bound_input_addr_ = 0;  // address of the directly-bound input MTLBuffer this call, 0 if converter ran
  int64_t direct_binds_ = 0;           // cumulative count (debug)
```

Emit (mirror however this file sends optional outputs; tag string `"ZERO_COPY_DEBUG"`):

```cpp
    if (cc->Outputs().HasTag("ZERO_COPY_DEBUG")) {
      cc->Outputs().Tag("ZERO_COPY_DEBUG")
          .AddPacket(MakePacket<int64_t>(last_bound_input_addr_)
                         .At(cc->InputTimestamp()));
    }
```

- [ ] **Step 5: Re-add + extend the unit test.** Recover `inference_calculator_metal_zero_copy_test.cc` from `ae0676c` (it has `DirectBindMatchesNormalPath`, which runs `1x256x256x3_softmax.tflite` through both paths and asserts equal output within `1e-4`). Add one test proving the guard:

```cpp
// N>1 in direct mode must fail loudly, not fall back or corrupt.
TEST_F(InferenceMetalZeroCopyTest, RejectsBatchGreaterThanOne) {
  // Build a graph node with metal_external_input_zero_copy: true and feed a
  // physical [2,H,W,C4] tensor (N=2). Expect graph Run/Process to return a
  // non-OK status mentioning batch N==1. (Reuse RunOnce's harness; add an
  // N=2 entry point, or assert the calculator Open/Process status is not OK.)
  // The exact construction mirrors RunOnce(zero_copy=true) but with N=2.
  ...
}
```

Recover the BUILD target `inference_calculator_metal_zero_copy_test` from `ae0676c` (apple-gated cc_test with the metal copts + the `1x256x256x3_softmax.tflite` data dep).

- [ ] **Step 6: Build + run (real Metal, not the pipe exit).**

Run: `bazel test //mediapipe/calculators/tensor:inference_calculator_metal_zero_copy_test --test_output=all`
Expected: `DirectBindMatchesNormalPath` PASSES (direct-bind output == normal-path output) and `RejectsBatchGreaterThanOne` PASSES (non-OK status). Read the gtest summary, not a piped `exit 0`.

- [ ] **Step 7: Commit.**

```bash
git add mediapipe/calculators/tensor/inference_calculator.proto \
        mediapipe/calculators/tensor/inference_calculator_metal.cc \
        mediapipe/calculators/tensor/inference_calculator_metal_zero_copy_test.cc \
        mediapipe/calculators/tensor/BUILD
git commit -m "feat(metal): revive InferenceCalculatorMetal direct external-input zero-copy, gated N==1

Restores the ae0676c direct-bind path (removed in 2fa11df) with a permanent
RET_CHECK that input batch N==1 (N>1 corrupts the batch-innermost delegate
layout) and a minimal ZERO_COPY_DEBUG int64 output (bound buffer address, 0
when the normal converter ran). Default path byte-identical (option off)."
```

---

### Task 3: Add a physical-PHWC4 output mode to the Metal tiled writer

**Files:**
- Modify: `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal.{h,cc}`
- Modify: `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal_test.cc`

The current `TiledBatchMetalWriter` (compute pipeline) writes 3 floats/pixel into a logical `[N,H,W,3]` buffer at `3*(tile_row*out_h*out_w + y*out_w + x)`. Add a mode that writes **4 floats/pixel** (RGB + 0) into a physical PHWC4 `[1,H,W,4]` buffer at `4*(y*out_w + x)` (single batch row), matching the Metal delegate's N=1 input layout. This is a small shader/stride delta — do NOT revive the old render-pipeline writer.

- [ ] **Step 1: Read the current writer** `streaming_tiles_to_tensor_batch_metal.{h,cc}` end to end. Note `Create(device, out_w, out_h, channels, ...)`, `WriteTileRow(texture, sub_rect, tile_row, alpha, beta, command_buffer, dest)`, and the compute shader's write index/stride.

- [ ] **Step 2: Add a physical mode to `Create`.** Add a trailing `bool physical_phwc4 = false` parameter to `TiledBatchMetalWriter::Create` and a `bool physical_phwc4_` member. When true, the compute pipeline uses a shader variant that writes `4` floats/pixel (`pixel.rgb` then `0.0` for the 4th) at index `4*(tile_row*out_h*out_w + y*out_w + x)`; when false, the existing 3-float logical write is unchanged. Keep the matrix sampling identical. (Implement as either a `#define`/function-constant variant of the existing shader source or a second short shader string selected by `physical_phwc4`.)

- [ ] **Step 3: Write the failing test** in `streaming_tiles_to_tensor_batch_metal_test.cc` (apple/ObjC++): a `PhysicalPhwc4WriterWritesPaddedRgba` test that creates a known gradient `MTLTexture`, allocates a physical `Tensor(kFloat32, Shape{1, out_h, out_w, 4})`, runs `WriteTileRow(..., tile_row=0, alpha=1, beta=0, ...)` with `physical_phwc4=true`, commits + waits, reads back via `GetCpuReadView()`, and asserts each pixel is `[R,G,B,0]` matching the CPU crop/resize/normalize of the sub-rect (channel 4 == 0 exactly; RGB within tolerance).

- [ ] **Step 4: Run it to confirm it fails** (before Step 2 is wired, or with `physical_phwc4=false`).

Run: `bazel test //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_metal_test --test_output=all`
Expected: the new test FAILS (4th channel wrong / wrong stride) until Step 2 is correct.

- [ ] **Step 5: Make it pass** (finish Step 2's shader variant). Re-run → the new test and the existing logical-writer test both PASS.

- [ ] **Step 6: Commit.**

```bash
git add mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal.h \
        mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal.cc \
        mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal_test.cc
git commit -m "feat(metal): physical-PHWC4 [1,H,W,4] output mode for the tiled Metal writer

Adds an opt-in mode writing 4 floats/pixel (RGB + zero pad) for the N=1 delegate
input layout, alongside the default logical [N,H,W,3] compute write."
```

---

### Task 4: Wire `ProcessMetal` to emit physical PHWC4 in direct mode

**Files:**
- Modify: `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.proto`
- Modify: `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc`

- [ ] **Step 1: Add the proto option.** In `StreamingTilesToTensorBatchCalculatorOptions` (last field today is `allow_gpu_readback_fallback = 10`), add:

```proto
  // Apple/Metal only. When true (and enable_gpu_zero_copy + IMAGE_GPU), emit a
  // PHYSICAL PHWC4 float32 [1,H,W,RoundUp(C,4)] Metal tensor per batch for
  // direct delegate binding (pair with InferenceCalculatorMetal
  // metal_external_input_zero_copy=true). Requires N==1 and C<=4; fails loudly
  // otherwise. Default off => the logical [N,H,W,C] Metal path (unchanged).
  optional bool metal_direct_delegate_input = 11 [default = false];
```

Run: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_cc_proto` → builds (proto-only change is CPU-safe).

- [ ] **Step 2: Branch `ProcessMetal`.** In the emit loop of `ProcessMetal` (the one that today builds `Tensor(kFloat32, Shape{N, H, W, C})` and calls `metal_writer_->WriteTileRow(...)` per row), add a direct-mode branch. When `options_.metal_direct_delegate_input()`:

```cpp
    if (options_.metal_direct_delegate_input()) {
      // Direct delegate input: physical PHWC4, N==1 per batch (fixed-batch-1
      // model). Guard loudly — N>1 / C>4 cannot be direct-bound.
      RET_CHECK_EQ(N, 1) << "metal_direct_delegate_input requires batch N==1 "
                            "(set batch_capacity=1); got N=" << N;
      RET_CHECK_LE(C, 4) << "metal_direct_delegate_input requires C<=4; got C=" << C;
      const int c4 = 4;  // RoundUp(C,4) for C<=4
      Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{1, H, W, c4});
      // ... obtain the Metal command buffer + write the single tile row 0 with
      //     the physical writer (physical_phwc4=true), alpha=1, beta=0 ...
      MP_RETURN_IF_ERROR(metal_writer_->WriteTileRow(
          texture, rr, /*tile_row=*/0, /*alpha=*/1.0f, /*beta=*/0.0f,
          command_buffer, write_view.buffer()));
      // commit; emit `tensor` on kOutTensors as a single-element vector exactly
      // as the logical branch does; BATCH_INFO identical (valid_count=1, etc.).
    } else {
      // ... existing logical [N,H,W,C] path (unchanged) ...
    }
```

The `metal_writer_` must have been created with `physical_phwc4=true` when `metal_direct_delegate_input` is set — do this where `metal_writer_` is constructed (in `Open()`/lazy-init), selecting the mode from the option. Everything else (the per-tile multi-batch loop, `TensorBatchInfo`: `total_batches`, `batch_index`, `valid_count`, synthetic `batch_ts_`) stays exactly as the logical path so the merge accumulator behaves identically.

- [ ] **Step 3: Build the calculator (apple).**

Run: `bazel build //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator` (default apple config, Metal on)
Expected: builds. Also confirm the CPU build is untouched: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator`.

- [ ] **Step 4: Commit.**

```bash
git add mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.proto \
        mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc
git commit -m "feat(metal): metal_direct_delegate_input mode emits physical PHWC4 [1,H,W,4]

Pairs with InferenceCalculatorMetal direct external-input for true zero-copy;
RET_CHECKs N==1 and C<=4. Default off => logical Metal path unchanged."
```

---

### Task 5: End-to-end Metal zero-copy detection test

**Files:**
- Create: `mediapipe/calculators/tensor/tiled_zero_copy_detection_metal_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

Assemble the full tiled OBB pipeline twice on `boats.jpg` and assert equivalence + true-zero-copy. Pipeline (per variant): `front → InferenceCalculator(Metal) → YoloObbTensorsToOrientedDetectionsCalculator → MergeTileDetectionsAccumulatorCalculator → RotatedNonMaxSuppressionCalculator`.

- [ ] **Step 1: Read references for graph wiring + helpers.**
  - `mediapipe/calculators/tensor/streaming_to_inference_metal_zero_copy_test.cc` (how it builds the graph, sets `METADATA`, makes a `GpuBuffer`, and runs under Metal — reuse its image→`GpuBuffer` helper).
  - `mediapipe/calculators/tensor/tiled_obb_pipeline_test.cc` (merge + `RotatedNonMaxSuppressionCalculator` node configs, `OrientedDetection` field access).
  - `mediapipe/calculators/tensor/video_tile_scheduler_pipeline_test.cc` (`TilePlan` construction + `StreamingTilesToTensorBatchCalculator` node options).
  - `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_graph.cc` (the OBB decoder node options: `num_classes=15`, `CHANNELS_FIRST`, `conf_threshold`).

- [ ] **Step 2: Gate + load fixtures.** `GTEST_SKIP()` if `yolov8n-obb.tflite` or `boats.jpg` is missing (paths via the same helper the sibling tests use). Decode `boats.jpg` → RGB `ImageFrame`; build a matching opaque-alpha RGBA `GpuBuffer`. Assert model metadata `input_height=640,input_width=640,input_channels=3,batch_capacity=1`.

- [ ] **Step 3: Build a deterministic 2-tile `TilePlan`** over the frame: left `x∈[0.0,0.6]`, right `x∈[0.4,1.0]`, full height (overlap exercises global NMS). Build the matching `InferenceMetadata{640,640,3, batch_capacity=1, dynamic=false}`.

- [ ] **Step 4: Define the two graph configs (text proto), differing only in the front + inference options.**

Variant 1 (Metal control): `StreamingTilesToTensorBatchCalculator` CPU front (`IMAGE`, `enable_gpu_zero_copy=false`) → `InferenceCalculator` (Metal delegate, `metal_external_input_zero_copy=false`) → decode → merge → NMS.

Variant 2 (true zero-copy): `StreamingTilesToTensorBatchCalculator` (`IMAGE_GPU`, `enable_gpu_zero_copy=true`, `metal_direct_delegate_input=true`, `emit_cache_stats=true`, `allow_gpu_readback_fallback=false`) → `InferenceCalculator` (Metal, `metal_external_input_zero_copy=true`, `ZERO_COPY_DEBUG` output connected) → decode → merge → NMS. Additionally fan the front's `TENSORS` and `CACHE_STATS` out to graph `output_stream`s so the test can observe them (a stream may feed both the inference node and a graph output).

Both decode nodes: `num_classes:15`, layout `CHANNELS_FIRST`, `conf_threshold:0.30` (a margin above 0.25 so detections are unambiguous — see Step 6 count guard).

- [ ] **Step 5: Run both variants** at the same source timestamp and tile plan. Collect: the final `std::vector<OrientedDetection>` for each. For variant 2 also collect, per batch (2 batches): the `ZERO_COPY_DEBUG` int64 (bound-input buffer address); the front `TENSORS` packet's Metal buffer address (`tensor.GetMtlBufferReadView(command_buffer).buffer()` cast via `reinterpret_cast<int64_t>((__bridge void*)buffer)`) plus `ready_as_metal_buffer()` / `ready_on_cpu()`; and the single `CACHE_STATS` (`TilingCacheStats`) packet at the source frame.

- [ ] **Step 6: Assertions.**

// (a) Converter skipped: every inference batch bound a real MTLBuffer.
//     total_batches == 2 -> two non-zero debug values.
ASSERT_EQ(zero_copy_debug.size(), 2u);
for (int64_t addr : zero_copy_debug) EXPECT_NE(addr, 0) << "converter ran / no direct bind";
// (a') Direct buffer IDENTITY: the buffers bound to the delegate are exactly the
//      front-tensor buffers (compare as multisets; batch order is deterministic
//      by synthetic timestamp, so sorted-equal is sufficient).
std::sort(zero_copy_debug.begin(), zero_copy_debug.end());
std::sort(front_buffer_addrs.begin(), front_buffer_addrs.end());
ASSERT_EQ(front_buffer_addrs.size(), zero_copy_debug.size());
EXPECT_EQ(front_buffer_addrs, zero_copy_debug) << "delegate bound a different buffer than the front produced";
// (b) GPU residency: front tensors are Metal-resident, not CPU-materialized.
for (const auto& ft : front_tensors_variant2) {
  EXPECT_TRUE(ft.ready_as_metal_buffer());
  EXPECT_FALSE(ft.ready_on_cpu());
}
// (b') No silent fallback: the front never fell back to a CPU readback.
EXPECT_EQ(cache_stats_variant2.gpu_to_cpu_fallbacks, 0);
// (c) Merge emits exactly once per source frame.
ASSERT_EQ(final_packets_variant2.size(), 1u);
// (d) Equivalence: variant 2 ~= variant 1. Sort by score desc, then label_id, then cx.
ASSERT_EQ(dets2.size(), dets1.size());  // unambiguous detections (conf 0.30) => stable count
for (size_t i = 0; i < dets2.size(); ++i) {
  EXPECT_EQ(dets2[i].label_id(0), dets1[i].label_id(0));
  EXPECT_NEAR(dets2[i].cx(), dets1[i].cx(), kPxTol);
  EXPECT_NEAR(dets2[i].cy(), dets1[i].cy(), kPxTol);
  EXPECT_NEAR(dets2[i].width(), dets1[i].width(), kPxTol);
  EXPECT_NEAR(dets2[i].height(), dets1[i].height(), kPxTol);
  EXPECT_NEAR(dets2[i].rotation(), dets1[i].rotation(), kRotTol);
  EXPECT_NEAR(dets2[i].score(0), dets1[i].score(0), kScoreTol);
}
// (e) Sanity: a ship (DOTA class 1) is detected.
EXPECT_TRUE(std::any_of(dets2.begin(), dets2.end(),
            [](const OrientedDetection& d){ return d.label_id(0) == 1; }));
```

Pick `kPxTol` (a few px in the output's coordinate units), `kRotTol` (~0.02 rad), `kScoreTol` (~0.02). If `dets2.size() != dets1.size()` proves flaky from a borderline box, switch the equivalence to "exact match on the subset with `score >= 0.35`" and document it (spec "count guard").

- [ ] **Step 7: Add the apple-gated `cc_test` target** in `mediapipe/calculators/tensor/BUILD` mirroring `streaming_tiles_to_tensor_batch_calculator_metal_test`'s copts (`-x objective-c++`, Metal frameworks) and adding `data = ["//mediapipe/tasks/testdata/vision:yolo_obb_test_model"]` plus deps on the streaming/inference/decode/merge/NMS calculators, `//mediapipe/framework/formats:oriented_detection_cc_proto`, and the GPU test helpers.

- [ ] **Step 8: Build + run on Metal.**

Run: `bazel test //mediapipe/calculators/tensor:tiled_zero_copy_detection_metal_test --test_output=all`
Expected: RUNS (fixtures present) and PASSES — two non-zero `ZERO_COPY_DEBUG` values, one final packet, variant-2 ≈ variant-1, ship detected. Read the real gtest summary.

- [ ] **Step 9: Commit.**

```bash
git add mediapipe/calculators/tensor/tiled_zero_copy_detection_metal_test.cc \
        mediapipe/calculators/tensor/BUILD
git commit -m "test(metal): end-to-end true zero-copy tiled OBB detection (M2 Max)

GPU frame -> direct-PHWC4 tiled front -> direct Metal delegate input -> OBB
decode -> tile->global merge -> global rotated NMS. Asserts converter-skipped
direct bind (ZERO_COPY_DEBUG), one merged frame result, equivalence vs the
CPU-front Metal-inference control, and a ship detection."
```

---

### Task 6: Final verification

- [ ] **Step 1: Metal suite green.**

Run: `bazel test //mediapipe/calculators/tensor:inference_calculator_metal_zero_copy_test //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_metal_test //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_metal_test //mediapipe/calculators/tensor:tiled_zero_copy_detection_metal_test --test_output=errors`
Expected: all pass on M2 Max.

- [ ] **Step 2: CPU build still green (no regression, Metal excluded).**

Run: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator //mediapipe/calculators/tensor:inference_calculator` and `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test`
Expected: builds + the CPU Group-1 test passes; the apple-only targets are excluded.

- [ ] **Step 3: `git status` clean** except the gitignored model fixtures.

---

## Self-review checklist
- Default paths byte-identical: `metal_external_input_zero_copy` (default false), `metal_direct_delegate_input` (default false), the writer's `physical_phwc4` (default false), and `ZERO_COPY_DEBUG` (optional, unconnected) all leave existing behavior unchanged; the existing logical Metal tests + CPU Group-1 tests stay green.
- The N==1 guard exists in BOTH new code paths (inference direct-bind AND the writer/`ProcessMetal` physical mode) and fails loudly (RET_CHECK), never silently falls back.
- The oracle is variant 2 vs variant 1 (both Metal inference); the CPU-inference path is not the equivalence oracle.
- `total_batches=2`, `batch_capacity=1`, merge emits once at source ts (uses the existing, verified merge accumulator unchanged).
- Field numbers don't collide: `inference_calculator.proto` Gpu field `14`; streaming options field `11`.
- True zero-copy is proven by a real output (`ZERO_COPY_DEBUG` non-zero ⇒ direct bind, converter skipped), not inferred.

## Out of scope
N>1 direct binding (needs an SHWBC4-aware writer); output zero-copy; GLES end-to-end; axis-aligned YOLO tiled detection; public Tasks API wiring; the bounded GPU input-tensor pool (separate Phase-5 follow-up).
