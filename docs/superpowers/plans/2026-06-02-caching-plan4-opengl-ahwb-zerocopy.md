# Caching — Plan 4: OpenGL/AHWB zero-copy tiled preprocessing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> ⚠️ **BUILD/RUN ENVIRONMENT — READ FIRST.** This plan's code is GPU (OpenGL ES 3.1 SSBO + AHardwareBuffer) and was authored on a machine where it could be **neither compiled nor run** (desktop build has `MEDIAPIPE_DISABLE_GPU=1`; the Android cross-compile toolchain does not resolve in this hybrid WORKSPACE+Bzlmod checkout — `No matching toolchains found for @bazel_tools//tools/cpp:toolchain_type`). **Every task here MUST be built and tested on a GPU-capable environment** — an Android device/emulator (`--config=android_arm64`, GLES 3.1) or a Linux EGL desktop GPU build — before it is considered done. Do not mark a task complete on a build that excludes the GPU code path. The CPU-buildable scaffolding subset (Task 1) is the only part verifiable under `MEDIAPIPE_DISABLE_GPU=1`.

**Goal:** Add an OpenGL ES 3.1 zero-copy path to `StreamingTilesToTensorBatchCalculator` so that, for GPU-backed input and a GPU-capable inference delegate, tiled crop/resize/normalize runs entirely on the GPU — each tile written directly into its row of a GPU/AHWB-backed `[N,H,W,C]` tensor batch — with no `GetCpuReadView()`/`GetCpuWriteView()`/`MatView`/`cv::resize`/CPU upload, plus context-scoped resource pools and an explicit in-flight ownership model so buffers are reused only after downstream packet release **and** GPU fence completion.

**Architecture:** Adapt the existing `ImageToTensorGlBufferConverter` (`image_to_tensor_converter_gl_buffer.cc`, GLES 3.1 compute shader → SSBO) into a *tiled batch* converter: one compute dispatch per valid tile, each writing into a **row offset** of a single batch SSBO obtained from `Tensor::GetOpenGlBufferWriteView()`. The output tensor's GlSync fence (created by the write view) gives the write→read GPU sync for free. GPU/AHWB tensor buffers come from a context-scoped, bounded pool (Cache 5) governed by an in-flight ownership state machine; GL programs/surfaces come from a second context-scoped cache (Cache 4). Everything is gated by `enable_gpu_zero_copy` and falls back to the (now shipped) CPU path per explicit policy. The transform per tile reuses the **same** `GetRotatedSubRectToRectTransformMatrix` convention already used by `tiling_matrix_utils` (Plan 2) and `image_to_tensor` — so CPU and GPU projections are identical.

**Tech Stack:** C++20, MediaPipe GPU (`GlCalculatorHelper`, `GlContext`, `GpuBuffer`), `Tensor` OpenGL-buffer + AHardwareBuffer views, TFLite GPU GL helpers (`tflite::gpu::gl::{GlProgram,GlShader,GlBuffer,CommandQueue}`), GLES 3.1 compute shaders, Bazel (`--config=android_arm64` or Linux EGL GPU). GoogleTest on device/emulator.

**Spec:** `docs/superpowers/specs/2026-06-02-caching-tile-geometry-design.md` — "Cache 4: OpenGL tile-surface cache", "Cache 5: OpenGL/AHWB tensor-buffer cache", "OpenGL resource scope key", "In-flight GPU/OpenGL ownership model", "Zero-copy OpenGL path".

**Builds on:** Plan 1 (`tiling_cache_utils.h`), Plan 2 (multi-batch + `TileBatchGeometry` + `tiling_matrix_utils`), Plan 3 (CPU pool + `TilingCacheStats` + `CACHE_STATS`). **Last caching plan in the M8 sequence.**

---

## Verification environment (NON-NEGOTIABLE)

A task is "done" only when it builds AND its tests pass in a GPU-capable config:

- **Preferred:** Android device or emulator with GLES 3.1. Build/test e.g. `bazel build --config=android_arm64 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator` and run instrumented GPU tests on the device. (This requires the android NDK toolchain to resolve, which it currently does NOT on the authoring machine — fix toolchain resolution first, or use a properly-configured CI.)
- **Alternative:** Linux desktop with EGL + a GPU exposing GLES 3.1, GPU NOT disabled.
- **GPU test discipline (from the spec):** zero-copy/perf tests set `allow_gpu_readback_fallback = false`; any CPU readback (`MatView`, `GetCpuReadView`, `GetCpuWriteView`) in the GPU path is a test FAILURE. Add a readback detector (Task 8).
- Each task commits separately; co-author trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Branch `dev`.

## Reference facts (verified against shipped code)

- **`image_to_tensor_converter_gl_buffer.cc` is THE template.** Whole file is guarded `#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31`.
  - Compute shader `kShaderCode` (lines 94–152): `layout(binding=0) writeonly buffer B0 { float elements[]; } output_data;` uniforms `out_size` (ivec2), `alpha`, `beta`, `transform_matrix` (mat4), `input_data` (sampler2D). Per invocation: normalize gid → `tc = transform_matrix * vec4(nx,ny,0,1)` → `texture(input_data, tc.xy)*alpha+beta` → write 3 floats at `3*(gid.y*out_width+gid.x)`. Workgroup `{8,8,1}`.
  - `SubRectExtractorGl::ExtractSubRectToBuffer(...)` (line 154): builds the transform via `GetRotatedSubRectToRectTransformMatrix(sub_rect, texture.w, texture.h, flip, &mat)`, binds the texture sampler + the destination `GlBuffer` to binding 0, sets uniforms, `command_queue->Dispatch(program_, DivideRoundUp({W,H,1}, {8,8,1}))`.
  - `ImageToTensorGlBufferConverter::Convert(...)` (line 267): `auto buffer_view = output_tensor.GetOpenGlBufferWriteView(); tflite::gpu::gl::GlBuffer output(GL_SHADER_STORAGE_BUFFER, buffer_view.name(), bytes, /*offset=*/0, /*has_ownership=*/false);` then `ExtractSubRectToBuffer(...)`. Runs inside `gl_helper_.RunInGlContext(...)`. `gl_helper_` is a `mediapipe::GlCalculatorHelper`.
- **`Tensor` GPU/AHWB views (`tensor.h`):**
  - `OpenGlBufferView GetOpenGlBufferWriteView(...) const` (line 455) and `GetOpenGlBufferReadView() const` (454). The write view, on destruction, creates a GlSync fence; the next read view waits on it on the GPU (doc lines 441–447). `OpenGlBufferView::name()` returns the SSBO `GLuint`.
  - `AHardwareBufferView GetAHardwareBufferWriteView() const` (307) / `...ReadView()` (306). `AHardwareBufferView::handle()` → `AHardwareBuffer*`; it carries `TensorAhwbUsage*` for read-finished callbacks/fences (lines 233–302).
  - `Tensor` ctor takes `MemoryManager*` (Plan 3 added the CPU pool there; AHWB pool already there via `GetAndroidHardwareBufferPool()`).
- **`tiling_matrix_utils.h` (Plan 2):** `std::array<float,16> TileToImageMatrix(const TilePixelRoi&, int fw, int fh)` and `ApplyMatrix`/`InvertAffine2d`. The GPU shader's `transform_matrix` is exactly a `TileToImageMatrix`-style row-major mat4 (tile/tensor-normalized → image-normalized) — but note the ImageToTensor shader maps *output tensor coords → input image coords* (the INVERSE direction of `TileToImageMatrix`'s detection projection). Use `GetRotatedSubRectToRectTransformMatrix` directly on the tile's `effective_pixel_roi` as the existing converter does (it already produces the output→input sampling matrix); do NOT reuse `TileToImageMatrix` for the shader uniform (that one maps tile→frame for detection projection, opposite direction). Keep the two clearly separated and unit-test that the GPU sampling matrix == the converter's matrix for a full-frame tile.
- **`StreamingTilesToTensorBatchCalculator` (post-Plan-3):** api2 `Node`, `namespace mediapipe::api2`, contract `(kInImage[IMAGE, ImageFrame], kInPlan[TILE_PLAN], kSideMeta[METADATA], kOutTensors[TENSORS], kOutInfo[BATCH_INFO], kOutStats[CACHE_STATS]::Optional, TimestampChange::Arbitrary())`. Builds `TileBatchGeometry` (cached), emits per-batch `[N,H,W,C]` CPU tensors via the pixel loop, supports `T>cap` multi-batch at synthetic `batch_ts_`, has a calculator-local CPU-pool `MemoryManager`. proto fields: `dynamic_batch=1`, `max_cached_tile_matrices=2`, `max_cpu_tensor_workspaces=3`, `emit_cache_stats=4`.
- **`tiling_cache_stats.h` (Plan 3):** `struct TilingCacheStats { CacheStats tile_plan, tile_matrix; CpuBufferPoolStats cpu_tensor_pool; };` — Plan 4 extends it with GPU fields.
- **GpuBuffer / GL input:** GPU calculators take `Image`/`GpuBuffer`; `GlCalculatorHelper::CreateSourceTexture(gpu_buffer)` → `GlTexture` (name + width/height). `gl_helper_.Open(cc)` in `Open()`, work inside `gl_helper_.RunInGlContext(...)`.

## Cross-cutting conventions

- All GL/AHWB code lives behind `#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31` (SSBO) / `#ifdef MEDIAPIPE_TENSOR_USE_AHWB`. The CPU path (shipped) is the `#else`/fallback and MUST remain byte-identical and the default.
- All new options default to `0`/`false`. Negative capacities rejected with `RET_CHECK_GE(...,0)` before signed→unsigned conversion. `enable_gpu_zero_copy=false` ⇒ existing behavior exactly.
- No cached entry holds pixels; reused surfaces/buffers are fully overwritten (every valid row) before emission; padding rows cleared on GPU.
- A buffer/surface emitted in a packet is not reusable until BOTH downstream packet release AND GPU fence completion are observed.
- GL resources are scoped to `GpuResourceScopeKey`; never reused across incompatible contexts; cleared in `Close()` and on context teardown.
- In-editor clang errors are FALSE POSITIVES; only a GPU-capable bazel build is authoritative.

## Public options (add to `StreamingTilesToTensorBatchCalculatorOptions`)

```proto
optional int32 max_gpu_tensor_buffers = 5 [default = 0];
optional int32 max_tile_surfaces = 6 [default = 0];
optional bool enable_gpu_zero_copy = 7 [default = false];
optional bool allow_cpu_input_fallback = 8 [default = true];
optional int32 max_in_flight_gpu_batches = 9 [default = 0];
optional bool allow_gpu_readback_fallback = 10 [default = false];
```
(Fields 1–4 are taken by Plans 2–3. Use 5–10 exactly as above to match the design doc.)

## File Structure

- **Create** `mediapipe/calculators/tensor/tiling_gpu_resource.h` — `GpuResourceScopeKey`, `GpuResourceState` enum, `GpuBufferPoolStats`/`TileSurfaceStats` (extend stats), and the in-flight ownership accounting type. CPU-buildable (pure types) — verifiable under `MEDIAPIPE_DISABLE_GPU=1`.
- **Create** `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_gl.{h,cc}` — the GLES 3.1 tiled-batch SSBO converter (adapted from `image_to_tensor_converter_gl_buffer.cc`), entirely under the GLES 3.1 guard. Writes each tile into a row offset of one batch SSBO.
- **Modify** `streaming_tiles_to_tensor_batch_calculator.{proto,cc}` — GPU input contract (`IMAGE_GPU`), option plumbing, GPU vs CPU dispatch, fences, fallback, GPU stats.
- **Modify** `tiling_cache_stats.h` — add `gpu_tensor_buffer`, `tile_surface` `CacheStats` fields.
- **Create** GPU tests (instrumented/device): `streaming_tiles_to_tensor_batch_gl_test.cc` and additions to the calculator test.
- **Modify** `mediapipe/calculators/tensor/BUILD` — GPU deps under a `select()`/`MEDIAPIPE_DISABLE_GPU` gate mirroring how `image_to_tensor_calculator` wires its GL converters; new targets.

> Before writing the BUILD, READ how `image_to_tensor_calculator`'s BUILD conditionally includes `image_to_tensor_converter_gl_buffer` (it uses a `select()` keyed on `//mediapipe/gpu:disable_gpu` or platform). Mirror that exact pattern so the CPU build never compiles the GL TU.

---

### Task 1: GPU scaffolding types + option plumbing (CPU-buildable)

**Files:** Create `tiling_gpu_resource.h`; modify `tiling_cache_stats.h`, `streaming_tiles_to_tensor_batch_calculator.proto`, `...calculator.cc` (Open-time validation only), `BUILD`.

This task contains NO GL code and MUST build under `MEDIAPIPE_DISABLE_GPU=1` (the one task verifiable on the authoring machine). It lays the contract/types so later GPU tasks slot in.

- [ ] **Step 1:** Create `tiling_gpu_resource.h` (full Apache header):
```cpp
#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TILING_GPU_RESOURCE_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TILING_GPU_RESOURCE_H_

#include <cstdint>

namespace mediapipe {

// Context-scope identity for GPU resource caches. A resource created under one
// key must never be reused under a different key. gl_context_identity is the
// GlContext* (or a stable share-group id if available); api_version is the
// GLES major*10+minor. See spec "OpenGL resource scope key".
struct GpuResourceScopeKey {
  const void* gl_context_identity = nullptr;
  const void* share_group_identity = nullptr;
  int api_version = 0;
  bool operator==(const GpuResourceScopeKey& o) const {
    return gl_context_identity == o.gl_context_identity &&
           share_group_identity == o.share_group_identity &&
           api_version == o.api_version;
  }
};

// Ownership lifecycle of a pooled GPU resource (spec "In-flight ... ownership").
enum class GpuResourceState {
  kFree,                // safe to acquire
  kAcquiredForWrite,    // owned by the current Process()
  kSubmitted,           // GL writes enqueued; waiting on write fence/sync
  kInFlightDownstream,  // emitted in a packet
  kReclaimable,         // packet released AND GPU work complete
};

}  // namespace mediapipe
#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TILING_GPU_RESOURCE_H_
```

- [ ] **Step 2:** Extend `tiling_cache_stats.h` with GPU fields:
```cpp
struct TilingCacheStats {
  CacheStats tile_plan;
  CacheStats tile_matrix;
  CpuBufferPoolStats cpu_tensor_pool;
  CacheStats gpu_tensor_buffer;  // Cache 5
  CacheStats tile_surface;       // Cache 4
  int64_t in_flight_gpu_batches = 0;
  int64_t gpu_to_cpu_fallbacks = 0;
};
```

- [ ] **Step 3:** proto — add the six options listed under "Public options" above (fields 5–10).

- [ ] **Step 4:** `.cc` `Open()` — add validation only (no behavior yet):
```cpp
    RET_CHECK_GE(options_.max_gpu_tensor_buffers(), 0);
    RET_CHECK_GE(options_.max_tile_surfaces(), 0);
    RET_CHECK_GE(options_.max_in_flight_gpu_batches(), 0);
    // Zero-copy with no finite buffer/surface capacity is a config error
    // (spec): if enabled and max_in_flight_gpu_batches==0, both
    // max_gpu_tensor_buffers and max_tile_surfaces must be > 0.
    if (options_.enable_gpu_zero_copy() &&
        options_.max_in_flight_gpu_batches() == 0) {
      RET_CHECK_GT(options_.max_gpu_tensor_buffers(), 0)
          << "enable_gpu_zero_copy requires a finite GPU buffer/in-flight "
             "capacity";
    }
```

- [ ] **Step 5:** BUILD — `cc_library(name="tiling_gpu_resource", hdrs=["tiling_gpu_resource.h"])`; add it + the extended stats to the calculator deps. **Verify under CPU build:**
`bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator` — must build (this is the one CPU-verifiable gate in Plan 4).
- [ ] **Step 6:** Commit `feat(tensor-cache): GPU zero-copy scaffolding types + options (CPU-buildable)`.

---

### Task 2: Tiled GLES 3.1 batch SSBO converter — single tile into a row offset

**Files:** Create `streaming_tiles_to_tensor_batch_gl.{h,cc}` (+ device test) under the GLES 3.1 guard; modify BUILD.

**[GPU env required.]** Adapt `image_to_tensor_converter_gl_buffer.cc`. The one structural change vs. the reference: write into a **row offset** of a `[N,H,W,C]` batch SSBO instead of a single `[H,W,C]` tensor. Add a `row_base` uniform = `tile_row * H * W` so `first_component_index = 3 * (row_base + gid.y*out_width + gid.x)`.

- [ ] **Step 1:** Header `streaming_tiles_to_tensor_batch_gl.h` (guard the whole body with `#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31`). Declare:
```cpp
// Writes one tile's crop/resize/normalize result into row `tile_row` of a
// [N,H,W,C] float SSBO bound at binding 0. RGB (C==3) only for v1.
class TiledBatchGlWriter {
 public:
  static absl::StatusOr<std::unique_ptr<TiledBatchGlWriter>> Create(
      const mediapipe::GlContext& gl_context, int out_w, int out_h, int channels,
      BorderMode border_mode, bool input_starts_at_bottom);

  // sub_rect: tile's RotatedRect over the source texture (effective pixel ROI,
  // built via GetRotatedSubRectToRectTransformMatrix-compatible RotatedRect).
  // alpha/beta: normalization (1/255, 0 for [0,1]). dest: the batch SSBO.
  absl::Status WriteTileRow(const tflite::gpu::gl::GlTexture& texture,
                            const tflite::gpu::HW& texture_size,
                            const RotatedRect& sub_rect, int tile_row,
                            float alpha, float beta,
                            tflite::gpu::gl::CommandQueue* command_queue,
                            tflite::gpu::gl::GlBuffer* dest);
};
```

- [ ] **Step 2:** `.cc` — copy `kShaderCode` from the reference and add `uniform int row_base;` and change the write index:
```glsl
    int linear_index = row_base + gid.y * out_width + gid.x;
    int first_component_index = 3 * linear_index;
```
Set `row_base = tile_row * out_h * out_w` as a program uniform before each dispatch (alongside `transform_matrix`, `out_size`, `alpha`, `beta`). Reuse `SubRectExtractorGl::Create` shader-assembly verbatim except for the modified shader body and the extra uniform. `WriteTileRow` mirrors `ExtractSubRectToBuffer` but sets `row_base` and does NOT reset the destination between tiles (the caller dispatches all tiles into the same `dest` before reading).

- [ ] **Step 3:** Device test `streaming_tiles_to_tensor_batch_gl_test.cc` (guarded; instrumented): create a GlContext, upload a known RGBA texture, allocate a `Tensor` `[2,H,W,3]`, `GetOpenGlBufferWriteView()` → `GlBuffer`, call `WriteTileRow` twice (rows 0 and 1) with two sub-rects, then read back via `GetOpenGlBufferReadView()`/CPU and assert each row contains the expected resized/normalized pixels for its sub-rect. Assert the GPU sampling matrix equals `GetRotatedSubRectToRectTransformMatrix` for a full-frame sub-rect.
- [ ] **Step 4:** BUILD — new `cc_library(name="streaming_tiles_to_tensor_batch_gl", ...)` gated like `image_to_tensor_converter_gl_buffer` (deps: `:image_to_tensor_utils`, `:image_to_tensor_converter_gl_utils`, `//mediapipe/gpu:gl_calculator_helper`, the tflite gpu gl deps). Device test target.
- [ ] **Step 5:** Build+test on GPU env. Commit `feat(tensor-cache): tiled GLES 3.1 SSBO writer (one tile per batch row)`.

---

### Task 3: GPU input contract + zero-copy dispatch in the calculator

**Files:** modify `...calculator.{proto,cc}` (+ device test). **[GPU env required.]**

- [ ] **Step 1:** Add GPU input. Mirror `ImageToTensorCalculator`'s dual input: keep `kInImage{"IMAGE"}` (CPU `ImageFrame`/CPU `Image`) and add `static constexpr Input<mediapipe::GpuBuffer>::Optional kInImageGpu{"IMAGE_GPU"};` (or accept GPU-backed `Image` on `IMAGE` — match whatever ImageToTensor does; READ it). Add `kInImageGpu` to the contract. Add `GlCalculatorHelper gl_helper_;` member; `gl_helper_.Open(cc)` in `Open()` when a GPU input is connected and `enable_gpu_zero_copy`.
- [ ] **Step 2:** In `Process()`, branch: if `enable_gpu_zero_copy` AND GPU input present AND (delegate GPU-capable — for v1 assume yes when enabled): run the GPU path inside `gl_helper_.RunInGlContext([&]{...})`:
  1. `GlTexture src = gl_helper_.CreateSourceTexture(*kInImageGpu(cc));`
  2. Acquire a GPU/AHWB-backed batch `Tensor` `[N,H,W,C]` (Task 4 pool; for this task, allocate directly).
  3. `auto wv = tensor.GetOpenGlBufferWriteView(); GlBuffer dest(GL_SHADER_STORAGE_BUFFER, wv.name(), bytes, 0, false);`
  4. For each valid row `r`: build the tile's `RotatedRect` from `geom->effective_pixel_rois[r]` (same ROI as CPU), `writer_->WriteTileRow(src, {fw,fh}, sub_rect, r, 1.0/255.0, 0.0, command_queue, &dest);`
  5. Clear padding rows `[valid_count, N)` on GPU (a memset dispatch or `glClearBufferSubData`).
  6. Emit the tensor on `kOutTensors` at `batch_ts_` (the write view's destruction creates the fence; downstream inference's read view waits on it).
- [ ] **Step 3:** Fallback policy (spec): if GPU input absent → CPU path iff `allow_cpu_input_fallback`, else `RET_CHECK`-fail with an actionable message. If GPU input present but zero-copy disabled → CPU path. Record `gpu_to_cpu_fallbacks` in stats.
- [ ] **Step 4:** Device test: drive the calculator with a GPU `Image` + a 2-tile plan; assert the emitted tensor (read back) matches the CPU path's tensor within tolerance, and that NO CPU view was taken in the GPU path (Task 8 detector). Commit `feat(tensor-cache): OpenGL zero-copy tiled preprocessing path`.

---

### Task 4: Cache 5 — context-scoped GPU/AHWB tensor-buffer pool + in-flight ownership

**Files:** modify `...calculator.cc`; possibly extend `MemoryManager`/`Tensor` for OpenGL-buffer pooling if release tracking is insufficient. **[GPU env required.]**

- [ ] **Step 1:** AHWB first (release tracking already exists). When `MEDIAPIPE_TENSOR_USE_AHWB` and a `MemoryManager` with an AHWB pool is available, construct batch tensors with that `MemoryManager` so AHWB buffers are pooled and returned on packet release + read-finished fence (`AHardwareBufferView` release funcs). Bound by `max_gpu_tensor_buffers`/`max_in_flight_gpu_batches`.
- [ ] **Step 2:** Non-AHWB OpenGL SSBO pooling: per the spec, do NOT pool with calculator-local reuse unless BOTH packet release AND GPU fence completion are observable. Implement the `GpuResourceState` machine: a resource goes `kFree→kAcquiredForWrite→kSubmitted(write fence)→kInFlightDownstream(emitted)→kReclaimable(packet released + fence signaled)→kFree`. Track in-flight count; never exceed the effective limit; never block in `Process()`.
- [ ] **Step 3:** `GpuResourceScopeKey`: key the pool by `(GlContext* identity, share-group id if available, api_version)`. A resource from one key is never reused under another. Clear pools in `Close()` and on context teardown.
- [ ] **Step 4:** Exhaustion: if no free resource and at the limit → for GPU input, `ResourceExhausted` with in-flight count/capacity/timestamp/batch-index UNLESS `allow_gpu_readback_fallback`. Never read back in zero-copy perf mode.
- [ ] **Step 5:** Device tests: pool reuse after warmup (no per-frame GPU alloc) with a slow downstream consumer proving no reuse-before-release; two non-shared GlContexts do not reuse each other's resources; exhaustion behavior both modes. Commit `feat(tensor-cache): context-scoped GPU/AHWB tensor-buffer pool + in-flight ownership`.

---

### Task 5: Cache 4 — OpenGL program/tile-surface cache

**Files:** modify `streaming_tiles_to_tensor_batch_gl.{h,cc}`, `...calculator.cc`. **[GPU env required.]**

- [ ] **Step 1:** Cache the compiled `GlProgram` (and any sampler/state objects, intermediate surfaces if used) by a key of `(GpuResourceScopeKey, source format/origin, out W/H, dtype/layout/channels, border mode, shader variant flags)`. The program is expensive to compile; reuse across frames with stable config.
- [ ] **Step 2:** Context-scoped (same rules as Task 4); destroy on `Close()`/teardown; overwrite reused surfaces before read; bound by `max_tile_surfaces`.
- [ ] **Step 3:** Device test: program built once across many stable frames (assert via stats `tile_surface.hits`), two contexts isolated. Commit `feat(tensor-cache): context-scoped GL program/tile-surface cache`.

---

### Task 6: Wire GPU stats into `CACHE_STATS`

**Files:** modify `...calculator.cc`. **[partially CPU-buildable]**

- [ ] **Step 1:** Fill `gpu_tensor_buffer`, `tile_surface`, `in_flight_gpu_batches`, `gpu_to_cpu_fallbacks` in the `TilingCacheStats` emitted on `CACHE_STATS` (guarded so the fields stay zero in CPU builds). Keep allocation-free when disabled.
- [ ] **Step 2:** Device test asserts GPU stats reflect hits/in-flight under stable GPU tiling. Commit `feat(tensor-cache): expose GPU pool/surface stats on CACHE_STATS`.

---

### Task 7: Fallback + dynamic-batch + padding correctness on GPU

**Files:** `...calculator.cc` + device tests. **[GPU env required.]**

- [ ] **Step 1:** Fixed batch: clear rows `[valid_count, N)` on GPU. Dynamic batch: emit `N==valid_count`, no padding. Multi-batch `T>cap`: same synthetic-`batch_ts_` emission as the CPU path, one GPU tensor per batch.
- [ ] **Step 2:** Device tests: `T<cap`/`T==cap`/`T>cap`/empty-frame on the GPU path; fixed vs dynamic batch; GPU output == CPU output within tolerance for all. Commit `feat(tensor-cache): GPU path padding + dynamic/multi-batch parity with CPU`.

---

### Task 8: CPU-readback detector + perf/safety instrumentation

**Files:** test helper + device tests. **[GPU env required.]**

- [ ] **Step 1:** A test-only hook/guard that fails if `MatView`/`GetCpuReadView`/`GetCpuWriteView` is called on the GPU path while `allow_gpu_readback_fallback==false`. (Implement via a debug counter on Tensor under a test flag, or by asserting the output tensor was never CPU-materialized — choose the least invasive approach that actually observes readback.)
- [ ] **Step 2:** Perf tests: stable GPU multi-tile video shows no per-frame GPU tensor allocation after warmup (stats), and no CPU readback. Bounded memory within configured capacities. Commit `test(tensor-cache): CPU-readback detector + GPU perf/bounded-memory tests`.

---

## Self-review checklist (before final review)
- CPU path unchanged and default; `enable_gpu_zero_copy=false` ⇒ byte-identical to Plan 3.
- GPU sampling matrix == `GetRotatedSubRectToRectTransformMatrix` (CPU/GPU projection parity), verified by test.
- Each tile writes its own row (`row_base`); padding rows cleared; no cross-frame/cross-tile mixing.
- No buffer/surface reuse before BOTH packet release AND GPU fence completion.
- Resources are `GpuResourceScopeKey`-scoped; two contexts proven isolated; pools cleared on teardown.
- Exhaustion + fallback policies match the spec; no CPU readback when `allow_gpu_readback_fallback==false`.
- All new options default off; negative capacities rejected.
- **Every GPU task built+tested on a GPU-capable env (NOT on the authoring machine's CPU build).**

## Final verification (GPU-capable env)
- [ ] GLES 3.1 device/emulator: all `streaming_tiles_to_tensor_batch_gl_test` + GPU calculator tests pass.
- [ ] CPU build (`MEDIAPIPE_DISABLE_GPU=1`): all existing tests still pass; Task-1 scaffolding compiles; GL TUs are excluded by the BUILD `select()`.
- [ ] CPU-readback detector green (no readback in zero-copy mode).
- [ ] `git status` clean.

## Done criteria
- Zero-copy OpenGL tiled preprocessing verified on a GL-capable environment (or explicitly marked not-tested with CPU-only tests passing, per spec) — removes GPU→CPU readback, CPU crop/resize/normalize, and CPU→GPU upload for GPU inference.
- GPU/AHWB tensor-buffer + program/surface pools are context-scoped, bounded, and reuse only after release+fence.
- GPU output matches CPU output within tolerance for all tile-count and batch-mode cases.
- CACHE_STATS exposes GPU hit rates, in-flight count, and fallbacks.
- No stale pixels, no cross-frame mixing, no reuse before downstream release.

## Known environment blocker (carried from authoring)
The android cross-compile toolchain does not resolve in this checkout (hybrid WORKSPACE+Bzlmod; `No matching toolchains found for @bazel_tools//tools/cpp:toolchain_type`), and desktop GPU is disabled, so NONE of Tasks 2–8 can be built or run here. Resolve the android NDK toolchain (or use a GPU CI/Linux-EGL/device) before executing this plan. Task 1 (scaffolding) is the only part buildable under `MEDIAPIPE_DISABLE_GPU=1`.
