# M6 Zero-Copy GPU Design (M6a OpenGL-on-TFLite + M6b LiteRT-Next follow-on)

- **Date:** 2026-06-01
- **Status:** Approved (design). **Design-only — implementation deferred** until a Linux/Android OpenGL build+test environment is available (this macOS dev machine cannot build/run the desktop-GL GPU path; all of Group 1 was built with `--define MEDIAPIPE_DISABLE_GPU=1`).
- **Scope:** **M6a** (OpenGL-on-TFLite zero-copy input path) is the implementable core. **M6b** (LiteRT-Next backend-wide buffer interop) is designed here as a sequenced follow-on phase but not implemented.
- **Build model:** Fork / custom build of MediaPipe.

> This is feature 6 ("Zero-copy performance optimization — LiteRT-Next GPU") from the parent request. It was deferred out of Group 1 (see `2026-06-01-detection-core-yolo-obb-tiling-design.md` §14.2) with the Group-1 calculator contracts kept stable so this attaches without breaking them.

---

## 1. Background & goal

Group 1 built a CPU tiling→inference pipeline: `StreamingTilesToTensorBatchCalculator` crops/resizes/normalizes externally-supplied tiles into a CPU metadata-shaped `[N,H,W,C]` tensor, which `InferenceCalculator` runs. When frames are already GPU-resident (camera/GL pipelines deliver `GpuBuffer`/`IMAGE_GPU`), the CPU path forces a GPU→CPU download for tiling and a CPU→GPU upload for GPU inference — two full-frame copies per frame on the hot path.

**M6a goal:** when the input frame is GPU-resident, crop/resize/normalize tiles **directly on the GPU** into the inference input tensor's OpenGL-backed storage, then bind that storage directly to the TFLite GL delegate input. "Zero-copy" here means no CPU↔GPU copy and no extra GL-buffer-to-GL-buffer copy between the tiled input tensor and inference.

**Non-goal (this milestone):** output-side zero-copy, GPU-side NMS, and non-OpenGL GPU backends (Metal/Vulkan/CUDA) — those are M6b / future.

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope of this spec | Design both; M6a implementable core + M6b sequenced follow-on | M6b (LiteRT-Next) has no foundation in the repo and depends on M5; M6a builds on existing primitives. |
| Primary GPU API | OpenGL (GLES + desktop GL) | Matches the existing `Tensor::OpenGlTexture2dView`/`OpenGlBufferView`, the GL `image_to_tensor` converters, and the TFLite GL delegate. |
| Verification | **Design-only now**; implementation deferred to a Linux/Android GL env; code structured to split CPU-testable logic from GL-context code | This macOS machine can't build/run desktop-GL GPU calculators; Group 1's local TDD loop doesn't extend to the GL runtime. |
| Inference runner | Use `inference_calculator_gl_advanced` or an equivalent direct-SSBO binding path for true zero-copy | The standard `inference_calculator_gl.cc` currently copies input with `glCopyBufferSubData`; it is a GPU-with-copy fallback, not the M6a zero-copy path. |
| Structure | Reuse the existing `ImageToTensorConverter` row-offset interface first; add only thin tile/batch helpers around it if needed | The existing converter already accepts `tensor_buffer_offset` and the GL buffer converter writes one ROI into an SSBO offset. Avoid a parallel converter abstraction unless implementation proves it necessary. |
| Caching | M6a owns a **bounded, fence-gated GL input-tensor pool** (§5.1); the other cache layers stay in the separate **M8** milestone (detection-core spec §14.1) | Zero-copy removes the CPU↔GPU *copy*, but allocating a fresh GL SSBO every frame pays a per-frame GPU *allocation* cost that erases much of the win — so buffer reuse is intrinsic to M6a's value (and §5's double/triple-buffering already half-mandates it). M8's layer 3 was already labeled "primary zero-copy cache in GPU mode." Tile-geometry, output/decoder, and model/delegate caches remain orthogonal M8 perf. |

## 3. Current-state findings (codebase, verified)

- `Tensor` exposes GPU-resident storage views: `OpenGlTexture2dView` (read/write; **BHWC-only**, per the header note) and `OpenGlBufferView` (read/write; the write view creates a `GlSync` fence). These are the zero-copy destinations. Do **not** introduce a parallel GPU memory abstraction.
- The image→tensor pattern to mirror already exists: `image_to_tensor_converter.h` (interface) with `image_to_tensor_converter_gl_buffer.{cc,h}`, `…_gl_texture.{cc,h}`, `…_metal.{cc,h}`, `…_opencv.{cc,h}` impls, selected at runtime by `image_to_tensor_calculator.cc` under `#if !MEDIAPIPE_DISABLE_GPU`. The GL converters already do single-image crop+resize+normalize via GL shaders.
- `inference_calculator_gl.cc` and `inference_calculator_gl_advanced.cc` are both TFLite GL-delegate inference paths, but they are not equivalent for M6a:
  - `inference_calculator_gl.cc` explicitly copies each input tensor into an internal delegate buffer with `glCopyBufferSubData`; this does not satisfy input zero-copy.
  - `inference_calculator_gl_advanced.cc` binds the caller-provided SSBO to the delegate input via `BindSSBOToInputTensor`; this is the preferred M6a path.
- GPU code throughout is gated by `#if !MEDIAPIPE_DISABLE_GPU` and uses `GlCalculatorHelper` / `mediapipe/gpu/gpu_buffer.h`.
- **No LiteRT-Next in the repo:** no `litert::TensorBuffer` / buffer-interop references and no LiteRT module dependency. M6b requires adding that dependency and depends on M5 (pluggable backends).
- Group 1's `StreamingTilesToTensorBatchCalculator` is CPU-only (OpenCV crop/resize → CPU `Tensor`), consumes `InferenceMetadata` (from M4) for `H/W/C`/dtype/layout/`batch_capacity`/`is_dynamic_batch`, and **ships a single-batch-per-frame guard** `RET_CHECK_LE(T, batch_capacity)` (`streaming_tiles_to_tensor_batch_calculator.cc:77`). Multi-batch emission (`T > batch_capacity`) was **deferred** in Plan 3b — emitting multiple batch packets at one input timestamp would violate output-stream timestamp monotonicity (a deep CHECK-failure), so the guard converts it into a clean boundary error. **M6 inherits this guard; it must not assume multi-batch streaming exists.** (If multi-batch is ever wanted, it needs a timestamp-offsetting scheme — its own milestone, GPU or CPU.)

## 4. Architecture — converter reuse + batcher ownership

The batcher owns frame/tile bookkeeping, batch sizing, padding, `TensorBatchInfo`, tile indices, and batch emission (one batch per frame under the shipped guard). Pixel conversion is delegated to the existing image-to-tensor converter seam.

```
                 InferenceMetadata (side packet, from M4: H,W,C,batch_capacity,dynamic)
                                   │
Frame (ImageFrame | GpuBuffer) ───┤
TilePlan (from TileSpecToTilePlan) ┤
                                   ▼
   StreamingTilesToTensorBatchCalculator  (bookkeeping + batch emission)
     • batch sizing, padding, TensorBatchInfo, tile_indices
     • one batch per frame: T <= batch_capacity guard (RET_CHECK_LE);
       fixed-batch pads to N == batch_capacity; multi-batch deferred
     • selects converter by input type:
         ImageFrame → ImageToTensorConverterOpenCv / existing CPU path
         GpuBuffer  → ImageToTensorGlBufferConverter / GL path
                                   │
                                   ▼
            Tensor [N,H,W,C]  (CPU-backed OR OpenGL-backed)
                                   ▼
   InferenceCalculator(CPU)  OR  InferenceCalculatorGlAdvanced/direct SSBO
                                   ▼
            raw outputs ──► YOLO / OBB decode (CPU) ──► merge ──► global NMS
```

### 4.1 Existing converter seam

Prefer the existing `ImageToTensorConverter` API:

```cpp
absl::Status Convert(const mediapipe::Image& input,
                     const RotatedRect& roi,
                     float range_min,
                     float range_max,
                     int tensor_buffer_offset,
                     Tensor& output_tensor);
```

The batcher converts each validated `TileGeometry` into the converter's absolute `RotatedRect`, computes `tensor_buffer_offset = row_index * bytes_per_row`, and calls `Convert()` once per valid row in the current emitted batch. This matches the existing GL buffer converter, which already writes one ROI into an SSBO at the supplied offset.

If later profiling shows per-row dispatch overhead is too high, add an optional bulk helper that takes `std::vector<TileGeometry>` and emits the same row-offset writes internally. That helper must be an optimization over the existing seam, not a replacement for the calculator contract.

`TileToTensorParams` remains an internal convenience struct only if needed by the batcher/helper. It must mirror `InferenceMetadata`: target `H/W/C`, dtype, layout, normalization range, and keep-aspect-ratio behavior.

### 4.2 Batcher changes (behavior-preserving for CPU)
`StreamingTilesToTensorBatchCalculator` keeps the Group-1 contract verbatim: no cross-frame batching, one batch per frame under the shipped `RET_CHECK_LE(T, batch_capacity)` guard (multi-batch streaming deferred), fixed/dynamic `N`, padding (`N == batch_capacity` for fixed-batch models after padding), `TensorBatchInfo`, `tile_indices`. The CPU path should be a behavior-preserving refactor at most; existing Group-1 batcher tests must stay green unchanged.

## 5. The GL converter path (zero-copy core)

- Runs in the calculator's `GlCalculatorHelper` GL context.
- Allocates / receives the `[N,H,W,C]` input `Tensor` and takes `Tensor::GetOpenGlBufferWriteView()` (SSBO) as the GL-resident destination. SSBO is the primary M6a target because `inference_calculator_gl_advanced` can bind caller-provided SSBOs directly. `OpenGlTexture2dView` remains a fallback only if the selected delegate path explicitly supports that layout.
- Initial M6a supports `float32` GL input tensors. If `InferenceMetadata` reports another input dtype, fail clearly until quantized/integer GL packing is designed and tested.
- For each valid tile row `r`: call the GL buffer image-to-tensor converter with that tile's ROI and `tensor_buffer_offset = r * H * W * C * sizeof(dtype)`. The converter samples the source `GpuBuffer` texture, bilinear-resizes to `H×W`, applies the normalization range, and writes into the row-`r` SSBO offset.
- Fixed-batch padding rows are cleared to zero once (e.g., `glClearBufferSubData`/zero-fill before the tile loop).
- **No CPU readback:** when input is `GpuBuffer`, the converter never calls `GetCpuReadView`/`MatView`/materializes `ImageFrame`.
- **Sync, memory visibility, and lifetime:**
  - `OpenGlBufferView` fences/release callbacks handle cross-context ordering and object lifetime, but they do **not** make SSBO writes visible by themselves.
  - After zero-fill and tile shader writes, issue the required GL memory barrier before inference reads the SSBO (at minimum `GL_SHADER_STORAGE_BARRIER_BIT`; include `GL_BUFFER_UPDATE_BARRIER_BIT` when the padding clear path uses buffer update commands).
  - Use double/triple-buffered input tensors (the bounded pool of §5.1) plus release callbacks so an input tensor is not reused for another frame or batch until downstream inference packets release it.
  - Do not call `GetOpenGlBufferWriteView()` multiple times on the same already-valid `Tensor` instance as a cache reuse mechanism; `Tensor` is designed for single writes. Pooling reuses the underlying GL *storage* behind fresh logical tensor packets — see §5.1.

### 5.1 GL input-tensor pool (bounded, fence-gated) — part of M6a

The double/triple-buffering above is concretely a small bounded pool of GL-backed input tensors, reused across frames so the zero-copy path does not re-allocate SSBO storage every frame. This is **in M6a** (it is what makes zero-copy fast); the broader cache subsystem stays in M8 (§ "scope boundary" below).

- **What is pooled:** the GL SSBO storage for the `[N,H,W,C]` input tensor. Because `InferenceMetadata` is static (M4), the shape/dtype/layout key is effectively constant per graph — so the pool is a fixed-size ring of identically-shaped buffers, not a general keyed map. (Key still includes shape/dtype/layout/memory-type so a metadata change forces a rebuild; metadata does not change post-init.)
- **Single-write correctness:** `Tensor` is single-write, so the pool does **not** rewrite a live `Tensor`. It recycles the underlying GL buffer object behind a **fresh logical `Tensor`/packet** each frame (or an explicitly synchronized buffer wrapper). A buffer returns to the pool only after (a) its `OpenGlBufferView` `GlSync` fence has signaled and (b) the downstream inference packet that consumed it has been released (the release-callback discipline already required in §5). This couples the pool to the fence/barrier lifetime — it is not a plain map.
- **Bounds:** explicit options `max_in_flight_frames` (default 1) and `max_input_tensors` (the ring depth, default 2–3). With `max_in_flight_frames=1` the ring is just the double/triple buffer; allowing more frames in flight grows the ring proportionally. The pool blocks/allocates-up-to-bound rather than growing unbounded.
- **Scope boundary (what is NOT here):** the tile-geometry/matrix cache, tile-surface cache, output/decoder-buffer cache, and model/delegate cache remain **M8** (detection-core spec §14.1), cross-referenced — they help CPU and GPU alike and are not specific to zero-copy. The CPU path needs no pool change for M6a (Group-1 per-frame CPU allocation is acceptable; CPU buffer reuse, if wanted, is M8).
- **Cache-correctness rules inherited from M8 §14.1:** a pooled buffer must never be reused before release; pooling must not mix tiles/detections across source frames; the pool is GL-context-scoped. Default the pool to a safe minimal depth and make depth an explicit option.

## 6. Inference consumption + the hard invariant

- The GL-backed input `Tensor` must flow into `inference_calculator_gl_advanced` or an equivalent direct-SSBO binding runner. The standard `inference_calculator_gl.cc` path copies input with `glCopyBufferSubData`, so it is allowed only as an explicit "GPU-with-copy" fallback and must not be labeled zero-copy.
- Output tensors are comparatively small; YOLO/OBB **decode stays on CPU** (out of M6a scope — M6a is the input path). Decoders read the output tensors' CPU view as today.
- **Invariant:** GPU-in → GPU tile packing → direct SSBO-bound GPU inference with no implicit CPU readback and no extra GL input-buffer copy. Explicit fallbacks are allowed only when named in graph/options: CPU fallback (download once, OpenCV converter + CPU inference) or GPU-with-copy fallback (standard GL runner). Silent fallback is not allowed.

## 7. M4 metadata reuse + M6b seam

- The GL batcher consumes the same `InferenceMetadata` side packet (from M4) as the CPU path — `H/W/C`, dtype, layout, `batch_capacity`, and `is_dynamic_batch` drive GPU tensor sizing identically. No new metadata.
- **M6b (sequenced follow-on, not implemented now):** add a LiteRT-Next/platform buffer writer that writes tiles into a `litert::TensorBuffer` (AHWB on Android, OpenCL, or Metal) per the LiteRT-Next GPU buffer-interop API (https://ai.google.dev/edge/litert/next/gpu), so zero-copy generalizes across the M5 pluggable backends (TensorRT/CoreML/ONNX) rather than only the TFLite GL delegate. The reusable seam is the Group-1 tensor batch contract plus the row-offset pixel-write abstraction. Sequencing: **M5 → M6b**; M6a has no M5 dependency.

## 8. Error handling / fallback

- `GpuBuffer` input but GPU disabled / no GL context available → clear `RET_CHECK` with an actionable message, or a configured explicit CPU fallback (upload/download made explicit, never implicit).
- Channel/layout/dtype mismatch between `InferenceMetadata`, the source frame, and the GL tensor → fail clearly. M6a accepts float32 `[N,H,W,C]` output from the tile packer initially; non-float input metadata requires an explicit later design.
- Multi-batch (`T > batch_capacity`) inherits the shipped Group-1 guard: `RET_CHECK_LE(T, batch_capacity)` fails cleanly; multi-batch streaming is deferred (not a GPU feature to add here). Never combine tiles from multiple frames into one input tensor.
- `OpenGlTexture2dView` is BHWC-only (header constraint) — if a texture-view destination is chosen, assert BHWC and prove delegate compatibility; otherwise use the buffer (SSBO) view.

## 9. Testing strategy (split for the deferred environment)

**Locally runnable now (CPU, this machine):**
- Behavior-preserving refactor: after routing CPU tile writes through the existing `ImageToTensorConverter` row-offset seam or a thin wrapper over it, the existing Group-1 batcher tests (`streaming_tiles_to_tensor_batch_calculator_test`) must pass **unchanged** — proves the CPU path is untouched in behavior.
- Pure-logic unit tests (no GL context): tile-ROI to `RotatedRect` conversion, destination row-offset computation, fixed/dynamic `N`, the `T <= batch_capacity` guard (and that `T > batch_capacity` fails cleanly), padding bookkeeping, and dtype/layout validation. (Some already exist on the batcher; extend as the converter seam is introduced.)

**Build-gated / CI / on-device (written, NOT run in the local loop):**
- GL image-to-tensor row-offset golden tests: a GPU-cropped/resized tile row matches the OpenCV converter's output within a tolerance (same source + tile + metadata → equivalent tensor row).
- GPU batch tests: `T < batch_capacity` and `T == batch_capacity` emit the same `TensorBatchInfo`/`tile_indices` semantics as CPU (no cross-frame batching); `T > batch_capacity` fails with the same `RET_CHECK_LE` guard as CPU.
- No-readback assertions: GpuBuffer input takes the GL path and never calls `GetCpuReadView`/materializes `ImageFrame`.
- Direct-bind assertion: the zero-copy graph uses `inference_calculator_gl_advanced` or an equivalent direct-SSBO runner and does not execute the standard `inference_calculator_gl.cc` input `glCopyBufferSubData` path.
- GL memory visibility / lifetime: required memory barriers are issued after SSBO writes and before inference reads; an input tensor is not overwritten until downstream release; double/triple-buffer correctness.
- Input-tensor pool (§5.1): *CPU-testable now* — ring depth/bound enforcement, key match against `InferenceMetadata`, no buffer handed out twice while in flight. *GL/on-device* — a pooled buffer is recycled only after its fence signals and the downstream packet releases; never reused before release; pool stays bounded by `max_input_tensors`/`max_in_flight_frames`.
- End-to-end GPU graph: `GpuBuffer` frame → GL batcher → direct-SSBO GL inference → decode → merge → global NMS, asserting parity with the CPU pipeline on the same input.

These GPU tests are authored alongside the GL converter but executed only where an OpenGL build+test environment exists (Linux/Android CI or on-device), per the verification decision.

## 10. File structure (for the eventual implementation plan)

- Prefer reusing: `mediapipe/calculators/tensor/image_to_tensor_converter.h` and existing `image_to_tensor_converter_{opencv,gl_buffer,gl_texture}.{cc,h}`.
- Modify: `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc` — select CPU/GL converter by input type, convert `TileGeometry` to `RotatedRect`, compute row offsets, preserve the existing batch bookkeeping + `T <= batch_capacity` guard.
- Create only if needed: `mediapipe/calculators/tensor/tile_batch_to_tensor_helper.{cc,h}` — thin helper for tile-to-ROI conversion, row-offset math, dtype/layout validation, and repeated calls into `ImageToTensorConverter`. This helper must not introduce a second public converter abstraction.
- Modify or configure: GL inference graph/calculator wiring to use `inference_calculator_gl_advanced` or an equivalent direct-SSBO binding runner for the M6a zero-copy graph.
- Create (GL env): the bounded GL input-tensor pool (§5.1) — likely a small `GlInputTensorPool` owned by the batcher/GL converter (ring of GL buffers + fence-gated recycling). Its bookkeeping (ring index, bound, key match) is CPU-unit-testable; the GL recycling is build-gated. Reuse MediaPipe's existing `GpuBuffer`/GL buffer pools if they fit rather than inventing storage.
- Modify: `mediapipe/calculators/tensor/BUILD` — build-gated GL deps/tests.
- Tests: local CPU seam/helper tests, build-gated GL row-offset golden tests, GPU multi-batch tests, direct-bind/no-readback tests, and `tiled_gpu_pipeline_test.cc` (build-gated/on-device).

## 11. Risks & open items

1. **Local un-testability of the GL path** — the dominant constraint. Mitigation: reuse the existing converter seam; keep tile/row-offset logic CPU-testable; GL parts wait for a Linux/Android env. Implementation is explicitly deferred.
2. **macOS is Metal, not desktop GL** — this machine can't even compile-verify the GL path. A Metal converter is possible but is M6b/parallel scope, not M6a.
3. **Wrong GL inference runner silently breaking zero-copy** — standard `inference_calculator_gl.cc` performs an input copy. M6a graphs must use `inference_calculator_gl_advanced`/direct SSBO binding, with tests or instrumentation proving the copy path is not used.
4. **GL tensor layout vs delegate expectation** — the TFLite GL delegate's expected input buffer layout (BHWC, alignment) must match what the converter writes; verify on-device. Texture-view is BHWC-only.
5. **GL memory barrier / fence / lifetime correctness, incl. the input-tensor pool (§5.1)** — incorrect visibility or synchronization causes stale reads/races; the pool must recycle a GL buffer only after its fence signals and the downstream packet releases (never reuse-before-release). Needs explicit SSBO memory barriers, fence-gated recycling, bounded depth, and on-device tests. Without the pool, zero-copy still works but re-allocates GPU storage per frame (perf, not correctness).
6. **dtype support is narrower than metadata** — M6a initially supports float32 input packing. If metadata reports quantized/integer input, fail until that packing is designed.
7. **Refactor regression risk** — changing CPU pixel-write plumbing could change CPU behavior; mitigated by keeping the Group-1 batcher tests green unchanged (verifiable now).
8. **M6b depends on M5 + a new LiteRT-Next dependency** — large, separate effort; only the Group-1 batch contract and row-offset seam are committed to now.

## 12. Sequencing

1. **(Now, locally verifiable)** Route CPU tile writes through the existing `ImageToTensorConverter` row-offset seam or a thin helper; preserve the shipped `T <= batch_capacity` guard; Group-1 batcher tests stay green.
2. **(Deferred → GL env)** Enable GL batcher selection using the GL buffer converter, add the bounded fence-gated input-tensor pool (§5.1) + explicit memory barriers, and write build-gated GL row-offset + batch + pool tests.
3. **(Deferred → GL env)** Wire the zero-copy graph to `inference_calculator_gl_advanced`/direct SSBO binding and prove the standard input-copy path is not used.
4. **(Deferred → on-device/CI)** End-to-end GPU graph parity test.
5. **(M6b, after M5)** LiteRT-Next/platform buffer writer + dependency for backend-wide zero-copy.

## Appendix — relation to the roadmap

From the detection-core spec's roadmap: `M6 | Zero-copy GPU: OpenGL-on-TFLite input path + backend-wide LiteRT-Next/platform backends | depends on M3 (contract), M5`. This spec realizes the **M3-contract-dependent OpenGL-on-TFLite slice (M6a)** and designs the **M5-dependent LiteRT-Next slice (M6b)** as a follow-on. Group-1 contracts (`InferenceMetadata` side packet, `StreamingTilesToTensorBatchCalculator` bookkeeping + `T <= batch_capacity` guard, `TensorBatchInfo`) are reused unchanged.
