# Spec — Phase 5 (Metal): zero-copy tiled preprocessing on macOS

Date: 2026-06-03
Status: Design (pre-plan)
Branch: `dev` (long-lived integration branch; commit only when asked)
Roadmap: `docs/superpowers/specs/2026-06-01-roadmap.md` Phase 5 (OpenGL zero-copy).
Relation: the **Metal** realization of the GLES design in
`docs/superpowers/plans/2026-06-03-caching-plan4-opengl-ahwb-zerocopy.md` (Plan 4
targets GLES 3.1 SSBO for Android/Linux; macOS has no GLES/AHWB, so the macOS
zero-copy path is Metal + CVPixelBuffer).

## Goal

When the streaming tiled-detection path runs on **GPU-backed input on macOS**, do
crop/resize/normalize **in Metal** and write the result directly into the
**same `MTLBuffer` that the TFLite Metal delegate consumes as its input**. This
means no GPU→CPU→GPU round-trip and no intermediate BHWC→BPHWC4 input conversion
inside `InferenceCalculatorMetal`. Default-off; the shipped CPU path is
byte-identical when disabled. This is the MediaPipe TFLite tiled-inference
pipeline's GPU path (`StreamingTilesToTensorBatchCalculator` +
`InferenceCalculatorMetal`); it is unrelated to the torch `.pt` demo (which
preprocesses in Python, not MediaPipe).

## Prerequisite (done)

macOS GPU build config: `mediapipe/gpu/gpu_buffer_format.h` was changed to define
`MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER` on macOS (upstream gates it to iOS via
`!TARGET_OS_OSX`). This enables the CVPixelBuffer-backed `GpuBuffer` and
`MPPMetalHelper`'s `metalTextureWithGpuBuffer:` bridge on macOS. Compile-verified
(no regression) in commit `9c13613`. **Runtime is NOT yet proven** — the first
task of this work is a Metal runtime smoke (below).

## Why Metal (not GLES/AHWB/Vulkan) on macOS

- **GLES 3.1 SSBO** (Plan 4 core): macOS has no OpenGL ES and desktop GL is frozen
  at 4.1 (no compute shaders) → unavailable.
- **AHardwareBuffer**: Android-only NDK API → unavailable.
- **Vulkan**: not native (only via MoltenVK→Metal); and MediaPipe `Tensor` has **no
  Vulkan view** → unusable.
- **Metal**: native; `Tensor` has `MtlBufferView`; `image_to_tensor_converter_metal.cc`
  is a working reference. This is the one viable native zero-copy backend here.

## Verified codebase facts

- `MtlBufferView` (`mediapipe/framework/formats/tensor_mtl_buffer_view.h`):
  `static MtlBufferView GetWriteView(const Tensor&, id<MTLCommandBuffer>)`,
  `GetWriteView(const Tensor&, id<MTLDevice>)`, `GetReadView(const Tensor&,
  id<MTLCommandBuffer>)`; `id<MTLBuffer> buffer()`. So a CPU-constructed `Tensor`
  becomes Metal-backed on first Metal view; the underlying `MTLBuffer` is read by
  inference with no CPU copy.
- `MPPMetalHelper` (`mediapipe/gpu/MPPMetalHelper.h`): `initWithCalculatorContext:`,
  `+ updateContract:`, `- commandBuffer`, `@property mtlDevice`,
  `- metalTextureWithGpuBuffer:(const mediapipe::GpuBuffer&)` (now compiled on
  macOS), `- newLibraryWithResourceName:error:`.
- `image_to_tensor_converter_metal.cc` — reference Metal preprocessing: builds the
  sub-rect transform via `GetRotatedSubRectToRectTransformMatrix`, gets the input
  `MTLTexture` from the `GpuBuffer`, encodes a Metal command buffer that samples
  the sub-rect into the output `Tensor` Metal buffer, normalizes. The reference
  supports only `dims[3]==4` and `tensor_buffer_offset==0`; adapt this to write
  one tile per batch row into delegate input storage.
- TFLite Metal delegate external input:
  `TFLGpuDelegateBindMetalBufferToTensor(delegate, tensor_index, buffer)` binds a
  user-prepared `MTLBuffer` after `Interpreter::ModifyGraphWithDelegate()`. In the
  delegate implementation, a bound input is marked external and the delegate skips
  its normal BHWC→BPHWC4 converter. The current `InferenceCalculatorMetal` binds
  an internal input buffer and copies/converts packet tensors into it, so it is not
  true zero-copy until direct packet-buffer binding is added.
- TFLite GPU PHWC4/BPHWC4 storage groups channels by 4. For logical model input
  `[N,H,W,C]`, the direct Metal input buffer size is
  `N*H*W*RoundUp(C,4)` elements. For common detector inputs (`C=3`), physical
  storage is `[N,H,W,4]` with the padded channel zeroed; for `C=4`, physical
  storage matches logical BHWC. `C>4` requires multi-plane PHWC4 writes and is
  deferred until a compute-shader path covers it.
- `StreamingTilesToTensorBatchCalculator` (Plans 2–3, CPU): inputs `IMAGE`
  (`ImageFrame`) + `TILE_PLAN` + side `METADATA` (`InferenceMetadata`,
  `batch_capacity`/H/W/C); outputs `TENSORS` (`std::vector<Tensor>`) + `BATCH_INFO`
  (`TensorBatchInfo` carrying `TileBatchGeometry`: per-row `effective_pixel_rois`,
  `tile_to_image_matrices`); supports multi-batch at synthetic timestamps; has the
  default-off matrix/geometry cache + CPU tensor pool.
- `Tensor` shapes are BHWC `[N,H,W,C]`; `tile_to_image_matrices` use the same
  row-major `GetRotatedSubRectToRectTransformMatrix` convention the Metal shader's
  transform uniform expects.

## Scope

In scope:
1. A Metal runtime smoke (feasibility gate): allocate a Metal-backed `Tensor`, run
   a trivial Metal kernel, read back, verify — proving the macOS Metal runtime +
   `MtlBufferView` execute on this machine.
2. `StreamingTilesToTensorBatchMetalConverter` (ObjC++ `.mm`): Metal shader + per
   tile dispatch writing crop/resize/normalize into the Metal delegate input
   buffer layout at each tile's row offset, from a GPU input texture.
3. `StreamingTilesToTensorBatchCalculator` GPU path: add `IMAGE_GPU` (`GpuBuffer`)
   input + `enable_gpu_zero_copy` option; when enabled and GPU input present, run
   the Metal converter and emit a physical `[N,H,W,RoundUp(C,4)]` Metal tensor for
   logical model input `[N,H,W,C]`.
4. `InferenceCalculatorMetal` direct external-input mode: bind each packet's
   `MTLBuffer` to the Metal delegate input and skip the input `TFLBufferConvert`.
5. Tests: runtime smoke; delegate external-buffer smoke; GPU-vs-CPU parity after
   CPU output is converted to PHWC4; no-readback/no-input-converter assertion;
   single-batch and `T > batch_capacity` multi-batch.

Out of scope (deferred / YAGNI):
- Metal-backed-`Tensor` pooling + in-flight ownership (Plan 4 "Cache 5"): allocate
  per batch first; pooling is a follow-up.
- Rotated tiles (axis-aligned only, matching the shipped geometry).
- Model inputs with `C>4` (requires true multi-plane PHWC4 writes).
- Direct Metal output zero-copy; this milestone only requires delegate direct
  consumption of the tiled input buffer.
- GLES/AHWB/Vulkan backends (not viable on macOS — see above).
- The torch `.pt` pipeline (separate; preprocesses in Python).

## Non-negotiable rules

- **Default-off:** `enable_gpu_zero_copy=false` (default) ⇒ the shipped CPU path,
  byte-identical. GPU path only when the option is set AND a GPU input is wired.
- **True zero-copy:** the `MTLBuffer` emitted by tiled preprocessing is the same
  `MTLBuffer` bound to the TFLite Metal delegate input. A Metal-ready logical
  `[N,H,W,C]` tensor that `InferenceCalculatorMetal` later copies/converts into
  its internal BPHWC4 buffer is not zero-copy.
- **No CPU readback in the GPU path:** the input `GpuBuffer`/texture and the
  delegate-layout Metal-backed input tensor must not be CPU-materialized
  (`GetCpuReadView`/`GetCpuWriteView`/`MatView`) before explicit test readback.
- **Correctness parity:** GPU output equals the CPU path within tolerance for the
  same input + tiles after the CPU result is converted to PHWC4/BPHWC4.
- **Padding/valid-count + multi-batch semantics** match the CPU path
  (`valid_count`, `batch_size`, padded rows, synthetic batch timestamps).
- macOS-gated: the Metal `.mm` + GPU path compile under `MEDIAPIPE_METAL_ENABLED`
  / Apple; the calculator's CPU build (`--define MEDIAPIPE_DISABLE_GPU=1`) is
  unchanged and the GPU code is excluded there.

## Architecture

```
IMAGE_GPU: GpuBuffer (CVPixelBuffer-backed on macOS)
   │  MPPMetalHelper metalTextureWithGpuBuffer:  → id<MTLTexture> src (GPU)
   ▼
StreamingTilesToTensorBatchMetalConverter  [NEW .mm]
   │  for each valid tile row r in the batch:
   │    transform = GetRotatedSubRectToRectTransformMatrix(effective_pixel_roi[r], fw, fh)
   │    encode Metal shader: sample src sub-rect (linear filter → crop+resize),
   │      normalize (×1/255), write H*W*C4 floats at row offset r in the delegate buffer
   │  output physical Tensor batch [N,H,W,C4], Metal-backed via MtlBufferView::GetWriteView
   ▼
StreamingTilesToTensorBatchCalculator (GPU branch)  [MODIFIED]
   │  emits TENSORS (delegate-layout Metal-backed) + BATCH_INFO (same as CPU path)
   ▼
InferenceCalculatorMetal direct-input mode  [MODIFIED]
   │  TFLGpuDelegateBindMetalBufferToTensor(packet.MTLBuffer)
   │  skips BHWC→BPHWC4 input TFLBufferConvert
   ▼
TFLite Metal delegate consumes the same MTLBuffer
```

Decision logic: if `enable_gpu_zero_copy` AND `IMAGE_GPU` is connected and present
→ Metal path; else the existing CPU path (`IMAGE` ImageFrame). The geometry
(`TileBatchGeometry`, ROIs, matrices) is computed/cached exactly as today and
reused to drive the per-tile Metal transforms. Metal inference must also opt into
direct external input; otherwise the graph is a GPU-preprocessing path with an
input conversion copy, not zero-copy.

## Testing (macOS, Metal GpuResources)

- **Runtime smoke** (gate, runs first): Metal-backed `Tensor` write+read via
  `MtlBufferView` + a trivial kernel → proves the runtime.
- **Delegate binding smoke**: bind a user `MTLBuffer` to a small TFLite Metal
  model input with `TFLGpuDelegateBindMetalBufferToTensor()` and invoke.
- **Converter parity**: feed a known input as both `ImageFrame` (CPU path) and
  `GpuBuffer` (Metal path) with the same `TilePlan`; convert CPU output to PHWC4
  and assert the emitted Metal physical tensor is equal within tolerance.
- **No-readback/no-input-copy**: assert the GPU path produced a Metal-ready (not
  CPU) tensor, never CPU-materialized the input, and `InferenceCalculatorMetal`
  direct mode skipped input `TFLBufferConvert`.
- **Buffer identity**: assert the exact `id<MTLBuffer>` produced by the
  preprocessor is the one bound to the delegate input for every batch.
- **Tile-count cases**: single tile, `T<cap`, `T==cap`, `T>cap` (multi-batch);
  fixed-batch padding rows zeroed on GPU; dynamic batch `N==valid_count`.
- All under a Metal `GpuResources` context (the runtime smoke confirms it works
  headless on this Mac).

## Done criteria

- Metal runtime smoke passes on this Mac (runtime proven, not just compile).
- GPU-backed input → Metal tiled crop/resize/normalize → delegate-layout
  Metal-backed input tensor → TFLite Metal delegate consumes the same `MTLBuffer`,
  no CPU readback and no input conversion copy, parity with the CPU path within
  tolerance after PHWC4 conversion.
- `enable_gpu_zero_copy` default-off; CPU path unchanged; CPU build excludes the
  Metal code.
- Single-batch and `T>cap` multi-batch covered; Metal-Tensor pooling explicitly
  deferred to a follow-up.
