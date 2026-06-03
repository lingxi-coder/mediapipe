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
crop/resize/normalize **in Metal** and write the result directly into a
**Metal-backed `Tensor`** batch, so downstream inference reads the same `MTLBuffer`
— no GPU→CPU→GPU round-trip. Default-off; the shipped CPU path is byte-identical
when disabled. This is the MediaPipe TFLite tiled-inference pipeline's GPU path
(`StreamingTilesToTensorBatchCalculator`); it is unrelated to the torch `.pt`
demo (which preprocesses in Python, not MediaPipe).

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
  the sub-rect into the output `Tensor` Metal buffer, normalizes. Adapt this to
  write **one tile per batch row**.
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
   tile dispatch writing crop/resize/normalize into a Metal-backed `Tensor` batch
   at each tile's row offset, from a GPU input texture.
3. `StreamingTilesToTensorBatchCalculator` GPU path: add `IMAGE_GPU` (`GpuBuffer`)
   input + `enable_gpu_zero_copy` option; when enabled and GPU input present, run
   the Metal converter and emit a Metal-backed `Tensor` with no CPU readback.
4. Tests: runtime smoke; GPU-vs-CPU parity (Metal output read back equals the CPU
   pixel-loop output within tolerance); no-CPU-readback assertion; single-batch
   then `T > batch_capacity` multi-batch.

Out of scope (deferred / YAGNI):
- Metal-backed-`Tensor` pooling + in-flight ownership (Plan 4 "Cache 5"): allocate
  per batch first; pooling is a follow-up.
- Rotated tiles (axis-aligned only, matching the shipped geometry).
- GLES/AHWB/Vulkan backends (not viable on macOS — see above).
- The torch `.pt` pipeline (separate; preprocesses in Python).

## Non-negotiable rules

- **Default-off:** `enable_gpu_zero_copy=false` (default) ⇒ the shipped CPU path,
  byte-identical. GPU path only when the option is set AND a GPU input is wired.
- **No CPU readback in the GPU path:** the input `GpuBuffer`/texture and the output
  Metal-backed `Tensor` must not be CPU-materialized (`GetCpuReadView`/
  `GetCpuWriteView`/`MatView`) on the zero-copy path. A test asserts this.
- **Correctness parity:** GPU output equals the CPU path within tolerance for the
  same input + tiles (same `GetRotatedSubRectToRectTransformMatrix` geometry).
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
   │      normalize (×1/255), write H*W*C floats at row offset r in the batch buffer
   │  output Tensor batch [N,H,W,C], Metal-backed via MtlBufferView::GetWriteView
   ▼
StreamingTilesToTensorBatchCalculator (GPU branch)  [MODIFIED]
   │  emits TENSORS (Metal-backed) + BATCH_INFO (same as CPU path)
   ▼
downstream inference reads the MTLBuffer directly (no CPU readback)
```

Decision logic: if `enable_gpu_zero_copy` AND `IMAGE_GPU` is connected and present
→ Metal path; else the existing CPU path (`IMAGE` ImageFrame). The geometry
(`TileBatchGeometry`, ROIs, matrices) is computed/cached exactly as today and
reused to drive the per-tile Metal transforms.

## Testing (macOS, Metal GpuResources)

- **Runtime smoke** (gate, runs first): Metal-backed `Tensor` write+read via
  `MtlBufferView` + a trivial kernel → proves the runtime.
- **Converter parity**: feed a known input as both `ImageFrame` (CPU path) and
  `GpuBuffer` (Metal path) with the same `TilePlan`; assert the emitted tensors are
  equal within tolerance (small, due to GPU sampler interpolation rounding).
- **No-readback**: assert the GPU path produced a Metal-ready (not CPU) tensor and
  never CPU-materialized the input.
- **Tile-count cases**: single tile, `T<cap`, `T==cap`, `T>cap` (multi-batch);
  fixed-batch padding rows zeroed on GPU; dynamic batch `N==valid_count`.
- All under a Metal `GpuResources` context (the runtime smoke confirms it works
  headless on this Mac).

## Done criteria

- Metal runtime smoke passes on this Mac (runtime proven, not just compile).
- GPU-backed input → Metal tiled crop/resize/normalize → Metal-backed `Tensor`,
  no CPU readback, parity with the CPU path within tolerance.
- `enable_gpu_zero_copy` default-off; CPU path unchanged; CPU build excludes the
  Metal code.
- Single-batch and `T>cap` multi-batch covered; Metal-Tensor pooling explicitly
  deferred to a follow-up.
