# M6 Zero-Copy GPU Design (M6a OpenGL-on-TFLite + M6b LiteRT-Next follow-on)

- **Date:** 2026-06-01
- **Status:** Approved (design). **Design-only — implementation deferred** until a Linux/Android OpenGL build+test environment is available (this macOS dev machine cannot build/run the desktop-GL GPU path; all of Group 1 was built with `--define MEDIAPIPE_DISABLE_GPU=1`).
- **Scope:** **M6a** (OpenGL-on-TFLite zero-copy input path) is the implementable core. **M6b** (LiteRT-Next backend-wide buffer interop) is designed here as a sequenced follow-on phase but not implemented.
- **Build model:** Fork / custom build of MediaPipe.

> This is feature 6 ("Zero-copy performance optimization — LiteRT-Next GPU") from the parent request. It was deferred out of Group 1 (see `2026-06-01-detection-core-yolo-obb-tiling-design.md` §14.2) with the Group-1 calculator contracts kept stable so this attaches without breaking them.

---

## 1. Background & goal

Group 1 built a CPU tiling→inference pipeline: `StreamingTilesToTensorBatchCalculator` crops/resizes/normalizes externally-supplied tiles into a CPU `[N,H,W,C]` float32 tensor, which `InferenceCalculator` runs. When frames are already GPU-resident (camera/GL pipelines deliver `GpuBuffer`/`IMAGE_GPU`), the CPU path forces a GPU→CPU download for tiling and a CPU→GPU upload for GPU inference — two full-frame copies per frame on the hot path.

**M6a goal:** when the input frame is GPU-resident, crop/resize/normalize tiles **directly on the GPU** into the inference input tensor's OpenGL-backed storage, so the TFLite GL delegate (`inference_calculator_gl`) runs inference with **no CPU↔GPU copy on the input path**.

**Non-goal (this milestone):** output-side zero-copy, GPU-side NMS, and non-OpenGL GPU backends (Metal/Vulkan/CUDA) — those are M6b / future.

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope of this spec | Design both; M6a implementable core + M6b sequenced follow-on | M6b (LiteRT-Next) has no foundation in the repo and depends on M5; M6a builds on existing primitives. |
| Primary GPU API | OpenGL (GLES + desktop GL) | Matches the existing `Tensor::OpenGlTexture2dView`/`OpenGlBufferView`, the GL `image_to_tensor` converters, and the TFLite GL delegate. |
| Verification | **Design-only now**; implementation deferred to a Linux/Android GL env; code structured to split CPU-testable logic from GL-context code | This macOS machine can't build/run desktop-GL GPU calculators; Group 1's local TDD loop doesn't extend to the GL runtime. |
| Structure | **Converter abstraction**: `TileToTensorConverter` interface with OpenCV (CPU) + GL impls | Mirrors the repo's existing `ImageToTensorConverter` (gl_buffer/gl_texture/opencv); isolates GL code into one unit; keeps batcher bookkeeping CPU-testable. |

## 3. Current-state findings (codebase, verified)

- `Tensor` exposes GPU-resident storage views: `OpenGlTexture2dView` (read/write; **BHWC-only**, per the header note) and `OpenGlBufferView` (read/write; the write view creates a `GlSync` fence). These are the zero-copy destinations. Do **not** introduce a parallel GPU memory abstraction.
- The image→tensor pattern to mirror already exists: `image_to_tensor_converter.h` (interface) with `image_to_tensor_converter_gl_buffer.{cc,h}`, `…_gl_texture.{cc,h}`, `…_metal.{cc,h}`, `…_opencv.{cc,h}` impls, selected at runtime by `image_to_tensor_calculator.cc` under `#if !MEDIAPIPE_DISABLE_GPU`. The GL converters already do single-image crop+resize+normalize via GL shaders.
- `inference_calculator_gl.cc` / `inference_calculator_gl_advanced.cc` are the TFLite GL-delegate inference paths that already consume GL-backed input tensors.
- GPU code throughout is gated by `#if !MEDIAPIPE_DISABLE_GPU` and uses `GlCalculatorHelper` / `mediapipe/gpu/gpu_buffer.h`.
- **No LiteRT-Next in the repo:** no `litert::TensorBuffer` / buffer-interop references and no LiteRT module dependency. M6b requires adding that dependency and depends on M5 (pluggable backends).
- Group 1's `StreamingTilesToTensorBatchCalculator` is CPU-only (OpenCV crop/resize → CPU `Tensor`), consumes `InferenceMetadata` (from M4) for `H/W/C`/`batch_capacity`/`is_dynamic_batch`, enforces a single-batch-per-frame guard (`RET_CHECK_LE(T, cap)`) and a channel-count guard.

## 4. Architecture — converter abstraction

Introduce a converter interface that owns *only* the per-tile pixel write; the batcher owns everything else.

```
                 InferenceMetadata (side packet, from M4: H,W,C,batch_capacity,dynamic)
                                   │
Frame (ImageFrame | GpuBuffer) ───┤
TilePlan (from TileSpecToTilePlan) ┤
                                   ▼
   StreamingTilesToTensorBatchCalculator  (bookkeeping only — unchanged logic)
     • batch sizing, padding, TensorBatchInfo, tile_indices, guards
     • selects converter by input type:
         ImageFrame → TileToTensorConverterOpenCv   (CPU; refactor of Group-1 logic)
         GpuBuffer  → TileToTensorConverterGl        (NEW; zero-copy)
                                   │
                                   ▼
            Tensor [N,H,W,C]  (CPU-backed  OR  OpenGL-backed)
                                   ▼
   InferenceCalculator(Cpu)  OR  InferenceCalculatorGl  (TFLite GL delegate)
                                   ▼
            raw outputs ──► YOLO / OBB decode (CPU) ──► merge ──► global NMS
```

### 4.1 `TileToTensorConverter` interface
```
class TileToTensorConverter {
 public:
  virtual ~TileToTensorConverter() = default;
  // Writes the cropped/resized/normalized pixels of each valid tile into the
  // corresponding row of `batch` ([N,H,W,C]). `tiles` are the valid TileGeometry
  // for this batch (size == valid_count). Implementations must not touch padded
  // rows. `params` carries normalization range + layout from InferenceMetadata.
  virtual absl::Status Convert(const <SourceImage>& image,
                               const std::vector<TileGeometry>& tiles,
                               const TileToTensorParams& params,
                               Tensor* batch) = 0;
};
```
- `<SourceImage>` is the source frame; the OpenCV impl takes `ImageFrame`, the GL impl takes `GpuBuffer` (or `mediapipe::Image`, which wraps both). The batcher passes whichever it received.
- `TileToTensorParams`: target `H/W/C`, normalization range, keep-aspect-ratio flag (future), tensor layout (`BHWC`).

### 4.2 Batcher changes (behavior-preserving for CPU)
`StreamingTilesToTensorBatchCalculator` keeps its Group-1 bookkeeping verbatim (batch sizing, fixed/dynamic `N`, padding, `TensorBatchInfo`, `tile_indices`, the `T≤cap` and channel-count guards). The only change: extract the inline OpenCV crop/resize/normalize loop into `TileToTensorConverterOpenCv` behind the interface, and add converter selection by input type. **The existing Group-1 batcher unit tests must stay green unchanged** — this is the one piece verifiable on the current machine.

## 5. The GL converter (zero-copy core) — `TileToTensorConverterGl`

- Runs in the calculator's `GlCalculatorHelper` GL context.
- Allocates / receives the `[N,H,W,C]` float32 `Tensor` and takes `Tensor::GetOpenGlBufferWriteView()` (SSBO) as the GL-resident destination (texture-2d view is an alternative where BHWC layout suits the delegate; SSBO chosen because the batch dim makes a flat buffer natural).
- For each valid tile row `r`: a GL program samples the source `GpuBuffer` texture over the tile's normalized ROI, bilinear-resizes to `H×W`, applies the normalization range, and writes into the row-`r` offset of the SSBO. This is the existing `image_to_tensor_converter_gl_buffer` shader generalized with (a) a per-tile ROI and (b) a destination row offset.
- Fixed-batch padding rows are cleared to zero once (e.g., `glClear`/zero-fill before the tile loop).
- **No CPU readback:** when input is `GpuBuffer`, the converter never calls `GetCpuReadView`/`MatView`/materializes `ImageFrame`.
- **Sync & lifetime:** rely on the `OpenGlBufferView` `GlSync` fence; use double/triple-buffered input tensors plus release callbacks so an input tensor is not reused for the next frame until the GL delegate has finished reading it. (Mirrors the existing GL converter lifetime handling.)

## 6. Inference consumption + the hard invariant

- The GL-backed input `Tensor` flows into `InferenceCalculatorGl` (TFLite GL delegate), which already consumes GL tensors — so inference runs on GPU with no input copy.
- Output tensors are comparatively small; YOLO/OBB **decode stays on CPU** (out of M6a scope — M6a is the input path). Decoders read the output tensors' CPU view as today.
- **Invariant:** GPU-in → GPU-through-inference with no implicit readback. An explicit CPU fallback (download once, run the OpenCV converter + CPU inference) is taken only when the input is CPU-only or GPU mode is unavailable — never silently.

## 7. M4 metadata reuse + M6b seam

- The GL batcher consumes the same `InferenceMetadata` side packet (from M4) as the CPU path — `H/W/C`, `batch_capacity`, `is_dynamic_batch` drive GPU tensor sizing identically. No new metadata.
- **M6b (sequenced follow-on, not implemented now):** add `TileToTensorConverterLiteRt`, writing tiles into a `litert::TensorBuffer` (AHWB on Android, OpenCL, or Metal) per the LiteRT-Next GPU buffer-interop API (https://ai.google.dev/edge/litert/next/gpu), so zero-copy generalizes across the M5 pluggable backends (TensorRT/CoreML/ONNX) rather than only the TFLite GL delegate. The `TileToTensorConverter` interface is the seam; M6b = "a new converter impl + the LiteRT-Next dependency + M5." Sequencing: **M5 → M6b**; M6a has no M5 dependency.

## 8. Error handling / fallback

- `GpuBuffer` input but GPU disabled / no GL context available → clear `RET_CHECK` with an actionable message, or a configured explicit CPU fallback (upload/download made explicit, never implicit).
- Channel/layout mismatch between `InferenceMetadata` and the GL tensor → fail clearly (mirror the Group-1 CPU channel-count guard `RET_CHECK_EQ(frame.NumberOfChannels(), C)`).
- Multi-batch (`T > batch_capacity`) GPU emission inherits the Group-1 single-batch-per-frame guard (`RET_CHECK_LE(T, cap)`); multi-batch emission remains deferred.
- `OpenGlTexture2dView` is BHWC-only (header constraint) — if a texture-view destination is chosen, assert BHWC; otherwise use the buffer (SSBO) view.

## 9. Testing strategy (split for the deferred environment)

**Locally runnable now (CPU, this machine):**
- Behavior-preserving refactor: after extracting `TileToTensorConverterOpenCv`, the existing Group-1 batcher tests (`streaming_tiles_to_tensor_batch_calculator_test`) must pass **unchanged** — proves the CPU path is untouched in behavior.
- Pure-logic unit tests (no GL context): tile-ROI pixel math, destination row-offset computation, fixed/dynamic `N` + padding bookkeeping. (Some already exist on the batcher; extend as the converter seam is introduced.)

**Build-gated / CI / on-device (written, NOT run in the local loop):**
- `TileToTensorConverterGl` golden tests: a GPU-cropped/resized tile matches the OpenCV converter's output within a tolerance (same source + tile + metadata → equivalent tensor rows).
- No-readback assertions: GpuBuffer input takes the GL path and never calls `GetCpuReadView`/materializes `ImageFrame`.
- GL fence / lifetime: an input tensor is not overwritten until downstream release; double/triple-buffer correctness.
- End-to-end GPU graph: `GpuBuffer` frame → GL batcher → `InferenceCalculatorGl` → decode → merge → global NMS, asserting parity with the CPU pipeline on the same input.

These GPU tests are authored alongside the GL converter but executed only where an OpenGL build+test environment exists (Linux/Android CI or on-device), per the verification decision.

## 10. File structure (for the eventual implementation plan)

- Create: `mediapipe/calculators/tensor/tile_to_tensor_converter.h` — the interface + `TileToTensorParams`.
- Create: `mediapipe/calculators/tensor/tile_to_tensor_converter_opencv.{cc,h}` — CPU impl (refactored from the Group-1 batcher).
- Create: `mediapipe/calculators/tensor/tile_to_tensor_converter_gl.{cc,h}` — GL impl (build-gated).
- Modify: `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc` — delegate the pixel write to a converter selected by input type; bookkeeping unchanged.
- Modify: `mediapipe/calculators/tensor/BUILD` — converter libs (GL lib under the GPU config), gpu deps.
- Tests: `tile_to_tensor_converter_opencv_test.cc` (local), `tile_to_tensor_converter_gl_test.cc` (build-gated), `tiled_gpu_pipeline_test.cc` (build-gated/on-device).

## 11. Risks & open items

1. **Local un-testability of the GL path** — the dominant constraint. Mitigation: the converter seam isolates GL code; the CPU refactor is verified now; GL parts wait for a Linux/Android env. Implementation is explicitly deferred.
2. **macOS is Metal, not desktop GL** — this machine can't even compile-verify the GL path. A Metal converter is possible but is M6b/parallel scope, not M6a.
3. **GL tensor layout vs delegate expectation** — the TFLite GL delegate's expected input buffer layout (BHWC, alignment) must match what the converter writes; verify on-device. Texture-view is BHWC-only.
4. **GL fence / lifetime correctness** — incorrect synchronization causes tearing/races; needs the double/triple-buffer + release-callback discipline and on-device tests.
5. **Refactor regression risk** — extracting the OpenCV converter could change CPU behavior; mitigated by keeping the Group-1 batcher tests green unchanged (verifiable now).
6. **M6b depends on M5 + a new LiteRT-Next dependency** — large, separate effort; only the seam is committed to now.

## 12. Sequencing

1. **(Now, locally verifiable)** Refactor the CPU pixel-write into `TileToTensorConverterOpenCv` behind the new interface; Group-1 batcher tests stay green. *(This sub-step alone could be done before a GL env exists, de-risking the seam.)*
2. **(Deferred → GL env)** Implement `TileToTensorConverterGl` + GL batcher selection + build-gated GL tests.
3. **(Deferred → on-device/CI)** End-to-end GPU graph parity test.
4. **(M6b, after M5)** `TileToTensorConverterLiteRt` + LiteRT-Next dependency for backend-wide zero-copy.

## Appendix — relation to the roadmap

From the detection-core spec's roadmap: `M6 | Zero-copy GPU: OpenGL-on-TFLite input path + backend-wide LiteRT-Next/platform backends | depends on M3 (contract), M5`. This spec realizes the **M3-contract-dependent OpenGL-on-TFLite slice (M6a)** and designs the **M5-dependent LiteRT-Next slice (M6b)** as a follow-on. Group-1 contracts (`InferenceMetadata` side packet, `StreamingTilesToTensorBatchCalculator` bookkeeping, `TensorBatchInfo`) are reused unchanged.
