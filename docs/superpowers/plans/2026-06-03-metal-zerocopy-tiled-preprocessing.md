# Metal Zero-Copy Tiled Preprocessing (Phase 5, macOS) Implementation Plan

> **OUTCOME UPDATE (2026-06-03):** implemented, then revised. Tasks 1–3 (Metal smoke,
> tiled writer, calculator branch) shipped; Task 4 (delegate direct external-input)
> shipped then was **reverted**. Deep review showed the physical-PHWC4 + direct-bind
> approach is **batch-1 only** (delegate input is SHWBC4/batch-innermost; a contiguous
> packet is batch-outermost — match only at N=1). Final design (commit `2fa11df`)
> switched to the **GL-style logical-tensor** approach: the Metal writer is a **compute**
> shader emitting logical BHWC `[N,H,W,C]`; the **unmodified** `InferenceCalculatorMetal`
> converts it (correct for all N). The `metal_external_input_zero_copy` proto field and
> Task 4 were removed. Task 4 below + the "true zero-copy delegate input" goal are
> historical; "zero-copy" now means no-CPU-round-trip (one cheap on-GPU conversion stays).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a true Metal zero-copy tiled path: GPU-backed input is cropped/resized/normalized per tile in Metal, written directly into the exact `MTLBuffer` layout consumed by the TFLite Metal delegate, and `InferenceCalculatorMetal` invokes the delegate on that same buffer — no CPU readback and no intermediate BHWC→BPHWC4 input copy.

**Architecture:** A new ObjC++ Metal converter (`streaming_tiles_to_tensor_batch_metal`) adapts the shipped `image_to_tensor_converter_metal.cc` (render-pipeline quad + `metalTextureWithGpuBuffer:` input texture + `MtlBufferView::GetWriteView` output) to render each tile into its row offset of the **delegate input buffer layout**. For model input `[N,H,W,C]`, the Metal delegate expects PHWC4/BPHWC4 storage with `C4=RoundUp(C,4)` and `N*H*W*C4` elements (for common `C<=4`, this is contiguous `[N,H,W,C4]` with zero-filled padded channels). `StreamingTilesToTensorBatchCalculator` gains an `IMAGE_GPU` (`GpuBuffer`) input + `enable_gpu_zero_copy` option and emits a Metal-backed physical input tensor `[N,H,W,C4]`; `InferenceCalculatorMetal` gains a direct external-input mode that binds the packet's `MTLBuffer` with `TFLGpuDelegateBindMetalBufferToTensor()` and skips `TFLBufferConvert` input conversion. Default-off keeps the shipped CPU path byte-identical. A Metal runtime + delegate binding smoke gates the whole effort.

**Tech Stack:** Objective-C++ (Metal), `MPPMetalHelper`, `MtlBufferView`, `Tensor` (logical BHWC and delegate PHWC4/BPHWC4 physical layout), `GetRotatedSubRectToRectTransformMatrix`, TFLite Metal delegate external buffer binding, MediaPipe `GpuResources`/Metal, Bazel (Apple/Metal build, NOT `--define MEDIAPIPE_DISABLE_GPU=1`), GoogleTest.

**Spec:** `docs/superpowers/specs/2026-06-03-metal-zerocopy-tiled-preprocessing-design.md`.

**Prerequisite (done):** `mediapipe/gpu/gpu_buffer_format.h` enables `MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER` on macOS (commit `9c13613`), so the CVPixelBuffer `GpuBuffer` + `metalTextureWithGpuBuffer:` bridge compile here.

---

## Reference facts (verified)

- **Build flavor:** Metal code builds WITHOUT `--define MEDIAPIPE_DISABLE_GPU=1` (that flag forces `MEDIAPIPE_METAL_ENABLED 0`). On macOS, GPU build ⇒ Metal. Use `bazel {build,test} -c opt <target>` (no disable-gpu) for Metal targets. The CPU calculator tests still use `--define MEDIAPIPE_DISABLE_GPU=1` and must stay green.
- **`MtlBufferView`** (`mediapipe/framework/formats/tensor_mtl_buffer_view.h`): `static MtlBufferView GetWriteView(const Tensor&, id<MTLCommandBuffer>)`, `GetWriteView(const Tensor&, id<MTLDevice>)`, `GetReadView(const Tensor&, id<MTLCommandBuffer>)`; instance `id<MTLBuffer> buffer()`. Header guarded by `MEDIAPIPE_METAL_ENABLED`.
- **`MPPMetalHelper`** (`mediapipe/gpu/MPPMetalHelper.h`): `- (instancetype)initWithCalculatorContext:(CalculatorContext*)cc`, `+ (absl::Status)updateContract:(CalculatorContract*)cc`, `- (id<MTLCommandBuffer>)commandBuffer`, `@property(readonly) id<MTLDevice> mtlDevice`, `- (id<MTLTexture>)metalTextureWithGpuBuffer:(const mediapipe::GpuBuffer&)`.
- **Reference `image_to_tensor_converter_metal.cc`:** `Convert(const Image& input, const RotatedRect& roi, float range_min, float range_max, int tensor_buffer_offset, Tensor& output_tensor)`. It: gets `texture = [metal_helper_ metalTextureWithGpuBuffer:input.GetGpuBuffer()]`; `command_buffer = [metal_helper_ commandBuffer]`; `buffer_view = MtlBufferView::GetWriteView(output_tensor, command_buffer)`; `extractor_->Execute(texture, roi, flip, scale, offset, HW(out_h,out_w), command_buffer, buffer_view.buffer())`; `[command_buffer commit]`. **It hard-asserts `output_shape.dims[0]==1`, `output_shape.dims[3]==4`, and `tensor_buffer_offset==0`.** The inner `SubRectExtractorMetal` renders a quad (positions/tex-coords/transform vertex buffers, input texture as fragment sampler) into a render target aliased over the destination `MTLBuffer`. The tiling delta = render each tile into a per-row buffer offset, with the physical channel count padded to `C4=RoundUp(C,4)`.
- **TFLite Metal delegate external buffers:** `TFLGpuDelegateBindMetalBufferToTensor(delegate, tensor_index, id<MTLBuffer>)` must be called after `Interpreter::ModifyGraphWithDelegate()`. The delegate implementation marks the graph input/output `set_externally=true`; during invoke it skips the built-in BHWC→BPHWC4 converter for externally set inputs and consumes the bound `MetalSpatialTensor` buffer directly. Current `InferenceCalculatorMetal` does **not** satisfy true zero-copy because it binds an internal `gpu_buffers_in_` buffer in `Open()` and, in `Process()`, copies/converts packet input with `TFLBufferConvert` before invoking. This plan must add a direct external-input mode and tests proving the input converter is bypassed.
- **TFLite PHWC4/BPHWC4 layout:** TFLite GPU common `ConvertToPHWC4` defines PHWC4 as channels grouped by 4; `GetElementsSizeForPHWC4(BHWC)` returns `N*H*W*RoundUp(C,4)`. For `C=3`, the physical buffer is `[N,H,W,4]` with alpha/padded channel zeroed. For `C=4`, physical storage matches logical BHWC. For `C>4`, planes are channel groups of 4; support is out of scope until a compute shader path writes true PHWC4 planes.
- **`StreamingTilesToTensorBatchCalculator`** (`streaming_tiles_to_tensor_batch_calculator.cc`, Plans 2–3): api2 `Node`, `namespace mediapipe::api2`. Inputs `IMAGE`(`ImageFrame`), `TILE_PLAN`(`TilePlan`), side `METADATA`(`InferenceMetadata`: `input_height/width/channels`, `batch_capacity`, `is_dynamic_batch`). Outputs `TENSORS`(`std::vector<Tensor>`), `BATCH_INFO`(`TensorBatchInfo`), `CACHE_STATS`(optional). `Process()` builds/caches `TileBatchGeometry` (per valid row: `effective_pixel_rois` (`TilePixelRoi{x,y,width,height}`), `tile_geometries`, `tile_to_image_matrices`), then a CPU pixel loop crops/resizes/normalizes each row into a CPU `Tensor`, emits at synthetic `batch_ts_`. proto fields: `dynamic_batch=1`, `max_cached_tile_matrices=2`, `max_cpu_tensor_workspaces=3`, `emit_cache_stats=4`.
- **`GetRotatedSubRectToRectTransformMatrix(const RotatedRect& sub_rect, int rect_w, int rect_h, bool flip, std::array<float,16>*)`** (`image_to_tensor_utils.h`): the transform the Metal shader's `transform_matrix` vertex uniform expects; `RotatedRect{center_x,center_y,width,height,rotation}`. The CPU path's `TilePixelRoi{x,y,width,height}` maps to a `RotatedRect{center_x=x+width/2, center_y=y+height/2, width, height, rotation=0}` (same as `tiling_matrix_utils::TileToImageMatrix`).
- **Metal `.cc`/`.mm` BUILD pattern** (`image_to_tensor_converter_metal` target, `mediapipe/calculators/tensor/BUILD:1879`): `cc_library` with `copts = select({apple: ["-x", "objective-c++", "-fobjc-arc"], default: []})` and `deps = ["//mediapipe/framework:port"] + select({apple: [":image_to_tensor_converter", ":image_to_tensor_utils", "//mediapipe/gpu:MPPMetalHelper", "//mediapipe/gpu:gpu_buffer", "//mediapipe/framework/formats:tensor", "@google_toolbox_for_mac//:GTM_Defines"], default: []})`. READ the exact target before writing yours and copy the apple `select`/copts/deps shape.

## Cross-cutting conventions

- Metal build/test: `bazel test -c opt //mediapipe/calculators/tensor:<metal_target> --test_output=errors` (NO disable-gpu). CPU regression: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test`.
- In-editor clang errors are FALSE POSITIVES; only bazel is authoritative. **`exit 0` from a piped build is NOT success — always read the build/test output tail** (this fork's GPU stack has surfaced misleading pipe-exit-0s).
- Default-off: `enable_gpu_zero_copy=false` (default) ⇒ shipped CPU path, byte-identical. GPU path only when set AND `IMAGE_GPU` connected+present AND downstream Metal inference is configured for direct delegate input.
- All Metal code Apple-gated; the CPU build (`MEDIAPIPE_DISABLE_GPU=1`) must not compile the Metal `.mm` and must stay green.
- True zero-copy means the same `id<MTLBuffer>` produced by tiled preprocessing is passed to `TFLGpuDelegateBindMetalBufferToTensor()` and consumed by the delegate. Merely emitting a Metal-ready `[N,H,W,C]` tensor that `InferenceCalculatorMetal` later converts into its internal BPHWC4 buffer is not zero-copy.
- The logical model input remains `[N,H,W,C]`; the Metal zero-copy packet is physical `[N,H,W,C4]` for `C<=4`, where `C4=RoundUp(C,4)`. `TensorBatchInfo` remains logical batch/tile metadata; inference validation compares packet shape to model input shape with channel padding.
- Each task commits separately; co-author trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Branch `dev`.

## File Structure

- **Create** `mediapipe/calculators/tensor/metal_tensor_smoke_test.cc` — Metal runtime gate (ObjC++).
- **Create** `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal.{h,cc}` — the Metal tiled converter (`.cc` compiled ObjC++ via copts, matching the repo pattern). One responsibility: render tiles → delegate-layout Metal input rows.
- **Create** `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal_test.cc` — converter unit test (input texture + tiles → buffer, read back, compare to CPU math).
- **Modify** `streaming_tiles_to_tensor_batch_calculator.{proto,cc}` + test — `IMAGE_GPU` input, `enable_gpu_zero_copy`, GPU dispatch; parity + no-readback tests.
- **Modify** `inference_calculator_metal.{cc,proto if needed}` + test — direct external Metal input mode; bind packet `MTLBuffer` to delegate input and skip input `TFLBufferConvert`.
- **Modify** `mediapipe/calculators/tensor/BUILD` — Metal targets (apple-gated copts/deps), test targets.

---

## Task 1: Metal runtime + delegate binding smoke (feasibility GATE)

**Files:** Create `metal_tensor_smoke_test.cc`; modify `BUILD`. This proves the macOS Metal runtime + `MtlBufferView` actually execute here (the config fix only proved compile), and that the local TFLite Metal delegate can consume a user-bound `MTLBuffer`. **If either smoke fails at runtime, STOP and report — the rest of the plan is blocked.**

- [ ] **Step 1: Write the smoke test** `metal_tensor_smoke_test.cc` (full Apache header). No `GpuResources`/graph needed — just a Metal device + a Metal-backed `Tensor`.

```cpp
#import <Metal/Metal.h>

#include <vector>

#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/formats/tensor_mtl_buffer_view.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

// Proves: a Tensor can be Metal-backed, a Metal command runs on this machine,
// and the result is observable. Writes known floats into the Tensor's MTLBuffer
// on the GPU (via a blit from a source buffer) and reads them back.
TEST(MetalTensorSmokeTest, WriteAndReadBackMetalBackedTensor) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ASSERT_NE(device, nil) << "no Metal device";
  id<MTLCommandQueue> queue = [device newCommandQueue];

  constexpr int kN = 8;
  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{kN});

  // Source buffer with known values.
  std::vector<float> src(kN);
  for (int i = 0; i < kN; ++i) src[i] = i + 0.5f;
  id<MTLBuffer> src_buf =
      [device newBufferWithBytes:src.data()
                          length:sizeof(float) * kN
                         options:MTLResourceStorageModeShared];

  id<MTLCommandBuffer> cb = [queue commandBuffer];
  {
    // GetWriteView allocates/links the Tensor's MTLBuffer.
    auto wv = MtlBufferView::GetWriteView(tensor, cb);
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src_buf
            sourceOffset:0
                toBuffer:wv.buffer()
       destinationOffset:0
                    size:sizeof(float) * kN];
    [blit endEncoding];
  }
  [cb commit];
  [cb waitUntilCompleted];

  auto cpu = tensor.GetCpuReadView();
  const float* out = cpu.buffer<float>();
  for (int i = 0; i < kN; ++i) EXPECT_FLOAT_EQ(out[i], i + 0.5f);
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 2: BUILD** — add an apple-gated ObjC++ test target (mirror the metal converter's copts). In `mediapipe/calculators/tensor/BUILD`:

```python
cc_test(
    name = "metal_tensor_smoke_test",
    srcs = ["metal_tensor_smoke_test.cc"],
    copts = select({
        "@build_bazel_apple_support//constraints:apple": [
            "-x",
            "objective-c++",
            "-fobjc-arc",
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@build_bazel_apple_support//constraints:apple": [
            "-framework Metal",
            "-framework Foundation",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
    ] + select({
        "@build_bazel_apple_support//constraints:apple": [
            "//mediapipe/framework/formats:tensor_mtl_buffer_view",
        ],
        "//conditions:default": [],
    }),
)
```
> Confirm the apple constraint label by checking the metal converter target's `select` keys (it may be `//mediapipe:apple` or `@build_bazel_apple_support//constraints:apple` or `//mediapipe/gpu:metal_*`). Use whatever the shipped `image_to_tensor_converter_metal` target uses. Confirm `tensor_mtl_buffer_view` is a separate target or part of `:tensor`.

- [ ] **Step 3: Build + run (read the real output, not the pipe exit).**

Run: `bazel test -c opt //mediapipe/calculators/tensor:metal_tensor_smoke_test --test_output=all`
Expected: PASS. If it fails to build/run (Metal device unavailable in sandbox, link error, `MtlBufferView` runtime issue), STOP — report the exact error; the GPU path is blocked until resolved.

- [ ] **Step 4: Add a delegate external-buffer smoke.** Use an existing tiny float model fixture (prefer `mediapipe/calculators/tensor/testdata/1x256x256x3_softmax.tflite`; use `1x3_square_float32.tflite` only if it delegates on Metal). Build an interpreter with the Metal delegate, call `ModifyGraphWithDelegate`, allocate a user `MTLBuffer` sized as `GetElementsSizeForPHWC4(BHWC{N,H,W,C}) * sizeof(float)` (or half when `allow_precision_loss=true`), bind it with `TFLGpuDelegateBindMetalBufferToTensor(delegate, input_index, buffer)`, set an external command buffer, invoke, and verify the delegate does not require CPU input. This test need not validate tiled preprocessing; it only proves the external binding API works on this Mac.

- [ ] **Step 5: Commit** `feat(metal): prove Metal Tensor and delegate input binding`.

---

## Task 2: Metal tiled converter writes delegate input layout

**Files:** Create `streaming_tiles_to_tensor_batch_metal.{h,cc}` + `..._test.cc`; modify `BUILD`. Adapt `image_to_tensor_converter_metal.cc` to render **each tile into its row offset** of the Metal delegate input buffer.

- [ ] **Step 1: Read the references** end-to-end:
  - `image_to_tensor_converter_metal.cc` — the `SubRectExtractorMetal` (shader source `kShaderLibHeader`, `MTLRenderPipelineState`, vertex buffers, `ExtractSubRectToBuffer`-equivalent `Execute`) and `Convert`. Note it renders into a render target aliased over the destination `MTLBuffer` and asserts `tensor_buffer_offset==0` / `dims[0]==1` / `dims[3]==4`.
  - TFLite GPU `ConvertToPHWC4` / `GetElementsSizeForPHWC4` — this defines the physical buffer the delegate consumes. For this plan, support model input `C<=4`; write `C4=4` floats per pixel and zero padded channels. Do not claim support for `C>4` until a compute shader writes multi-plane PHWC4.

- [ ] **Step 2: Write the header** `streaming_tiles_to_tensor_batch_metal.h` (full Apache header), guarded `#if MEDIAPIPE_METAL_ENABLED`:

```cpp
#ifndef MEDIAPIPE_CALCULATORS_TENSOR_STREAMING_TILES_TO_TENSOR_BATCH_METAL_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_STREAMING_TILES_TO_TENSOR_BATCH_METAL_H_

#include "mediapipe/framework/port.h"
#if MEDIAPIPE_METAL_ENABLED

#import <Metal/Metal.h>

#include <array>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/gpu/gpu_buffer.h"

namespace mediapipe {

// Renders each valid tile of `geometry` from `input` (a GPU texture) into the
// row offset of `output_tensor` in Metal delegate input layout. Logical model
// shape is [N,H,W,C]; physical storage is [N,H,W,C4] for C<=4 where C4=4 and
// padded channels are zero. One command buffer, committed by the caller. No CPU
// readback.
class StreamingTilesToTensorBatchMetal {
 public:
  static absl::StatusOr<std::unique_ptr<StreamingTilesToTensorBatchMetal>>
  Create(id<MTLDevice> device, int out_h, int out_w, int logical_channels,
         bool use_float16_delegate_buffer);

  // `input_texture` is the source frame (e.g. from
  // metalTextureWithGpuBuffer:). `frame_w/h` size the transform. Writes rows
  // [0, geometry.effective_pixel_rois.size()).
  absl::Status RenderTiles(id<MTLTexture> input_texture, int frame_w,
                           int frame_h, const TileBatchGeometry& geometry,
                           id<MTLCommandBuffer> command_buffer,
                           Tensor& output_tensor);
};

}  // namespace mediapipe
#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // header guard
```

- [ ] **Step 3: Write the impl** `streaming_tiles_to_tensor_batch_metal.cc` (ObjC++ via copts), guarded `#if MEDIAPIPE_METAL_ENABLED`. Reuse the reference's shader + render-pipeline setup for `C<=4`, but make the destination explicitly delegate-layout:
  - Validate `logical_channels` is 1, 3, or 4 for now; reject `C>4` with a clear error.
  - Allocate/expect `output_tensor.shape() == [N,H,W,4]` for both `C=3` and `C=4` (`ElementType::kFloat32` first; add float16 only if the test proves `MTLPixelFormatRGBA16Float` parity and the inference delegate is configured with `allow_precision_loss=true`).
  - Take `MtlBufferView::GetWriteView(output_tensor, command_buffer)` exactly once per output tensor, then encode all row renders into that buffer.
  - For each valid row `r`, build `RotatedRect` from `geometry.effective_pixel_rois[r]`, compute the transform, and render into row byte offset `r*out_h*out_w*4*sizeof(element)`.
  - Use `bytesPerRow = out_w*4*sizeof(element)` only after checking it and `row_offset` satisfy `[device minimumLinearTextureAlignmentForPixelFormat:pixel_format]`; if not aligned, fail and switch the plan to a compute-shader writer rather than silently producing an invalid buffer texture.
  - Normalize with the same value-range transform as the reference. For `C=3`, write RGB and zero alpha/padded channel. For `C=1`, write R and zero GBA, or reject until covered by a test.
  - Do NOT commit the command buffer here. Zero padded batch rows `[valid_count,N)` and padded channels on GPU.

> This lifts the reference's `tensor_buffer_offset==0` restriction by computing a per-row offset into the delegate physical buffer. The logical model input is still `[N,H,W,C]`; only the Metal packet storage is `[N,H,W,4]` for `C<=4`.

- [ ] **Step 4: Converter test** `streaming_tiles_to_tensor_batch_metal_test.cc` (ObjC++): create a Metal device; build a known input `MTLTexture` (e.g. a gradient BGRA texture via `newTextureWithDescriptor` + `replaceRegion`); build a `TileBatchGeometry` for 2 tiles over a known frame size; allocate a physical `[2,H,W,4]` `Tensor` for a logical `C=3` model; run `RenderTiles`; commit + wait; read back via `tensor.GetCpuReadView()`; assert rows match CPU crop/resize/normalize converted to PHWC4 (`RGB0` per pixel) within tolerance. Cover row-offset correctness and padded-channel zeroing.

- [ ] **Step 5: BUILD** — `cc_library(name="streaming_tiles_to_tensor_batch_metal", srcs=[".cc"], hdrs=[".h"], copts=<apple objc++ select>, deps=["//mediapipe/framework:port", ":tiling_types"] + select(apple:[":image_to_tensor_utils", "//mediapipe/gpu:MPPMetalHelper", "//mediapipe/gpu:gpu_buffer", "//mediapipe/framework/formats:tensor", "//mediapipe/framework/formats:tensor_mtl_buffer_view"]))`. Mirror the shipped metal converter target's apple select exactly. Test target apple-gated.

- [ ] **Step 6: Build + test, commit.**
Run: `bazel test -c opt //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_metal_test --test_output=all` → PASS.
```bash
git add mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal.h mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal.cc mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'feat(metal): render tiled input directly into delegate layout\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 3: Calculator GPU branch emits delegate-layout tensors

**Files:** Modify `streaming_tiles_to_tensor_batch_calculator.{proto,cc}` + test; modify `BUILD`.

- [ ] **Step 1: proto** — add `optional bool enable_gpu_zero_copy = 5 [default = false];` (field 5; 1–4 used). Comment: when true AND an `IMAGE_GPU` input is connected+present, run the Metal delegate-input path (macOS/Metal); else the CPU path.

- [ ] **Step 2: cc** — guard the Metal code `#if MEDIAPIPE_METAL_ENABLED`. Make `IMAGE` and `IMAGE_GPU` mutually-exclusive optional inputs in the contract; CPU mode requires `IMAGE`, GPU mode requires `IMAGE_GPU`. Add (Metal-only) members: an `MPPMetalHelper* metal_helper_` and a `std::unique_ptr<StreamingTilesToTensorBatchMetal>` per `(H,W,C,element_type)`. In `GetContract`, when Metal-enabled, call `[MPPMetalHelper updateContract:cc requestGpuAsOptional:true]` so default-off CPU graphs do not require GPU service. In `Open()`, init the helper only when `enable_gpu_zero_copy` is set and the GPU service is available; otherwise fail clearly if the graph requested GPU mode but cannot provide `IMAGE_GPU`/GpuResources. In `Process()`:
  - If `options_.enable_gpu_zero_copy()` AND `MEDIAPIPE_METAL_ENABLED` AND `kInImageGpu(cc).IsConnected() && !kInImageGpu(cc).IsEmpty()`: run the GPU branch — build/reuse the same `TileBatchGeometry` (existing cache code), allocate the physical batch `Tensor` `[N,H,W,4]` for logical model `[N,H,W,C]` with `C<=4`, get `id<MTLTexture> src = [metal_helper_ metalTextureWithGpuBuffer:*kInImageGpu(cc)]`, `cb = [metal_helper_ commandBuffer]`, `converter->RenderTiles(src, fw, fh, *geom, cb, tensor)`, `[cb commit]`, emit the Metal-backed physical tensor + `BATCH_INFO` exactly as the CPU path (same `valid_count`/`batch_size`/timestamps). Do NOT call any CPU view on input or output.
  - Else: the existing CPU pixel loop (unchanged).
  Keep the current CPU multi-batch behavior. Do not add a temporary `T>cap` guard on the GPU path; GPU mode must loop over batches with the same synthetic timestamps from the first functional commit.

- [ ] **Step 3: BUILD** — add `streaming_tiles_to_tensor_batch_calculator` deps (apple-gated): `:streaming_tiles_to_tensor_batch_metal`, `//mediapipe/gpu:MPPMetalHelper`, `//mediapipe/gpu:gpu_buffer`, `//mediapipe/framework/formats:tensor_mtl_buffer_view`; add the apple objc++ copts to the calculator target (it becomes ObjC++ on Apple). Ensure the CPU (`MEDIAPIPE_DISABLE_GPU=1`) build still excludes all Metal deps via the select.

- [ ] **Step 4: Parity + no-readback test** — in a Metal-enabled test (`streaming_tiles_to_tensor_batch_metal_pipeline_test.cc` or extend the converter test): feed a known image as a `GpuBuffer` (build one from a `cv::Mat`/`ImageFrame` via `GpuBufferStorageCvPixelBuffer` or an upload helper) AND as an `ImageFrame`; run the calculator both ways (GPU branch vs CPU branch); convert the CPU result to PHWC4 in test and compare with the GPU physical tensor within tolerance (e.g. `1e-2`, GPU sampler interpolation). Before any readback, assert `ready_as_metal_buffer()` and `!ready_on_cpu()` for the GPU output. Include `T=5, batch_capacity=2` so the first GPU implementation proves multi-batch immediately.

- [ ] **Step 5: CPU regression + commit.**
Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test --test_output=errors` (CPU path unchanged) AND `bazel test -c opt //mediapipe/calculators/tensor:<metal_pipeline_test> --test_output=all`.
```bash
git add mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.proto mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc mediapipe/calculators/tensor/*metal_pipeline_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'feat(metal): emit delegate-layout tiled Metal tensors\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 4: InferenceCalculatorMetal direct external input

**Files:** Modify `inference_calculator.proto` if an option is needed, `inference_calculator_metal.cc`, tests, and BUILD.

- [ ] **Step 1: option/contract** — add an experimental Metal-only option under `InferenceCalculatorOptions::Delegate::Gpu`, e.g. `optional bool use_metal_direct_input_buffer = <next field> [default=false];`. It means input tensor packets are already in the TFLite Metal delegate physical layout (`PHWC4/BPHWC4`) and must be bound directly, not converted from logical BHWC.

- [ ] **Step 2: implementation** — in `InferenceCalculatorMetal`:
  - In `CreateConverters`, keep output binding/conversion unchanged for now, but when direct input is enabled do not allocate `gpu_buffers_in_` and do not instantiate/use `converter_to_BPHWC4_`.
  - In `Process()`, get the packet tensor's Metal buffer via `MtlBufferView::GetReadView(tensor_span[i], command_buffer).buffer()`, validate element type and physical shape against model input (`[N,H,W,RoundUp(C,4)]` for `C<=4`), call `TFLGpuDelegateBindMetalBufferToTensor(delegate_.get(), input_indices[i], packet_buffer)` before `Invoke()`, set the same command buffer on the delegate, and invoke. The command buffer must encode preprocessing-before-inference ordering without CPU waits; if preprocessing committed a prior command buffer, rely on Metal command queue ordering or explicit synchronization documented in the test.
  - In non-direct mode, keep the existing `TFLBufferConvert` input copy path unchanged.

- [ ] **Step 3: direct-bind test** — build a graph where `StreamingTilesToTensorBatchCalculator(enable_gpu_zero_copy=true)` feeds `InferenceCalculator(delegate { gpu { use_metal_direct_input_buffer:true } })`. Use a model with logical input `C=3`; assert the preprocessor output buffer pointer equals the buffer passed to `TFLGpuDelegateBindMetalBufferToTensor` (instrument with a test-only observer/counter if needed), and assert no `converter_to_BPHWC4_` command is encoded in direct mode. Verify output numerically enough to prove inference ran.

- [ ] **Step 4: negative tests** — direct mode rejects CPU tensors, non-Metal tensors, wrong physical channel count, `C>4` until compute PHWC4 support exists, and missing GPU service. Default non-direct Metal inference still passes existing tests.

- [ ] **Step 5: commit** `feat(metal): bind tiled input buffers directly to Metal delegate`.

---

## Task 5: End-to-end true zero-copy multi-batch graph

**Files:** Metal pipeline integration test(s), graph snippets, BUILD.

- [ ] **Step 1:** Test `T=5, batch_capacity=2` end-to-end: GPU tiled preprocessor emits 3 physical Metal input tensors (`valid_count` 2/2/1), each packet's `MTLBuffer` is directly rebound to the delegate input and invoked, and `TensorBatchInfo` keeps the same source frame timestamp across all batches.
- [ ] **Step 2:** Assert true zero-copy invariants: no CPU input materialization, no CPU output materialization before test readback, no input `TFLBufferConvert`, and delegate-bound input buffer identity equals the preprocessor packet buffer for every batch.
- [ ] **Step 3:** CPU regression remains green with `--define MEDIAPIPE_DISABLE_GPU=1`; Metal non-direct inference remains green; default-off graph does not require GPU service.
- [ ] **Step 4:** Commit `feat(metal): true zero-copy tiled preprocessing into Metal inference`.

---

## Self-review checklist
- Metal runtime smoke and TFLite Metal delegate external-buffer smoke pass on this Mac (runtime proven, not just compile).
- GPU branch parity with CPU-converted-to-PHWC4 within tolerance; no CPU view of input/output on the GPU path before explicit test readback.
- Default-off: `enable_gpu_zero_copy=false` ⇒ CPU path byte-identical; CPU build (`MEDIAPIPE_DISABLE_GPU=1`) excludes all Metal code and stays green.
- Per-tile row-offset rendering correct (row r at byte offset `r*H*W*4*sizeof(element)` for `C<=4` delegate layout); padded batch rows and padded channels zeroed.
- `InferenceCalculatorMetal` direct mode binds the packet `MTLBuffer` to the delegate input and skips the BHWC→BPHWC4 `TFLBufferConvert` path.
- `T=5,batch_capacity=2` multi-batch is covered end-to-end; geometry/matrices reused from the shipped cache.

## Final verification
- [ ] `bazel test -c opt //mediapipe/calculators/tensor:metal_tensor_smoke_test //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_metal_test <metal_direct_bind_test> <metal_pipeline_test> --test_output=errors` — pass (Metal).
- [ ] `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test --test_output=errors` — CPU path unchanged.
- [ ] `git status` clean.

## Done criteria
- macOS Metal runtime proven; GPU-backed input → Metal tiled crop/resize/normalize → delegate-layout Metal `Tensor` → TFLite Metal delegate consumes the same `MTLBuffer`.
- No CPU readback and no BHWC→BPHWC4 input copy on the zero-copy path; buffer identity is asserted in tests.
- Default-off; CPU path + CPU build unaffected.
- Single + multi-batch covered; Metal-`Tensor` pooling explicitly deferred to a follow-up.
