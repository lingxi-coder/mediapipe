# Metal Zero-Copy Tiled Preprocessing (Phase 5, macOS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Metal GPU path to `StreamingTilesToTensorBatchCalculator` so GPU-backed input is cropped/resized/normalized per-tile in Metal and written directly into a Metal-backed `Tensor` batch — no CPU readback — verifiable on this Mac.

**Architecture:** A new ObjC++ Metal converter (`streaming_tiles_to_tensor_batch_metal`) adapts the shipped `image_to_tensor_converter_metal.cc` (render-pipeline quad + `metalTextureWithGpuBuffer:` input texture + `MtlBufferView::GetWriteView` output) to render **each tile into its row offset** of the `[N,H,W,C]` batch buffer. The calculator gains an `IMAGE_GPU` (`GpuBuffer`) input + `enable_gpu_zero_copy` option; default-off keeps the shipped CPU path byte-identical. A Metal runtime smoke gates the whole effort.

**Tech Stack:** Objective-C++ (Metal), `MPPMetalHelper`, `MtlBufferView`, `Tensor` (BHWC), `GetRotatedSubRectToRectTransformMatrix`, MediaPipe `GpuResources`/Metal, Bazel (Apple/Metal build, NOT `--define MEDIAPIPE_DISABLE_GPU=1`), GoogleTest.

**Spec:** `docs/superpowers/specs/2026-06-03-metal-zerocopy-tiled-preprocessing-design.md`.

**Prerequisite (done):** `mediapipe/gpu/gpu_buffer_format.h` enables `MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER` on macOS (commit `9c13613`), so the CVPixelBuffer `GpuBuffer` + `metalTextureWithGpuBuffer:` bridge compile here.

---

## Reference facts (verified)

- **Build flavor:** Metal code builds WITHOUT `--define MEDIAPIPE_DISABLE_GPU=1` (that flag forces `MEDIAPIPE_METAL_ENABLED 0`). On macOS, GPU build ⇒ Metal. Use `bazel {build,test} -c opt <target>` (no disable-gpu) for Metal targets. The CPU calculator tests still use `--define MEDIAPIPE_DISABLE_GPU=1` and must stay green.
- **`MtlBufferView`** (`mediapipe/framework/formats/tensor_mtl_buffer_view.h`): `static MtlBufferView GetWriteView(const Tensor&, id<MTLCommandBuffer>)`, `GetWriteView(const Tensor&, id<MTLDevice>)`, `GetReadView(const Tensor&, id<MTLCommandBuffer>)`; instance `id<MTLBuffer> buffer()`. Header guarded by `MEDIAPIPE_METAL_ENABLED`.
- **`MPPMetalHelper`** (`mediapipe/gpu/MPPMetalHelper.h`): `- (instancetype)initWithCalculatorContext:(CalculatorContext*)cc`, `+ (absl::Status)updateContract:(CalculatorContract*)cc`, `- (id<MTLCommandBuffer>)commandBuffer`, `@property(readonly) id<MTLDevice> mtlDevice`, `- (id<MTLTexture>)metalTextureWithGpuBuffer:(const mediapipe::GpuBuffer&)`.
- **Reference `image_to_tensor_converter_metal.cc`:** `Convert(const Image& input, const RotatedRect& roi, float range_min, float range_max, int tensor_buffer_offset, Tensor& output_tensor)`. It: gets `texture = [metal_helper_ metalTextureWithGpuBuffer:input.GetGpuBuffer()]`; `command_buffer = [metal_helper_ commandBuffer]`; `buffer_view = MtlBufferView::GetWriteView(output_tensor, command_buffer)`; `extractor_->Execute(texture, roi, flip, scale, offset, HW(out_h,out_w), command_buffer, buffer_view.buffer())`; `[command_buffer commit]`. **It hard-asserts `output_shape.dims[0]==1` and `tensor_buffer_offset==0`.** The inner `SubRectExtractorMetal` renders a quad (positions/tex-coords/transform vertex buffers, input texture as fragment sampler) into a render target aliased over the destination `MTLBuffer`. The tiling delta = render each tile into a per-row buffer offset.
- **`StreamingTilesToTensorBatchCalculator`** (`streaming_tiles_to_tensor_batch_calculator.cc`, Plans 2–3): api2 `Node`, `namespace mediapipe::api2`. Inputs `IMAGE`(`ImageFrame`), `TILE_PLAN`(`TilePlan`), side `METADATA`(`InferenceMetadata`: `input_height/width/channels`, `batch_capacity`, `is_dynamic_batch`). Outputs `TENSORS`(`std::vector<Tensor>`), `BATCH_INFO`(`TensorBatchInfo`), `CACHE_STATS`(optional). `Process()` builds/caches `TileBatchGeometry` (per valid row: `effective_pixel_rois` (`TilePixelRoi{x,y,width,height}`), `tile_geometries`, `tile_to_image_matrices`), then a CPU pixel loop crops/resizes/normalizes each row into a CPU `Tensor`, emits at synthetic `batch_ts_`. proto fields: `dynamic_batch=1`, `max_cached_tile_matrices=2`, `max_cpu_tensor_workspaces=3`, `emit_cache_stats=4`.
- **`GetRotatedSubRectToRectTransformMatrix(const RotatedRect& sub_rect, int rect_w, int rect_h, bool flip, std::array<float,16>*)`** (`image_to_tensor_utils.h`): the transform the Metal shader's `transform_matrix` vertex uniform expects; `RotatedRect{center_x,center_y,width,height,rotation}`. The CPU path's `TilePixelRoi{x,y,width,height}` maps to a `RotatedRect{center_x=x+width/2, center_y=y+height/2, width, height, rotation=0}` (same as `tiling_matrix_utils::TileToImageMatrix`).
- **Metal `.cc`/`.mm` BUILD pattern** (`image_to_tensor_converter_metal` target, `mediapipe/calculators/tensor/BUILD:1879`): `cc_library` with `copts = select({apple: ["-x", "objective-c++", "-fobjc-arc"], default: []})` and `deps = ["//mediapipe/framework:port"] + select({apple: [":image_to_tensor_converter", ":image_to_tensor_utils", "//mediapipe/gpu:MPPMetalHelper", "//mediapipe/gpu:gpu_buffer", "//mediapipe/framework/formats:tensor", "@google_toolbox_for_mac//:GTM_Defines"], default: []})`. READ the exact target before writing yours and copy the apple `select`/copts/deps shape.

## Cross-cutting conventions

- Metal build/test: `bazel test -c opt //mediapipe/calculators/tensor:<metal_target> --test_output=errors` (NO disable-gpu). CPU regression: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test`.
- In-editor clang errors are FALSE POSITIVES; only bazel is authoritative. **`exit 0` from a piped build is NOT success — always read the build/test output tail** (this fork's GPU stack has surfaced misleading pipe-exit-0s).
- Default-off: `enable_gpu_zero_copy=false` (default) ⇒ shipped CPU path, byte-identical. GPU path only when set AND `IMAGE_GPU` connected+present.
- All Metal code Apple-gated; the CPU build (`MEDIAPIPE_DISABLE_GPU=1`) must not compile the Metal `.mm` and must stay green.
- Each task commits separately; co-author trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Branch `dev`.

## File Structure

- **Create** `mediapipe/calculators/tensor/metal_tensor_smoke_test.cc` — Metal runtime gate (ObjC++).
- **Create** `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal.{h,cc}` — the Metal tiled converter (`.cc` compiled ObjC++ via copts, matching the repo pattern). One responsibility: render tiles → Metal-backed Tensor rows.
- **Create** `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal_test.cc` — converter unit test (input texture + tiles → buffer, read back, compare to CPU math).
- **Modify** `streaming_tiles_to_tensor_batch_calculator.{proto,cc}` + test — `IMAGE_GPU` input, `enable_gpu_zero_copy`, GPU dispatch; parity + no-readback tests.
- **Modify** `mediapipe/calculators/tensor/BUILD` — Metal targets (apple-gated copts/deps), test targets.

---

## Task 1: Metal runtime smoke (feasibility GATE)

**Files:** Create `metal_tensor_smoke_test.cc`; modify `BUILD`. This proves the macOS Metal runtime + `MtlBufferView` actually execute here (the config fix only proved compile). **If this fails at runtime, STOP and report — the rest of the plan is blocked.**

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

- [ ] **Step 4: Commit** `feat(metal): Metal-backed Tensor runtime smoke (macOS GPU gate)`.

---

## Task 2: Metal tiled converter

**Files:** Create `streaming_tiles_to_tensor_batch_metal.{h,cc}` + `..._test.cc`; modify `BUILD`. Adapt `image_to_tensor_converter_metal.cc` to render **each tile into its row offset** of the batch buffer.

- [ ] **Step 1: Read the reference** `image_to_tensor_converter_metal.cc` end-to-end — the `SubRectExtractorMetal` (shader source `kShaderLibHeader`, `MTLRenderPipelineState`, vertex buffers, `ExtractSubRectToBuffer`-equivalent `Execute`) and `Convert`. Note it renders into a render target aliased over the destination `MTLBuffer` and asserts `tensor_buffer_offset==0` / `dims[0]==1`.

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

// Renders each valid tile of `geometry` from `input` (a GPU texture) into its
// row offset of `output_tensor` ([N,H,W,C], Metal-backed), normalizing to [0,1].
// One command buffer, committed by the caller's helper. No CPU readback.
class StreamingTilesToTensorBatchMetal {
 public:
  static absl::StatusOr<std::unique_ptr<StreamingTilesToTensorBatchMetal>>
  Create(id<MTLDevice> device, int out_h, int out_w, int channels);

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

- [ ] **Step 3: Write the impl** `streaming_tiles_to_tensor_batch_metal.cc` (ObjC++ via copts), guarded `#if MEDIAPIPE_METAL_ENABLED`. Reuse the reference's shader + render-pipeline setup verbatim (copy `kShaderLibHeader`, pipeline creation, positions/tex-coords buffers). The ONLY structural change: in `RenderTiles`, loop over valid rows and, per tile `r`, (a) build `RotatedRect sub_rect{center_x=roi.x+roi.width/2, center_y=roi.y+roi.height/2, width=roi.width, height=roi.height, rotation=0}` from `geometry.effective_pixel_rois[r]`, (b) compute the transform via `GetRotatedSubRectToRectTransformMatrix(sub_rect, frame_w, frame_h, false, &mat)`, (c) render that tile into the batch buffer at **row byte offset `r*out_h*out_w*channels*sizeof(float)`** — i.e., create the render-target texture aliased over `MtlBufferView::GetWriteView(output_tensor, command_buffer).buffer()` with `offset:` = that row offset and `bytesPerRow: out_w*channels*sizeof(float)` (via `[buffer newTextureWithDescriptor:descriptor offset:rowOffset bytesPerRow:...]`), and normalize ×(1/255) using the existing scale/offset uniform path (range_min=0, range_max=1 with input already 0..1, or scale=1/255 if input is 0..255 BGRA — match the reference's `GetValueRangeTransformation`). Do NOT commit the command buffer here (the caller does). Zero any padding rows `[valid_count, N)` (a blit-fill or a clear render).

> This lifts the reference's `tensor_buffer_offset==0` restriction by computing a per-row offset. Keep the shader identical; only the destination texture's buffer offset changes per tile.

- [ ] **Step 4: Converter test** `streaming_tiles_to_tensor_batch_metal_test.cc` (ObjC++): create a Metal device; build a known input `MTLTexture` (e.g. a solid-color or gradient BGRA texture via `newTextureWithDescriptor` + `replaceRegion`); build a `TileBatchGeometry` for 2 tiles over a known frame size; allocate a `[2,H,W,C]` `Tensor`; run `RenderTiles`; commit + wait; read back via `tensor.GetCpuReadView()`; assert each row's pixels match the expected crop/resize/normalize of the input sub-rect within tolerance (compare against a CPU `cv::resize`+normalize of the same sub-rect, or for a solid-color input assert the constant value). Cover the row-offset correctness (row 1 differs from row 0 for different tiles).

- [ ] **Step 5: BUILD** — `cc_library(name="streaming_tiles_to_tensor_batch_metal", srcs=[".cc"], hdrs=[".h"], copts=<apple objc++ select>, deps=["//mediapipe/framework:port", ":tiling_types"] + select(apple:[":image_to_tensor_utils", "//mediapipe/gpu:MPPMetalHelper", "//mediapipe/gpu:gpu_buffer", "//mediapipe/framework/formats:tensor", "//mediapipe/framework/formats:tensor_mtl_buffer_view"]))`. Mirror the shipped metal converter target's apple select exactly. Test target apple-gated.

- [ ] **Step 6: Build + test, commit.**
Run: `bazel test -c opt //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_metal_test --test_output=all` → PASS.
```bash
git add mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal.h mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal.cc mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'feat(metal): tiled crop/resize/normalize into Metal-backed Tensor rows\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 3: Calculator GPU branch (single-batch) + parity

**Files:** Modify `streaming_tiles_to_tensor_batch_calculator.{proto,cc}` + test; modify `BUILD`.

- [ ] **Step 1: proto** — add `optional bool enable_gpu_zero_copy = 5 [default = false];` (field 5; 1–4 used). Comment: when true AND an `IMAGE_GPU` input is connected+present, run the Metal zero-copy path (macOS/Metal); else the CPU path.

- [ ] **Step 2: cc** — guard the Metal code `#if MEDIAPIPE_METAL_ENABLED`. Add an optional GPU input `static constexpr Input<mediapipe::GpuBuffer>::Optional kInImageGpu{"IMAGE_GPU"};` to the contract. Add (Metal-only) members: an `MPPMetalHelper* metal_helper_` and a `std::unique_ptr<StreamingTilesToTensorBatchMetal>` per (H,W,C). In `GetContract`, when Metal-enabled, call `[MPPMetalHelper updateContract:cc]`. In `Open()`, when `enable_gpu_zero_copy` and Metal-enabled, init `metal_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc]`. In `Process()`:
  - If `options_.enable_gpu_zero_copy()` AND `MEDIAPIPE_METAL_ENABLED` AND `kInImageGpu(cc).IsConnected() && !kInImageGpu(cc).IsEmpty()`: run the GPU branch — build/reuse the same `TileBatchGeometry` (existing cache code), allocate the batch `Tensor` `[N,H,W,C]`, get `id<MTLTexture> src = [metal_helper_ metalTextureWithGpuBuffer:*kInImageGpu(cc)]`, `cb = [metal_helper_ commandBuffer]`, `converter->RenderTiles(src, fw, fh, *geom, cb, tensor)`, `[cb commit]`, emit the (Metal-backed) `Tensor` + `BATCH_INFO` exactly as the CPU path (same `valid_count`/`batch_size`/timestamps). Do NOT call any CPU view on input or output.
  - Else: the existing CPU pixel loop (unchanged).
  Keep single-batch (`T<=cap`) for this task (multi-batch GPU is Task 4); for `T>cap` on the GPU branch, `RET_CHECK_LE` for now with a clear message, OR fall through to CPU. Choose RET_CHECK and note it.

- [ ] **Step 3: BUILD** — add `streaming_tiles_to_tensor_batch_calculator` deps (apple-gated): `:streaming_tiles_to_tensor_batch_metal`, `//mediapipe/gpu:MPPMetalHelper`, `//mediapipe/gpu:gpu_buffer`, `//mediapipe/framework/formats:tensor_mtl_buffer_view`; add the apple objc++ copts to the calculator target (it becomes ObjC++ on Apple). Ensure the CPU (`MEDIAPIPE_DISABLE_GPU=1`) build still excludes all Metal deps via the select.

- [ ] **Step 4: Parity + no-readback test** — in a Metal-enabled test (`streaming_tiles_to_tensor_batch_metal_pipeline_test.cc` or extend the converter test): feed a known image as a `GpuBuffer` (build one from a `cv::Mat`/`ImageFrame` via `GpuBufferStorageCvPixelBuffer` or an upload helper) AND as an `ImageFrame`; run the calculator both ways (GPU branch vs CPU branch); read back both tensors; assert element-wise equality within tolerance (e.g. `1e-2`, GPU sampler interpolation). Assert the GPU branch's output tensor reports Metal-ready (`ready_as_metal_buffer()` or that a CPU view was never taken before read-back). Build + test.

- [ ] **Step 5: CPU regression + commit.**
Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test --test_output=errors` (CPU path unchanged) AND `bazel test -c opt //mediapipe/calculators/tensor:<metal_pipeline_test> --test_output=all`.
```bash
git add mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.proto mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc mediapipe/calculators/tensor/*metal_pipeline_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'feat(metal): GPU zero-copy branch in StreamingTilesToTensorBatch (single-batch)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 4: Multi-batch on the GPU path (`T > batch_capacity`)

**Files:** Modify `streaming_tiles_to_tensor_batch_calculator.cc` + the Metal pipeline test.

- [ ] **Step 1:** Remove the single-batch `RET_CHECK_LE` on the GPU branch; loop over batches exactly like the CPU path (same `batch_ts_` synthetic timestamps, `batch_index`/`total_batches`/`valid_count`), allocating + rendering + emitting one Metal-backed `Tensor` per batch. Padding rows `[valid_count, N)` zeroed on GPU.
- [ ] **Step 2:** Test: `T=5, batch_capacity=2` on the GPU branch → 3 Metal-backed tensors, `valid_count` 2/2/1, same `source_frame_timestamp`, parity with the CPU path per batch. Build + test.
- [ ] **Step 3:** Commit `feat(metal): multi-batch (T>cap) on the GPU zero-copy path`.

---

## Self-review checklist
- Metal runtime smoke passes on this Mac (runtime proven, not just compile).
- GPU branch parity with CPU within tolerance; no CPU view of input/output on the GPU path.
- Default-off: `enable_gpu_zero_copy=false` ⇒ CPU path byte-identical; CPU build (`MEDIAPIPE_DISABLE_GPU=1`) excludes all Metal code and stays green.
- Per-tile row-offset rendering correct (row r at byte offset `r*H*W*C*4`); padding rows zeroed.
- Single-batch then `T>cap` multi-batch; geometry/matrices reused from the shipped cache.

## Final verification
- [ ] `bazel test -c opt //mediapipe/calculators/tensor:metal_tensor_smoke_test //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_metal_test <metal_pipeline_test> --test_output=errors` — pass (Metal).
- [ ] `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test --test_output=errors` — CPU path unchanged.
- [ ] `git status` clean.

## Done criteria
- macOS Metal runtime proven; GPU-backed input → Metal tiled crop/resize/normalize → Metal-backed `Tensor`, no CPU readback, parity with CPU.
- Default-off; CPU path + CPU build unaffected.
- Single + multi-batch covered; Metal-`Tensor` pooling explicitly deferred to a follow-up.
