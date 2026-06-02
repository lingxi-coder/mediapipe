# Spec - Phase 3 (M8): Tile, matrix, tensor-buffer, and zero-copy cache design

Date: 2026-06-02
Status: Design (pre-plan)
Branch: `dev` (long-lived integration branch; commit only when asked)
Roadmap: `docs/superpowers/specs/2026-06-01-roadmap.md` Phase 3 (M8);
expands the cache sketch in
`docs/superpowers/specs/2026-06-01-detection-core-yolo-obb-tiling-design.md`
section 14.1.

## Goal

Build a bounded, default-off performance layer for the tiled detection path that
actually reduces repeated work in video/streaming workloads:

- avoid rebuilding stable `TilePlan`s;
- avoid recomputing tile-to-tensor matrices for stable image/model/tile geometry;
- avoid repeated batch tensor buffer allocation where MediaPipe packet lifetime
  permits safe reuse;
- avoid CPU readback/copies in GPU mode by adding an OpenGL zero-copy path;
- keep all caches bounded, context-scoped where needed, and correctness-neutral.

The previous "TilePlan-only" slice is safe but too small to materially improve
runtime. This spec keeps that low-risk piece as M8.1 and adds the missing pieces
needed for a useful M8 performance milestone.

## Current shipped constraints

- `TileSpecToTilePlanCalculator` currently consumes `std::vector<NormalizedRect>`
  and emits `TilePlan`; it validates non-empty dimensions, rejects non-zero
  rotation, assigns `tile_index`, and has only `max_tiles_per_frame`.
- `StreamingTilesToTensorBatchCalculator` currently consumes `IMAGE` +
  `TILE_PLAN` + `InferenceMetadata`, emits `TENSORS` and `BATCH_INFO`, and uses a
  CPU-only OpenCV path (`ImageFrame` -> crop/resize/normalize -> CPU `Tensor`).
- Current streaming code rejects `T > batch_capacity`; true multi-batch
  per-frame emission is not implemented yet.
- Current streaming code does not compute or emit tile-to-tensor matrices.
- `ImageToTensorCalculator` already defines the relevant matrix convention:
  row-major `std::array<float, 16>` mapping input image points to output tensor
  points, invertible by downstream code.
- `Tensor` supports CPU, OpenGL texture, OpenGL buffer, Metal, WebGPU, and AHWB
  views. Packet lifetime matters: a tensor buffer cannot be reused after
  `Send()` until downstream packets release it.
- `MemoryManager` currently provides pooled AHardwareBuffer support when
  `MEDIAPIPE_TENSOR_USE_AHWB` is enabled. Plain desktop CPU buffer reuse and
  non-AHWB OpenGL buffer reuse require either new pool ownership hooks or a
  release-callback extension.

## Scope

In scope:

1. `TilePlan` cache in `TileSpecToTilePlanCalculator`.
2. Multi-tile and multi-batch contract repair for the streaming tiled path.
3. Tile-to-tensor matrix generation and cache.
4. CPU tensor workspace reuse where safe, plus a clear Tensor/MemoryManager
   extension for true CPU buffer pooling.
5. OpenGL zero-copy tiled preprocessing path for `GpuBuffer`/GPU-backed `Image`.
6. OpenGL/AHWB tensor-buffer cache, scoped to GL context and packet release.
7. Tests and perf instrumentation proving hit rates, allocation reduction, and
   absence of stale pixels/cross-frame mixing.

Out of scope:

- Output decoder/NMS scratch-buffer caching except for interface compatibility.
- Model/delegate startup cache (`cached_kernel_path`, NNAPI cache dir, etc.).
- Metal/WebGPU/Vulkan/CUDA implementations. The design must not block them, but
  M8 implementation targets CPU + OpenGL/AHWB first.

## Non-negotiable correctness rules

- No cache entry may contain detections or semantic frame results.
- Pixel content is never cached by geometry key. Reusable tensor/surface caches
  cache storage only; every valid row must be overwritten before emission.
- Padding rows must be zeroed or marked invalid and ignored by merge.
- No batch may mix tiles from different source frames.
- Final merge/NMS still runs once per source frame.
- Any reusable buffer sent in a packet must not be reused until downstream
  release is observed or a proven framework-owned pool returns it.
- OpenGL resources are scoped to their GL context/share group. They must not be
  reused across incompatible contexts.
- All new cache options default to `0` or `false`.
- Negative capacities are rejected before signed-to-unsigned conversion.

## Public options

Add to `TileSpecToTilePlanCalculatorOptions`:

```proto
optional int32 max_cached_tile_plans = 2 [default = 0];
optional bool emit_cache_stats = 3 [default = false];
```

Add to `StreamingTilesToTensorBatchCalculatorOptions`:

```proto
optional int32 max_cached_tile_matrices = 2 [default = 0];
optional int32 max_cpu_tensor_workspaces = 3 [default = 0];
optional int32 max_gpu_tensor_buffers = 4 [default = 0];
optional int32 max_tile_surfaces = 5 [default = 0];
optional bool enable_gpu_zero_copy = 6 [default = false];
optional bool allow_cpu_input_fallback = 7 [default = true];
optional bool emit_cache_stats = 8 [default = false];
optional int32 max_in_flight_gpu_batches = 9 [default = 0];
optional bool allow_gpu_readback_fallback = 10 [default = false];
```

Validation:

```cpp
RET_CHECK_GE(options_.max_cached_tile_plans(), 0);
RET_CHECK_GE(options_.max_cached_tile_matrices(), 0);
RET_CHECK_GE(options_.max_cpu_tensor_workspaces(), 0);
RET_CHECK_GE(options_.max_gpu_tensor_buffers(), 0);
RET_CHECK_GE(options_.max_tile_surfaces(), 0);
RET_CHECK_GE(options_.max_in_flight_gpu_batches(), 0);
```

Stats exposure:

- Unit tests read `CacheStats` directly from helper-owned test instances.
- Calculator tests use an optional `CACHE_STATS` output stream when connected.
- Production graphs can leave stats disconnected; counters must not allocate on
  the hot path when `emit_cache_stats == false`.
- `CACHE_STATS` packet type is a C++ struct:

```cpp
struct TilingCacheStats {
  CacheStats tile_plan;
  CacheStats tile_matrix;
  CacheStats cpu_tensor_workspace;
  CacheStats gpu_tensor_buffer;
  CacheStats tile_surface;
};
```

- `TileSpecToTilePlanCalculator` emits cumulative stats at the input timestamp
  after processing the tile list.
- `StreamingTilesToTensorBatchCalculator` emits cumulative stats at the source
  frame timestamp after all batches for that source frame have been emitted.
- Stats packets are diagnostic only; they must not be used to align inference or
  merge streams.

## Shared cache primitives

Create `mediapipe/calculators/tensor/tiling_cache_utils.h`.

```cpp
namespace mediapipe {

struct StableCacheKey {
  std::string bytes;        // type-tagged, length-delimited field encoding
  uint64_t fingerprint = 0; // deterministic lookup hash only
};

struct CacheStats {
  int64_t hits = 0;
  int64_t misses = 0;
  int64_t inserts = 0;
  int64_t evictions = 0;
  int64_t in_flight = 0;
  int64_t reclaimed = 0;
  int64_t pool_exhausted = 0;
  int64_t gpu_to_cpu_fallbacks = 0;
};

class StableKeyBuilder {
 public:
  StableKeyBuilder& AddInt(int64_t value);
  StableKeyBuilder& AddBool(bool value);
  StableKeyBuilder& AddFloat(float value);  // std::bit_cast<uint32_t>
  StableKeyBuilder& AddString(std::string_view value);
  StableCacheKey Build() const;
};

template <typename Value>
class BoundedLruCache {
 public:
  explicit BoundedLruCache(size_t capacity);
  bool enabled() const;
  const Value* Get(const StableCacheKey& key);
  void Put(StableCacheKey key, Value value);
  CacheStats stats() const;
};

}  // namespace mediapipe
```

The LRU map must compare full encoded key bytes, not only the 64-bit
fingerprint, so a hash collision cannot return the wrong value.

## Contract repair for multi-tile and multi-batch

Current `StreamingTilesToTensorBatchCalculator` rejects `T > batch_capacity`.
That must be fixed before caches can be called complete.

Add/extend `TensorBatchInfo`:

```cpp
struct TilePixelRoi {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

struct TileBatchGeometry {
  std::vector<int> tile_indices;
  std::vector<TileGeometry> tile_geometries;
  std::vector<TilePixelRoi> effective_pixel_rois;
  std::vector<std::array<float, 16>> image_to_tensor_matrices;
  std::vector<std::array<float, 16>> tensor_to_image_matrices;
};

struct TensorBatchInfo {
  int64_t source_frame_timestamp = 0;
  int64_t batch_timestamp = 0;
  int batch_index = 0;
  int total_batches = 1;
  int batch_capacity = 1;
  int batch_size = 0;     // emitted N dimension
  int valid_count = 0;   // valid rows [0, valid_count)
  std::shared_ptr<const TileBatchGeometry> geometry;
};
```

Emission rules:

- `source_frame_timestamp` remains the original frame timestamp.
- `batch_timestamp` is a monotonic calculator-local timestamp used only to align
  `TENSORS` and `BATCH_INFO` packets for each emitted batch.
- Multi-batch output must use explicit packet timestamps. Do not rely on the
  api2 `Send()` helper if it would emit multiple packets at the same input
  timestamp.
- `batch_index` and `total_batches` group batches back to one source frame.
- `batch_capacity` is the configured/model maximum capacity. `batch_size` is the
  emitted tensor's N dimension. For fixed-batch models, `batch_size ==
  batch_capacity`; for dynamic-batch models, `batch_size == valid_count`.
- Each batch carries the tile geometry and matrices for its valid rows. This
  avoids requiring a full-frame `TilePlan` packet at every synthetic batch
  timestamp.
- `TileBatchGeometry` is immutable and may be shared with a cache entry. Avoid
  copying matrix/geometry vectors into every `TensorBatchInfo` packet.
- `geometry` must be non-null when `valid_count > 0`, and its vector sizes must
  all equal `valid_count`.
- Merge calculators consume batch-local geometry/matrices from `TensorBatchInfo`;
  they no longer require a same-timestamp full `TilePlan` input for projection.
  This is a public calculator contract change: existing graph configs and tests
  that connect `TILE_PLAN` to merge must be migrated.
- Empty tile lists emit one `BATCH_INFO` packet at the source timestamp with
  `valid_count == 0`, `total_batches == 0`, no `TENSORS` packet, and no
  detections packet. Merge must be `BATCH_INFO`-driven and treat this packet as
  an empty source-frame result.
- Merge must emit final detections at `Timestamp(info.source_frame_timestamp)`,
  not at the synthetic batch timestamp. It must manage output timestamp bounds so
  delayed final outputs are legal and monotonic by source frame.
- The streaming calculator emits all batches for one source frame before
  emitting batches for the next source frame. This keeps merge output ordering
  deterministic and avoids interleaved frame completion.

This solves multi-tile frames, `T < cap`, `T == cap`, and `T > cap` without
timestamp monotonicity violations.

## Cache 1: TilePlan cache

Owner: `TileSpecToTilePlanCalculator`.

Value:

```cpp
TilePlan
```

Key:

- `max_tiles_per_frame`;
- tile count;
- for each `NormalizedRect`: `has_x_center`, `x_center`, `has_y_center`,
  `y_center`, `has_width`, `width`, `has_height`, `height`, `has_rotation`,
  `rotation`.

Notes:

- Current validation rejects non-zero rotation. Explicit zero rotation and absent
  rotation can produce equivalent plans but different keys; that is a harmless
  miss.
- On cache hit, the calculator still sends a fresh packet at the current
  timestamp. The value may be copied from the cached `TilePlan`; later, if
  packet-level sharing is useful, introduce a shared immutable payload helper.

Expected benefit:

- Small CPU and allocation savings for stable tile policies.
- Not enough by itself to make M8 worthwhile.

## Cache 2: tile-to-tensor matrix cache

Owner: `StreamingTilesToTensorBatchCalculator`.

Value:

```cpp
struct TileMatrixSet {
  std::vector<std::array<float, 16>> image_to_tensor;
  std::vector<std::array<float, 16>> tensor_to_image;
};
```

Key:

- source image width/height;
- tile count and all `TileGeometry` fields;
- effective pixel ROI for each tile after the exact CPU/GPU crop rounding and
  clamping rules (`TilePixelRoi`). Matrix values must describe the pixels
  actually sampled, not only the requested normalized tile rectangle;
- model input height/width/channels;
- batch mode only if it affects row layout;
- tensor layout (`BHWC` today);
- tensor dtype;
- keep-aspect/letterbox policy;
- normalization range only if matrix semantics include normalization-dependent
  coordinates (normally they should not);
- matrix convention version.

Matrix generation:

- Use the same row-major convention as `ImageToTensorCalculator`:
  image-normalized/input image coordinates -> tensor coordinates.
- Also store inverse matrices to avoid repeated inversion during merge.
- For axis-aligned tiles, matrix generation is scale + translation over the
  effective pixel ROI. CPU and GPU paths must either share the same ROI helper or
  prove equivalent rounding/clamping in tests.
- Boundary tiles that extend outside the source frame are projected using the
  clamped sampled ROI. The original requested tile rectangle may still be kept
  for diagnostics but must not drive the matrix.
- If rotated tiles are enabled later, use the same rotated-rect helper family as
  `ImageToTensorCalculator` and add rotation to `TileGeometry`.

Expected benefit:

- Avoids repeated matrix derivation/inversion for stable video tiling.
- More important than `TilePlan` cache, but still smaller than tensor/GPU buffer
  reuse.

## Cache 3: CPU tensor workspace and buffer reuse

Owner: `StreamingTilesToTensorBatchCalculator`, with framework support where
needed.

Current problem:

- The CPU path constructs a new `Tensor` per emitted batch and gets a CPU write
  view. For plain CPU buffers, current `Tensor` ownership does not expose a safe
  public "return this buffer to a pool when packet is released" hook.

M8 design:

1. Short-term safe improvement:
   - Reuse per-calculator temporary OpenCV `cv::Mat` workspaces for ROI/resized
     intermediates when shape/type match.
   - Avoid allocating temporary vectors inside the hot row loop.
   - Pre-size `TileBatchGeometry` vectors (`tile_indices`, matrices,
     geometries, effective pixel ROIs).
   - Construct `std::vector<Tensor>` with capacity 1 or replace output with a
     single `Tensor` stream if the downstream contract permits it.

2. Real CPU tensor-buffer pooling:
   - Extend `MemoryManager` or `Tensor` with a CPU buffer pool keyed by
     `(element_type, shape, alignment)`.
   - A `Tensor` allocated from the pool returns its CPU buffer when the Tensor is
     destroyed after packet release.
   - The pool is bounded by `max_cpu_tensor_workspaces`.
   - The calculator must never reuse a buffer by itself after `Send()`.

3. Padding discipline:
   - Fixed batch: zero rows `[valid_count, batch_capacity)`.
   - Dynamic batch: emit `N == valid_count`; no padding rows.

Expected benefit:

- Short-term workspace reuse reduces temporary OpenCV/vector allocations.
- True CPU tensor-buffer pooling reduces the expensive per-batch tensor backing
  allocation, but requires Tensor/MemoryManager ownership support.

## Cache 4: OpenGL tile-surface cache

Owner: GPU implementation of `StreamingTilesToTensorBatchCalculator`.

Resources:

- intermediate crop/resize render targets, if the implementation uses tile
  surfaces;
- shader/program objects for tiled crop/resize/normalize;
- sampler/state objects where applicable.

Key:

- `GpuResourceScopeKey` (defined below);
- source format and origin;
- tile output size;
- tensor dtype/layout/channel count;
- interpolation/border mode;
- keep-aspect/letterbox policy;
- shader variant flags.

Rules:

- Cache is context-scoped. Destroy all entries on context teardown.
- Do not cache pixel content. Reused surfaces are overwritten before read.
- If a surface is sent downstream or referenced by a tensor packet, it is not
  reusable until a release callback/fence proves downstream is done.
- If release tracking is unavailable, do not pool that resource; allocate per
  packet or use a bounded in-flight pool that only releases through Tensor/
  MemoryManager ownership.

Expected benefit:

- Avoids repeated GL texture/FBO/program setup in stable GPU tiling.

## Cache 5: OpenGL/AHWB tensor-buffer cache

Owner: `MemoryManager`/`Tensor`, consumed by the streaming calculator.

Resources:

- AHWB-backed tensors on Android where `MEDIAPIPE_TENSOR_USE_AHWB` is enabled;
- OpenGL SSBO or texture-backed tensors when release-safe pooling is added.

Key:

- `GpuResourceScopeKey` (defined below);
- tensor shape, dtype, layout;
- memory kind (`AHWB`, `OpenGlBuffer`, `OpenGlTexture2D`);
- backend/delegate compatibility;
- alignment/usage flags.

Lifecycle:

- Acquire buffer for a batch.
- Write tile rows on GPU.
- Attach release callback or framework pool ownership to the emitted `Tensor`.
- Downstream inference reads the same GPU/AHWB storage.
- Only after downstream release and GPU fence completion may the buffer return to
  the cache.

Important constraint:

- Existing AHWB paths already have release callback concepts. Non-AHWB OpenGL
  SSBO/texture pooling may need a Tensor/MemoryManager extension; do not
  implement it with calculator-local reuse unless packet release and GPU fence
  completion are both observable.

Expected benefit:

- This is the main allocation win in GPU mode.

## OpenGL resource scope key

Define a concrete key for context-scoped caches:

```cpp
struct GpuResourceScopeKey {
  const void* gl_context_identity = nullptr;
  const void* share_group_identity = nullptr;
  int api_version = 0;
};
```

Rules:

- Prefer a stable share-group identity from `GlContext` if MediaPipe exposes one.
- If no share-group id is available, use the current `GlContext*` identity as a
  conservative key. That may reduce sharing but avoids unsafe cross-context
  reuse.
- A resource created under one `GpuResourceScopeKey` must never be reused under a
  different key.
- Clear context-scoped pools in `Close()` and on any available GL context teardown
  hook. Tests must cover two distinct contexts not sharing cached resources.

## In-flight GPU/OpenGL ownership model

OpenGL cache capacity is not the same thing as reusable capacity. A buffer can
exist in the cache but still be in flight downstream. M8 must track ownership
explicitly.

Resource states:

```cpp
enum class GpuResourceState {
  kFree,              // safe to acquire for a new batch
  kAcquiredForWrite,  // owned by the current Process() call
  kSubmitted,         // GL writes enqueued; waiting on write fence/sync
  kInFlightDownstream,// emitted in a MediaPipe packet
  kReclaimable,       // packet released and GPU work complete
};
```

Lifecycle:

1. Acquire only `kFree` resources.
2. Move to `kAcquiredForWrite` while crop/resize/normalize writes tile rows.
3. After GL dispatch, insert the required memory barrier and write fence/sync,
   then move to `kSubmitted`.
4. When the tensor packet is sent, move to `kInFlightDownstream`.
5. Packet release marks downstream ownership complete. GPU fence completion marks
   GL work complete. Only when both are true may the resource become
   `kReclaimable` and then `kFree`.
6. If either signal is missing, the resource is not reusable.

In-flight budget:

- `max_in_flight_gpu_batches` bounds tensor batches emitted but not yet released.
- When `max_in_flight_gpu_batches == 0`, the effective in-flight limit is derived
  from `max_gpu_tensor_buffers` for GPU tensors and `max_tile_surfaces` for tile
  surfaces. If zero-copy is enabled with no finite buffer/surface capacity, graph
  initialization fails.
- If no free resource exists and live resources are below the effective limit,
  allocate a new pooled resource.
- If the effective limit is reached:
  - for CPU-only input, use the CPU path only when
    `allow_cpu_input_fallback == true`;
  - for GPU-backed input, do **not** read back to CPU unless
    `allow_gpu_readback_fallback == true`;
  - when fallback is not allowed, return `ResourceExhausted` with the current
    in-flight count, capacity, source timestamp, and batch index.
- Never block indefinitely inside `Process()` waiting for downstream release.

Stats must report `in_flight`, `reclaimed`, `pool_exhausted`, and
`gpu_to_cpu_fallbacks` in addition to hit/miss/eviction counts.

## Zero-copy OpenGL path

Add GPU-capable inputs to `StreamingTilesToTensorBatchCalculator`:

- CPU path: `IMAGE` as `ImageFrame` or CPU-backed `Image`.
- GPU path: `IMAGE_GPU` as `GpuBuffer`, or `IMAGE` as GPU-backed `Image`.

When `enable_gpu_zero_copy == true`:

1. If input is GPU-backed and compatible with the selected inference delegate,
   run tiled crop/resize/normalize in OpenGL.
2. Write directly into a GPU/AHWB-backed `Tensor` batch buffer.
3. Emit that tensor to inference without `GetCpuReadView()`, `GetCpuWriteView()`,
   `formats::MatView`, or CPU `cv::resize`.
4. Use GL fences/sync tokens so downstream reads observe completed writes.
5. If input is CPU-only:
   - if `allow_cpu_input_fallback == true`, use the CPU path;
   - otherwise fail with an actionable error.
6. If downstream inference is CPU-only, zero-copy is not available; fail or
   read back according to `allow_gpu_readback_fallback`.
7. Perf/zero-copy tests must set `allow_gpu_readback_fallback == false`; any CPU
   readback in that mode is a failure.

Batch row layout:

- One tensor batch contains rows `[0, valid_count)`.
- Each tile writes to its row offset in the same GPU buffer.
- The shader receives source texture, tile rect/matrix, output `H/W/C`, and row
  offset.
- Fixed-batch padding rows are cleared on GPU.

Expected benefit:

- Removes GPU->CPU readback, CPU crop/resize, CPU normalization, and CPU->GPU
  upload for GPU inference.
- This is the main runtime win for OpenGL-backed streaming.

## Implementation order

1. Contract repair and tests:
   - extend `TensorBatchInfo`;
   - support `T > batch_capacity`;
   - batch-local tile geometry/matrices;
   - make merge `BATCH_INFO`-driven and emit final output at source timestamp;
   - define and test empty-tile source frames;
   - update merge calculators and tests.

2. Shared cache utilities:
   - stable key builder;
   - collision-safe bounded LRU;
   - cache stats.

3. `TilePlan` cache:
   - add options;
   - negative validation;
   - cache-on/off correctness tests.

4. Matrix generation and matrix cache:
   - row-major matrix convention tests against `ImageToTensorCalculator`;
   - axis-aligned multi-tile tests;
   - effective pixel ROI rounding/clamping tests;
   - changed image size/tile/model geometry misses.

5. CPU allocation reduction:
   - pre-sized `TileBatchGeometry` vectors;
   - reusable OpenCV workspaces;
   - optional Tensor/MemoryManager CPU pool design patch if accepted.

6. OpenGL zero-copy:
   - add GPU input contract;
   - add GL tiled preprocessing shader;
   - write directly into GPU/AHWB tensor batch;
   - context-scoped surface/tensor caches;
   - define `GpuResourceScopeKey` and teardown cleanup;
   - fence/release lifecycle tests.

7. Perf and safety instrumentation:
   - cache hit/miss counters;
   - allocation counters or benchmark hooks;
   - CPU readback detector in GPU tests;
   - bounded memory tests.

## Testing

CPU correctness:

- cache off equals current behavior for existing tests;
- cache on equals cache off for single tile, multiple tiles, repeated frames;
- `T < cap`, `T == cap`, `T > cap`, empty tile list;
- `TensorBatchInfo` invariants: `batch_size`, `batch_capacity`, `valid_count`,
  and `geometry` vector lengths are consistent;
- merge emits final output at `source_frame_timestamp`, not synthetic
  `batch_timestamp`;
- empty tile list produces an empty source-frame output without requiring a
  tensor or detections packet;
- fixed-batch padding rows ignored;
- dynamic batch emits `N == valid_count`;
- changed tile rect/image size/model metadata invalidates matrix cache;
- negative capacities fail.

Matrix correctness:

- full-frame tile matrix equals identity-equivalent behavior;
- half-frame/overlap tiles project boxes back to expected full-frame coords;
- edge tiles that require rounding/clamping project using the effective sampled
  ROI;
- inverse matrix round-trip error within tolerance;
- matrix convention matches `ImageToTensorCalculator` row-major output.

CPU allocation/perf:

- repeated stable multi-tile frames show fewer temporary allocations after
  workspace reuse;
- if CPU Tensor pool is implemented, repeated frames reuse buffers only after
  packet release.

OpenGL/zero-copy:

- GPU-backed input does not call CPU `MatView`, `GetCpuReadView()`, or CPU
  `GetCpuWriteView()` in the GPU path;
- output tensor is ready on GPU/AHWB for the selected inference path;
- GL cache entries are context-scoped;
- two non-shared GL contexts do not reuse each other's cached resources;
- buffers are not reused before release callback/fence completion;
- in-flight GPU batches are bounded by `max_in_flight_gpu_batches` or the derived
  tensor/surface capacity;
- pool exhaustion behavior is tested for both fallback and `ResourceExhausted`
  modes;
- CPU input fallback and GPU readback fallback are separate and tested; zero-copy
  perf tests disable GPU readback fallback.

Perf acceptance:

- Stable CPU multi-tile video: measurable reduction in per-frame allocations and
  no regression in output.
- Stable GPU multi-tile video: no CPU readback and no per-frame GL tensor buffer
  allocation after warmup when pooling support is available.
- GPU stress test with slow downstream consumer proves in-flight resources are not
  reused before packet release.
- Cache stats output is typed, optional, cumulative, and does not allocate when
  disabled.
- Cache memory remains within configured capacities.

## Done criteria

- All new options default off.
- Existing CPU tests pass with caching off.
- Multi-tile and multi-batch tests pass.
- Matrix projection tests pass.
- Cache stats prove hits under stable tiling and misses on key changes.
- CPU workspace allocation reduction is measured.
- GPU zero-copy path is verified on a GL-capable environment, or explicitly
  marked not-tested on this machine with CPU-only tests passing.
- In-flight GPU resource accounting is tested under delayed downstream release.
- No stale pixels, no cross-frame batch mixing, no buffer reuse before downstream
  release.
