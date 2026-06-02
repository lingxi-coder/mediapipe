# Spec — Phase 3 (M8): Tile geometry/matrix cache

Date: 2026-06-02
Status: Design (pre-plan)
Branch: `dev` (long-lived integration branch; commit only when asked)
Roadmap: `docs/superpowers/specs/2026-06-01-roadmap.md` Phase 3 (M8); expands the
§14.1 sketch in `docs/superpowers/specs/2026-06-01-detection-core-yolo-obb-tiling-design.md`.

## Goal

A bounded, **default-off** cache that eliminates per-frame recomputation of
deterministic tiling geometry — the `TilePlan` and the tile-to-tensor affine
matrices — when the inputs (image size, tile layout, model geometry) are stable
across frames (the common case in video/streaming with a fixed tiling policy).
CPU-only; fully build + test verifiable on this machine.

## Scope

**In scope:** cache #1 from §14.1 — tile geometry/matrix caching, via a shared
header-only helper used by the two existing tiling calculators.

**Out of scope / deferred (unchanged from §14.1):**
- #2 tile-surface cache, #3 input-tensor cache — GPU/GL-bound; need a device env.
- #5 model/delegate on-disk kernel cache — `cached_kernel_path` /
  `serialized_model_dir` / NNAPI `cache_dir` are GPU/NNAPI-**delegate** knobs on
  `InferenceCalculatorOptions`; their runtime effect cannot be exercised in this
  CPU-only (`MEDIAPIPE_DISABLE_GPU=1`) environment, so #5 is reclassified with the
  deferred delegate/GPU caches.
- #4 output/decoder-buffer cache — a separate follow-on spec.

## Background (verified against the shipped Group-1 code)

- `mediapipe/calculators/tensor/tiling_types.h` — `TileGeometry`, `TilePlan`
  (`std::vector<TileGeometry>`), `TensorBatchInfo`.
- `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.{cc,proto}` —
  builds `TilePlan` from the per-frame tile-spec list; validates + assigns
  `tile_index`. Current option: `max_tiles_per_frame` (default 0 = no limit).
- `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.{cc,proto}` —
  crops/resizes/normalizes tiles into batch tensor rows and produces the
  tile-to-tensor affine matrices. Reads `InferenceMetadata` as a **side packet**
  (constant for the graph lifetime: input H/W/C, dtype, layout, batch_capacity).
- These are shipped, CPU, tested (`*_test.cc` exist). The cache attaches without
  changing any calculator's stream contract (a requirement called out in §8 of the
  detection-core spec).

## Design

### Section 1 — Files
- **Create** `mediapipe/calculators/tensor/tiling_geometry_cache.h` (header-only):
  `BoundedLruCache<Value>` + a stable 64-bit key builder.
- **Create** `mediapipe/calculators/tensor/tiling_geometry_cache_test.cc`.
- **Modify** `tile_spec_to_tile_plan_calculator.proto` — add
  `optional int32 max_cached_tile_plans = 2 [default = 0];`.
- **Modify** `tile_spec_to_tile_plan_calculator.cc` — memoize the computed
  `TilePlan` keyed by the tile-spec list + `max_tiles_per_frame`.
- **Modify** `streaming_tiles_to_tensor_batch_calculator.proto` — add
  `optional int32 max_cached_tile_geometries = <next free field> [default = 0];`.
- **Modify** `streaming_tiles_to_tensor_batch_calculator.cc` — memoize the
  tile-to-tensor matrix set keyed by image size + tile geometry (+ the constant
  model geometry/layout/normalization captured at `Open()`).
- **Modify** the two `*_test.cc` to add cache-on correctness + hit/miss cases.
- **Modify** the calculators' `BUILD` rule to depend on `:tiling_geometry_cache`
  (a new `cc_library` for the header) and the test deps.

### Section 2 — Shared helper API (`tiling_geometry_cache.h`)
```cpp
namespace mediapipe {

// Bounded LRU memo. capacity == 0 ⇒ disabled (enabled() == false): Get always
// misses and Put is a no-op, so callers behave exactly as if there were no cache.
// Calculator-local and single-threaded (one Process at a time); no locking.
template <typename Value>
class BoundedLruCache {
 public:
  explicit BoundedLruCache(size_t capacity);
  bool enabled() const;                 // capacity_ > 0
  const Value* Get(uint64_t key);       // nullptr on miss; promotes on hit
  void Put(uint64_t key, Value value);  // inserts, evicts LRU past capacity
 private:
  // intrusive LRU list + absl::flat_hash_map<uint64_t, iterator>
};

// Stable key folding. Floats are bit-cast to uint32 before mixing so identical
// geometry hashes identically across calls/process runs (no float drift).
class GeometryKeyBuilder {
 public:
  GeometryKeyBuilder& AddInt(int64_t v);
  GeometryKeyBuilder& AddFloat(float v);   // absl::bit_cast<uint32_t>
  GeometryKeyBuilder& AddString(absl::string_view v);
  uint64_t Build() const;                  // e.g. absl::HashOf / FNV-1a fold
};

}  // namespace mediapipe
```
Rationale: one small, isolated, unit-testable unit; each calculator gains only a
lookup/store wrapper. `Value` is `TilePlan` for one calculator and the
matrix-set struct for the other.

### Section 3 — Keying & invalidation
- **TilePlan cache** (`tile_spec_to_tile_plan`): key =
  `AddInt(max_tiles_per_frame)` + for each input tile rect
  `AddFloat(cx).AddFloat(cy).AddFloat(w).AddFloat(h)` (+ rotation if present) +
  `AddInt(tile_count)`. The `TilePlan` output is a pure function of these.
- **Matrix cache** (`streaming_tiles`): key = `AddInt(image_w).AddInt(image_h)` +
  the tile geometry list + the constant model fields captured at `Open()`
  (`input_h/w/c`, layout, dtype, keep-aspect, normalization params). Because
  `InferenceMetadata` is a constant side packet and options are fixed at `Open()`,
  the per-frame-varying key reduces to image size + tile geometry → a constant key
  under stable video tiling ⇒ hit after the first frame.
  - **CRITICAL — caches the affine matrices ONLY, never the packed tensor.** The
    calculator emits both the tile-to-tensor matrices (geometry, frame-independent)
    and the batch `Tensor` (packed **pixels**, frame content). Only the matrix set
    is the cache `Value`. The crop/resize/normalize that fills the tensor rows with
    actual pixels **always runs every frame** — caching it would serve stale
    pixels. The cache thus saves the matrix derivation, not the per-pixel packing.
- **Invalidation is implicit:** every field in §14.1's invalidation list (image
  size, tile policy, model id/geometry, layout, dtype, thresholds, keep-aspect) is
  part of the key, so any change ⇒ key miss ⇒ recompute. There is **no explicit
  invalidation code** and no time-based expiry.

### Section 4 — Default-off wiring
Both new options default to **0**. `BoundedLruCache(0).enabled() == false`, so
`Get` always misses and `Put` is a no-op: the calculator computes exactly as it
does today. Caching is strictly opt-in (roadmap: "default off until covered by
tests"). With the option unset, there is zero behavior change and zero added
per-frame allocation.

### Section 5 — Correctness & invariants
- The cache stores **pure deterministic geometry** — no detections, no pixels, no
  cross-frame data. A hit returns the identical value the calculator would have
  computed, so it is **correctness-neutral by construction** and cannot affect the
  one-global-NMS-per-frame or no-cross-frame-mixing invariants (those govern the
  deferred detection/tensor caches, not this one).
- Bounded memory: at most `capacity` entries; LRU eviction. Geometry entries are
  small (vectors of floats/ints). **The cache never retains image, pixel, or
  `Tensor` data** — only `TilePlan`s and affine matrices. This is both a memory
  bound and the guarantee that no stale pixel content can be served.

### Section 6 — Testing (all CPU, `--define MEDIAPIPE_DISABLE_GPU=1`)
- `tiling_geometry_cache_test.cc`:
  - `BoundedLruCache`: hit, miss, LRU eviction at capacity, capacity-0 disabled
    (Get misses, Put no-ops).
  - `GeometryKeyBuilder`: identical inputs ⇒ equal key; any single field change ⇒
    different key; float bit-cast stability (e.g. `0.1f` repeatable).
- `tile_spec_to_tile_plan_calculator_test.cc` (extend): with
  `max_cached_tile_plans > 0`, feeding the same tile specs across N frames yields
  output identical to caching-off; changing a tile rect forces a recompute (still
  correct). 
- `streaming_tiles_to_tensor_batch_calculator_test.cc` (extend): with
  `max_cached_tile_geometries > 0`, repeated identical (image size, TilePlan)
  frames produce matrices/tensors identical to caching-off; a changed image size
  forces recompute.
- Cross-cutting correctness test: **cached output == uncached output** for a
  representative multi-tile sequence (the core safety claim).

### Section 7 — Build order
1. `tiling_geometry_cache.h` + `tiling_geometry_cache_test.cc` + `cc_library` —
   pure unit, no calculator deps. Build + test.
2. TilePlan cache: proto option → `.cc` memoization → extend test. Build + test.
3. Matrix cache: proto option → `.cc` memoization → extend test. Build + test.

## Done criteria
- `tiling_geometry_cache` library + test build and pass.
- Both calculators build; their tests pass with caching **off** (unchanged
  behavior) and **on** (output identical to off across repeated frames; recompute
  on input change).
- No stream-contract change to either calculator; no change to any other
  calculator or the detector graphs.
- Defaults keep caching off; `git` tree clean.
