# Caching — Plan 3: CPU tensor-buffer pool + workspace reuse + cache-stats observability

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real, bounded, default-off CPU tensor-buffer pool to the core `Tensor`/`MemoryManager` framework (returning buffers on packet release), opt the tiled streaming calculator into it with reusable OpenCV workspaces, and expose cumulative cache statistics from both tiling calculators via an optional `CACHE_STATS` output.

**Architecture:** A new `CpuBufferPool` (bounded LRU-ish free-list keyed by `(bytes, alignment)`, thread-safe) is owned by `MemoryManager` behind a `std::shared_ptr` and **co-owned by every `Tensor`** that uses it — exactly mirroring the existing AHWB `hardware_buffer_pool_` pattern, so the pool always outlives in-flight tensors regardless of teardown order. `Tensor::AllocateCpuBuffer`/`FreeCpuBuffer` route through the pool when present; since `~Tensor()` runs when a packet is released, returning the buffer in `FreeCpuBuffer` *is* the "return-on-release" semantic. Everything is default-off: no `MemoryManager`, or a `MemoryManager` with a disabled (capacity-0 ⇒ null) pool, behaves byte-for-byte like today. Separately, the tiling calculators gain a `TilingCacheStats` `CACHE_STATS` output that surfaces the Plan-1/Plan-2 cache hit rates.

**Tech Stack:** C++20, MediaPipe `Tensor`/`MemoryManager`/`kMemoryManagerService`, api2 calculators, `aligned_malloc_and_free`, Bazel (`--define MEDIAPIPE_DISABLE_GPU=1`), GoogleTest.

**Spec:** `docs/superpowers/specs/2026-06-02-caching-tile-geometry-design.md` — "Cache 3: CPU tensor workspace and buffer reuse" (lines 345–383), "Public options" / "Stats exposure" (lines 86–143). User scope decision (2026-06-02): include the core `Tensor`/`MemoryManager` CPU pool.

**Builds on:** Plan 1 (`tiling_cache_utils.h`: `BoundedLruCache`, `CacheStats`, shipped) + Plan 2 (matrix cache, multi-batch, shipped). **Followed by:** Plan 4 (OpenGL/AHWB zero-copy).

---

## Reference facts (verified against shipped code)

- `mediapipe/framework/formats/tensor.cc`:
  - `AllocateCpuBuffer() const` (line ~767): after an AHWB early-return (`#ifdef MEDIAPIPE_TENSOR_USE_AHWB`), allocates `cpu_buffer_` with `aligned_malloc(std::max(memory_alignment_, bytes()), memory_alignment_)` when `memory_alignment_ > 0`, else `malloc(bytes())`. `RET_CHECK(cpu_buffer_)`.
  - `FreeCpuBuffer() const` (line ~792): `aligned_free(cpu_buffer_)` when `memory_alignment_ > 0` else `free(cpu_buffer_)`; sets `cpu_buffer_ = nullptr`.
  - `~Tensor()` → `Invalidate()` → `FreeCpuBuffer()` (both Metal and non-Metal `Invalidate` call it). So a buffer returns to the pool exactly when the Tensor is destroyed, i.e. when its packet is released.
  - Constructors (lines 478–503) take `MemoryManager* memory_manager`; today they only do `hardware_buffer_pool_ = memory_manager->GetAndroidHardwareBufferPool();` under `#ifdef MEDIAPIPE_TENSOR_USE_AHWB`.
  - `Move(Tensor* src)` (line 444) transfers `cpu_buffer_ = std::exchange(src->cpu_buffer_, nullptr)`, `memory_alignment_`, etc. New members MUST be moved here too.
  - `aligned_malloc`/`aligned_free` come from `#include "mediapipe/framework/port/aligned_malloc_and_free.h"`.
  - Members: `mutable void* cpu_buffer_ = nullptr;` (tensor.h ~570), `int memory_alignment_;`, `std::shared_ptr<HardwareBufferPool> hardware_buffer_pool_;` (AHWB only).
- `mediapipe/framework/memory_manager.h`: class `MemoryManager`; default ctor; under AHWB holds `std::shared_ptr<HardwareBufferPool>`. **Empty in non-AHWB CPU builds today.** Used via `kMemoryManagerService` (`mediapipe/framework/memory_manager_service.h`, `GraphService<MemoryManager>`, `kDisallowDefaultInitialization`).
- `Tensor::Shape::num_elements()` and `Tensor::bytes()` give the buffer size; `Tensor` is move-only, non-copyable.
- `mediapipe/calculators/tensor/tiling_cache_utils.h` (Plan 1): `struct CacheStats { int64_t hits, misses, inserts, evictions; }` (NOTE: Plan-1 shipped `CacheStats` has ONLY these four fields, not the extended GPU fields in the design doc — use the shipped four). `BoundedLruCache<V>::stats()` returns `const CacheStats&`.
- `StreamingTilesToTensorBatchCalculator` (Plan 2): has `BoundedLruCache<std::shared_ptr<const TileBatchGeometry>> matrix_cache_`; option `max_cached_tile_matrices` (proto field 2). Constructs `Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{N,H,W,C})` with NO memory_manager. Has reusable-per-row `cv::Mat roi/resized/f32` locals in the pixel loop. proto: `dynamic_batch=1`, `max_cached_tile_matrices=2`.
- `TileSpecToTilePlanCalculator` (Plan 1): has a `BoundedLruCache<TilePlan>`; option `max_cached_tile_plans` (proto field 2). proto fields used: `max_tiles_per_frame=1`, `max_cached_tile_plans=2`.
- BUILD: `//mediapipe/framework:memory_manager` (target name — confirm with `grep -n "name = \"memory_manager\"" mediapipe/framework/BUILD`), `//mediapipe/framework/formats:tensor`, `//mediapipe/framework/port:aligned_malloc_and_free`. The tensor cc_library is `//mediapipe/framework/formats:tensor`; its test is `//mediapipe/framework/formats:tensor_test` (a large regression suite — must stay green).

## Cross-cutting conventions

- Build/test: `bazel {build,test} -c opt --define MEDIAPIPE_DISABLE_GPU=1 <target> --test_output=errors`.
- In-editor clang errors are FALSE POSITIVES; only bazel is authoritative.
- **Default-off is non-negotiable.** With no `MemoryManager`, or a `MemoryManager` whose CPU pool capacity is 0 (⇒ the pool `shared_ptr` is null), `Tensor` CPU allocation must be byte-for-byte identical to today. The full `//mediapipe/framework/formats:tensor_test` suite must pass unchanged after Task 3.
- New cache/pool options default to `0`/`false`. Negative capacities rejected with `RET_CHECK_GE(..., 0)` before any signed→unsigned conversion.
- License header: full Apache-2.0 header (copy from any neighboring file) on every new file.
- Each task commits separately; co-author trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Branch is `dev` (commit on `dev`, do NOT branch).

## File Structure

- **Create** `mediapipe/framework/formats/cpu_buffer_pool.{h,cc}` + `cpu_buffer_pool_test.cc` — the standalone bounded CPU buffer pool (size+alignment keyed free-list, thread-safe, stats). Self-contained + unit-testable.
- **Modify** `mediapipe/framework/memory_manager.h` — own an optional `std::shared_ptr<CpuBufferPool>` (all builds); ctor capacity arg; accessor.
- **Modify** `mediapipe/framework/formats/tensor.{h,cc}` — co-own `cpu_buffer_pool_` (shared_ptr) + `bool cpu_buffer_from_pool_`; route `AllocateCpuBuffer`/`FreeCpuBuffer`; transfer in `Move`.
- **Modify** `mediapipe/framework/formats/tensor_test.cc` — add a pool reuse/round-trip test.
- **Modify** `streaming_tiles_to_tensor_batch_calculator.{proto,cc}` + test — opt into the pool via a calculator-local `MemoryManager`; reusable `cv::Mat` workspace members; `emit_cache_stats` + `CACHE_STATS`.
- **Modify** `tile_spec_to_tile_plan_calculator.{proto,cc}` + test — `emit_cache_stats` + `CACHE_STATS`.
- **Create** `mediapipe/calculators/tensor/tiling_cache_stats.h` — the `TilingCacheStats` packet struct shared by both calculators.
- **Modify** `mediapipe/calculators/tensor/BUILD` and `mediapipe/framework/{,formats/}BUILD` — deps + new targets.

---

### Task 1: `CpuBufferPool` — standalone bounded buffer pool

**Files:** Create `mediapipe/framework/formats/cpu_buffer_pool.{h,cc}` + `cpu_buffer_pool_test.cc`; modify `mediapipe/framework/formats/BUILD`.

- [ ] **Step 1: Write the failing test** `cpu_buffer_pool_test.cc` (full Apache header):

```cpp
#include "mediapipe/framework/formats/cpu_buffer_pool.h"

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

TEST(CpuBufferPoolTest, DisabledNeverPools) {
  CpuBufferPool pool(/*capacity=*/0);
  EXPECT_FALSE(pool.enabled());
  void* p = pool.Acquire(/*bytes=*/64, /*alignment=*/0);
  EXPECT_NE(p, nullptr);                 // still allocates
  EXPECT_FALSE(pool.Release(p, 64, 0));  // refuses to retain; caller-freed=false
  EXPECT_EQ(pool.stats().inserts, 0);
}

TEST(CpuBufferPoolTest, RecyclesSameSizeBuffer) {
  CpuBufferPool pool(/*capacity=*/2);
  void* a = pool.Acquire(256, 0);
  ASSERT_NE(a, nullptr);
  EXPECT_TRUE(pool.Release(a, 256, 0));   // retained
  void* b = pool.Acquire(256, 0);         // same key -> same pointer
  EXPECT_EQ(b, a);
  EXPECT_EQ(pool.stats().hits, 1);
  EXPECT_EQ(pool.stats().misses, 1);      // the first Acquire missed
  pool.Release(b, 256, 0);
}

TEST(CpuBufferPoolTest, DifferentKeyDoesNotAlias) {
  CpuBufferPool pool(2);
  void* a = pool.Acquire(256, 0);
  pool.Release(a, 256, 0);
  void* b = pool.Acquire(512, 0);   // different size -> miss, distinct buffer
  EXPECT_NE(b, a);
  void* c = pool.Acquire(256, 16);  // same size, different alignment -> miss
  EXPECT_NE(c, a);
  pool.Release(b, 512, 0);
  pool.Release(c, 256, 16);
}

TEST(CpuBufferPoolTest, CapacityBoundsRetainedBuffers) {
  CpuBufferPool pool(/*capacity=*/1);
  void* a = pool.Acquire(256, 0);
  void* b = pool.Acquire(256, 0);          // miss, both live
  EXPECT_TRUE(pool.Release(a, 256, 0));    // retained (pool now full: 1)
  EXPECT_FALSE(pool.Release(b, 256, 0));   // over capacity -> not retained
  EXPECT_EQ(pool.stats().evictions, 0);    // we never inserted-then-evicted
}

TEST(CpuBufferPoolTest, AlignedBuffersAreAligned) {
  CpuBufferPool pool(2);
  void* p = pool.Acquire(100, 64);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);
  pool.Release(p, 100, 64);
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 2: Write the header** `cpu_buffer_pool.h` (full Apache header):

```cpp
#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_CPU_BUFFER_POOL_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_CPU_BUFFER_POOL_H_

#include <cstddef>
#include <cstdint>
#include <list>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"

namespace mediapipe {

// Diagnostic counters for one CPU buffer pool.
struct CpuBufferPoolStats {
  int64_t hits = 0;       // Acquire served from the free-list
  int64_t misses = 0;     // Acquire had to allocate
  int64_t inserts = 0;    // Release retained a buffer
  int64_t evictions = 0;  // Release freed a buffer because the pool was full
};

// A bounded, thread-safe pool of raw CPU buffers keyed by (byte size,
// alignment). `capacity == 0` => disabled: Acquire always allocates, Release
// always frees (returns false), behaving exactly as if there were no pool.
// Allocation uses the same aligned_malloc/aligned_free routines as Tensor, so
// pooled buffers are interchangeable with directly-allocated ones.
//
// Lifetime contract: the pool is intended to be co-owned (shared_ptr) by every
// Tensor that draws from it, so it always outlives in-flight buffers. The pool
// frees everything it still holds on destruction.
class CpuBufferPool {
 public:
  explicit CpuBufferPool(size_t capacity) : capacity_(capacity) {}
  ~CpuBufferPool();

  CpuBufferPool(const CpuBufferPool&) = delete;
  CpuBufferPool& operator=(const CpuBufferPool&) = delete;

  bool enabled() const { return capacity_ > 0; }

  // Returns a buffer of exactly `bytes` with the given `alignment`
  // (alignment 0 => unaligned malloc, matching Tensor). Never returns nullptr
  // unless allocation itself fails. Served from the free-list on a key match,
  // otherwise freshly allocated.
  void* Acquire(size_t bytes, int alignment);

  // Returns `buffer` (which must have been produced by Acquire with the same
  // bytes+alignment) to the pool. Retains it if the pool has spare capacity and
  // returns true; otherwise frees it and returns false. A disabled pool always
  // frees and returns false.
  bool Release(void* buffer, size_t bytes, int alignment);

  CpuBufferPoolStats stats() const;

 private:
  struct Key {
    size_t bytes;
    int alignment;
    bool operator==(const Key& o) const {
      return bytes == o.bytes && alignment == o.alignment;
    }
    template <typename H>
    friend H AbslHashValue(H h, const Key& k) {
      return H::combine(std::move(h), k.bytes, k.alignment);
    }
  };
  static void* Allocate(size_t bytes, int alignment);
  static void Free(void* buffer, int alignment);

  mutable absl::Mutex mu_;
  const size_t capacity_;
  size_t retained_ ABSL_GUARDED_BY(mu_) = 0;  // total buffers held across keys
  absl::flat_hash_map<Key, std::list<void*>> free_ ABSL_GUARDED_BY(mu_);
  CpuBufferPoolStats stats_ ABSL_GUARDED_BY(mu_);
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_CPU_BUFFER_POOL_H_
```

- [ ] **Step 3: Write the impl** `cpu_buffer_pool.cc` (full Apache header). Match Tensor's allocation exactly: aligned path uses `aligned_malloc(std::max<size_t>(alignment, bytes), alignment)`.

```cpp
#include "mediapipe/framework/formats/cpu_buffer_pool.h"

#include <algorithm>
#include <cstdlib>

#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port/aligned_malloc_and_free.h"

namespace mediapipe {

void* CpuBufferPool::Allocate(size_t bytes, int alignment) {
  if (alignment > 0) {
    return aligned_malloc(std::max<size_t>(alignment, bytes), alignment);
  }
  return malloc(bytes);
}

void CpuBufferPool::Free(void* buffer, int alignment) {
  if (buffer == nullptr) return;
  if (alignment > 0) {
    aligned_free(buffer);
  } else {
    free(buffer);
  }
}

CpuBufferPool::~CpuBufferPool() {
  // capacity_ keyed by (bytes, alignment): free everything still retained.
  for (auto& [key, buffers] : free_) {
    for (void* b : buffers) Free(b, key.alignment);
  }
}

void* CpuBufferPool::Acquire(size_t bytes, int alignment) {
  if (capacity_ > 0) {
    absl::MutexLock lock(&mu_);
    auto it = free_.find(Key{bytes, alignment});
    if (it != free_.end() && !it->second.empty()) {
      void* b = it->second.back();
      it->second.pop_back();
      --retained_;
      ++stats_.hits;
      return b;
    }
    ++stats_.misses;
  }
  return Allocate(bytes, alignment);
}

bool CpuBufferPool::Release(void* buffer, size_t bytes, int alignment) {
  if (buffer == nullptr) return true;
  if (capacity_ > 0) {
    absl::MutexLock lock(&mu_);
    if (retained_ < capacity_) {
      free_[Key{bytes, alignment}].push_back(buffer);
      ++retained_;
      ++stats_.inserts;
      return true;
    }
    ++stats_.evictions;
  }
  Free(buffer, alignment);
  return false;
}

CpuBufferPoolStats CpuBufferPool::stats() const {
  absl::MutexLock lock(&mu_);
  return stats_;
}

}  // namespace mediapipe
```

- [ ] **Step 4: BUILD** — add to `mediapipe/framework/formats/BUILD`:

```python
cc_library(
    name = "cpu_buffer_pool",
    srcs = ["cpu_buffer_pool.cc"],
    hdrs = ["cpu_buffer_pool.h"],
    deps = [
        "//mediapipe/framework/port:aligned_malloc_and_free",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "cpu_buffer_pool_test",
    srcs = ["cpu_buffer_pool_test.cc"],
    deps = [
        ":cpu_buffer_pool",
        "//mediapipe/framework/port:gtest_main",
    ],
)
```
> Confirm the `aligned_malloc_and_free` target name with `grep -n "aligned_malloc_and_free" mediapipe/framework/port/BUILD`. Use the exact label it defines.

- [ ] **Step 5: Run the test**

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/framework/formats:cpu_buffer_pool_test --test_output=all`
Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add mediapipe/framework/formats/cpu_buffer_pool.h mediapipe/framework/formats/cpu_buffer_pool.cc mediapipe/framework/formats/cpu_buffer_pool_test.cc mediapipe/framework/formats/BUILD
git commit -m "$(printf 'feat(tensor): bounded thread-safe CPU buffer pool (default-off)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: Own the CPU pool in `MemoryManager`

**Files:** Modify `mediapipe/framework/memory_manager.h`; modify `mediapipe/framework/BUILD`.

- [ ] **Step 1: Add the pool member + ctor arg + accessor.** Edit `memory_manager.h`. Add `#include <cstddef>` and `#include "mediapipe/framework/formats/cpu_buffer_pool.h"` at the top of the includes (these are unconditional, NOT under the AHWB `#ifdef`). Change the class so a CPU pool is created when a non-zero capacity is requested. The CPU pool is available in ALL builds (not gated by `MEDIAPIPE_TENSOR_USE_AHWB`).

Replace the class body with:

```cpp
class MemoryManager {
 public:
  // cpu_buffer_pool_capacity == 0 (default) => no CPU pooling; GetCpuBufferPool()
  // returns nullptr and Tensor CPU allocation is unchanged.
  explicit MemoryManager(size_t cpu_buffer_pool_capacity = 0) {
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
    hardware_buffer_pool_ = std::make_shared<HardwareBufferPool>();
#endif
    if (cpu_buffer_pool_capacity > 0) {
      cpu_buffer_pool_ =
          std::make_shared<CpuBufferPool>(cpu_buffer_pool_capacity);
    }
  }

#ifdef MEDIAPIPE_TENSOR_USE_AHWB
  std::shared_ptr<HardwareBufferPool> GetAndroidHardwareBufferPool() const {
    return hardware_buffer_pool_;
  }

  explicit MemoryManager(const MultiPoolOptions& options,
                         size_t cpu_buffer_pool_capacity = 0)
      : hardware_buffer_pool_(std::make_shared<HardwareBufferPool>(options)) {
    if (cpu_buffer_pool_capacity > 0) {
      cpu_buffer_pool_ =
          std::make_shared<CpuBufferPool>(cpu_buffer_pool_capacity);
    }
  }
#endif

  // Null when CPU pooling is disabled.
  std::shared_ptr<CpuBufferPool> GetCpuBufferPool() const {
    return cpu_buffer_pool_;
  }

 private:
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
  std::shared_ptr<HardwareBufferPool> hardware_buffer_pool_;
#endif
  std::shared_ptr<CpuBufferPool> cpu_buffer_pool_;
};
```

(Keep the existing file-level comment block. Note the default ctor still works: `MemoryManager()` == `MemoryManager(0)` == no CPU pool, so existing AHWB callers are unaffected.)

- [ ] **Step 2: BUILD dep.** In `mediapipe/framework/BUILD`, add `"//mediapipe/framework/formats:cpu_buffer_pool"` to the `deps` of the `memory_manager` `cc_library` (find it: `grep -n "name = \"memory_manager\"" mediapipe/framework/BUILD`). It is a header-only-consuming dep but the target builds a header lib; add it unconditionally.

- [ ] **Step 3: Build the dependents**

Run: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/framework:memory_manager //mediapipe/framework:memory_manager_service`
Expected: builds.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/framework/memory_manager.h mediapipe/framework/BUILD
git commit -m "$(printf 'feat(tensor): MemoryManager owns optional CPU buffer pool (all builds)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 3: Route `Tensor` CPU allocation through the pool

**Files:** Modify `mediapipe/framework/formats/tensor.{h,cc}` + `tensor_test.cc`; modify `mediapipe/framework/formats/BUILD`.

This is the highest-regression task. The whole `//mediapipe/framework/formats:tensor_test` suite must stay green. Default-off: when `cpu_buffer_pool_` is null (no MemoryManager, or capacity-0 MemoryManager), behavior is byte-identical to today.

- [ ] **Step 1: Add members to `tensor.h`.** Near the other CPU members (around `mutable void* cpu_buffer_ = nullptr;`, ~line 570), add:

```cpp
  // Co-owned CPU buffer pool (null when pooling disabled). Mirrors the AHWB
  // hardware_buffer_pool_ pattern so the pool outlives this Tensor's buffer.
  std::shared_ptr<CpuBufferPool> cpu_buffer_pool_;
  // True when cpu_buffer_ was drawn from cpu_buffer_pool_ and must be returned
  // there (not free()d) in FreeCpuBuffer().
  mutable bool cpu_buffer_from_pool_ = false;
```

Add `#include "mediapipe/framework/formats/cpu_buffer_pool.h"` and `#include <memory>` to `tensor.h` includes if not already present. (`MemoryManager` is already forward-declared / included for the constructor param; confirm `cpu_buffer_pool.h` include compiles — it's lightweight.)

- [ ] **Step 2: Populate the pool in both constructors** (`tensor.cc` ~line 478 and ~490). After the existing AHWB block, add (UNCONDITIONALLY, both ctors):

```cpp
  if (memory_manager) {
    cpu_buffer_pool_ = memory_manager->GetCpuBufferPool();
  }
```

- [ ] **Step 3: Transfer the new members in `Move()`** (`tensor.cc` ~line 444). Add after the `cpu_buffer_ = std::exchange(...)` line:

```cpp
  cpu_buffer_pool_ = std::move(src->cpu_buffer_pool_);
  cpu_buffer_from_pool_ = std::exchange(src->cpu_buffer_from_pool_, false);
```

- [ ] **Step 4: Route `AllocateCpuBuffer()`** (`tensor.cc` ~line 767). Replace the non-Metal allocation block so the pool is tried first when present. The Metal branch (`MEDIAPIPE_METAL_ENABLED`) is unchanged.

```cpp
absl::Status Tensor::AllocateCpuBuffer() const {
  if (!cpu_buffer_) {
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
    if (use_ahwb_ && AllocateAHardwareBuffer().ok()) return absl::OkStatus();
#endif  // MEDIAPIPE_TENSOR_USE_AHWB
#if MEDIAPIPE_METAL_ENABLED
    cpu_buffer_ = AllocateVirtualMemory(bytes());
#else
    if (cpu_buffer_pool_ && cpu_buffer_pool_->enabled()) {
      cpu_buffer_ = cpu_buffer_pool_->Acquire(bytes(), memory_alignment_);
      cpu_buffer_from_pool_ = true;
    } else if (memory_alignment_ > 0) {
      cpu_buffer_ = aligned_malloc(std::max(memory_alignment_, bytes()),
                                   memory_alignment_);
    } else {
      cpu_buffer_ = malloc(bytes());
    }
    RET_CHECK(cpu_buffer_) << "Failed to allocate CPU buffer.";
#endif  // MEDIAPIPE_METAL_ENABLED
  }
  return absl::OkStatus();
}
```

- [ ] **Step 5: Route `FreeCpuBuffer()`** (`tensor.cc` ~line 792). Return pooled buffers to the pool:

```cpp
void Tensor::FreeCpuBuffer() const {
  if (cpu_buffer_ == nullptr) {
    return;
  }
#if MEDIAPIPE_METAL_ENABLED
  free(cpu_buffer_);
#else
  if (cpu_buffer_from_pool_) {
    cpu_buffer_pool_->Release(cpu_buffer_, bytes(), memory_alignment_);
  } else if (memory_alignment_ > 0) {
    aligned_free(cpu_buffer_);
  } else {
    free(cpu_buffer_);
  }
#endif  // MEDIAPIPE_METAL_ENABLED
  cpu_buffer_from_pool_ = false;
  cpu_buffer_ = nullptr;
}
```

> Note: `bytes()` and `memory_alignment_` are unchanged between Allocate and Free for a given Tensor (they depend only on shape+dtype+alignment, fixed at construction), so the Release key matches the Acquire key exactly.

- [ ] **Step 6: BUILD dep.** In `mediapipe/framework/formats/BUILD`, add `":cpu_buffer_pool"` to the `tensor` `cc_library` deps.

- [ ] **Step 7: Add the reuse round-trip test** to `tensor_test.cc`. This proves real pooling with packet-release == Tensor-destruction semantics. Add near the other CPU tests (match the file's namespace/style; include `"mediapipe/framework/memory_manager.h"`):

```cpp
TEST(TensorCpuPoolTest, DestroyedTensorReturnsBufferToPoolAndIsReused) {
  MemoryManager mm(/*cpu_buffer_pool_capacity=*/2);
  void* first_ptr = nullptr;
  {
    Tensor t(Tensor::ElementType::kFloat32, Tensor::Shape{16}, &mm);
    auto view = t.GetCpuWriteView();
    first_ptr = view.buffer<float>();
    ASSERT_NE(first_ptr, nullptr);
  }  // t destroyed here -> buffer returned to pool (simulates packet release).
  Tensor t2(Tensor::ElementType::kFloat32, Tensor::Shape{16}, &mm);
  auto view2 = t2.GetCpuWriteView();
  EXPECT_EQ(view2.buffer<float>(), first_ptr);  // same buffer reused
}

TEST(TensorCpuPoolTest, NullPoolDisabledIsUnchanged) {
  MemoryManager mm(/*cpu_buffer_pool_capacity=*/0);
  EXPECT_EQ(mm.GetCpuBufferPool(), nullptr);
  Tensor t(Tensor::ElementType::kFloat32, Tensor::Shape{16}, &mm);
  auto view = t.GetCpuWriteView();
  EXPECT_NE(view.buffer<float>(), nullptr);  // allocates normally, no crash
}
```
Add `"//mediapipe/framework:memory_manager"` to the `tensor_test` target deps if not already present (`grep -n "name = \"tensor_test\"" mediapipe/framework/formats/BUILD`).

- [ ] **Step 8: Run the FULL tensor suite (regression gate) + the new tests**

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/framework/formats:tensor_test --test_output=errors`
Expected: ALL pass (existing + 2 new). If ANY pre-existing test regresses, STOP — the default-off guarantee is violated; do not loosen tests.

- [ ] **Step 9: Commit**

```bash
git add mediapipe/framework/formats/tensor.h mediapipe/framework/formats/tensor.cc mediapipe/framework/formats/tensor_test.cc mediapipe/framework/formats/BUILD
git commit -m "$(printf 'feat(tensor): route CPU buffer alloc/free through optional pool on release\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 4: Streaming calc opts into the CPU pool + reusable OpenCV workspaces

**Files:** Modify `streaming_tiles_to_tensor_batch_calculator.{proto,cc}` + test; modify `mediapipe/calculators/tensor/BUILD`.

- [ ] **Step 1: proto** — add `optional int32 max_cpu_tensor_workspaces = 3 [default = 0];` (field 3 is next free; fields 1,2 are `dynamic_batch`,`max_cached_tile_matrices`). Comment: 0 = disabled (per-batch alloc, unchanged); N>0 = pool up to N batch tensor buffers in a calculator-local MemoryManager, reused after each batch packet is released.

- [ ] **Step 2: cc** — in `Open()`, after the existing matrix-cache init: `RET_CHECK_GE(options_.max_cpu_tensor_workspaces(), 0);` and, when `> 0`, construct a calculator-local pool-owning MemoryManager:

```cpp
    if (options_.max_cpu_tensor_workspaces() > 0) {
      memory_manager_ = std::make_shared<MemoryManager>(
          static_cast<size_t>(options_.max_cpu_tensor_workspaces()));
    }
```

Add members + includes:
```cpp
#include <memory>
#include "mediapipe/framework/memory_manager.h"
// ...
  std::shared_ptr<MemoryManager> memory_manager_;  // null unless pooling enabled
  cv::Mat resized_workspace_;  // reused across rows/batches when shape matches
  cv::Mat f32_workspace_;
```

Change the `Tensor` construction in `Process()` to pass the manager (nullptr when disabled is fine — the ctor accepts a null `MemoryManager*`):

```cpp
      Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{N, H, W, C},
                    memory_manager_.get());
```

In the pixel loop, reuse the `cv::Mat` workspaces instead of declaring `resized`/`f32` locals each row (OpenCV `resize`/`convertTo` will reuse the buffer when the size/type already match):

```cpp
      for (int r = 0; r < rows; ++r) {
        const TilePixelRoi& proi = geom->effective_pixel_rois[r];
        cv::Mat roi = src(cv::Rect(proi.x, proi.y, proi.width, proi.height));
        cv::resize(roi, resized_workspace_, cv::Size(W, H));
        resized_workspace_.convertTo(f32_workspace_, CV_32FC(C), 1.0 / 255.0);
        std::memcpy(buf + static_cast<size_t>(r) * H * W * C,
                    f32_workspace_.ptr<float>(0), sizeof(float) * H * W * C);
      }
```

- [ ] **Step 3: BUILD** — add `"//mediapipe/framework:memory_manager"` to the calculator's cc_library deps.

- [ ] **Step 4: test** — add `CpuPoolOnVsOffIdenticalResults`: run identical (image, plan) for several frames with `max_cpu_tensor_workspaces=2` vs `0`; assert output tensors byte-identical and BATCH_INFO geometry identical. (Buffer-reuse itself is proven at the Tensor level in Task 3; here we only prove the pooled path produces identical results and does not crash across multiple frames.) Build + test:

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test --test_output=all`
Expected: all existing tests + the new one PASS.

- [ ] **Step 5: commit**

```bash
git add mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.proto mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'feat(tensor-cache): opt streaming calc into CPU tensor pool + cv::Mat workspace reuse\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 5: `TilingCacheStats` struct + `CACHE_STATS` on the streaming calc

**Files:** Create `mediapipe/calculators/tensor/tiling_cache_stats.h`; modify `streaming_tiles_to_tensor_batch_calculator.{proto,cc}` + test; modify `BUILD`.

- [ ] **Step 1: Create the shared struct** `tiling_cache_stats.h` (full Apache header). Use the shipped `CacheStats` (4 fields) from `tiling_cache_utils.h`.

```cpp
#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TILING_CACHE_STATS_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TILING_CACHE_STATS_H_

#include "mediapipe/calculators/tensor/tiling_cache_utils.h"

namespace mediapipe {

// Diagnostic-only snapshot of the tiling caches. Emitted on an optional
// CACHE_STATS output; never used to align inference or merge streams.
struct TilingCacheStats {
  CacheStats tile_plan;
  CacheStats tile_matrix;
  CpuBufferPoolStats cpu_tensor_pool;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TILING_CACHE_STATS_H_
```
> This pulls `CpuBufferPoolStats` from `cpu_buffer_pool.h` — add `#include "mediapipe/framework/formats/cpu_buffer_pool.h"`. (Keep `tile_surface`/`gpu_tensor_buffer` fields OUT until Plan 4 adds those caches — YAGNI.)

- [ ] **Step 2: proto** — add `optional bool emit_cache_stats = 4 [default = false];` (field 4). Comment: when true AND a `CACHE_STATS` output is connected, emit cumulative stats once per source frame at the source timestamp; must not allocate on the hot path when false/disconnected.

- [ ] **Step 3: cc** — add an optional output `static constexpr Output<TilingCacheStats>::Optional kOutStats{"CACHE_STATS"};` to the contract (api2 optional output syntax — verify the exact form against an existing optional output in the repo, e.g. `grep -rn "::Optional" mediapipe/calculators | head`). Add the includes for `tiling_cache_stats.h` and `tiling_cache_utils.h` (already present). When `options_.emit_cache_stats()` AND `kOutStats(cc).IsConnected()`, after emitting all batches for the frame, send one `TilingCacheStats` at the source timestamp:

```cpp
    if (options_.emit_cache_stats() && kOutStats(cc).IsConnected()) {
      TilingCacheStats stats;
      stats.tile_matrix = matrix_cache_.stats();
      if (memory_manager_ && memory_manager_->GetCpuBufferPool()) {
        stats.cpu_tensor_pool = memory_manager_->GetCpuBufferPool()->stats();
      }
      kOutStats(cc).Send(mediapipe::api2::MakePacket<TilingCacheStats>(stats)
                             .At(Timestamp(ts)));
    }
```
(Emit at the source-frame input `ts`; this calc is single-Process-per-frame on the input timestamp for stats purposes. `tile_plan` stays default — it belongs to the other calculator.)

- [ ] **Step 4: BUILD** — new `cc_library(name="tiling_cache_stats", hdrs=["tiling_cache_stats.h"], deps=[":tiling_cache_utils", "//mediapipe/framework/formats:cpu_buffer_pool"])`; add `:tiling_cache_stats` to the streaming calc cc_library + test deps.

- [ ] **Step 5: test** — `CacheStatsReportsMatrixHits`: connect `CACHE_STATS`, feed the SAME (image, plan) twice with `max_cached_tile_matrices=4` and `emit_cache_stats=true`; assert the last CACHE_STATS packet shows `tile_matrix.hits >= 1` and `tile_matrix.misses >= 1`. This is the direct cache-hit assertion the Plan-2 Task-4 review wanted. Build + test.

- [ ] **Step 6: commit**

```bash
git add mediapipe/calculators/tensor/tiling_cache_stats.h mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.proto mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'feat(tensor-cache): optional CACHE_STATS output exposes matrix + CPU pool hit rates\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 6: `CACHE_STATS` on `TileSpecToTilePlanCalculator`

**Files:** Modify `tile_spec_to_tile_plan_calculator.{proto,cc}` + test; modify `BUILD`.

- [ ] **Step 1: proto** — add `optional bool emit_cache_stats = 3 [default = false];` (field 3; fields 1,2 are `max_tiles_per_frame`,`max_cached_tile_plans`).

- [ ] **Step 2: cc** — add optional `Output<TilingCacheStats>::Optional kOutStats{"CACHE_STATS"};`, include `tiling_cache_stats.h`. When enabled + connected, after sending the TilePlan, emit `TilingCacheStats` with `stats.tile_plan = tile_plan_cache_.stats();` at the input timestamp. (Confirm the cache member name in the calculator; the Plan-1 cache member — `grep -n "BoundedLruCache" mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.cc`.)

- [ ] **Step 3: BUILD** — add `:tiling_cache_stats` to the calculator cc_library + test deps.

- [ ] **Step 4: test** — `CacheStatsReportsTilePlanHits`: feed the same tile list twice with `max_cached_tile_plans=4` + `emit_cache_stats=true`, connect `CACHE_STATS`, assert `tile_plan.hits >= 1`. Build + test.

- [ ] **Step 5: commit**

```bash
git add mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.proto mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.cc mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'feat(tensor-cache): optional CACHE_STATS output on TileSpecToTilePlanCalculator\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Self-review checklist (run before final review)
- CPU pool is default-off: `MemoryManager()` / capacity-0 ⇒ `GetCpuBufferPool() == nullptr` ⇒ Tensor path byte-identical; full `tensor_test` suite green.
- Pool is co-owned (`shared_ptr`) by Tensor like AHWB, so no use-after-free at teardown regardless of order.
- `FreeCpuBuffer` returns pooled buffers to the pool (not `free()`); `bytes()`+`memory_alignment_` key matches Acquire.
- `Move()` transfers `cpu_buffer_pool_` + `cpu_buffer_from_pool_`.
- Streaming calc CPU-pool path produces byte-identical tensors vs. disabled; `cv::Mat` workspace reuse preserves output.
- `CACHE_STATS` is optional, diagnostic-only, emitted at the frame timestamp, and does not allocate when disabled/disconnected.
- All new options default to 0/false; negative capacities `RET_CHECK_GE`-rejected.

## Final verification
- [ ] `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/framework/formats:cpu_buffer_pool_test //mediapipe/framework/formats:tensor_test //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test //mediapipe/calculators/tensor:tile_spec_to_tile_plan_calculator_test //mediapipe/calculators/tensor:merge_tile_detections_accumulator_calculator_test //mediapipe/calculators/tensor:tiled_obb_pipeline_test --test_output=errors` — all pass.
- [ ] `git status` clean.

## Done criteria
- Bounded, thread-safe, default-off `CpuBufferPool` shipped + unit-tested.
- `MemoryManager` optionally owns it; `Tensor` co-owns + uses it, returning buffers on destruction (packet release); no regression in the tensor suite.
- Streaming calc opts in via a calculator-local pool-owning `MemoryManager` and reuses `cv::Mat` workspaces; output unchanged.
- Both tiling calculators expose cumulative `TilingCacheStats` on an optional, allocation-free-when-disabled `CACHE_STATS` output; tests assert real cache hits.
- True CPU buffer pooling reuse proven at the Tensor level (destroyed tensor's buffer reused by the next same-shape allocation).
