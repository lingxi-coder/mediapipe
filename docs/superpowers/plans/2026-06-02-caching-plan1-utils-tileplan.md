# Caching — Plan 1: shared cache utils + TilePlan cache

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the collision-safe bounded-LRU cache primitive and the first concrete cache (TilePlan memoization) — a default-off, additive, fully desktop-build+test-verifiable foundation for the merged caching/zero-copy design.

**Architecture:** A header-only cache utility (`tiling_cache_utils.h`: `StableKeyBuilder` → byte-exact `StableCacheKey`, `BoundedLruCache<Value>`, `CacheStats`) used by tiling calculators. The first user is `TileSpecToTilePlanCalculator`, which memoizes its deterministic `TilePlan` keyed by the input tile rects. Capacity 0 ⇒ disabled ⇒ byte-identical current behavior.

**Tech Stack:** C++20, MediaPipe api2 calculators, Bazel (`--define MEDIAPIPE_DISABLE_GPU=1`), absl, GoogleTest.

**Spec:** `docs/superpowers/specs/2026-06-02-caching-tile-geometry-design.md` (the merged cache + zero-copy design).

---

## Plan sequence (the full merged design, in dependency order)

This is **Plan 1 of 4**. The merged spec is implemented as sequential plans, each producing working, testable software, all under the one unified design (which keeps the shared `TensorBatchInfo`/matrix/buffer contracts aligned):

1. **Plan 1 (this file): shared cache utils + TilePlan cache.** CPU, additive, no contract change. Desktop build+test.
2. **Plan 2: tiled multi-batch + matrix contract repair + matrix cache.** Implements `T > batch_capacity` emission, tile-to-tensor matrix generation, batch-local geometry in `TensorBatchInfo`, the `BATCH_INFO`-driven merge migration (public contract change), and the matrix cache. CPU, desktop build+test.
3. **Plan 3: CPU tensor workspace + buffer pooling.** Workspace reuse (safe) + the `Tensor`/`MemoryManager` CPU pool extension (core-framework; design-gated). CPU, desktop build+test.
4. **Plan 4: OpenGL/AHWB zero-copy.** GPU tiled preprocessing → GPU/AHWB tensor batch, tile-surface + tensor-buffer caches, in-flight ownership model. Built for Android/Linux GPU targets; **Android cross-compile build-check here (NDK present)** + runtime on device/CI.

Each plan is written as its own focused pass (Plan 2's matrix/merge work and Plan 4's GPU-infra need their own code-reading before their tasks are written). Plan 1 is fully self-contained and independently shippable.

## Cross-cutting conventions
- Build: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 <target>` ; test: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 <target> --test_output=all`.
- In-editor clang "file not found" diagnostics are FALSE POSITIVES (non-Bazel language server). Only bazel results are authoritative.
- Default-off invariant: every new option defaults to 0/false; with defaults, behavior is byte-identical to today.
- TDD where it fits the unit (the helper is pure and testable first); calculator changes are build+behavior-verified.

## File Structure
- **Create** `mediapipe/calculators/tensor/tiling_cache_utils.h` — header-only: `CacheStats`, `StableCacheKey`, `StableKeyBuilder`, `BoundedLruCache<Value>`.
- **Create** `mediapipe/calculators/tensor/tiling_cache_utils_test.cc` — unit tests.
- **Modify** `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.proto` — add `max_cached_tile_plans`.
- **Modify** `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.cc` — memoize `TilePlan`.
- **Modify** `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator_test.cc` — cache on/off correctness + recompute-on-change.
- **Modify** `mediapipe/calculators/tensor/BUILD` — add `tiling_cache_utils` `cc_library` + its `cc_test`; add the dep to `tile_spec_to_tile_plan_calculator`.

---

### Task 1: Shared cache utilities (`tiling_cache_utils.h`)

**Files:**
- Create: `mediapipe/calculators/tensor/tiling_cache_utils.h`
- Create: `mediapipe/calculators/tensor/tiling_cache_utils_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

- [ ] **Step 1: Write the header**

```cpp
// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TILING_CACHE_UTILS_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TILING_CACHE_UTILS_H_

#include <cstdint>
#include <cstring>
#include <list>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"

namespace mediapipe {

// Diagnostic counters for one cache instance.
struct CacheStats {
  int64_t hits = 0;
  int64_t misses = 0;
  int64_t inserts = 0;
  int64_t evictions = 0;
};

// A stable, collision-safe cache key. `bytes` is a type-tagged, length-delimited
// encoding of all key fields; equality/lookup compares the FULL bytes (never a
// truncated hash), so two different inputs cannot collide to the same entry.
// `fingerprint` is a fast 64-bit digest kept for diagnostics only.
struct StableCacheKey {
  std::string bytes;
  uint64_t fingerprint = 0;
  bool operator==(const StableCacheKey& other) const {
    return bytes == other.bytes;
  }
};

// Folds key fields into a StableCacheKey. Floats are bit-cast to their integer
// representation so identical values encode identically across calls and runs
// (no float formatting drift). Each field is type-tagged so e.g. AddInt(1) and
// AddString("\x01") do not encode to the same bytes.
class StableKeyBuilder {
 public:
  StableKeyBuilder& AddInt(int64_t value) {
    buf_.push_back('\x01');
    AppendRaw(&value, sizeof(value));
    return *this;
  }
  StableKeyBuilder& AddBool(bool value) {
    buf_.push_back('\x02');
    const uint8_t b = value ? 1 : 0;
    AppendRaw(&b, sizeof(b));
    return *this;
  }
  StableKeyBuilder& AddFloat(float value) {
    buf_.push_back('\x03');
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));  // bit-cast, stable encoding
    AppendRaw(&bits, sizeof(bits));
    return *this;
  }
  StableKeyBuilder& AddString(absl::string_view value) {
    buf_.push_back('\x04');
    const int64_t n = static_cast<int64_t>(value.size());
    AppendRaw(&n, sizeof(n));  // length-delimited: avoids boundary ambiguity
    buf_.append(value.data(), value.size());
    return *this;
  }
  StableCacheKey Build() const {
    StableCacheKey key;
    key.bytes = buf_;
    key.fingerprint = Fnv1a(buf_);
    return key;
  }

 private:
  void AppendRaw(const void* p, size_t n) {
    buf_.append(reinterpret_cast<const char*>(p), n);
  }
  static uint64_t Fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
      h ^= c;
      h *= 1099511628211ull;
    }
    return h;
  }
  std::string buf_;
};

// Bounded LRU memo. capacity == 0 ⇒ disabled: Get always misses, Put is a no-op,
// stats stay zero — so a caller behaves exactly as if there were no cache.
// Calculator-local and single-threaded (one Process at a time); no locking.
// The map is keyed on the full key bytes, so a fingerprint collision cannot
// return the wrong value.
template <typename Value>
class BoundedLruCache {
 public:
  explicit BoundedLruCache(size_t capacity) : capacity_(capacity) {}

  bool enabled() const { return capacity_ > 0; }

  const Value* Get(const StableCacheKey& key) {
    if (capacity_ == 0) return nullptr;
    auto it = index_.find(key.bytes);
    if (it == index_.end()) {
      ++stats_.misses;
      return nullptr;
    }
    order_.splice(order_.begin(), order_, it->second);  // promote to MRU
    ++stats_.hits;
    return &it->second->second;
  }

  void Put(const StableCacheKey& key, Value value) {
    if (capacity_ == 0) return;
    auto it = index_.find(key.bytes);
    if (it != index_.end()) {
      it->second->second = std::move(value);
      order_.splice(order_.begin(), order_, it->second);
      return;
    }
    order_.emplace_front(key.bytes, std::move(value));
    index_[key.bytes] = order_.begin();
    ++stats_.inserts;
    if (index_.size() > capacity_) {
      const std::string& lru_key = order_.back().first;
      index_.erase(lru_key);
      order_.pop_back();
      ++stats_.evictions;
    }
  }

  const CacheStats& stats() const { return stats_; }

 private:
  using Entry = std::pair<std::string, Value>;  // (key bytes, value)
  size_t capacity_;
  std::list<Entry> order_;  // front = most-recently-used
  absl::flat_hash_map<std::string, typename std::list<Entry>::iterator> index_;
  CacheStats stats_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TILING_CACHE_UTILS_H_
```

- [ ] **Step 2: Write the failing test**

```cpp
// Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0.
#include "mediapipe/calculators/tensor/tiling_cache_utils.h"

#include <string>

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

StableCacheKey IntKey(int64_t v) {
  return StableKeyBuilder().AddInt(v).Build();
}

TEST(StableKeyBuilderTest, IdenticalInputsEqualKeys) {
  EXPECT_EQ(StableKeyBuilder().AddInt(7).AddFloat(0.1f).AddString("a").Build(),
            StableKeyBuilder().AddInt(7).AddFloat(0.1f).AddString("a").Build());
}

TEST(StableKeyBuilderTest, AnyFieldChangeChangesKey) {
  const StableCacheKey base =
      StableKeyBuilder().AddInt(7).AddFloat(0.1f).Build();
  EXPECT_FALSE(base == StableKeyBuilder().AddInt(8).AddFloat(0.1f).Build());
  EXPECT_FALSE(base == StableKeyBuilder().AddInt(7).AddFloat(0.2f).Build());
}

TEST(StableKeyBuilderTest, TypeTaggingPreventsCrossTypeCollision) {
  // AddInt(1) must not encode the same bytes as AddString("\x01...").
  EXPECT_FALSE(StableKeyBuilder().AddInt(1).Build() ==
               StableKeyBuilder().AddString("\x01").Build());
}

TEST(BoundedLruCacheTest, DisabledNeverHits) {
  BoundedLruCache<int> cache(0);
  EXPECT_FALSE(cache.enabled());
  cache.Put(IntKey(1), 100);
  EXPECT_EQ(cache.Get(IntKey(1)), nullptr);
  EXPECT_EQ(cache.stats().hits, 0);
  EXPECT_EQ(cache.stats().inserts, 0);
}

TEST(BoundedLruCacheTest, HitAndMiss) {
  BoundedLruCache<int> cache(2);
  EXPECT_EQ(cache.Get(IntKey(1)), nullptr);
  cache.Put(IntKey(1), 100);
  const int* v = cache.Get(IntKey(1));
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(*v, 100);
  EXPECT_EQ(cache.stats().hits, 1);
  EXPECT_EQ(cache.stats().misses, 1);
}

TEST(BoundedLruCacheTest, EvictsLeastRecentlyUsed) {
  BoundedLruCache<int> cache(2);
  cache.Put(IntKey(1), 1);
  cache.Put(IntKey(2), 2);
  ASSERT_NE(cache.Get(IntKey(1)), nullptr);  // 1 now MRU, 2 is LRU
  cache.Put(IntKey(3), 3);                   // evicts 2
  EXPECT_EQ(cache.Get(IntKey(2)), nullptr);
  ASSERT_NE(cache.Get(IntKey(1)), nullptr);
  ASSERT_NE(cache.Get(IntKey(3)), nullptr);
  EXPECT_EQ(cache.stats().evictions, 1);
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 3: Add BUILD targets** in `mediapipe/calculators/tensor/BUILD`:

```python
cc_library(
    name = "tiling_cache_utils",
    hdrs = ["tiling_cache_utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "tiling_cache_utils_test",
    srcs = ["tiling_cache_utils_test.cc"],
    deps = [
        ":tiling_cache_utils",
        "//mediapipe/framework/port:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
```

- [ ] **Step 4: Build + run the test**

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tiling_cache_utils_test --test_output=all`
Expected: PASS (all cases).

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/tiling_cache_utils.h mediapipe/calculators/tensor/tiling_cache_utils_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "feat(tensor-cache): collision-safe bounded-LRU cache utility for tiling

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: TilePlan cache in `TileSpecToTilePlanCalculator`

**Files:**
- Modify: `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.proto`
- Modify: `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.cc`
- Modify: `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD` (add `:tiling_cache_utils` dep to the calculator)

- [ ] **Step 1: Add the option to the proto**

In `tile_spec_to_tile_plan_calculator.proto`, inside `TileSpecToTilePlanCalculatorOptions`, after `max_tiles_per_frame`:

```proto
  // Bounded LRU cache of computed TilePlans, keyed by the input tile rects +
  // max_tiles_per_frame. 0 = disabled (no caching; default). When > 0, repeated
  // identical tile lists (stable video tiling) reuse the validated plan instead
  // of re-validating and rebuilding it.
  optional int32 max_cached_tile_plans = 2 [default = 0];
```

- [ ] **Step 2: Add the failing test** (cache-on equals cache-off; recompute on change)

The existing test file uses `CalculatorRunner` + `ParseTextProtoOrDie<...Node>` + a local
`Rect(xc, yc, w, h)` helper. Add this `#include` near the top of the file:

```cpp
#include "absl/strings/str_format.h"
```

Then add this helper + test inside the anonymous namespace (the helper reuses the
existing `Rect`; it sets the `max_cached_tile_plans` option via the proto
extension and drives one `Run()` over all frames at increasing timestamps):

```cpp
// Runs the calculator over `frames` tile-lists (one per timestamp) with the
// given cache capacity; returns the emitted TilePlans in order.
std::vector<TilePlan> RunPlan(
    int max_cached_tile_plans,
    const std::vector<std::vector<NormalizedRect>>& frames) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      absl::StrFormat(R"pb(
        calculator: "TileSpecToTilePlanCalculator"
        input_stream: "TILES:tiles"
        output_stream: "TILE_PLAN:plan"
        options {
          [mediapipe.TileSpecToTilePlanCalculatorOptions.ext] {
            max_cached_tile_plans: %d
          }
        }
      )pb",
                      max_cached_tile_plans)));
  for (int t = 0; t < static_cast<int>(frames.size()); ++t) {
    auto tiles = std::make_unique<std::vector<NormalizedRect>>(frames[t]);
    runner.MutableInputs()->Tag("TILES").packets.push_back(
        Adopt(tiles.release()).At(Timestamp(t)));
  }
  MP_EXPECT_OK(runner.Run());
  std::vector<TilePlan> out;
  for (const Packet& p : runner.Outputs().Tag("TILE_PLAN").packets) {
    out.push_back(p.Get<TilePlan>());
  }
  return out;
}

// Caching on must produce TilePlans identical to caching off, across repeated
// identical frames (cache hits) and a changed frame (cache miss / recompute).
TEST(TileSpecToTilePlanCalculatorTest, CacheEnabledMatchesDisabled) {
  const std::vector<NormalizedRect> a = {Rect(0.25f, 0.25f, 0.5f, 0.5f),
                                         Rect(0.75f, 0.75f, 0.5f, 0.5f)};
  const std::vector<NormalizedRect> b = {Rect(0.5f, 0.5f, 0.4f, 0.4f)};
  const std::vector<std::vector<NormalizedRect>> frames = {a, a, b, a};

  const std::vector<TilePlan> off = RunPlan(/*max_cached_tile_plans=*/0, frames);
  const std::vector<TilePlan> on = RunPlan(/*max_cached_tile_plans=*/4, frames);

  ASSERT_EQ(off.size(), frames.size());
  ASSERT_EQ(on.size(), off.size());
  for (size_t f = 0; f < off.size(); ++f) {
    ASSERT_EQ(on[f].tiles.size(), off[f].tiles.size());
    for (size_t i = 0; i < off[f].tiles.size(); ++i) {
      EXPECT_EQ(on[f].tiles[i].tile_index, off[f].tiles[i].tile_index);
      EXPECT_FLOAT_EQ(on[f].tiles[i].x_center, off[f].tiles[i].x_center);
      EXPECT_FLOAT_EQ(on[f].tiles[i].y_center, off[f].tiles[i].y_center);
      EXPECT_FLOAT_EQ(on[f].tiles[i].width, off[f].tiles[i].width);
      EXPECT_FLOAT_EQ(on[f].tiles[i].height, off[f].tiles[i].height);
    }
  }
}
```

If the test target does not already pull in `str_format`, add
`"@com_google_absl//absl/strings:str_format"` to the `tile_spec_to_tile_plan_calculator_test`
deps in `BUILD`.

- [ ] **Step 3: Run the test to verify it fails**

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tile_spec_to_tile_plan_calculator_test --test_output=all`
Expected: FAIL to compile (option `max_cached_tile_plans` not yet read) — or the new test absent from the cc binary until the impl wires it.

- [ ] **Step 4: Implement the cache in the calculator**

In `tile_spec_to_tile_plan_calculator.cc`:
- add include `#include "mediapipe/calculators/tensor/tiling_cache_utils.h"`;
- add a member `BoundedLruCache<TilePlan> plan_cache_{0};` initialized in `Open()` from the option;
- validate the option non-negative;
- in `Process()`, build the key, try the cache, compute+store on miss.

Replace `Open()` and `Process()` with:

```cpp
  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<mediapipe::TileSpecToTilePlanCalculatorOptions>();
    RET_CHECK_GE(options_.max_cached_tile_plans(), 0);
    plan_cache_ = BoundedLruCache<TilePlan>(
        static_cast<size_t>(options_.max_cached_tile_plans()));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& tiles = *kInTiles(cc);

    // Cache key: max_tiles_per_frame + the full tile-rect list. The TilePlan is
    // a pure function of these, so a hit returns exactly what we would compute.
    StableCacheKey key;
    if (plan_cache_.enabled()) {
      StableKeyBuilder kb;
      kb.AddInt(options_.max_tiles_per_frame());
      kb.AddInt(static_cast<int64_t>(tiles.size()));
      for (const NormalizedRect& r : tiles) {
        kb.AddFloat(r.x_center()).AddFloat(r.y_center());
        kb.AddFloat(r.width()).AddFloat(r.height());
        kb.AddBool(r.has_rotation()).AddFloat(r.rotation());
      }
      key = kb.Build();
      if (const TilePlan* hit = plan_cache_.Get(key)) {
        kOutPlan(cc).Send(std::make_unique<TilePlan>(*hit));
        return absl::OkStatus();
      }
    }

    // Miss (or caching disabled): validate + build, exactly as before.
    if (options_.max_tiles_per_frame() > 0) {
      RET_CHECK_LE(static_cast<int>(tiles.size()),
                   options_.max_tiles_per_frame())
          << "tile count exceeds max_tiles_per_frame";
    }
    TilePlan plan;
    plan.tiles.reserve(tiles.size());
    for (int i = 0; i < static_cast<int>(tiles.size()); ++i) {
      const NormalizedRect& r = tiles[i];
      RET_CHECK(std::isfinite(r.x_center()) && std::isfinite(r.y_center()) &&
                std::isfinite(r.width()) && std::isfinite(r.height()))
          << "tile " << i << " has non-finite values";
      RET_CHECK_GT(r.width(), 0.0f) << "tile " << i << " width must be > 0";
      RET_CHECK_GT(r.height(), 0.0f) << "tile " << i << " height must be > 0";
      RET_CHECK(!r.has_rotation() || r.rotation() == 0.0f)
          << "rotated tiles are not supported yet (tile " << i << ")";
      TileGeometry g;
      g.tile_index = i;
      g.x_center = r.x_center();
      g.y_center = r.y_center();
      g.width = r.width();
      g.height = r.height();
      RET_CHECK(g.x0() < 1.0f && g.y0() < 1.0f && g.x0() + g.width > 0.0f &&
                g.y0() + g.height > 0.0f)
          << "tile " << i << " does not intersect the frame";
      plan.tiles.push_back(g);
    }

    if (plan_cache_.enabled()) {
      plan_cache_.Put(key, plan);  // store a copy; key already built above
    }
    kOutPlan(cc).Send(std::make_unique<TilePlan>(std::move(plan)));
    return absl::OkStatus();
  }
```

Add the member and include:

```cpp
 private:
  mediapipe::TileSpecToTilePlanCalculatorOptions options_;
  BoundedLruCache<TilePlan> plan_cache_{0};
```

NOTE: validation now runs only on a miss. That is correct: a cached plan was already validated when first built, and the key fully determines the plan. Inputs that would fail validation never get cached (the miss path returns the error before `Put`).

- [ ] **Step 5: Add the dep in BUILD**

Add `":tiling_cache_utils",` to the `deps` of the `tile_spec_to_tile_plan_calculator` `cc_library` in `mediapipe/calculators/tensor/BUILD`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tile_spec_to_tile_plan_calculator_test --test_output=all`
Expected: PASS (existing tests unchanged; new `CacheEnabledMatchesDisabled` passes).

- [ ] **Step 7: Commit**

```bash
git add mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.proto mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.cc mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "feat(tensor-cache): default-off TilePlan cache in TileSpecToTilePlanCalculator

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final verification (after both tasks)
- [ ] `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tiling_cache_utils_test //mediapipe/calculators/tensor:tile_spec_to_tile_plan_calculator_test --test_output=errors` — both pass.
- [ ] Confirm default-off: with `max_cached_tile_plans` unset, the calculator's existing tests are unchanged (no behavior diff).
- [ ] `git status` clean.

## Done criteria
- `tiling_cache_utils` library + unit test build and pass (hit/miss/eviction/disabled/key-stability/collision-safety).
- TilePlan cache: cache-on output == cache-off output across repeated + changed inputs; default-off byte-identical to today; no stream-contract change.
- Foundation ready for Plan 2 (matrix cache reuses `tiling_cache_utils.h`).
