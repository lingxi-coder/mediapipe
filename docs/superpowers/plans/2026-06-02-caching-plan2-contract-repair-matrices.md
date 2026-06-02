# Caching — Plan 2: tiled contract repair + matrices + matrix cache + multi-batch

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the tiled detection contract — carry batch-local geometry + tile-to-tensor matrices in `TensorBatchInfo`, migrate merge to be self-contained, add the matrix cache, and enable true multi-batch (`T > batch_capacity`) emission via the `BeginLoop`-style synthetic-timestamp pattern.

**Architecture:** `StreamingTilesToTensorBatchCalculator` becomes `BeginLoop`-shaped: it emits one tensor batch per `batch_capacity` tiles at a calculator-local monotonic "batch timestamp" (declared `TimestampChange::Arbitrary()`), each carrying an immutable `TileBatchGeometry` (tile indices + geometries + effective pixel ROIs + image↔tensor matrices) and the original `source_frame_timestamp`. `MergeTileDetectionsAccumulatorCalculator` becomes `EndLoop`-shaped: it regroups by `source_frame_timestamp`, projects with the carried geometry (no `TILE_PLAN` input), and emits the merged result at the source timestamp. Matrices are generated via the existing `GetRotatedSubRectToRectTransformMatrix` convention and memoized with the Plan-1 `BoundedLruCache`.

**Tech Stack:** C++20, MediaPipe api2 (`TimestampChange::Arbitrary`, `SetProcessTimestampBounds`), `image_to_tensor_utils`, Bazel (`--define MEDIAPIPE_DISABLE_GPU=1`), GoogleTest.

**Spec:** `docs/superpowers/specs/2026-06-02-caching-tile-geometry-design.md` (sections "Contract repair for multi-tile and multi-batch", "Cache 2: tile-to-tensor matrix cache").

**Builds on:** Plan 1 (`tiling_cache_utils.h` shipped). **Followed by:** Plan 3 (CPU buffer pooling), Plan 4 (OpenGL/AHWB zero-copy — the primary matrix consumer).

---

## Reference facts (verified against shipped code)
- `mediapipe/calculators/tensor/tiling_types.h` — `TileGeometry{tile_index,x_center,y_center,width,height, x0(),y0()}`, `TilePlan{vector<TileGeometry>}`, `TensorBatchInfo{source_frame_timestamp,batch_index,total_batches,batch_capacity,valid_count,tile_indices}`.
- `streaming_tiles_to_tensor_batch_calculator.cc` — api2 `Node`, inputs `IMAGE`(ImageFrame)+`TILE_PLAN`+`METADATA`(side), outputs `TENSORS`(vector<Tensor>)+`BATCH_INFO`. **Currently rejects `T > batch_capacity`** (`RET_CHECK_LE(T,cap)`), default 1:1 timestamp offset, emits no matrices.
- `merge_tile_detections_accumulator_calculator.cc` — api2 `Node`, inputs `ORIENTED_DETECTIONS`(vector<vector<OrientedDetection>>)+`BATCH_INFO`+`TILE_PLAN`, output `ORIENTED_DETECTIONS`. Projects with `g.x0()+d.cx()*g.width` scale+offset; accumulates by `source_frame_timestamp`; emits when `batches_seen>=total_batches`.
- `tiled_obb_pipeline_test.cc` — wires merge→RotatedNMS; feeds `dets`/`info`/`plan` at Timestamp(0); expects object at frame (0.5,0.5,0.2,0.2) → projected cx=0.5,width=0.2.
- `image_to_tensor_utils.h` — `struct RotatedRect{float center_x,center_y,width,height,rotation;}` and `void GetRotatedSubRectToRectTransformMatrix(const RotatedRect& sub_rect, int rect_width, int rect_height, bool flip_horizontally, std::array<float,16>* matrix)` — row-major 4x4 mapping sub_rect-normalized [0,1] → rect-normalized [0,1].
- `BeginLoopCalculator` (core) — the synthetic-timestamp precedent: calculator-local `loop_internal_timestamp_` incremented per emitted item; carries original timestamp for the companion to flush at; `SetProcessTimestampBounds(true)`.
- Multi-batch consumer scope: the only current `MergeTileDetectionsAccumulatorCalculator` / `TILE_PLAN→merge` wiring is `tiled_obb_pipeline_test.cc`. Before editing, run `grep -rn "MergeTileDetectionsAccumulator\|TILE_PLAN:" mediapipe/ ` to confirm no other graph wires `TILE_PLAN` into merge; migrate any that appear.

## Cross-cutting conventions
- Build/test: `bazel {build,test} -c opt --define MEDIAPIPE_DISABLE_GPU=1 <target> --test_output=errors`.
- In-editor clang errors are FALSE POSITIVES; only bazel is authoritative.
- New cache option defaults to 0 (off ⇒ unchanged). Multi-batch is a behavior addition, not gated by an option, but must preserve single-batch (`T<=cap`) outputs exactly.
- Each task commits separately; co-author trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure
- **Modify** `tiling_types.h` — add `TilePixelRoi`, `TileBatchGeometry`; extend `TensorBatchInfo`.
- **Create** `tiling_matrix_utils.h` + `tiling_matrix_utils_test.cc` — tile→image / image→tile matrix generation + point projection (wraps `GetRotatedSubRectToRectTransformMatrix`); isolated + unit-testable.
- **Modify** `streaming_tiles_to_tensor_batch_calculator.{proto,cc}` + test — matrices, matrix cache, multi-batch emission.
- **Modify** `merge_tile_detections_accumulator_calculator.cc` + test — self-contained projection, source-timestamp emission.
- **Modify** `tiled_obb_pipeline_test.cc` — migrate wiring (drop TILE_PLAN into merge) + add a `T>cap` case.
- **Modify** `BUILD` — deps (`:tiling_cache_utils`, `:tiling_matrix_utils`, `//mediapipe/calculators/tensor:image_to_tensor_utils`), new test targets.

---

### Task 1: Extend `TensorBatchInfo` with batch-local geometry

**Files:** Modify `mediapipe/calculators/tensor/tiling_types.h`

- [ ] **Step 1: Add the new types + fields** (append `TilePixelRoi` + `TileBatchGeometry` before `TensorBatchInfo`, and extend `TensorBatchInfo`):

```cpp
// Effective integer pixel ROI a tile actually sampled from the source frame,
// after rounding + clamping to the frame. Matrices describe THIS ROI, not the
// requested normalized rect, so boundary tiles project precisely.
struct TilePixelRoi {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

// Immutable per-batch geometry, shared (shared_ptr) so it travels on the
// BATCH_INFO packet without copying matrix/geometry vectors. All vectors are
// indexed by valid row [0, valid_count) in row order.
struct TileBatchGeometry {
  std::vector<int> tile_indices;
  std::vector<TileGeometry> tile_geometries;
  std::vector<TilePixelRoi> effective_pixel_rois;
  // Row-major 4x4. image_to_tensor maps tile-normalized [0,1] -> image-norm
  // [0,1]; tensor_to_image is its inverse (image-norm -> tile-norm). Stored so
  // merge projects detections (tile space) to frame space without re-deriving.
  std::vector<std::array<float, 16>> tile_to_image_matrices;
  std::vector<std::array<float, 16>> image_to_tile_matrices;
};
```

Add `#include <array>` and `#include <memory>` to the includes. Then change `TensorBatchInfo`:

```cpp
struct TensorBatchInfo {
  int64_t source_frame_timestamp = 0;
  int64_t batch_timestamp = 0;  // loop-internal monotonic ts this batch emitted at
  int batch_index = 0;
  int total_batches = 1;
  int batch_capacity = 1;
  int batch_size = 0;   // emitted N dimension of the tensor
  int valid_count = 0;  // valid rows [0, valid_count); rest is padding
  std::vector<int> tile_indices;  // retained for back-compat; mirrors geometry
  // Non-null when valid_count > 0; all its vectors have size == valid_count.
  std::shared_ptr<const TileBatchGeometry> geometry;
};
```

- [ ] **Step 2: Build the dependents to confirm the struct change compiles**

Run: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator //mediapipe/calculators/tensor:merge_tile_detections_accumulator_calculator`
Expected: builds (existing code still sets the old fields; new fields default).

- [ ] **Step 3: Commit**

```bash
git add mediapipe/calculators/tensor/tiling_types.h
git commit -m "feat(tiling): carry batch-local geometry + matrices in TensorBatchInfo"
```

---

### Task 2: Tile matrix utility (`tiling_matrix_utils.h`)

**Files:** Create `mediapipe/calculators/tensor/tiling_matrix_utils.h` + `..._test.cc`; modify `BUILD`.

- [ ] **Step 1: Write the header** — generates the tile→image matrix (and inverse) for an axis-aligned tile, and projects a normalized point through a row-major 4x4.

```cpp
// Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0.
#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TILING_MATRIX_UTILS_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TILING_MATRIX_UTILS_H_

#include <array>

#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/calculators/tensor/tiling_types.h"

namespace mediapipe {

// Builds the row-major 4x4 that maps a point in the tile's normalized [0,1]
// space to the source frame's normalized [0,1] space, for an AXIS-ALIGNED tile
// whose effective sampled region is `roi` pixels within a `frame_w` x `frame_h`
// frame. Uses the ImageToTensor convention (GetRotatedSubRectToRectTransform-
// Matrix) so it matches the rest of the codebase.
inline std::array<float, 16> TileToImageMatrix(const TilePixelRoi& roi,
                                               int frame_w, int frame_h) {
  RotatedRect sub_rect;
  sub_rect.center_x = roi.x + roi.width / 2.0f;
  sub_rect.center_y = roi.y + roi.height / 2.0f;
  sub_rect.width = static_cast<float>(roi.width);
  sub_rect.height = static_cast<float>(roi.height);
  sub_rect.rotation = 0.0f;
  std::array<float, 16> m;
  GetRotatedSubRectToRectTransformMatrix(sub_rect, frame_w, frame_h,
                                         /*flip_horizontally=*/false, &m);
  return m;
}

// Applies a row-major 4x4 to the homogeneous 2D point (x, y, 0, 1); returns the
// transformed (x', y') (w assumed 1 for affine tile transforms).
inline void ApplyMatrix(const std::array<float, 16>& m, float x, float y,
                        float* out_x, float* out_y) {
  *out_x = m[0] * x + m[1] * y + m[3];
  *out_y = m[4] * x + m[5] * y + m[7];
}

// Inverts the affine 2D part of a row-major 4x4 (rotation+scale+translation)
// into the reverse map (image-norm -> tile-norm). Tile transforms are affine,
// so this is exact.
std::array<float, 16> InvertAffine2d(const std::array<float, 16>& m);

}  // namespace mediapipe
#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TILING_MATRIX_UTILS_H_
```

Create `tiling_matrix_utils.cc` with `InvertAffine2d` (2x2 inverse of the [[m0,m1],[m4,m5]] block + translation):

```cpp
// Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0.
#include "mediapipe/calculators/tensor/tiling_matrix_utils.h"

namespace mediapipe {

std::array<float, 16> InvertAffine2d(const std::array<float, 16>& m) {
  const float a = m[0], b = m[1], c = m[4], d = m[5];
  const float tx = m[3], ty = m[7];
  const float det = a * d - b * c;
  const float inv_det = det != 0.0f ? 1.0f / det : 0.0f;
  const float ia = d * inv_det, ib = -b * inv_det;
  const float ic = -c * inv_det, id = a * inv_det;
  std::array<float, 16> r = {0};
  r[0] = ia; r[1] = ib; r[3] = -(ia * tx + ib * ty);
  r[4] = ic; r[5] = id; r[7] = -(ic * tx + id * ty);
  r[10] = 1.0f; r[15] = 1.0f;
  return r;
}

}  // namespace mediapipe
```

- [ ] **Step 2: Write the test** — for a half-frame tile, projecting tile-local (0.83333,0.5) maps to frame (0.5,0.5) and round-trips; matches the legacy scale+offset.

```cpp
// Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0.
#include "mediapipe/calculators/tensor/tiling_matrix_utils.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

// Left tile covering x[0,0.6] of a 100x100 frame -> roi {0,0,60,100}.
TEST(TilingMatrixUtilsTest, ProjectsTileLocalToFrameNormalized) {
  TilePixelRoi roi{/*x=*/0, /*y=*/0, /*width=*/60, /*height=*/100};
  const auto m = TileToImageMatrix(roi, /*frame_w=*/100, /*frame_h=*/100);
  // Object at frame (0.5,0.5) is tile-local cx = (0.5-0.0)/0.6 = 0.83333.
  float fx, fy;
  ApplyMatrix(m, 0.83333f, 0.5f, &fx, &fy);
  EXPECT_NEAR(fx, 0.5f, 1e-3);
  EXPECT_NEAR(fy, 0.5f, 1e-3);
  // Width scales by tile fraction: a tile-local width of 0.33333 -> 0.2 frame.
  float x0, y0, x1, y1;
  ApplyMatrix(m, 0.83333f - 0.33333f / 2, 0.5f, &x0, &y0);
  ApplyMatrix(m, 0.83333f + 0.33333f / 2, 0.5f, &x1, &y1);
  EXPECT_NEAR(x1 - x0, 0.2f, 1e-3);
}

TEST(TilingMatrixUtilsTest, InverseRoundTrips) {
  TilePixelRoi roi{10, 20, 60, 50};
  const auto m = TileToImageMatrix(roi, 100, 100);
  const auto inv = InvertAffine2d(m);
  float fx, fy, tx, ty;
  ApplyMatrix(m, 0.3f, 0.7f, &fx, &fy);
  ApplyMatrix(inv, fx, fy, &tx, &ty);
  EXPECT_NEAR(tx, 0.3f, 1e-4);
  EXPECT_NEAR(ty, 0.7f, 1e-4);
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 3: BUILD targets** (add to `mediapipe/calculators/tensor/BUILD`):

```python
cc_library(
    name = "tiling_matrix_utils",
    srcs = ["tiling_matrix_utils.cc"],
    hdrs = ["tiling_matrix_utils.h"],
    deps = [
        ":image_to_tensor_utils",
        ":tiling_types",  # if tiling_types.h has no cc_library yet, add a header-only one
    ],
)

cc_test(
    name = "tiling_matrix_utils_test",
    srcs = ["tiling_matrix_utils_test.cc"],
    deps = [
        ":tiling_matrix_utils",
        "//mediapipe/framework/port:gtest_main",
    ],
)
```
> `tiling_types.h` may currently be a bare header with no `cc_library`. If so, add `cc_library(name="tiling_types", hdrs=["tiling_types.h"])` and depend on it from the calculators that include it. Confirm with `grep -n "tiling_types" mediapipe/calculators/tensor/BUILD`.

- [ ] **Step 4: Build + test**

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tiling_matrix_utils_test --test_output=all`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/tiling_matrix_utils.h mediapipe/calculators/tensor/tiling_matrix_utils.cc mediapipe/calculators/tensor/tiling_matrix_utils_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "feat(tiling): tile->image matrix generation + affine inverse utility"
```

---

### Task 3: Streaming calc — populate geometry + matrices (single-batch first)

**Files:** Modify `streaming_tiles_to_tensor_batch_calculator.cc` + test.

Keep the `RET_CHECK_LE(T, cap)` for now (multi-batch is Task 5). In `Process()`, while filling each valid row, also compute the effective pixel ROI (the existing `rx,ry,rw,rh` clamp), the `TileGeometry`, and the matrices, and attach a `TileBatchGeometry` to the emitted `TensorBatchInfo`. Set `batch_size`, `source_frame_timestamp`, `batch_timestamp = ts` (single-batch: same as input ts for now).

- [ ] **Step 1:** add includes `#include "mediapipe/calculators/tensor/tiling_matrix_utils.h"` and `#include <array>`, `#include <memory>`.
- [ ] **Step 2:** in the row loop, after computing `rx,ry,rw,rh`, build the geometry. After the loop, attach it. Concretely, inside the batch loop replace the per-row body and the `TensorBatchInfo` setup with:

```cpp
      TensorBatchInfo info;
      info.source_frame_timestamp = ts;
      info.batch_timestamp = ts;  // single-batch: emit at input ts (Task 5 changes this)
      info.batch_index = emitted;
      info.total_batches = total_batches;
      info.batch_capacity = N;
      info.batch_size = N;
      info.valid_count = rows;
      auto geom = std::make_shared<TileBatchGeometry>();
      geom->tile_indices.reserve(rows);
      geom->tile_geometries.reserve(rows);
      geom->effective_pixel_rois.reserve(rows);
      geom->tile_to_image_matrices.reserve(rows);
      geom->image_to_tile_matrices.reserve(rows);

      for (int r = 0; r < rows; ++r) {
        const TileGeometry& g = plan.tiles[start + r];
        int rx = static_cast<int>(std::lround(g.x0() * fw));
        int ry = static_cast<int>(std::lround(g.y0() * fh));
        int rw = static_cast<int>(std::lround(g.width * fw));
        int rh = static_cast<int>(std::lround(g.height * fh));
        rx = std::clamp(rx, 0, fw - 1);
        ry = std::clamp(ry, 0, fh - 1);
        rw = std::clamp(rw, 1, fw - rx);
        rh = std::clamp(rh, 1, fh - ry);
        cv::Mat roi = src(cv::Rect(rx, ry, rw, rh));
        cv::Mat resized;
        cv::resize(roi, resized, cv::Size(W, H));
        cv::Mat f32;
        resized.convertTo(f32, CV_32FC(C), 1.0 / 255.0);
        std::memcpy(buf + static_cast<size_t>(r) * H * W * C,
                    f32.ptr<float>(0), sizeof(float) * H * W * C);
        info.tile_indices.push_back(g.tile_index);

        TilePixelRoi proi{rx, ry, rw, rh};
        const std::array<float, 16> t2i = TileToImageMatrix(proi, fw, fh);
        geom->tile_indices.push_back(g.tile_index);
        geom->tile_geometries.push_back(g);
        geom->effective_pixel_rois.push_back(proi);
        geom->tile_to_image_matrices.push_back(t2i);
        geom->image_to_tile_matrices.push_back(InvertAffine2d(t2i));
      }
      info.geometry = std::move(geom);
```

(The matrix cache in Task 4 will replace the per-row `TileToImageMatrix`/`InvertAffine2d` computation with a cache lookup keyed by the batch's geometry; for now compute directly.)

- [ ] **Step 3:** Build the calculator.

Run: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator`
Expected: builds.

- [ ] **Step 4:** Extend `streaming_tiles_to_tensor_batch_calculator_test.cc` with a check that `BATCH_INFO.geometry` is populated with `valid_count` rows and that `tile_to_image_matrices` project the tile center back to the tile's frame center (reuse the file's existing runner harness; assert `geom->tile_to_image_matrices.size()==valid_count` and an `ApplyMatrix` round-trip on row 0).

- [ ] **Step 5:** Build + test, then commit.

```bash
bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test --test_output=all
git add mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "feat(tiling): streaming calc emits batch-local geometry + tile matrices"
```

(Add `:tiling_matrix_utils` to the calculator's BUILD deps.)

---

### Task 4: Matrix cache in the streaming calc

**Files:** Modify `streaming_tiles_to_tensor_batch_calculator.{proto,cc}` + test.

- [ ] **Step 1:** proto — add `optional int32 max_cached_tile_matrices = <next free field number> [default = 0];` (read the proto; use the next free field number after the existing ones).
- [ ] **Step 2:** cc — add `#include "mediapipe/calculators/tensor/tiling_cache_utils.h"`; a member `BoundedLruCache<std::shared_ptr<const TileBatchGeometry>> matrix_cache_{0};` initialized in `Open()` (`RET_CHECK_GE(max_cached_tile_matrices,0)`). Cache the FULL `TileBatchGeometry` for a (frame size, tile set) since matrices depend only on geometry + frame size (NOT pixels). Before the per-row geometry build, compute a key over: `fw, fh, W, H, C`, and for each tile in this batch `g.x_center,g.y_center,g.width,g.height,g.tile_index`. On hit, reuse the cached `shared_ptr` for `info.geometry` and skip matrix/ROI recomputation — BUT still run the crop/resize/normalize pixel loop every frame (pixels are never cached). On miss, build geometry as in Task 3 and `Put` the shared_ptr.

Structure: split the row loop into (a) the pixel write (always runs) and (b) geometry build (cache-guarded). Concretely:

```cpp
      // Geometry (matrices + ROIs) depends only on frame size + tile set, not
      // pixels. Cache it; the pixel crop/resize below always runs.
      std::shared_ptr<const TileBatchGeometry> geom;
      StableCacheKey key;
      if (matrix_cache_.enabled()) {
        StableKeyBuilder kb;
        kb.AddInt(fw).AddInt(fh).AddInt(W).AddInt(H).AddInt(C);
        for (int r = 0; r < rows; ++r) {
          const TileGeometry& g = plan.tiles[start + r];
          kb.AddInt(g.tile_index).AddFloat(g.x_center).AddFloat(g.y_center)
            .AddFloat(g.width).AddFloat(g.height);
        }
        key = kb.Build();
        if (auto* hit = matrix_cache_.Get(key)) geom = *hit;
      }
      if (geom == nullptr) {
        geom = BuildBatchGeometry(plan, start, rows, fw, fh);  // Task 3 logic
        if (matrix_cache_.enabled()) matrix_cache_.Put(key, geom);
      }
      // ... pixel crop/resize/normalize loop (always runs) using geom or plan ...
      info.geometry = geom;
```

Extract the Task-3 geometry-build into a private `static std::shared_ptr<const TileBatchGeometry> BuildBatchGeometry(const TilePlan&, int start, int rows, int fw, int fh)`. The pixel loop reads `geom->effective_pixel_rois[r]` for the crop rect (so cached ROI and pixel sampling stay consistent).

- [ ] **Step 3:** test — extend with cache-on==cache-off: feed identical (image,plan) twice with `max_cached_tile_matrices=4` vs `0`, assert identical `geometry` matrices and identical output tensors. Build + test.
- [ ] **Step 4:** commit.

```bash
git add mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.proto mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "feat(tensor-cache): default-off tile-matrix/geometry cache in streaming calc"
```

---

### Task 5: Multi-batch emission (`T > batch_capacity`) via synthetic timestamps

**Files:** Modify `streaming_tiles_to_tensor_batch_calculator.cc` + test.

- [ ] **Step 1:** Change the node contract to declare arbitrary output timestamps:

```cpp
  MEDIAPIPE_NODE_CONTRACT(kInImage, kInPlan, kSideMeta, kOutTensors, kOutInfo,
                          ::mediapipe::api2::TimestampChange::Arbitrary());
```

- [ ] **Step 2:** Add a calculator-local monotonic batch timestamp member `Timestamp batch_ts_ = Timestamp(0);` and **remove** the `RET_CHECK_LE(T, cap)` guard. In the batch loop, emit each batch at `batch_ts_` and set `info.batch_timestamp = batch_ts_.Value()`, then `++batch_ts_`. Replace the two `Send(...)` calls with `.At(batch_ts_)`:

```cpp
    for (int start = 0; start < T; start += cap) {
      // ... build tensor + info (info.source_frame_timestamp = ts) ...
      info.batch_timestamp = batch_ts_.Value();
      // api2 emit-at-explicit-timestamp form (see sequence_shift_calculator.cc:
      // `kOut(cc).Send(packet.At(ts))`): build an api2 packet and stamp it.
      kOutTensors(cc).Send(
          mediapipe::api2::MakePacket<std::vector<Tensor>>(std::move(tensors))
              .At(batch_ts_));
      kOutInfo(cc).Send(
          mediapipe::api2::MakePacket<TensorBatchInfo>(std::move(info))
              .At(batch_ts_));
      ++batch_ts_;
      ++emitted;
    }
    // If T == 0, emit one empty BATCH_INFO at batch_ts_ so merge sees the frame.
    if (T == 0) {
      TensorBatchInfo info;
      info.source_frame_timestamp = ts;
      info.batch_timestamp = batch_ts_.Value();
      info.total_batches = 0;
      info.valid_count = 0;
      kOutInfo(cc).Send(
          mediapipe::api2::MakePacket<TensorBatchInfo>(std::move(info))
              .At(batch_ts_));
      ++batch_ts_;
    }
```

> The emit-at-timestamp form `kOut(cc).Send(api2::MakePacket<T>(...).At(ts))` is taken from `sequence_shift_calculator.cc:114` (the proven api2 precedent), paired with `TimestampChange::Arbitrary()` in the contract. Add `#include "mediapipe/framework/api2/packet.h"` if not already pulled in.

- [ ] **Step 3:** test — add a `T > cap` case (e.g. cap=2, 3 tiles) and assert: two BATCH_INFO packets emitted; `total_batches==2`; `batch_index` 0,1; `valid_count` 2 then 1; combined `tile_indices` cover all 3 tiles. Build + test.
- [ ] **Step 4:** commit.

```bash
git add mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator_test.cc
git commit -m "feat(tiling): multi-batch (T>batch_capacity) emission via synthetic timestamps"
```

---

### Task 6: Merge migration — self-contained projection + source-timestamp emission

**Files:** Modify `merge_tile_detections_accumulator_calculator.cc` + `tiled_obb_pipeline_test.cc` + `merge..._test.cc`.

- [ ] **Step 1:** Drop the `TILE_PLAN` input. New contract + projection using the carried geometry. Declare `TimestampChange::Arbitrary()` (output goes at source ts, inputs at batch ts):

```cpp
  static constexpr Input<std::vector<std::vector<OrientedDetection>>> kInDets{
      "ORIENTED_DETECTIONS"};
  static constexpr Input<TensorBatchInfo> kInInfo{"BATCH_INFO"};
  static constexpr Output<std::vector<OrientedDetection>> kOut{
      "ORIENTED_DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInDets, kInInfo, kOut,
                          ::mediapipe::api2::TimestampChange::Arbitrary());
```

Project each detection with the carried `tile_to_image_matrices` (falls back to `tile_geometries` scale+offset if `geometry` is null, preserving old behavior). For the box center use `ApplyMatrix`; for width/height scale by the tile fraction (`tile_geometries[r].width/height`) — identical to today for axis-aligned tiles. Emit at `Timestamp(info.source_frame_timestamp)`:

```cpp
  absl::Status Process(CalculatorContext* cc) override {
    const auto& batch = *kInDets(cc);
    const TensorBatchInfo& info = *kInInfo(cc);
    RET_CHECK(info.geometry != nullptr || info.valid_count == 0);
    auto& acc = pending_[info.source_frame_timestamp];
    const auto& geom = info.geometry;
    for (int r = 0; r < info.valid_count; ++r) {
      if (r >= static_cast<int>(batch.size())) continue;
      const TileGeometry& g = geom->tile_geometries[r];
      const std::array<float, 16>& m = geom->tile_to_image_matrices[r];
      for (const OrientedDetection& d : batch[r]) {
        OrientedDetection out = d;
        float fx, fy;
        ApplyMatrix(m, d.cx(), d.cy(), &fx, &fy);
        out.set_cx(fx);
        out.set_cy(fy);
        out.set_width(d.width() * g.width);
        out.set_height(d.height() * g.height);
        acc.received_dets.push_back(std::move(out));
      }
    }
    acc.batches_seen += 1;
    acc.total_batches = info.total_batches;
    if (acc.batches_seen >= acc.total_batches) {
      std::vector<OrientedDetection> merged = std::move(acc.received_dets);
      const int64_t src_ts = info.source_frame_timestamp;
      pending_.erase(info.source_frame_timestamp);
      kOut(cc).Send(
          mediapipe::api2::MakePacket<std::vector<OrientedDetection>>(
              std::move(merged))
              .At(Timestamp(src_ts)));
    }
    return absl::OkStatus();
  }
```

Add includes `#include "mediapipe/calculators/tensor/tiling_matrix_utils.h"` (for `ApplyMatrix`) and `#include "mediapipe/framework/api2/packet.h"`, plus the `:tiling_matrix_utils` BUILD dep. Handle `total_batches == 0` (empty frame): emit an empty vector at `src_ts`.

> Emit form is the same proven `api2::MakePacket<T>(...).At(ts)` as Task 5 (`sequence_shift_calculator.cc`). Emitting at `source_frame_timestamp` (< the batch timestamps consumed) is the EndLoop pattern; with `TimestampChange::Arbitrary` the framework permits it, and source_frame_timestamps are monotonic across frames so per-frame final emission stays ordered.

- [ ] **Step 2:** Migrate `tiled_obb_pipeline_test.cc`: remove the `plan` input stream + the `TILE_PLAN:plan` wire to merge; instead put the geometry on `info` (build a `TileBatchGeometry` with the two tiles' geometries + identity-free `TileToImageMatrix` for each tile's pixel ROI over a chosen frame size, e.g. 100x100, matching the existing tile rects). Keep the SAME expected output (cx≈0.5, width≈0.2, score 0.9). This proves the matrix projection reproduces the scale+offset result.
- [ ] **Step 3:** Update `merge_tile_detections_accumulator_calculator_test.cc` similarly (geometry on BATCH_INFO, no TILE_PLAN).
- [ ] **Step 4:** Build + run both tests.

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:merge_tile_detections_accumulator_calculator_test //mediapipe/calculators/tensor:tiled_obb_pipeline_test --test_output=all`
Expected: PASS with unchanged expected values.

- [ ] **Step 5:** commit.

```bash
git add mediapipe/calculators/tensor/merge_tile_detections_accumulator_calculator.cc mediapipe/calculators/tensor/tiled_obb_pipeline_test.cc mediapipe/calculators/tensor/merge_tile_detections_accumulator_calculator_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "feat(tiling): merge consumes batch-local geometry, emits at source timestamp"
```

---

### Task 7: End-to-end multi-batch pipeline test

**Files:** Modify `tiled_obb_pipeline_test.cc`.

- [ ] **Step 1:** Add a test that drives the FULL path with `T > cap`: feed an image + a TilePlan with 3 tiles through `StreamingTilesToTensorBatchCalculator` (cap=2 via metadata) → a stub/identity decode that emits per-row detections → merge → RotatedNMS, and assert the frame-global result is produced once at the source timestamp with detections from all 3 tiles deduped. (If wiring real inference is heavy, drive streaming→merge with a hand-built 2-batch sequence at synthetic timestamps `batch_timestamp` 0 and 1, both `source_frame_timestamp=0`, `total_batches=2`, and assert one merged output at Timestamp(0).)
- [ ] **Step 2:** Build + test, commit.

```bash
git add mediapipe/calculators/tensor/tiled_obb_pipeline_test.cc
git commit -m "test(tiling): end-to-end T>batch_capacity multi-batch merge at source timestamp"
```

---

## Self-review checklist (run before final review)
- `TensorBatchInfo.geometry` populated whenever `valid_count>0`; vectors sized `valid_count`.
- Matrix cache is default-off; caches geometry only (never pixels); pixel crop/resize runs every frame.
- Single-batch (`T<=cap`) outputs unchanged vs. pre-Plan-2 (the migrated pipeline test's expected values are identical).
- Multi-batch emits at strictly increasing `batch_ts_`; merge emits once per source frame at `source_frame_timestamp`.
- No remaining `TILE_PLAN` input on merge; all merge wirings migrated (grep).
- api2 emit uses `kOut(cc).Send(api2::MakePacket<T>(...).At(ts))` + `TimestampChange::Arbitrary()` (form proven by `sequence_shift_calculator.cc`); no nonexistent `Send(value, Timestamp)` 2-arg overload.

## Final verification
- [ ] `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tiling_matrix_utils_test //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test //mediapipe/calculators/tensor:merge_tile_detections_accumulator_calculator_test //mediapipe/calculators/tensor:tiled_obb_pipeline_test --test_output=errors` — all pass.
- [ ] `git status` clean.

## Done criteria
- Batch geometry + matrices carried in `TensorBatchInfo`; merge self-contained (no `TILE_PLAN`); projection reproduces prior axis-aligned results.
- Default-off matrix/geometry cache; pixels never cached.
- `T > batch_capacity` multi-batch emission works end-to-end via synthetic timestamps; one merged result per source frame.
- All four calculator/pipeline tests green; Plan-1 cache utility reused; no other calculators/graphs changed.
```
