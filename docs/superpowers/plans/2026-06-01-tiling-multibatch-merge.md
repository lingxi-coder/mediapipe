# Tiling + Multi-batch + Merge Implementation Plan (Group 1, Plan 3b of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement M1 (multi-batch) + M3 (external tiling): validate externally-supplied tiles, stream them into batched `[N,H,W,C]` tensors sized from `InferenceMetadata`, run inference per batch, then regroup decoded detections per source frame, drop padding, project tile-local boxes back to full-frame-normalized coords, and run one frame-global NMS.

**Architecture:** Three new CPU api2 calculators in `mediapipe/calculators/tensor/` and `mediapipe/calculators/image/`, plus a small shared header of internal tiling structs. Tiles flow per-frame; the batcher streams each filled batch straight to inference (no full-frame materialization). Per spec decisions, the only NMS is the final frame-global one (Plan 1 axis-aligned NMS / Plan 2 `RotatedNonMaxSuppressionCalculator`). **Scope guard:** tile projection supports **axis-aligned tiles** (`NormalizedRect.rotation == 0`), which is the SAHI grid / ROI case; rotated *tiles* are rejected with a clear error (a documented follow-on). Detections themselves still carry rotation (OBB).

**Tech Stack:** C++17, MediaPipe api2, OpenCV (crop/resize via port headers), `Tensor`, protobuf2, Bazel, GoogleTest via `CalculatorRunner`.

**Spec:** `docs/superpowers/specs/2026-06-01-detection-core-yolo-obb-tiling-design.md` (§4 data flow; §5.2–5.4 tile/batch types; §6 M1; §8 M3).

**Depends on:** Plan 1 (Detection decoder, `BatchDetections` type), Plan 2 (`OrientedDetection`, rotated NMS), Plan 3a (`InferenceMetadata` side packet). All must be present.

---

## File structure

- Create: `mediapipe/calculators/tensor/tiling_types.h` — `TileGeometry`, `TilePlan`, `TensorBatchInfo` structs.
- Create: `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.{cc,proto}` + test.
- Create: `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.{cc,proto}` + test.
- Create: `mediapipe/calculators/tensor/merge_tile_detections_accumulator_calculator.{cc,proto}` + test.
- Modify: `mediapipe/calculators/tensor/BUILD`.
- Create: `mediapipe/calculators/tensor/testdata/tiled_yolo_integration_test.cc` (graph round-trip).

All commands from repo root with `--define MEDIAPIPE_DISABLE_GPU=1`.

---

### Task 1: Shared tiling types header

**Files:**
- Create: `mediapipe/calculators/tensor/tiling_types.h`
- Modify: `mediapipe/calculators/tensor/BUILD`

These are plain C++ structs passed as packet payloads between the three calculators (no proto registration needed for intra-graph streams).

- [ ] **Step 1: Write the header**

Create `mediapipe/calculators/tensor/tiling_types.h`:

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
#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TILING_TYPES_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TILING_TYPES_H_

#include <vector>

namespace mediapipe {

// One validated tile: an axis-aligned crop over the source frame, in
// normalized [0,1] coordinates. (Rotated tiles are not supported yet.)
struct TileGeometry {
  int tile_index = 0;
  float x_center = 0.0f;  // normalized
  float y_center = 0.0f;
  float width = 0.0f;     // normalized, > 0
  float height = 0.0f;    // normalized, > 0

  // Top-left of the tile in normalized frame coords (derived).
  float x0() const { return x_center - width / 2.0f; }
  float y0() const { return y_center - height / 2.0f; }
};

// Per-frame validated tile list.
struct TilePlan {
  std::vector<TileGeometry> tiles;
};

// Travels with each emitted inference batch so the merge step can regroup and
// drop padding. Timestamps are in microseconds.
struct TensorBatchInfo {
  int64_t source_frame_timestamp = 0;
  int batch_index = 0;     // 0-based index of this batch within the frame
  int total_batches = 1;   // total batches the frame will emit
  int batch_capacity = 1;  // N dimension of the emitted tensor
  int valid_count = 0;     // valid rows [0, valid_count); rest is padding
  // Tile index for each valid row, in row order.
  std::vector<int> tile_indices;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TILING_TYPES_H_
```

- [ ] **Step 2: Add a header-only library** in `mediapipe/calculators/tensor/BUILD`:

```python
cc_library(
    name = "tiling_types",
    hdrs = ["tiling_types.h"],
)
```

- [ ] **Step 3: Build**

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tiling_types
```
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/calculators/tensor/tiling_types.h mediapipe/calculators/tensor/BUILD
git commit -m "feat(tiling): shared TileGeometry/TilePlan/TensorBatchInfo types"
```

---

### Task 2: `TileSpecToTilePlanCalculator` (validation)

**Files:**
- Create: `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.proto`
- Create: `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.cc`
- Create: `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

Input `TILES` is `std::vector<NormalizedRect>` (the spec's minimum representation). Output `TILE_PLAN` is `TilePlan`. Validates the §5.2 contract.

- [ ] **Step 1: Pick a unique extension id and write the options proto**

`grep -rhoE "ext = [0-9]+" mediapipe | sort -t= -k2 -n | tail -25` → pick unused (plan uses `471230004`).

Create `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.proto`:

```proto
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

syntax = "proto2";

package mediapipe;

import "mediapipe/framework/calculator.proto";

message TileSpecToTilePlanCalculatorOptions {
  extend .mediapipe.CalculatorOptions {
    optional TileSpecToTilePlanCalculatorOptions ext = 471230004;
  }
  // Reject a frame whose tile count exceeds this (0 = no limit).
  optional int32 max_tiles_per_frame = 1 [default = 0];
}
```

- [ ] **Step 2: Write the failing test**

Create `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator_test.cc`:

```cpp
// Copyright 2026 The MediaPipe Authors. Apache-2.0 (full header).
#include <memory>
#include <vector>

#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

NormalizedRect Rect(float xc, float yc, float w, float h) {
  NormalizedRect r;
  r.set_x_center(xc); r.set_y_center(yc); r.set_width(w); r.set_height(h);
  return r;
}

TEST(TileSpecToTilePlanCalculatorTest, ValidatesAndIndexesTiles) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TileSpecToTilePlanCalculator"
    input_stream: "TILES:tiles"
    output_stream: "TILE_PLAN:plan"
  )pb"));

  auto tiles = std::make_unique<std::vector<NormalizedRect>>();
  tiles->push_back(Rect(0.25f, 0.25f, 0.5f, 0.5f));
  tiles->push_back(Rect(0.75f, 0.75f, 0.5f, 0.5f));
  runner.MutableInputs()->Tag("TILES").packets.push_back(
      Adopt(tiles.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& plan =
      runner.Outputs().Tag("TILE_PLAN").packets[0].Get<TilePlan>();
  ASSERT_EQ(plan.tiles.size(), 2);
  EXPECT_EQ(plan.tiles[0].tile_index, 0);
  EXPECT_EQ(plan.tiles[1].tile_index, 1);
  EXPECT_NEAR(plan.tiles[0].x0(), 0.0f, 1e-5);
  EXPECT_NEAR(plan.tiles[0].width, 0.5f, 1e-5);
}

TEST(TileSpecToTilePlanCalculatorTest, RejectsNonPositiveSize) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TileSpecToTilePlanCalculator"
    input_stream: "TILES:tiles"
    output_stream: "TILE_PLAN:plan"
  )pb"));
  auto tiles = std::make_unique<std::vector<NormalizedRect>>();
  tiles->push_back(Rect(0.5f, 0.5f, 0.0f, 0.5f));  // zero width -> invalid
  runner.MutableInputs()->Tag("TILES").packets.push_back(
      Adopt(tiles.release()).At(Timestamp(0)));
  EXPECT_FALSE(runner.Run().ok());
}

TEST(TileSpecToTilePlanCalculatorTest, EmptyTilesEmitsEmptyPlan) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TileSpecToTilePlanCalculator"
    input_stream: "TILES:tiles"
    output_stream: "TILE_PLAN:plan"
  )pb"));
  runner.MutableInputs()->Tag("TILES").packets.push_back(
      Adopt(new std::vector<NormalizedRect>()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  EXPECT_TRUE(runner.Outputs().Tag("TILE_PLAN").packets[0]
                  .Get<TilePlan>().tiles.empty());
}

}  // namespace
}  // namespace mediapipe
```
(Use the full Apache header in the real file.)

- [ ] **Step 3: Write the calculator**

Create `mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.cc`:

```cpp
// Copyright 2026 The MediaPipe Authors. Apache-2.0 (full header).
#include <cmath>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.pb.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// Validates externally-supplied tiles and emits a TilePlan. Axis-aligned only.
class TileSpecToTilePlanCalculator : public Node {
 public:
  static constexpr Input<std::vector<NormalizedRect>> kInTiles{"TILES"};
  static constexpr Output<TilePlan> kOutPlan{"TILE_PLAN"};
  MEDIAPIPE_NODE_CONTRACT(kInTiles, kOutPlan);

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<mediapipe::TileSpecToTilePlanCalculatorOptions>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& tiles = *kInTiles(cc);
    if (options_.max_tiles_per_frame() > 0) {
      RET_CHECK_LE(static_cast<int>(tiles.size()),
                   options_.max_tiles_per_frame())
          << "tile count exceeds max_tiles_per_frame";
    }
    auto plan = std::make_unique<TilePlan>();
    plan->tiles.reserve(tiles.size());
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
      // Must intersect the frame.
      RET_CHECK(g.x0() < 1.0f && g.y0() < 1.0f &&
                g.x0() + g.width > 0.0f && g.y0() + g.height > 0.0f)
          << "tile " << i << " does not intersect the frame";
      plan->tiles.push_back(g);
    }
    kOutPlan(cc).Send(std::move(plan));
    return absl::OkStatus();
  }

 private:
  mediapipe::TileSpecToTilePlanCalculatorOptions options_;
};

MEDIAPIPE_REGISTER_NODE(TileSpecToTilePlanCalculator);

}  // namespace api2
}  // namespace mediapipe
```

- [ ] **Step 4: Add BUILD targets** (proto lib like prior tasks; calculator lib + test):

```python
mediapipe_proto_library(
    name = "tile_spec_to_tile_plan_calculator_proto",
    srcs = ["tile_spec_to_tile_plan_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "tile_spec_to_tile_plan_calculator",
    srcs = ["tile_spec_to_tile_plan_calculator.cc"],
    deps = [
        ":tile_spec_to_tile_plan_calculator_cc_proto",
        ":tiling_types",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "tile_spec_to_tile_plan_calculator_test",
    srcs = ["tile_spec_to_tile_plan_calculator_test.cc"],
    size = "small",
    deps = [
        ":tile_spec_to_tile_plan_calculator",
        ":tiling_types",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
    ],
)
```
Confirm `rect_cc_proto` label: `grep -n "name = \"rect_proto\"" mediapipe/framework/formats/BUILD`.

- [ ] **Step 5: Run the test (expect: first run after adding target shows the validation tests pass)**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tile_spec_to_tile_plan_calculator_test --test_output=all
```
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.proto mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator.cc mediapipe/calculators/tensor/tile_spec_to_tile_plan_calculator_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "feat(tiling): TileSpecToTilePlanCalculator with validation"
```

---

### Task 3: `StreamingTilesToTensorBatchCalculator` (CPU)

**Files:**
- Create: `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.proto`
- Create: `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc`
- Create: `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

Inputs: `IMAGE` (ImageFrame), `TILE_PLAN` (TilePlan), side input `METADATA` (InferenceMetadata from Plan 3a). Outputs (one set per emitted batch): `TENSORS` (`std::vector<Tensor>` with a single `[N,H,W,C]` float32 tensor) and `BATCH_INFO` (TensorBatchInfo). Crops each tile from the frame, resizes to `H×W`, normalizes to `[0,1]`, writes into batch rows.

- [ ] **Step 1: Confirm the ImageFrame→cv::Mat idiom and the batch-mode option**

Run:
```bash
grep -rn "MatView\|image_frame_opencv" mediapipe/framework/formats/image_frame_opencv.h | head
grep -rn "ImageFrame>\|formats:image_frame\b" mediapipe/calculators/image/BUILD | head -3
```
Use `mediapipe::formats::MatView(&image_frame)` (returns a `cv::Mat` view) — confirm the exact namespace/signature from the header.

- [ ] **Step 2: Write the options proto** (`471230005`):

```proto
// (full Apache header)
syntax = "proto2";
package mediapipe;
import "mediapipe/framework/calculator.proto";

message StreamingTilesToTensorBatchCalculatorOptions {
  extend .mediapipe.CalculatorOptions {
    optional StreamingTilesToTensorBatchCalculatorOptions ext = 471230005;
  }
  // If true, treat the model as dynamic-batch (emit N = valid_count, no
  // padding). If false (fixed batch), emit N = batch_capacity with padding.
  // When METADATA.is_dynamic_batch is available it overrides this.
  optional bool dynamic_batch = 1 [default = false];
}
```

- [ ] **Step 3: Write the failing test** (drives batching semantics with a synthetic 100x100 white frame and tile counts around capacity):

Create `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator_test.cc`:

```cpp
// (full Apache header)
#include <memory>
#include <vector>

#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

InferenceMetadata Meta(int batch_capacity, int h, int w, int c,
                       bool dynamic) {
  InferenceMetadata md;
  md.set_batch_capacity(batch_capacity);
  md.set_input_height(h); md.set_input_width(w); md.set_input_channels(c);
  md.set_is_dynamic_batch(dynamic);
  md.set_tensor_layout("BHWC");
  return md;
}

std::unique_ptr<ImageFrame> WhiteFrame(int w, int h) {
  auto f = std::make_unique<ImageFrame>(ImageFormat::SRGB, w, h);
  std::memset(f->MutablePixelData(), 255,
              f->Height() * f->WidthStep());
  return f;
}

TilePlan TwoTiles() {
  TilePlan p;
  TileGeometry a; a.tile_index = 0; a.x_center = .25; a.y_center = .5;
  a.width = .5; a.height = 1.0; p.tiles.push_back(a);
  TileGeometry b; b.tile_index = 1; b.x_center = .75; b.y_center = .5;
  b.width = .5; b.height = 1.0; p.tiles.push_back(b);
  return p;
}

// Fixed-batch capacity 4, 2 tiles -> one padded batch N=4, valid_count=2.
TEST(StreamingTilesTest, FixedBatchPadsToCapacity) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    input_side_packet: "METADATA:meta"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
  )pb"));
  runner.MutableSidePackets()->Tag("METADATA") =
      MakePacket<InferenceMetadata>(Meta(4, 8, 8, 3, /*dynamic=*/false));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      Adopt(WhiteFrame(16, 16).release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(TwoTiles()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& tpk = runner.Outputs().Tag("TENSORS").packets;
  const auto& ipk = runner.Outputs().Tag("BATCH_INFO").packets;
  ASSERT_EQ(tpk.size(), 1);
  ASSERT_EQ(ipk.size(), 1);
  const auto& tensors = tpk[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(tensors.size(), 1);
  EXPECT_EQ(tensors[0].shape().dims[0], 4);  // N == batch_capacity
  EXPECT_EQ(tensors[0].shape().dims[1], 8);
  EXPECT_EQ(tensors[0].shape().dims[2], 8);
  EXPECT_EQ(tensors[0].shape().dims[3], 3);
  const auto& info = ipk[0].Get<TensorBatchInfo>();
  EXPECT_EQ(info.batch_capacity, 4);
  EXPECT_EQ(info.valid_count, 2);
  EXPECT_EQ(info.total_batches, 1);
  ASSERT_EQ(info.tile_indices.size(), 2);
  EXPECT_EQ(info.tile_indices[0], 0);
}

// Dynamic batch, 2 tiles, capacity 4 -> one batch N=2 (no padding).
TEST(StreamingTilesTest, DynamicBatchNoPadding) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    input_side_packet: "METADATA:meta"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
  )pb"));
  runner.MutableSidePackets()->Tag("METADATA") =
      MakePacket<InferenceMetadata>(Meta(4, 8, 8, 3, /*dynamic=*/true));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      Adopt(WhiteFrame(16, 16).release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(TwoTiles()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& tensors =
      runner.Outputs().Tag("TENSORS").packets[0].Get<std::vector<Tensor>>();
  EXPECT_EQ(tensors[0].shape().dims[0], 2);  // N == valid_count
  const auto& info =
      runner.Outputs().Tag("BATCH_INFO").packets[0].Get<TensorBatchInfo>();
  EXPECT_EQ(info.valid_count, 2);
}

}  // namespace
}  // namespace mediapipe
```

Add `#include <cstring>` for `memset`.

- [ ] **Step 4: Write the calculator**

Create `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc`:

```cpp
// (full Apache header)
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.pb.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// Streams externally-supplied tiles into batched [N,H,W,C] float32 tensors.
// CPU implementation: crop -> resize -> normalize into batch rows.
class StreamingTilesToTensorBatchCalculator : public Node {
 public:
  static constexpr Input<ImageFrame> kInImage{"IMAGE"};
  static constexpr Input<TilePlan> kInPlan{"TILE_PLAN"};
  static constexpr SideInput<InferenceMetadata> kSideMeta{"METADATA"};
  static constexpr Output<std::vector<Tensor>> kOutTensors{"TENSORS"};
  static constexpr Output<TensorBatchInfo> kOutInfo{"BATCH_INFO"};
  MEDIAPIPE_NODE_CONTRACT(kInImage, kInPlan, kSideMeta, kOutTensors, kOutInfo);

  absl::Status Open(CalculatorContext* cc) override {
    options_ =
        cc->Options<mediapipe::StreamingTilesToTensorBatchCalculatorOptions>();
    meta_ = kSideMeta(cc).Get();
    RET_CHECK_GT(meta_.input_height(), 0);
    RET_CHECK_GT(meta_.input_width(), 0);
    RET_CHECK_GT(meta_.input_channels(), 0);
    RET_CHECK_GT(meta_.batch_capacity(), 0);
    dynamic_batch_ = meta_.is_dynamic_batch() || options_.dynamic_batch();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const ImageFrame& frame = *kInImage(cc);
    const TilePlan& plan = *kInPlan(cc);
    const int H = meta_.input_height(), W = meta_.input_width(),
              C = meta_.input_channels();
    const int cap = meta_.batch_capacity();
    cv::Mat src = formats::MatView(&frame);  // confirm namespace in Step 1
    const int fw = frame.Width(), fh = frame.Height();

    const int T = static_cast<int>(plan.tiles.size());
    const int total_batches = (T + cap - 1) / cap;  // 0 when T == 0
    const int64_t ts = cc->InputTimestamp().Value();

    int emitted = 0;
    for (int start = 0; start < T; start += cap) {
      const int rows = std::min(cap, T - start);
      const int N = dynamic_batch_ ? rows : cap;
      Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{N, H, W, C});
      auto write = tensor.GetCpuWriteView();
      float* buf = write.buffer<float>();
      std::memset(buf, 0, sizeof(float) * N * H * W * C);  // pad rows = 0

      TensorBatchInfo info;
      info.source_frame_timestamp = ts;
      info.batch_index = emitted;
      info.total_batches = total_batches;
      info.batch_capacity = N;
      info.valid_count = rows;

      for (int r = 0; r < rows; ++r) {
        const TileGeometry& g = plan.tiles[start + r];
        // Crop tile ROI from the source frame (clamped to image bounds).
        int rx = std::lround(g.x0() * fw), ry = std::lround(g.y0() * fh);
        int rw = std::lround(g.width * fw), rh = std::lround(g.height * fh);
        rx = std::clamp(rx, 0, fw - 1); ry = std::clamp(ry, 0, fh - 1);
        rw = std::clamp(rw, 1, fw - rx); rh = std::clamp(rh, 1, fh - ry);
        cv::Mat roi = src(cv::Rect(rx, ry, rw, rh));
        cv::Mat resized;
        cv::resize(roi, resized, cv::Size(W, H));
        cv::Mat f32;
        resized.convertTo(f32, CV_32FC(C), 1.0 / 255.0);  // normalize [0,1]
        // Copy HWC floats into row r of the batch buffer.
        std::memcpy(buf + static_cast<size_t>(r) * H * W * C,
                    f32.ptr<float>(0), sizeof(float) * H * W * C);
        info.tile_indices.push_back(g.tile_index);
      }

      auto tensors = std::make_unique<std::vector<Tensor>>();
      tensors->push_back(std::move(tensor));
      kOutTensors(cc).Send(std::move(tensors));
      kOutInfo(cc).Send(MakePacket<TensorBatchInfo>(std::move(info)));
      ++emitted;
    }
    return absl::OkStatus();
  }

 private:
  mediapipe::StreamingTilesToTensorBatchCalculatorOptions options_;
  InferenceMetadata meta_;
  bool dynamic_batch_ = false;
};

MEDIAPIPE_REGISTER_NODE(StreamingTilesToTensorBatchCalculator);

}  // namespace api2
}  // namespace mediapipe
```

Implementer notes:
- Confirm `formats::MatView` namespace/signature from Step 1; the channel count `C` should match the frame's channels (SRGB=3). If the model wants a different channel order/count, that is a later concern — the test uses 3-channel SRGB.
- Emitting multiple `(TENSORS, BATCH_INFO)` packet pairs in one `Process` at the same input timestamp will violate monotonic timestamps on a stream. For this CPU plan, the test uses `T <= cap` (single batch), so it is correct as written. **For `T > cap` (multiple batches per frame), the batches must carry distinct timestamps** — see Task 3b note below; if you implement multi-batch emission now, use `cc->Outputs()...At(Timestamp)` with per-batch timestamps and set `SetOffset`/timestamp bounds appropriately. Keep the single-batch path green first; add multi-batch emission only with its own test.

- [ ] **Step 5: BUILD targets** (proto + lib + test). The library deps must include:
```python
        ":streaming_tiles_to_tensor_batch_calculator_cc_proto",
        ":tiling_types",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:inference_metadata_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status",
```
The test deps add `:calculator_runner`, `formats:image_frame`, `formats:inference_metadata_cc_proto`, `formats:tensor`, and the port test libs. Confirm `image_frame_opencv` label via `grep -n "name = \"image_frame_opencv\"" mediapipe/framework/formats/BUILD`.

- [ ] **Step 6: Run the two tests**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test --test_output=all
```
Expected: PASS (2 tests: fixed-batch padding, dynamic no-padding).

- [ ] **Step 7: Commit**

```bash
git add mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.proto mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "feat(tiling): StreamingTilesToTensorBatchCalculator (CPU, single batch)"
```

---

### Task 4: `MergeTileDetectionsAccumulatorCalculator` (OBB)

**Files:**
- Create: `mediapipe/calculators/tensor/merge_tile_detections_accumulator_calculator.cc`
- Create: `mediapipe/calculators/tensor/merge_tile_detections_accumulator_calculator_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

Inputs: `ORIENTED_DETECTIONS` (`std::vector<std::vector<OrientedDetection>>`, batched per row), `BATCH_INFO` (TensorBatchInfo), `TILE_PLAN` (TilePlan). It drops padded rows, projects each valid row's tile-local boxes to full-frame-normalized coords, accumulates across the frame's batches, and emits the flattened `std::vector<OrientedDetection>` when `total_batches` have arrived. (Axis-aligned tiles → angle is unchanged by projection.)

For a single-batch frame (`total_batches == 1`) — the path Task 3 produces — accumulation is immediate. Multi-batch accumulation keyed by `source_frame_timestamp` is included but its multi-batch test is deferred with the multi-batch emitter (Task 3b note).

- [ ] **Step 1: Write the failing test** (single batch, two tiles, verify projection):

Create `mediapipe/calculators/tensor/merge_tile_detections_accumulator_calculator_test.cc`:

```cpp
// (full Apache header)
#include <memory>
#include <vector>

#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

OrientedDetection Obb(float cx, float cy, float w, float h) {
  OrientedDetection d;
  d.set_cx(cx); d.set_cy(cy); d.set_width(w); d.set_height(h);
  d.set_rotation(0.0f); d.add_score(0.9f); d.add_label_id(0);
  return d;
}

TEST(MergeTileAccumulatorTest, ProjectsTileLocalToFrame) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "MergeTileDetectionsAccumulatorCalculator"
    input_stream: "ORIENTED_DETECTIONS:dets"
    input_stream: "BATCH_INFO:info"
    input_stream: "TILE_PLAN:plan"
    output_stream: "ORIENTED_DETECTIONS:merged"
  )pb"));

  // Frame split into left/right halves. Left tile [x0=0,w=0.5], right [x0=0.5,w=0.5].
  TilePlan plan;
  TileGeometry l; l.tile_index=0; l.x_center=.25; l.y_center=.5; l.width=.5; l.height=1.0;
  TileGeometry r; r.tile_index=1; r.x_center=.75; r.y_center=.5; r.width=.5; r.height=1.0;
  plan.tiles = {l, r};

  // Row 0 (left tile): a box centered tile-local (0.5,0.5) size (0.4,0.4).
  // Row 1 (right tile): a box centered tile-local (0.5,0.5) size (0.4,0.4).
  auto batch = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch->push_back({Obb(0.5f, 0.5f, 0.4f, 0.4f)});
  batch->push_back({Obb(0.5f, 0.5f, 0.4f, 0.4f)});

  TensorBatchInfo info;
  info.source_frame_timestamp = 0; info.batch_index = 0; info.total_batches = 1;
  info.batch_capacity = 2; info.valid_count = 2; info.tile_indices = {0, 1};

  runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(batch.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(plan).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& merged = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                           .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(merged.size(), 2);
  // Left tile box center (0.5 local) -> frame x = 0 + 0.5*0.5 = 0.25.
  EXPECT_NEAR(merged[0].cx(), 0.25f, 1e-5);
  EXPECT_NEAR(merged[0].cy(), 0.5f, 1e-5);
  EXPECT_NEAR(merged[0].width(), 0.20f, 1e-5);   // 0.4 local * 0.5 tile = 0.2
  EXPECT_NEAR(merged[0].height(), 0.40f, 1e-5);  // 0.4 * 1.0
  // Right tile box center -> frame x = 0.5 + 0.5*0.5 = 0.75.
  EXPECT_NEAR(merged[1].cx(), 0.75f, 1e-5);
}

TEST(MergeTileAccumulatorTest, DropsPaddedRows) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "MergeTileDetectionsAccumulatorCalculator"
    input_stream: "ORIENTED_DETECTIONS:dets"
    input_stream: "BATCH_INFO:info"
    input_stream: "TILE_PLAN:plan"
    output_stream: "ORIENTED_DETECTIONS:merged"
  )pb"));

  TilePlan plan;
  TileGeometry t; t.tile_index=0; t.x_center=.5; t.y_center=.5; t.width=1.0; t.height=1.0;
  plan.tiles = {t};

  // Batch capacity 2, valid_count 1: row 1 is padding and must be ignored
  // even though the decoder emitted a (garbage) detection for it.
  auto batch = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch->push_back({Obb(0.5f, 0.5f, 0.2f, 0.2f)});
  batch->push_back({Obb(0.5f, 0.5f, 0.9f, 0.9f)});  // padded row -> dropped
  TensorBatchInfo info;
  info.source_frame_timestamp = 0; info.total_batches = 1; info.batch_capacity = 2;
  info.valid_count = 1; info.tile_indices = {0};

  runner.MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(batch.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("BATCH_INFO").packets.push_back(
      MakePacket<TensorBatchInfo>(info).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(plan).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& merged = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                           .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(merged.size(), 1);  // padded row dropped
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 2: Write the calculator**

Create `mediapipe/calculators/tensor/merge_tile_detections_accumulator_calculator.cc`:

```cpp
// (full Apache header)
#include <map>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// Regroups batched per-tile oriented detections back to the source frame,
// dropping padded rows and projecting tile-local boxes to full-frame coords.
// Emits the flattened frame detections once all batches arrive. Axis-aligned
// tiles only: projection is scale + offset; angle is preserved.
class MergeTileDetectionsAccumulatorCalculator : public Node {
 public:
  static constexpr Input<std::vector<std::vector<OrientedDetection>>> kInDets{
      "ORIENTED_DETECTIONS"};
  static constexpr Input<TensorBatchInfo> kInInfo{"BATCH_INFO"};
  static constexpr Input<TilePlan> kInPlan{"TILE_PLAN"};
  static constexpr Output<std::vector<OrientedDetection>> kOut{
      "ORIENTED_DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInDets, kInInfo, kInPlan, kOut);

  absl::Status Process(CalculatorContext* cc) override {
    const auto& batch = *kInDets(cc);
    const TensorBatchInfo& info = *kInInfo(cc);
    const TilePlan& plan = *kInPlan(cc);
    RET_CHECK_EQ(info.valid_count, static_cast<int>(info.tile_indices.size()));

    auto& acc = pending_[info.source_frame_timestamp];
    // Project only valid rows [0, valid_count); padded rows are ignored.
    for (int r = 0; r < info.valid_count; ++r) {
      const int tile_index = info.tile_indices[r];
      RET_CHECK_LT(tile_index, static_cast<int>(plan.tiles.size()));
      const TileGeometry& g = plan.tiles[tile_index];
      if (r >= static_cast<int>(batch.size())) continue;
      for (const OrientedDetection& d : batch[r]) {
        OrientedDetection out = d;
        out.set_cx(g.x0() + d.cx() * g.width);
        out.set_cy(g.y0() + d.cy() * g.height);
        out.set_width(d.width() * g.width);
        out.set_height(d.height() * g.height);
        // rotation preserved (axis-aligned tile).
        acc.received_dets.push_back(std::move(out));
      }
    }
    acc.batches_seen += 1;
    acc.total_batches = info.total_batches;

    if (acc.batches_seen >= acc.total_batches) {
      auto merged =
          std::make_unique<std::vector<OrientedDetection>>(std::move(acc.received_dets));
      pending_.erase(info.source_frame_timestamp);
      kOut(cc).Send(std::move(merged));
    }
    return absl::OkStatus();
  }

 private:
  struct FrameAcc {
    std::vector<OrientedDetection> received_dets;
    int batches_seen = 0;
    int total_batches = 1;
  };
  std::map<int64_t, FrameAcc> pending_;
};

MEDIAPIPE_REGISTER_NODE(MergeTileDetectionsAccumulatorCalculator);

}  // namespace api2
}  // namespace mediapipe
```

Note: emitting on the timestamp of the LAST batch is correct for single-batch frames (the common Task-3 path). For multi-batch frames the output should carry the source frame timestamp; when the multi-batch emitter (Task 3b note) lands, set the output timestamp explicitly to `info.source_frame_timestamp` via `kOut(cc).Send(..., Timestamp(info.source_frame_timestamp))` and add a multi-batch test.

- [ ] **Step 3: BUILD targets** (lib + test). Lib deps: `:tiling_types`, `calculator_framework`, `api2:node`, `formats:oriented_detection_cc_proto`, `port:ret_check`, `absl/status`. Test deps add `:calculator_runner`, `formats:oriented_detection_cc_proto`, port test libs, `:tiling_types`.

- [ ] **Step 4: Run the tests**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:merge_tile_detections_accumulator_calculator_test --test_output=all
```
Expected: PASS (2 tests: projection correctness, padded-row drop).

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/merge_tile_detections_accumulator_calculator.cc mediapipe/calculators/tensor/merge_tile_detections_accumulator_calculator_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "feat(tiling): MergeTileDetectionsAccumulator (project + drop padding)"
```

---

### Task 5: End-to-end tiling→merge→NMS graph round-trip

**Files:**
- Create: `mediapipe/calculators/tensor/tiled_obb_pipeline_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

This wires the tiling pipeline **without real inference** — feeding known tile-local detections directly into merge + rotated NMS — to prove the geometry + suppression compose into one frame-global result. (A true full-model test requires a YOLO-OBB `.tflite` asset; see the note.)

- [ ] **Step 1: Write the graph test**

Create `mediapipe/calculators/tensor/tiled_obb_pipeline_test.cc`:

```cpp
// (full Apache header)
#include <memory>
#include <vector>

#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

// merge -> rotated NMS. Two tiles each detect the SAME real-world object in
// their overlap; after projection the two boxes coincide and global NMS keeps
// one. Proves the single frame-global NMS dedups across tiles.
TEST(TiledObbPipelineTest, MergeThenGlobalNmsDedupsAcrossTiles) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    input_stream: "plan"
    output_stream: "out"
    node {
      calculator: "MergeTileDetectionsAccumulatorCalculator"
      input_stream: "ORIENTED_DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
      input_stream: "TILE_PLAN:plan"
      output_stream: "ORIENTED_DETECTIONS:merged"
    }
    node {
      calculator: "RotatedNonMaxSuppressionCalculator"
      input_stream: "ORIENTED_DETECTIONS:merged"
      output_stream: "ORIENTED_DETECTIONS:out"
      options {
        [mediapipe.RotatedNonMaxSuppressionCalculatorOptions.ext] {
          iou_threshold: 0.5
        }
      }
    }
  )pb");

  // Overlapping tiles: left covers x[0,0.6], right covers x[0.4,1.0].
  TilePlan plan;
  TileGeometry l; l.tile_index=0; l.x_center=.3; l.y_center=.5; l.width=.6; l.height=1.0;
  TileGeometry r; r.tile_index=1; r.x_center=.7; r.y_center=.5; r.width=.6; r.height=1.0;
  plan.tiles = {l, r};

  // Object truly at frame (0.5,0.5,0.2,0.2). In left tile local coords:
  //   cx = (0.5-0.0)/0.6 = 0.8333..., w = 0.2/0.6 = 0.3333...
  // In right tile local coords:
  //   cx = (0.5-0.4)/0.6 = 0.16667, w = 0.3333...
  auto obb = [](float cx, float cy, float w, float h, float s) {
    OrientedDetection d; d.set_cx(cx); d.set_cy(cy); d.set_width(w);
    d.set_height(h); d.set_rotation(0); d.add_score(s); d.add_label_id(0);
    return d;
  };
  auto batch = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
  batch->push_back({obb(0.83333f, 0.5f, 0.33333f, 0.2f, 0.9f)});   // left
  batch->push_back({obb(0.16667f, 0.5f, 0.33333f, 0.2f, 0.8f)});   // right
  TensorBatchInfo info;
  info.source_frame_timestamp=0; info.total_batches=1; info.batch_capacity=2;
  info.valid_count=2; info.tile_indices={0,1};

  std::vector<Packet> out_packets;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("out", [&](const Packet& p) {
    out_packets.push_back(p); return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "plan", MakePacket<TilePlan>(plan).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info", MakePacket<TensorBatchInfo>(info).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets", Adopt(batch.release()).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(out_packets.size(), 1);
  const auto& out = out_packets[0].Get<std::vector<OrientedDetection>>();
  // Both tiles projected to ~(0.5,0.5,0.2,0.2); global NMS keeps ONE.
  ASSERT_EQ(out.size(), 1);
  EXPECT_NEAR(out[0].cx(), 0.5f, 1e-3);
  EXPECT_NEAR(out[0].width(), 0.2f, 1e-3);
  EXPECT_NEAR(out[0].score(0), 0.9f, 1e-5);  // higher-scoring kept
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 2: BUILD target** — a `cc_test` `tiled_obb_pipeline_test` depending on `:merge_tile_detections_accumulator_calculator`, `//mediapipe/calculators/util:rotated_non_max_suppression_calculator`, `:tiling_types`, `formats:oriented_detection_cc_proto`, `:calculator_framework`, and port test libs.

- [ ] **Step 3: Run**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tiled_obb_pipeline_test --test_output=all
```
Expected: PASS — projection maps both tile detections onto the same frame box; one survives global NMS. If the two projected boxes do not reach IoU ≥ 0.5 (rounding), tighten the local-coord literals so the projected centers/sizes match within 1e-3.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/calculators/tensor/tiled_obb_pipeline_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "test(tiling): tile-merge + global rotated-NMS round trip"
```

---

## Done criteria (Plan 3b)

- All four test targets green: `tile_spec_to_tile_plan_calculator_test`, `streaming_tiles_to_tensor_batch_calculator_test`, `merge_tile_detections_accumulator_calculator_test`, `tiled_obb_pipeline_test`.
- Tiles are validated → packed into metadata-sized `[N,H,W,C]` batches (fixed-batch padding & dynamic no-padding) → decoded per row → padded rows dropped → projected to full-frame coords → one frame-global NMS.
- No existing code modified except additive BUILD entries.

## Deferred within Plan 3b (documented, not silently skipped)

- **Multi-batch-per-frame emission (`T > batch_capacity`)**: the streaming batcher and accumulator carry the timestamp/`batch_index`/`total_batches` fields and accumulation logic, but the multi-batch *emit-with-distinct-timestamps* path + its tests are deferred (Task 3/4 notes). The single-batch path is complete and tested.
- **Axis-aligned `Detection` variant** of the merge accumulator (Plan 1 path): mirror Task 4 for `std::vector<std::vector<Detection>>` → projected `std::vector<Detection>` → existing axis-aligned NMS. Same projection math on `RELATIVE_BOUNDING_BOX`. Add as a follow-on calculator when the detect (non-OBB) tiled graph is needed.
- **Rotated tiles**: rejected today (Task 2). Detections still carry rotation; only the tile crops are axis-aligned.
- **Full-model integration** with a real YOLO-OBB `.tflite` (tiles→batch→InferenceCalculator→OBB decode→merge→rotated NMS) requires adding a model asset to testdata; wire it once a fixture model is available, reusing the metadata side packet from Plan 3a.
