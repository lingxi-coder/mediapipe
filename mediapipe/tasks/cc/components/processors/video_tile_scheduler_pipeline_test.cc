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

// Integration test: VideoTileSchedulerCalculator →
//                  TileSpecToTilePlanCalculator →
//                  StreamingTilesToTensorBatchCalculator
//
// Tests the scheduler↔tiling seam, plus the SKIP/empty-frame protocol through
// the decoder-bound-propagation → merge-accumulator chain (no real inference;
// the decoder never fires on an empty frame, only its bound advances).

#include <cstring>
#include <memory>
#include <vector>

#include "mediapipe/framework/formats/tiling_cache_stats.h"
#include "mediapipe/framework/formats/tiling_types.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

// ---------------------------------------------------------------------------
// Helpers (mirrored from streaming_tiles_to_tensor_batch_calculator_test.cc)
// ---------------------------------------------------------------------------

InferenceMetadata Meta(int batch_capacity, int h, int w, int c, bool dynamic) {
  InferenceMetadata md;
  md.set_batch_capacity(batch_capacity);
  md.set_input_height(h);
  md.set_input_width(w);
  md.set_input_channels(c);
  md.set_is_dynamic_batch(dynamic);
  md.set_tensor_layout("BHWC");
  return md;
}

std::unique_ptr<ImageFrame> WhiteFrame(int w, int h) {
  auto f = std::make_unique<ImageFrame>(ImageFormat::SRGB, w, h);
  std::memset(f->MutablePixelData(), 255, f->Height() * f->WidthStep());
  return f;
}

// Five vertical strips spanning the full frame width and height.
// Each strip has width 0.2, height 1.0, centered at x = 0.1, 0.3, 0.5, 0.7,
// 0.9 respectively.
std::vector<NormalizedRect> FiveTiles() {
  std::vector<NormalizedRect> tiles;
  for (int i = 0; i < 5; ++i) {
    NormalizedRect r;
    r.set_x_center(0.1f + i * 0.2f);
    r.set_y_center(0.5f);
    r.set_width(0.2f);
    r.set_height(1.0f);
    tiles.push_back(r);
  }
  return tiles;
}

// Graph config for the single-frame multi-batch test.
//   VideoTileSchedulerCalculator (no TRACKING, detect_without_tracking=true)
//   → TileSpecToTilePlanCalculator
//   → StreamingTilesToTensorBatchCalculator (fixed batch, cap=2)
//
// With no TRACKING stream and detect_without_tracking defaulting to true, the
// scheduler always takes the DETECT path and emits all base tiles.
constexpr char kBaseGraphConfig[] = R"pb(
  input_stream: "image"
  input_stream: "base_tiles"
  input_stream: "priors"
  input_side_packet: "meta"
  node {
    calculator: "VideoTileSchedulerCalculator"
    input_stream: "TILES:base_tiles"
    input_stream: "PRIOR_DETECTIONS:priors"
    output_stream: "TILES:sched"
    output_stream: "REFRESH:refresh"
  }
  node {
    calculator: "TileSpecToTilePlanCalculator"
    input_stream: "TILES:sched"
    output_stream: "TILE_PLAN:plan"
  }
  node {
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    input_side_packet: "METADATA:meta"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
  }
)pb";

// Graph config for the cache-hit test.  TileSpec and StreamingTiles both have
// caches enabled and emit CACHE_STATS.
constexpr char kCacheGraphConfig[] = R"pb(
  input_stream: "image"
  input_stream: "base_tiles"
  input_stream: "priors"
  input_side_packet: "meta"
  node {
    calculator: "VideoTileSchedulerCalculator"
    input_stream: "TILES:base_tiles"
    input_stream: "PRIOR_DETECTIONS:priors"
    output_stream: "TILES:sched"
    output_stream: "REFRESH:refresh"
  }
  node {
    calculator: "TileSpecToTilePlanCalculator"
    input_stream: "TILES:sched"
    output_stream: "TILE_PLAN:plan"
    output_stream: "CACHE_STATS:plan_stats"
    options {
      [mediapipe.TileSpecToTilePlanCalculatorOptions.ext] {
        max_cached_tile_plans: 4
        emit_cache_stats: true
      }
    }
  }
  node {
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    input_side_packet: "METADATA:meta"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
    output_stream: "CACHE_STATS:matrix_stats"
    options {
      [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
        max_cached_tile_matrices: 8
        emit_cache_stats: true
      }
    }
  }
)pb";

// Graph config for the SKIP/empty-frame e2e. The scheduler is forced to SKIP
// every frame (no TRACKING wired + detect_without_tracking=false), so the
// batcher emits a BATCH_INFO-only empty frame (TENSORS bound advance, no
// packet). That bound must propagate through the never-firing OBB decoder so
// the merge accumulator emits an EMPTY result at each source timestamp
// instead of stalling or crashing on the missing detections packet.
constexpr char kSkipFrameGraphConfig[] = R"pb(
  input_stream: "image"
  input_stream: "base_tiles"
  input_stream: "priors"
  input_side_packet: "meta"
  node {
    calculator: "VideoTileSchedulerCalculator"
    input_stream: "TILES:base_tiles"
    input_stream: "PRIOR_DETECTIONS:priors"
    output_stream: "TILES:sched"
    output_stream: "REFRESH:refresh"
    options {
      [mediapipe.VideoTileSchedulerCalculatorOptions.ext] {
        detect_without_tracking: false
      }
    }
  }
  node {
    calculator: "TileSpecToTilePlanCalculator"
    input_stream: "TILES:sched"
    output_stream: "TILE_PLAN:plan"
  }
  node {
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    input_side_packet: "METADATA:meta"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
  }
  node {
    calculator: "YoloObbTensorsToOrientedDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "ORIENTED_DETECTIONS:dets"
    options {
      [mediapipe.YoloObbTensorsToOrientedDetectionsCalculatorOptions.ext] {
        num_classes: 15
      }
    }
  }
  node {
    calculator: "MergeTileDetectionsAccumulatorCalculator"
    input_stream: "ORIENTED_DETECTIONS:dets"
    input_stream: "BATCH_INFO:info"
    output_stream: "ORIENTED_DETECTIONS:merged"
  }
)pb";

// ---------------------------------------------------------------------------
// Test 1: T=5 tiles, batch_capacity=2 → 3 batches (2,2,1).
//
// NOTE on streaming vs. bulk materialization:
//   StreamingTilesToTensorBatchCalculator emits each batch via a separate
//   Send() inside a single Process() call (following the BeginLoopCalculator
//   synthetic-timestamp pattern).  In a single-threaded CPU graph the
//   ObserveOutputStream spy receives all 3 batch packets in batch_index order
//   after the Process() call returns.  Intra-Process ordering (e.g. "batch 0
//   was tensorized before batch 1 started") is not separately observable
//   without added instrumentation, and the shipped calculator is NOT modified
//   here.  The per-batch CONTRACT verified below (3 ordered batches with
//   correct valid_counts, shared source timestamp, padded tensor N=2) is the
//   appropriate streaming evidence for this integration-test scope.
// ---------------------------------------------------------------------------
TEST(VideoTileSchedulerPipelineTest, MultiBatchFiveTilesCapTwo) {
  auto config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(kBaseGraphConfig);

  std::vector<Packet> info_packets;
  std::vector<Packet> tensors_packets;

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "info", [&](const Packet& p) -> absl::Status {
        info_packets.push_back(p);
        return absl::OkStatus();
      }));
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "tensors", [&](const Packet& p) -> absl::Status {
        tensors_packets.push_back(p);
        return absl::OkStatus();
      }));

  // Side packet: batch_capacity=2, 8x8 crop target, 3 channels, fixed batch.
  constexpr int kH = 8, kW = 8, kC = 3;
  MP_ASSERT_OK(graph.StartRun({{"meta",
                                MakePacket<InferenceMetadata>(
                                    Meta(/*batch_capacity=*/2, kH, kW, kC,
                                         /*dynamic=*/false))}}));

  // Feed one source frame at Timestamp(0).
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image", Adopt(WhiteFrame(64, 64).release()).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "base_tiles",
      Adopt(new std::vector<NormalizedRect>(FiveTiles())).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "priors",
      Adopt(new std::vector<Detection>()).At(Timestamp(0))));

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  // 5 tiles / cap 2 → ceil(5/2) = 3 batches.
  ASSERT_EQ(info_packets.size(), 3u);
  ASSERT_EQ(tensors_packets.size(), 3u);

  // Verify per-batch contract.
  const int64_t kSourceTs = 0;
  const int expected_valid[] = {2, 2, 1};
  for (int i = 0; i < 3; ++i) {
    const TensorBatchInfo& info = info_packets[i].Get<TensorBatchInfo>();

    EXPECT_EQ(info.batch_index, i)
        << "batch_index mismatch at packet " << i;
    EXPECT_EQ(info.total_batches, 3)
        << "total_batches mismatch at packet " << i;
    EXPECT_EQ(info.valid_count, expected_valid[i])
        << "valid_count mismatch at packet " << i;
    EXPECT_EQ(info.source_frame_timestamp, kSourceTs)
        << "source_frame_timestamp mismatch at packet " << i;

    // Fixed batch → N=batch_capacity=2 for all batches (last padded).
    EXPECT_EQ(info.batch_size, 2)
        << "batch_size mismatch at packet " << i;

    // Check tensor dim[0] == 2 (fixed batch N=2).
    const auto& tensors = tensors_packets[i].Get<std::vector<Tensor>>();
    ASSERT_EQ(tensors.size(), 1u);
    EXPECT_EQ(tensors[0].shape().dims[0], 2)
        << "tensor dim[0] (N) mismatch at packet " << i;
    EXPECT_EQ(tensors[0].shape().dims[1], kH);
    EXPECT_EQ(tensors[0].shape().dims[2], kW);
    EXPECT_EQ(tensors[0].shape().dims[3], kC);
  }
}

// ---------------------------------------------------------------------------
// Test 2: Cache-hit test.
//
// Drive TWO frames (Timestamps 0 and 1) with identical image + identical 5
// tiles.  Both TileSpecToTilePlanCalculator and
// StreamingTilesToTensorBatchCalculator have caches enabled.  After the second
// frame the cumulative CACHE_STATS must show at least one hit in each cache.
//
// The scheduler is deterministic (no tracking, detect_without_tracking=true)
// so both frames produce identical scheduled tile sets → identical TilePlan →
// identical per-batch geometry keys → cache hits on frame 2.
// ---------------------------------------------------------------------------
TEST(VideoTileSchedulerPipelineTest, CacheHitsOnRepeatedIdenticalFrames) {
  auto config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(kCacheGraphConfig);

  std::vector<Packet> plan_stats_packets;
  std::vector<Packet> matrix_stats_packets;

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "plan_stats", [&](const Packet& p) -> absl::Status {
        plan_stats_packets.push_back(p);
        return absl::OkStatus();
      }));
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "matrix_stats", [&](const Packet& p) -> absl::Status {
        matrix_stats_packets.push_back(p);
        return absl::OkStatus();
      }));

  constexpr int kH = 8, kW = 8, kC = 3;
  MP_ASSERT_OK(graph.StartRun({{"meta",
                                MakePacket<InferenceMetadata>(
                                    Meta(/*batch_capacity=*/2, kH, kW, kC,
                                         /*dynamic=*/false))}}));

  // Feed two identical frames at Timestamps 0 and 1.
  for (int t = 0; t < 2; ++t) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image",
        Adopt(WhiteFrame(64, 64).release()).At(Timestamp(t))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "base_tiles",
        Adopt(new std::vector<NormalizedRect>(FiveTiles())).At(Timestamp(t))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "priors",
        Adopt(new std::vector<Detection>()).At(Timestamp(t))));
  }

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  // TileSpec emits one CACHE_STATS per source frame (2 total).
  ASSERT_GE(plan_stats_packets.size(), 1u);
  const TilingCacheStats& last_plan_stats =
      plan_stats_packets.back().Get<TilingCacheStats>();
  // Second frame's identical tile-set must hit the plan cache.
  EXPECT_GE(last_plan_stats.tile_plan.hits, 1)
      << "TileSpecToTilePlanCalculator: expected at least one tile_plan cache "
         "hit (second frame with identical tiles should hit the plan cache)";

  // StreamingTiles emits one CACHE_STATS per source frame (2 total).
  // Each frame has 3 batches, but CACHE_STATS is one packet per frame.
  ASSERT_GE(matrix_stats_packets.size(), 1u);
  const TilingCacheStats& last_matrix_stats =
      matrix_stats_packets.back().Get<TilingCacheStats>();
  // Second frame's per-batch geometry keys are identical to frame 1's →
  // at least one tile_matrix cache hit per batch group.
  EXPECT_GE(last_matrix_stats.tile_matrix.hits, 1)
      << "StreamingTilesToTensorBatchCalculator: expected at least one "
         "tile_matrix cache hit (second frame with identical geometry)";
}

// ---------------------------------------------------------------------------
// Test 3: SKIP/empty-frame e2e through the real chain.
//
// Every frame is SKIPped (no TRACKING + detect_without_tracking=false), so
// the producer emits a BATCH_INFO-only empty frame and advances the TENSORS
// bound. The decoder never fires (no TENSORS packet); its output bound
// propagation alone must let the merge accumulator emit an EMPTY merged
// result at each source timestamp — without stalling and without crashing on
// the missing detections packet.
// ---------------------------------------------------------------------------
TEST(VideoTileSchedulerPipelineTest, SkipFramesEmitEmptyMergedResults) {
  auto config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(kSkipFrameGraphConfig);

  std::vector<Packet> merged_packets;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "merged", [&](const Packet& p) -> absl::Status {
        merged_packets.push_back(p);
        return absl::OkStatus();
      }));

  constexpr int kH = 8, kW = 8, kC = 3;
  MP_ASSERT_OK(graph.StartRun({{"meta",
                                MakePacket<InferenceMetadata>(
                                    Meta(/*batch_capacity=*/2, kH, kW, kC,
                                         /*dynamic=*/false))}}));

  // Two frames; the scheduler SKIPs both → two empty TilePlans.
  for (int t = 0; t < 2; ++t) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image", Adopt(WhiteFrame(64, 64).release()).At(Timestamp(t))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "base_tiles",
        Adopt(new std::vector<NormalizedRect>(FiveTiles())).At(Timestamp(t))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "priors", Adopt(new std::vector<Detection>()).At(Timestamp(t))));
  }

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(merged_packets.size(), 2u);
  for (int t = 0; t < 2; ++t) {
    EXPECT_EQ(merged_packets[t].Timestamp(), Timestamp(t))
        << "merged result must come back at the source frame timestamp";
    EXPECT_TRUE(
        merged_packets[t].Get<std::vector<OrientedDetection>>().empty())
        << "a SKIPped frame must produce an empty detections result";
  }
}

}  // namespace
}  // namespace mediapipe
