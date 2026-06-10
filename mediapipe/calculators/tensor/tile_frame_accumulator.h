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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TILE_FRAME_ACCUMULATOR_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TILE_FRAME_ACCUMULATOR_H_

#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace mediapipe {

// Shared per-source-frame accumulation for tile-batch merge calculators
// (MergeTileDetectionsAccumulatorCalculator and its axis-aligned Detection
// counterpart): collects each batch's already-projected detections keyed by
// the source frame timestamp and flushes the frame once all total_batches
// have been seen. An empty frame (total_batches == 0) flushes on its first
// (BATCH_INFO-only) batch, producing an empty result.
//
// Entries for frames whose batches never all arrive stay in the map (no
// eviction); in the shipped graphs every batch of a frame is emitted from one
// producer Process() call and the queues flush on close, so an incomplete
// frame implies an upstream error that ends the graph anyway.
template <typename DetT>
class TileFrameAccumulator {
 public:
  // Appends one batch's detections to the frame's pending list. Returns the
  // completed frame's flattened detections when this was the last expected
  // batch (the pending entry is erased), or std::nullopt otherwise.
  std::optional<std::vector<DetT>> AddBatch(int64_t source_frame_timestamp,
                                            int total_batches,
                                            std::vector<DetT> batch_dets) {
    FrameAcc& acc = pending_[source_frame_timestamp];
    acc.received.insert(acc.received.end(),
                        std::make_move_iterator(batch_dets.begin()),
                        std::make_move_iterator(batch_dets.end()));
    acc.batches_seen += 1;
    acc.total_batches = total_batches;
    if (acc.batches_seen >= acc.total_batches) {
      std::vector<DetT> merged = std::move(acc.received);
      pending_.erase(source_frame_timestamp);
      return merged;
    }
    return std::nullopt;
  }

 private:
  struct FrameAcc {
    std::vector<DetT> received;
    int batches_seen = 0;
    int total_batches = 1;
  };
  std::map<int64_t, FrameAcc> pending_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TILE_FRAME_ACCUMULATOR_H_
