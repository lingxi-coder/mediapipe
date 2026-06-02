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

#include <array>
#include <map>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/tiling_matrix_utils.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace api2 {

// Regroups batched per-tile oriented detections to the source frame, drops
// padded rows, and projects tile-local boxes to full-frame coords using the
// TileBatchGeometry carried on BATCH_INFO. Emits the flattened frame
// detections at the original source frame timestamp once all batches arrive.
// Axis-aligned tiles only: rotation is preserved.
class MergeTileDetectionsAccumulatorCalculator : public Node {
 public:
  static constexpr Input<std::vector<std::vector<OrientedDetection>>> kInDets{
      "ORIENTED_DETECTIONS"};
  static constexpr Input<TensorBatchInfo> kInInfo{"BATCH_INFO"};
  static constexpr Output<std::vector<OrientedDetection>> kOut{
      "ORIENTED_DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInDets, kInInfo, kOut,
                          ::mediapipe::api2::TimestampChange::Arbitrary());

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
        // rotation preserved (axis-aligned tile).
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
              .At(::mediapipe::Timestamp(src_ts)));
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
