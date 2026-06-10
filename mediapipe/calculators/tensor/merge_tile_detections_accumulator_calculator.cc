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
    RET_CHECK(!kInInfo(cc).IsEmpty())
        << "ORIENTED_DETECTIONS arrived without a paired BATCH_INFO packet";
    // An empty frame (T==0, e.g. scheduler SKIP) carries a BATCH_INFO packet
    // with NO paired detections packet: the producer only advances the
    // TENSORS bound, so inference/decode emit nothing at this timestamp.
    // Treat the missing packet as an empty batch (api2 Get() on an empty
    // packet is fatal).
    const std::vector<std::vector<OrientedDetection>> empty_batch;
    const auto& batch = kInDets(cc).IsEmpty() ? empty_batch : *kInDets(cc);
    const TensorBatchInfo& info = *kInInfo(cc);
    RET_CHECK(info.geometry != nullptr || info.valid_count == 0);
    const auto& geom = info.geometry;
    if (info.valid_count > 0) {
      // The geometry's per-row vectors must cover every valid row; otherwise
      // the row-indexed access below would read out of bounds. The trusted
      // producer upholds this, but guard against a malformed BATCH_INFO.
      RET_CHECK_LE(info.valid_count,
                   static_cast<int>(geom->tile_geometries.size()));
      RET_CHECK_LE(info.valid_count,
                   static_cast<int>(geom->tile_to_image_matrices.size()));
    }
    auto& acc = pending_[info.source_frame_timestamp];
    for (int r = 0; r < info.valid_count; ++r) {
      if (r >= static_cast<int>(batch.size())) continue;
      const std::array<float, 16>& m = geom->tile_to_image_matrices[r];
      // Project the box with the tile's transform. The matrix is built from the
      // EFFECTIVE sampled pixel ROI, so for a clamped/rounded boundary tile its
      // x/y scales (m[0], m[5] for an axis-aligned tile = effective_roi.w/fw,
      // effective_roi.h/fh) differ from the requested normalized tile size.
      // Scale the box extent by the SAME matrix that projects the center;
      // using the requested tile width/height would put center and size in
      // different coordinate systems and corrupt boundary-tile boxes.
      for (const OrientedDetection& d : batch[r]) {
        OrientedDetection out = d;
        float fx, fy;
        ApplyMatrix(m, d.cx(), d.cy(), &fx, &fy);
        out.set_cx(fx);
        out.set_cy(fy);
        out.set_width(d.width() * m[0]);
        out.set_height(d.height() * m[5]);
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
