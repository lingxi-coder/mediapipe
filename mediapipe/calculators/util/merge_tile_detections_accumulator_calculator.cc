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
#include <cmath>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/util/tile_frame_accumulator.h"
#include "mediapipe/util/tiling_matrix_utils.h"
#include "mediapipe/framework/formats/tiling_types.h"
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
// Axis-aligned tiles only. Rotation passes through unchanged under isotropic
// tile->frame mapping; an anisotropic mapping (boundary-clamped tile)
// transforms the box axes, adjusting rotation and w/h accordingly.
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
    std::vector<OrientedDetection> projected;
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
        const float sx = m[0];
        const float sy = m[5];
        const float theta = d.rotation();
        if (sx == sy || theta == 0.0f) {
          // Isotropic mapping (square-ish tile) or axis-aligned box: per-axis
          // scaling is exact and the rotation passes through unchanged.
          out.set_width(d.width() * sx);
          out.set_height(d.height() * sy);
        } else {
          // Anisotropic mapping (e.g. boundary-clamped tile): transform the
          // box's edge DIRECTIONS through the scale instead of scaling w/h
          // per-axis with a preserved angle (which corrupts rotated boxes).
          // Exact for theta in {0, ±pi/2} and isotropic scales; the residual
          // shear a general anisotropic scale adds to a rotated rect has no
          // rotated-rect representation and is deliberately dropped.
          const float c = std::cos(theta);
          const float s = std::sin(theta);
          out.set_width(d.width() * std::hypot(sx * c, sy * s));
          out.set_height(d.height() * std::hypot(sx * s, sy * c));
          out.set_rotation(std::atan2(sy * s, sx * c));
        }
        projected.push_back(std::move(out));
      }
    }
    auto merged = accumulator_.AddBatch(
        info.source_frame_timestamp, info.total_batches, std::move(projected));
    if (merged.has_value()) {
      kOut(cc).Send(
          mediapipe::api2::MakePacket<std::vector<OrientedDetection>>(
              std::move(*merged))
              .At(::mediapipe::Timestamp(info.source_frame_timestamp)));
    }
    return absl::OkStatus();
  }

 private:
  TileFrameAccumulator<OrientedDetection> accumulator_;
};

MEDIAPIPE_REGISTER_NODE(MergeTileDetectionsAccumulatorCalculator);

}  // namespace api2
}  // namespace mediapipe
