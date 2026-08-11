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
#include <set>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/tile_frame_accumulator.h"
#include "mediapipe/calculators/tensor/tiling_matrix_utils.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace api2 {

// Axis-aligned counterpart of MergeTileDetectionsAccumulatorCalculator:
// regroups batched per-tile Detections to the source frame, drops padded
// rows, projects tile-local relative bounding boxes to full-frame coords
// using the TileBatchGeometry carried on BATCH_INFO, and emits ONE flattened
// result at the source frame timestamp once all batches arrive. This closes
// the tiled composition for the plain (non-OBB) YOLO pipeline, whose
// detections are mediapipe::Detection.
class MergeTileBoxDetectionsAccumulatorCalculator : public Node {
 public:
  static constexpr Input<std::vector<std::vector<Detection>>> kInDets{
      "DETECTIONS"};
  static constexpr Input<TensorBatchInfo> kInInfo{"BATCH_INFO"};
  static constexpr Output<std::vector<Detection>> kOut{"DETECTIONS"};
  static constexpr Output<std::vector<TilePixelRoi>>::Optional kOutRois{
      "OBSERVED_ROIS"};
  MEDIAPIPE_NODE_CONTRACT(kInDets, kInInfo, kOut, kOutRois,
                          ::mediapipe::api2::TimestampChange::Arbitrary());

  absl::Status Process(CalculatorContext* cc) override {
    RET_CHECK(!kInInfo(cc).IsEmpty())
        << "DETECTIONS arrived without a paired BATCH_INFO packet";
    // An empty frame (T==0, e.g. scheduler SKIP) carries a BATCH_INFO packet
    // with NO paired detections packet: the producer only advances the
    // TENSORS bound. Treat the missing packet as an empty batch (api2 Get()
    // on an empty packet is fatal).
    const std::vector<std::vector<Detection>> empty_batch;
    const auto& batch = kInDets(cc).IsEmpty() ? empty_batch : *kInDets(cc);
    const TensorBatchInfo& info = *kInInfo(cc);
    RET_CHECK(info.geometry != nullptr || info.valid_count == 0);
    const auto& geom = info.geometry;
    if (info.valid_count > 0) {
      RET_CHECK_LE(info.valid_count,
                   static_cast<int>(geom->tile_to_image_matrices.size()));
      RET_CHECK_LE(info.valid_count,
                   static_cast<int>(geom->effective_pixel_rois.size()));
    }

    std::vector<Detection> projected;
    for (int r = 0; r < info.valid_count; ++r) {
      if (r >= static_cast<int>(batch.size())) continue;
      const std::array<float, 16>& m = geom->tile_to_image_matrices[r];
      for (const Detection& d : batch[r]) {
        Detection out = d;
        auto* bb =
            out.mutable_location_data()->mutable_relative_bounding_box();
        float fx, fy;
        ApplyMatrix(m, bb->xmin(), bb->ymin(), &fx, &fy);
        bb->set_xmin(fx);
        bb->set_ymin(fy);
        // Axis-aligned tile mapping: scale the extents by the SAME matrix
        // that projects the corner (m[0]/m[5] come from the EFFECTIVE sampled
        // ROI, which for a clamped boundary tile differs from the requested
        // tile size).
        bb->set_width(bb->width() * m[0]);
        bb->set_height(bb->height() * m[5]);
        projected.push_back(std::move(out));
      }
    }

    std::vector<TilePixelRoi> observed_rois;
    if (info.valid_count > 0) {
      observed_rois.assign(geom->effective_pixel_rois.begin(),
                           geom->effective_pixel_rois.begin() +
                               info.valid_count);
    }

    auto merged = accumulator_.AddBatch(
        info.source_frame_timestamp, info.total_batches, std::move(projected));
    auto merged_rois = roi_accumulator_.AddBatch(
        info.source_frame_timestamp, info.total_batches,
        std::move(observed_rois));
    RET_CHECK_EQ(merged.has_value(), merged_rois.has_value())
        << "Detection and observed-ROI accumulators completed out of lockstep";
    if (merged.has_value()) {
      kOut(cc).Send(mediapipe::api2::MakePacket<std::vector<Detection>>(
                        std::move(*merged))
                        .At(::mediapipe::Timestamp(
                            info.source_frame_timestamp)));
      std::set<std::tuple<int, int, int, int>> seen;
      std::vector<TilePixelRoi> deduplicated;
      deduplicated.reserve(merged_rois->size());
      for (const TilePixelRoi& roi : *merged_rois) {
        if (seen.emplace(roi.x, roi.y, roi.width, roi.height).second) {
          deduplicated.push_back(roi);
        }
      }
      if (kOutRois(cc).IsConnected()) {
        kOutRois(cc).Send(
            mediapipe::api2::MakePacket<std::vector<TilePixelRoi>>(
                std::move(deduplicated))
                .At(::mediapipe::Timestamp(info.source_frame_timestamp)));
      }
    }
    return absl::OkStatus();
  }

 private:
  TileFrameAccumulator<Detection> accumulator_;
  TileFrameAccumulator<TilePixelRoi> roi_accumulator_;
};

MEDIAPIPE_REGISTER_NODE(MergeTileBoxDetectionsAccumulatorCalculator);

}  // namespace api2
}  // namespace mediapipe
