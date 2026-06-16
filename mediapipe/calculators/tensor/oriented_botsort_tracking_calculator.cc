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

#include <cmath>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "BoTSORT.h"
#include "DataType.h"
#include "GmcParams.h"
#include "ReIDParams.h"
#include "TrackerParams.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/botsort_tracking_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "opencv2/core.hpp"

namespace mediapipe {
namespace api2 {

namespace {
// AABB (pixel tlwh) enclosing a rotated rect given in normalized space.
cv::Rect_<float> OrientedToAabbTlwh(const OrientedDetection& d, float img_w,
                                    float img_h) {
  const float hw = 0.5f * d.width();
  const float hh = 0.5f * d.height();
  const float c = std::abs(std::cos(d.rotation()));
  const float s = std::abs(std::sin(d.rotation()));
  const float ax = hw * c + hh * s;  // normalized half-extent x
  const float ay = hw * s + hh * c;  // normalized half-extent y
  return cv::Rect_<float>((d.cx() - ax) * img_w, (d.cy() - ay) * img_h,
                          2.0f * ax * img_w, 2.0f * ay * img_h);
}

float AabbIoU(const cv::Rect_<float>& a, const cv::Rect_<float>& b) {
  const float inter = (a & b).area();
  const float uni = a.area() + b.area() - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}
}  // namespace

// Assigns stable BoTSORT track ids to oriented detections via ID-association on
// their axis-aligned bounding boxes. Output geometry is always the fresh rotated
// detection; rotation is never tracked; no gap-fill (a Kalman/AABB box has no
// angle). track_id is the proto string form of BoTSORT's int track id.
//
// NOTE: BoTSORT only emits a track_id for CONFIRMED tracks; the confirmation
// thresholds must be at/below the detector's score_threshold or no id is set.
//
// Inputs:  IMAGE (ImageFrame), ORIENTED_DETECTIONS (vector<OrientedDetection>).
// Output:  ORIENTED_DETECTIONS (same fresh detections, track_id set on matches).
class OrientedBotsortTrackingCalculator : public Node {
 public:
  static constexpr Input<ImageFrame> kImage{"IMAGE"};
  static constexpr Input<std::vector<OrientedDetection>> kDets{
      "ORIENTED_DETECTIONS"};
  static constexpr Output<std::vector<OrientedDetection>> kOut{
      "ORIENTED_DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kImage, kDets, kOut);

  absl::Status Open(CalculatorContext* cc) override {
    const auto& opts = cc->Options<BotsortTrackingCalculatorOptions>();
    TrackerParams params;
    params.track_high_thresh = opts.track_high_threshold();
    params.track_low_thresh = opts.track_low_threshold();
    params.new_track_thresh = opts.new_track_threshold();
    params.track_buffer = opts.track_buffer();
    params.match_thresh = opts.match_threshold();
    params.gmc_enabled = opts.enable_gmc();
    params.reid_enabled = false;
    Config<GMC_Params> gmc_config = std::monostate{};
    if (opts.enable_gmc()) {
      GMC_Params gmc;
      gmc.method_ = GMC_Method::SparseOptFlow;
      gmc.method_params_ = SparseOptFlow_Params{};
      gmc_config = gmc;
    }
    tracker_ = std::make_unique<BoTSORT>(
        Config<TrackerParams>(params), gmc_config,
        Config<ReIDParams>(std::monostate{}), "");
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const ImageFrame& image = *kImage(cc);
    cv::Mat frame = formats::MatView(&image);
    const float w = static_cast<float>(image.Width());
    const float h = static_cast<float>(image.Height());

    std::vector<OrientedDetection> out;
    std::vector<cv::Rect_<float>> aabbs;
    if (kDets(cc).IsConnected() && !kDets(cc).IsEmpty()) {
      out = *kDets(cc);
      aabbs.reserve(out.size());
      for (const OrientedDetection& d : out) {
        aabbs.push_back(OrientedToAabbTlwh(d, w, h));
      }
    }

    std::vector<::Detection> bs_dets;
    bs_dets.reserve(out.size());
    for (size_t i = 0; i < out.size(); ++i) {
      ::Detection bd;
      bd.bbox_tlwh = aabbs[i];
      bd.class_id = out[i].label_id_size() > 0 ? out[i].label_id(0) : 0;
      bd.confidence = out[i].score_size() > 0 ? out[i].score(0) : 0.0f;
      bs_dets.push_back(bd);
    }

    std::vector<std::shared_ptr<Track>> tracks = tracker_->track(bs_dets, frame);
    for (const std::shared_ptr<Track>& t : tracks) {
      const std::vector<float> tlwh = t->get_tlwh();
      const cv::Rect_<float> track_box(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);
      int best_idx = -1;
      float best_iou = 0.1f;
      for (size_t i = 0; i < out.size(); ++i) {
        if (out[i].has_track_id()) continue;
        const float iou = AabbIoU(track_box, aabbs[i]);
        if (iou >= best_iou) {
          best_iou = iou;
          best_idx = static_cast<int>(i);
        }
      }
      if (best_idx >= 0) {
        out[best_idx].set_track_id(std::to_string(t->track_id));
      }
    }
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<BoTSORT> tracker_;
};

MEDIAPIPE_REGISTER_NODE(OrientedBotsortTrackingCalculator);

}  // namespace api2
}  // namespace mediapipe
