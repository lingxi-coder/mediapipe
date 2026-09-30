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
#include <utility>
#include <variant>
#include <vector>

#include "BoTSORT.h"
#include "DataType.h"
#include "GmcParams.h"
#include "ReIDParams.h"
#include "TrackerParams.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/video/botsort_tracking_calculator.pb.h"
#include "mediapipe/framework/formats/tiling_types.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/status_macros.h"
#include "opencv2/core.hpp"

namespace mediapipe {
namespace api2 {

namespace {

absl::Status ValidateOptions(
    const BotsortTrackingCalculatorOptions& opts) {
  const auto in_unit_interval = [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  };
  if (!in_unit_interval(opts.track_high_threshold())) {
    return absl::InvalidArgumentError(
        "track_high_threshold must be finite and in [0, 1].");
  }
  if (!in_unit_interval(opts.track_low_threshold())) {
    return absl::InvalidArgumentError(
        "track_low_threshold must be finite and in [0, 1].");
  }
  if (!in_unit_interval(opts.new_track_threshold())) {
    return absl::InvalidArgumentError(
        "new_track_threshold must be finite and in [0, 1].");
  }
  if (!in_unit_interval(opts.match_threshold())) {
    return absl::InvalidArgumentError(
        "match_threshold must be finite and in [0, 1].");
  }
  if (opts.track_low_threshold() > opts.track_high_threshold()) {
    return absl::InvalidArgumentError(
        "track_low_threshold must be <= track_high_threshold.");
  }
  if (opts.track_buffer() < 0 || opts.track_buffer() > 255) {
    return absl::InvalidArgumentError("track_buffer must be in [0, 255].");
  }
  if (opts.nominal_frame_rate() < 1 || opts.nominal_frame_rate() > 255) {
    return absl::InvalidArgumentError(
        "nominal_frame_rate must be in [1, 255].");
  }
  const int effective_lost_window = static_cast<int>(std::floor(
      static_cast<double>(opts.nominal_frame_rate()) / 30.0 *
      opts.track_buffer()));
  if (effective_lost_window > 255) {
    return absl::InvalidArgumentError(
        "floor(nominal_frame_rate / 30 * track_buffer) must be <= 255.");
  }
  return absl::OkStatus();
}

absl::Status ValidateDetection(const Detection& detection) {
  if (!detection.has_location_data() ||
      detection.location_data().format() !=
          LocationData::RELATIVE_BOUNDING_BOX ||
      !detection.location_data().has_relative_bounding_box()) {
    return absl::InvalidArgumentError(
        "BoTSORT detections require RELATIVE_BOUNDING_BOX location data.");
  }
  if (detection.label_id_size() != 1 || detection.label_id(0) < 0 ||
      detection.label_id(0) > 255) {
    return absl::InvalidArgumentError(
        "BoTSORT detections require exactly one label_id in [0, 255].");
  }
  if (detection.score_size() != 1 ||
      !std::isfinite(detection.score(0)) || detection.score(0) < 0.0F ||
      detection.score(0) > 1.0F) {
    return absl::InvalidArgumentError(
        "BoTSORT detections require exactly one finite score in [0, 1].");
  }
  const auto& box = detection.location_data().relative_bounding_box();
  if (!std::isfinite(box.xmin()) || !std::isfinite(box.ymin()) ||
      !std::isfinite(box.width()) || !std::isfinite(box.height()) ||
      box.width() < 0.0F || box.height() < 0.0F) {
    return absl::InvalidArgumentError(
        "BoTSORT detection boxes must have finite coordinates and "
        "non-negative width/height.");
  }
  return absl::OkStatus();
}

absl::Status ValidateObservedRoi(const TilePixelRoi& roi, int width,
                                 int height) {
  if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
      roi.x > width || roi.y > height || roi.width > width - roi.x ||
      roi.height > height - roi.y) {
    return absl::InvalidArgumentError(
        "OBSERVED_ROIS must be positive, in-bounds source-pixel rectangles.");
  }
  return absl::OkStatus();
}

}  // namespace

// Wraps the vendored motion-only BoTSORT tracker as an api2 calculator.
//
// Inputs : IMAGE      (ImageFrame) - the current frame; used for global motion
//                                    compensation (sparse optical flow).
//          DETECTIONS (std::vector<mediapipe::Detection>) - fresh detections in
//                                    normalized RELATIVE_BOUNDING_BOX form.
//          REFRESH (bool) - true for observed inference, false for scheduler
//                           SKIP/prediction-only.
//          OBSERVED_ROIS (std::vector<TilePixelRoi>) - exact source-pixel
//                           inference coverage for this frame.
// Output : DETECTIONS (std::vector<mediapipe::Detection>) - the tracked boxes,
//                                    again normalized. The output preserves
//                                    label_id (<= 255) and score, and now writes
//                                    BoTSORT's persistent track id as the proto's
//                                    string track_id field. detection_id is left
//                                    unset.
//
// The vendored BoTSORT `Detection` type is in the global namespace; it is
// referenced as `::Detection` to disambiguate from `mediapipe::Detection`.
class BotsortTrackingCalculator : public Node {
 public:
  static constexpr Input<ImageFrame> kImage{"IMAGE"};
  static constexpr Input<std::vector<Detection>> kDetections{"DETECTIONS"};
  static constexpr Input<bool> kRefresh{"REFRESH"};
  static constexpr Input<std::vector<TilePixelRoi>> kObservedRois{
      "OBSERVED_ROIS"};
  static constexpr Output<std::vector<Detection>> kOut{"DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kImage, kDetections, kRefresh, kObservedRois, kOut);

  absl::Status Open(CalculatorContext* cc) override {
    const auto& opts = cc->Options<BotsortTrackingCalculatorOptions>();
    ABSL_RETURN_IF_ERROR(ValidateOptions(opts));
    TrackerParams params;
    params.track_high_thresh = opts.track_high_threshold();
    params.track_low_thresh = opts.track_low_threshold();
    params.new_track_thresh = opts.new_track_threshold();
    params.track_buffer = opts.track_buffer();
    params.match_thresh = opts.match_threshold();
    params.frame_rate = opts.nominal_frame_rate();
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
    if (kImage(cc).IsEmpty() || kDetections(cc).IsEmpty() ||
        kRefresh(cc).IsEmpty() || kObservedRois(cc).IsEmpty()) {
      return absl::InvalidArgumentError(
          "BoTSORT requires IMAGE, DETECTIONS, REFRESH, and OBSERVED_ROIS "
          "packets at every source timestamp.");
    }
    const ImageFrame& image = *kImage(cc);
    if (image.Width() <= 0 || image.Height() <= 0) {
      return absl::InvalidArgumentError("BoTSORT image dimensions must be positive.");
    }
    cv::Mat frame = formats::MatView(&image);
    const float width = static_cast<float>(image.Width());
    const float height = static_cast<float>(image.Height());

    const std::vector<Detection>& dets = *kDetections(cc);
    std::vector<::Detection> bs_dets;
    bs_dets.reserve(dets.size());
    for (const Detection& d : dets) {
      ABSL_RETURN_IF_ERROR(ValidateDetection(d));
      const auto& rbb = d.location_data().relative_bounding_box();
      ::Detection bd;
      bd.bbox_tlwh =
          cv::Rect_<float>(rbb.xmin() * width, rbb.ymin() * height,
                           rbb.width() * width, rbb.height() * height);
      bd.class_id = d.label_id(0);
      bd.confidence = d.score(0);
      bs_dets.push_back(bd);
    }

    const bool refresh = *kRefresh(cc);
    const std::vector<TilePixelRoi>& input_rois = *kObservedRois(cc);
    if (!refresh && (!bs_dets.empty() || !input_rois.empty())) {
      return absl::InvalidArgumentError(
          "REFRESH=false requires empty DETECTIONS and OBSERVED_ROIS.");
    }
    std::vector<std::shared_ptr<Track>> tracks;
    if (refresh) {
      if (input_rois.empty()) {
        return absl::InvalidArgumentError(
            "REFRESH=true requires non-empty OBSERVED_ROIS.");
      }
      std::vector<cv::Rect> rois;
      rois.reserve(input_rois.size());
      for (const TilePixelRoi& roi : input_rois) {
        ABSL_RETURN_IF_ERROR(ValidateObservedRoi(roi, image.Width(), image.Height()));
        rois.emplace_back(roi.x, roi.y, roi.width, roi.height);
      }
      tracks = tracker_->track_observed(bs_dets, frame, rois);
    } else {
      tracks = tracker_->predict_only(frame);
    }
    std::vector<Detection> out;
    out.reserve(tracks.size());
    for (const std::shared_ptr<Track>& t : tracks) {
      const std::vector<float> tlwh = t->get_tlwh();
      Detection det;
      det.add_score(t->get_score());
      det.add_label_id(static_cast<int>(t->get_class_id()));
      // Surface BoTSORT's persistent track id (an int) as the proto's string
      // track_id field ("part of a track"). Consumed downstream by
      // TiledFrameSuppression (id propagation) and the result containers.
      det.set_track_id(std::to_string(t->track_id));
      auto* loc = det.mutable_location_data();
      loc->set_format(LocationData::RELATIVE_BOUNDING_BOX);
      auto* box = loc->mutable_relative_bounding_box();
      box->set_xmin(tlwh[0] / width);
      box->set_ymin(tlwh[1] / height);
      box->set_width(tlwh[2] / width);
      box->set_height(tlwh[3] / height);
      out.push_back(std::move(det));
    }
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<BoTSORT> tracker_;
};

MEDIAPIPE_REGISTER_NODE(BotsortTrackingCalculator);

}  // namespace api2
}  // namespace mediapipe
