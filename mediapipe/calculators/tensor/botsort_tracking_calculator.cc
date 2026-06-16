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

#include <memory>
#include <utility>
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
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "opencv2/core.hpp"

namespace mediapipe {
namespace api2 {

// Wraps the vendored motion-only BoTSORT tracker as an api2 calculator.
//
// Inputs : IMAGE      (ImageFrame) - the current frame; used for global motion
//                                    compensation (sparse optical flow).
//          DETECTIONS (std::vector<mediapipe::Detection>) - fresh detections in
//                                    normalized RELATIVE_BOUNDING_BOX form.
// Output : DETECTIONS (std::vector<mediapipe::Detection>) - the tracked boxes,
//                                    again normalized. The output preserves
//                                    label_id (<= 255) and score but carries NO
//                                    track_id/detection_id (parity contract with
//                                    the optical-flow tracker output).
//
// The vendored BoTSORT `Detection` type is in the global namespace; it is
// referenced as `::Detection` to disambiguate from `mediapipe::Detection`.
class BotsortTrackingCalculator : public Node {
 public:
  static constexpr Input<ImageFrame> kImage{"IMAGE"};
  static constexpr Input<std::vector<Detection>> kDetections{"DETECTIONS"};
  static constexpr Output<std::vector<Detection>> kOut{"DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kImage, kDetections, kOut);

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
    const float width = static_cast<float>(image.Width());
    const float height = static_cast<float>(image.Height());

    std::vector<::Detection> bs_dets;
    if (kDetections(cc).IsConnected() && !kDetections(cc).IsEmpty()) {
      const std::vector<Detection>& dets = *kDetections(cc);
      bs_dets.reserve(dets.size());
      for (const Detection& d : dets) {
        if (!d.has_location_data() ||
            !d.location_data().has_relative_bounding_box()) {
          continue;
        }
        const auto& rbb = d.location_data().relative_bounding_box();
        ::Detection bd;
        bd.bbox_tlwh =
            cv::Rect_<float>(rbb.xmin() * width, rbb.ymin() * height,
                             rbb.width() * width, rbb.height() * height);
        bd.class_id = d.label_id_size() > 0 ? d.label_id(0) : 0;
        bd.confidence = d.score_size() > 0 ? d.score(0) : 0.0f;
        bs_dets.push_back(bd);
      }
    }

    std::vector<std::shared_ptr<Track>> tracks = tracker_->track(bs_dets, frame);
    std::vector<Detection> out;
    out.reserve(tracks.size());
    for (const std::shared_ptr<Track>& t : tracks) {
      const std::vector<float> tlwh = t->get_tlwh();
      Detection det;
      det.add_score(t->get_score());
      det.add_label_id(static_cast<int>(t->get_class_id()));
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
