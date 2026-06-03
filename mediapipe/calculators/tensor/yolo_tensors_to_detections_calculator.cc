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
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/detection_nms_util.h"
#include "mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// Decodes Ultralytics YOLOv8/v11 detect-head tensors into batched Detections.
class YoloTensorsToDetectionsCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  static constexpr Output<std::vector<std::vector<Detection>>> kOutDetections{
      "DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInTensors, kOutDetections);

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<mediapipe::YoloTensorsToDetectionsCalculatorOptions>();
    RET_CHECK_GT(options_.num_classes(), 0)
        << "num_classes must be set and > 0";
    RET_CHECK_GE(options_.input_width(), 0) << "input_width must be >= 0";
    RET_CHECK_GE(options_.input_height(), 0) << "input_height must be >= 0";
    RET_CHECK_EQ(options_.input_width() > 0, options_.input_height() > 0)
        << "input_width and input_height must both be set (>0) or both unset; "
           "got width=" << options_.input_width()
        << " height=" << options_.input_height();
    if (options_.input_width() > 0 && options_.input_height() > 0) {
      inv_w_ = 1.0f / static_cast<float>(options_.input_width());
      inv_h_ = 1.0f / static_cast<float>(options_.input_height());
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& tensors = *kInTensors(cc);
    RET_CHECK_EQ(tensors.size(), 1) << "expected exactly one output tensor";
    const Tensor& t = tensors[0];
    RET_CHECK(t.element_type() == Tensor::ElementType::kFloat32)
        << "only float32 output is supported";
    const auto& dims = t.shape().dims;
    RET_CHECK_EQ(dims.size(), 3) << "expected a rank-3 tensor";

    int N = dims[0];
    int num_classes = options_.num_classes();
    int channels = 4 + num_classes;
    int A;
    if (options_.layout() ==
        mediapipe::YoloTensorsToDetectionsCalculatorOptions::CHANNELS_LAST) {
      A = dims[1];
      RET_CHECK_EQ(dims[2], channels);
    } else {  // CHANNELS_FIRST (proto default; LAYOUT_UNSPECIFIED also lands here)
      RET_CHECK_EQ(dims[1], channels);
      A = dims[2];
    }

    auto view = t.GetCpuReadView();
    const float* data = view.buffer<float>();

    auto out = std::make_unique<std::vector<std::vector<Detection>>>();
    out->resize(N);
    for (int n = 0; n < N; ++n) {
      DecodeRow(data, n, channels, A, num_classes, &(*out)[n]);
    }

    kOutDetections(cc).Send(std::move(out));
    return absl::OkStatus();
  }

 private:
  // Decodes batch row `n` into `dets`.
  void DecodeRow(const float* data, int n, int channels, int A, int num_classes,
                 std::vector<Detection>* dets) {
    const bool channels_last =
        options_.layout() ==
        mediapipe::YoloTensorsToDetectionsCalculatorOptions::CHANNELS_LAST;
    for (int a = 0; a < A; ++a) {
      auto at = [&](int c) -> float {
        return channels_last ? data[(n * A + a) * channels + c]
                             : data[(n * channels + c) * A + a];
      };
      float cx = at(0), cy = at(1), w = at(2), h = at(3);
      if (inv_w_ > 0.0f) {  // model emits pixel-space boxes -> normalize to [0,1]
        cx *= inv_w_;
        w *= inv_w_;
        cy *= inv_h_;
        h *= inv_h_;
      }
      int best = 0;
      float best_score = at(4);
      for (int c = 1; c < num_classes; ++c) {
        const float s = at(4 + c);
        if (s > best_score) {
          best_score = s;
          best = c;
        }
      }
      if (best_score < options_.conf_threshold()) continue;

      Detection d;
      auto* loc = d.mutable_location_data();
      loc->set_format(LocationData::RELATIVE_BOUNDING_BOX);
      auto* bb = loc->mutable_relative_bounding_box();
      bb->set_xmin(cx - w / 2.0f);
      bb->set_ymin(cy - h / 2.0f);
      bb->set_width(w);
      bb->set_height(h);
      d.add_score(best_score);
      d.add_label_id(best);
      dets->push_back(std::move(d));
    }
    const int k = options_.max_detections_before_nms();
    if (k >= 0 && static_cast<int>(dets->size()) > k) {
      std::partial_sort(
          dets->begin(), dets->begin() + k, dets->end(),
          [](const Detection& l, const Detection& r) {
            return l.score(0) > r.score(0);
          });
      dets->resize(k);
    }

    TileLocalNms(dets);
    const int post = options_.max_detections_after_tile_nms();
    if (post > 0 && static_cast<int>(dets->size()) > post) {
      // dets already in descending-score order from TileLocalNms (or unsorted
      // if NMS was disabled — guard that case).
      if (options_.tile_local_nms_iou_threshold() <= 0.0f) {
        std::partial_sort(dets->begin(), dets->begin() + post, dets->end(),
                          [](const Detection& l, const Detection& r) {
                            return l.score(0) > r.score(0);
                          });
      }
      dets->resize(post);
    }
  }

  // Greedy axis-aligned NMS within one row (tile), via the shared util. Per-row
  // only (DecodeRow is called once per row); never compares across rows.
  // Disabled when threshold <= 0. Result is in descending-score order.
  void TileLocalNms(std::vector<Detection>* dets) const {
    const float thr = options_.tile_local_nms_iou_threshold();
    if (thr <= 0.0f || dets->size() < 2) return;
    *dets = GreedyDetectionNms(std::move(*dets), thr,
                               options_.tile_local_nms_class_agnostic());
  }

  mediapipe::YoloTensorsToDetectionsCalculatorOptions options_;
  float inv_w_ = 0.0f;  // 1/input_width when normalizing pixel-space boxes
  float inv_h_ = 0.0f;  // 1/input_height; 0 => boxes already normalized
};

MEDIAPIPE_REGISTER_NODE(YoloTensorsToDetectionsCalculator);

}  // namespace api2
}  // namespace mediapipe
