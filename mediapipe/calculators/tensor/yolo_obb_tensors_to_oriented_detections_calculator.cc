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
#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/yolo_obb_tensors_to_oriented_detections_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// Decodes Ultralytics YOLOv8/v11-OBB tensors into batched OrientedDetections.
class YoloObbTensorsToOrientedDetectionsCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  static constexpr Output<std::vector<std::vector<OrientedDetection>>>
      kOutDetections{"ORIENTED_DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInTensors, kOutDetections);

  absl::Status Open(CalculatorContext* cc) override {
    options_ =
        cc->Options<mediapipe::YoloObbTensorsToOrientedDetectionsCalculatorOptions>();
    RET_CHECK_GT(options_.num_classes(), 0) << "num_classes must be set and > 0";
    RET_CHECK(options_.allow_classes().empty() || options_.ignore_classes().empty())
        << "allow_classes and ignore_classes are mutually exclusive";
    for (int c : options_.allow_classes()) allow_classes_.insert(c);
    for (int c : options_.ignore_classes()) ignore_classes_.insert(c);
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

    const int N = dims[0];
    const int num_classes = options_.num_classes();
    const int channels = 4 + num_classes + 1;  // + angle
    int A;
    if (options_.layout() ==
        mediapipe::YoloObbTensorsToOrientedDetectionsCalculatorOptions::
            CHANNELS_LAST) {
      A = dims[1];
      RET_CHECK_EQ(dims[2], channels);
    } else {  // CHANNELS_FIRST (proto default; LAYOUT_UNSPECIFIED also lands here)
      RET_CHECK_EQ(dims[1], channels);
      A = dims[2];
    }

    auto view = t.GetCpuReadView();
    const float* data = view.buffer<float>();

    auto out = std::make_unique<std::vector<std::vector<OrientedDetection>>>();
    out->resize(N);
    for (int n = 0; n < N; ++n) {
      DecodeRow(data, n, channels, A, num_classes, &(*out)[n]);
    }
    kOutDetections(cc).Send(std::move(out));
    return absl::OkStatus();
  }

 private:
  void DecodeRow(const float* data, int n, int channels, int A, int num_classes,
                 std::vector<OrientedDetection>* dets) {
    const bool channels_last =
        options_.layout() ==
        mediapipe::YoloObbTensorsToOrientedDetectionsCalculatorOptions::
            CHANNELS_LAST;
    const int angle_idx = 4 + num_classes;  // last channel
    for (int a = 0; a < A; ++a) {
      auto at = [&](int c) -> float {
        return channels_last ? data[(n * A + a) * channels + c]
                             : data[(n * channels + c) * A + a];
      };
      const float cx = at(0), cy = at(1), w = at(2), h = at(3);
      int best = 0;
      float best_score = at(4);
      for (int c = 1; c < num_classes; ++c) {
        const float s = at(4 + c);
        if (s > best_score) {
          best_score = s;
          best = c;
        }
      }
      if (!allow_classes_.empty() && !allow_classes_.contains(best)) continue;
      if (ignore_classes_.contains(best)) continue;
      if (best_score < options_.conf_threshold()) continue;

      OrientedDetection d;
      d.set_cx(cx);
      d.set_cy(cy);
      d.set_width(w);
      d.set_height(h);
      d.set_rotation(at(angle_idx));
      d.add_score(best_score);
      d.add_label_id(best);
      dets->push_back(std::move(d));
    }

    const int k = options_.max_detections_before_nms();
    if (k >= 0 && static_cast<int>(dets->size()) > k) {
      std::partial_sort(dets->begin(), dets->begin() + k, dets->end(),
                        [](const OrientedDetection& l, const OrientedDetection& r) {
                          return l.score(0) > r.score(0);
                        });
      dets->resize(k);
    }
  }

  mediapipe::YoloObbTensorsToOrientedDetectionsCalculatorOptions options_;
  absl::flat_hash_set<int> allow_classes_;
  absl::flat_hash_set<int> ignore_classes_;
};

MEDIAPIPE_REGISTER_NODE(YoloObbTensorsToOrientedDetectionsCalculator);

}  // namespace api2
}  // namespace mediapipe
