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
//
// pybind bridge: runs a real MediaPipe CalculatorGraph
//   YoloTensorsToDetectionsCalculator -> YoloBatchDetectionsToSingleCalculator
//   -> NonMaxSuppressionCalculator
// over a raw YOLOv8 detect-head output tensor supplied from Python (e.g. the
// output of torch running yolov8n.pt). This is the runnable realization of the
// "PyTorch backend": torch produces the raw tensor in Python; MediaPipe's C++
// graph does the anchor-free decode + NMS.

#include <cstring>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace {

// Runs the YOLO decode+NMS graph once over a raw [1, 4+num_classes, A] output
// tensor (CHANNELS_FIRST). Box channels are expected pre-normalized to [0,1].
// Returns an (M, 6) float array: [xmin, ymin, width, height, score, label_id]
// in the same normalized space, after NMS.
py::array_t<float> RunYoloGraph(
    py::array_t<float, py::array::c_style | py::array::forcecast> raw,
    int num_classes, float conf_threshold, float iou_threshold) {
  if (raw.ndim() != 3 || raw.shape(0) != 1) {
    throw std::invalid_argument("raw must have shape [1, 4+num_classes, A]");
  }
  const int channels = static_cast<int>(raw.shape(1));
  const int anchors = static_cast<int>(raw.shape(2));
  if (channels != 4 + num_classes) {
    throw std::invalid_argument(absl::StrFormat(
        "channel dim %d != 4 + num_classes (%d)", channels, 4 + num_classes));
  }

  const std::string config_text = absl::StrFormat(
      R"pb(
        input_stream: "raw_tensors"
        output_stream: "final_dets"
        node {
          calculator: "YoloTensorsToDetectionsCalculator"
          input_stream: "TENSORS:raw_tensors"
          output_stream: "DETECTIONS:batched"
          options {
            [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
              num_classes: %d
              conf_threshold: %f
              layout: CHANNELS_FIRST
            }
          }
        }
        node {
          calculator: "YoloBatchDetectionsToSingleCalculator"
          input_stream: "DETECTIONS:batched"
          output_stream: "DETECTIONS:flat"
        }
        node {
          calculator: "NonMaxSuppressionCalculator"
          input_stream: "flat"
          output_stream: "final_dets"
          options {
            [mediapipe.NonMaxSuppressionCalculatorOptions.ext] {
              min_suppression_threshold: %f
              overlap_type: JACCARD
              return_empty_detections: true
              multiclass_nms: true
            }
          }
        }
      )pb",
      num_classes, conf_threshold, iou_threshold);

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          config_text);

  mediapipe::CalculatorGraph graph;
  absl::Status status = graph.Initialize(config);
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));

  std::vector<mediapipe::Detection> result;
  status = graph.ObserveOutputStream(
      "final_dets", [&result](const mediapipe::Packet& p) {
        result = p.Get<std::vector<mediapipe::Detection>>();
        return absl::OkStatus();
      });
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));

  status = graph.StartRun({});
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));

  // Build the input tensor [1, channels, anchors] and copy the numpy buffer in.
  mediapipe::Tensor tensor(
      mediapipe::Tensor::ElementType::kFloat32,
      mediapipe::Tensor::Shape{1, channels, anchors});
  {
    auto write = tensor.GetCpuWriteView();
    std::memcpy(write.buffer<float>(), raw.data(),
                sizeof(float) * channels * anchors);
  }
  std::vector<mediapipe::Tensor> tensors;
  tensors.push_back(std::move(tensor));

  status = graph.AddPacketToInputStream(
      "raw_tensors",
      mediapipe::MakePacket<std::vector<mediapipe::Tensor>>(std::move(tensors))
          .At(mediapipe::Timestamp(0)));
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));

  status = graph.CloseAllPacketSources();
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
  status = graph.WaitUntilDone();
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));

  py::array_t<float> out({static_cast<py::ssize_t>(result.size()),
                          static_cast<py::ssize_t>(6)});
  auto m = out.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(result.size()); ++i) {
    const auto& d = result[i];
    const auto& b = d.location_data().relative_bounding_box();
    m(i, 0) = b.xmin();
    m(i, 1) = b.ymin();
    m(i, 2) = b.width();
    m(i, 3) = b.height();
    m(i, 4) = d.score_size() > 0 ? d.score(0) : 0.0f;
    m(i, 5) = d.label_id_size() > 0 ? static_cast<float>(d.label_id(0)) : -1.0f;
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(_yolo_pt_graph, m) {
  m.doc() =
      "MediaPipe YOLO decode+NMS graph bridge (drives the C++ calculators from "
      "Python over a raw YOLOv8 output tensor).";
  m.def("run_yolo_graph", &RunYoloGraph, py::arg("raw"),
        py::arg("num_classes"), py::arg("conf_threshold") = 0.25f,
        py::arg("iou_threshold") = 0.45f,
        "Run YoloTensorsToDetections -> flatten -> NonMaxSuppression over a "
        "raw [1,4+num_classes,A] tensor. Returns (M,6) "
        "[xmin,ymin,w,h,score,label_id].");
}
