// Motion-only stub for the vendored BoTSORT. Upstream ReID.h pulls in TensorRT;
// this CPU/desktop build runs motion-only (Kalman + IoU/lapjv + GMC), so ReID
// is never enabled at runtime. A complete no-op ReIDModel lets BoTSORT.{h,cpp}
// compile and link without CUDA/TensorRT/ONNXRuntime.
#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "DataType.h"
#include "ReIDParams.h"

class ReIDModel {
 public:
  ReIDModel(const ReIDParams& /*params*/,
            const std::string& /*onnx_model_path*/) {}
  ~ReIDModel() = default;

  void pre_process(cv::Mat& /*image*/) {}

  FeatureVector extract_features(cv::Mat& /*image*/) {
    return FeatureVector::Zero();
  }

  const std::string& get_distance_metric() const {
    static const std::string kMetric = "cosine";
    return kMetric;
  }
};
