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
#include <cstring>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.pb.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// Streams externally-supplied tiles into batched [N,H,W,C] float32 tensors.
// CPU: crop -> resize -> normalize into batch rows.
class StreamingTilesToTensorBatchCalculator : public Node {
 public:
  static constexpr Input<ImageFrame> kInImage{"IMAGE"};
  static constexpr Input<TilePlan> kInPlan{"TILE_PLAN"};
  static constexpr SideInput<InferenceMetadata> kSideMeta{"METADATA"};
  static constexpr Output<std::vector<Tensor>> kOutTensors{"TENSORS"};
  static constexpr Output<TensorBatchInfo> kOutInfo{"BATCH_INFO"};
  MEDIAPIPE_NODE_CONTRACT(kInImage, kInPlan, kSideMeta, kOutTensors, kOutInfo);

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<
        mediapipe::StreamingTilesToTensorBatchCalculatorOptions>();
    meta_ = kSideMeta(cc).Get();
    RET_CHECK_GT(meta_.input_height(), 0);
    RET_CHECK_GT(meta_.input_width(), 0);
    RET_CHECK_GT(meta_.input_channels(), 0);
    RET_CHECK_GT(meta_.batch_capacity(), 0);
    dynamic_batch_ = meta_.is_dynamic_batch() || options_.dynamic_batch();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const ImageFrame& frame = *kInImage(cc);
    const TilePlan& plan = *kInPlan(cc);
    const int H = meta_.input_height();
    const int W = meta_.input_width();
    const int C = meta_.input_channels();
    const int cap = meta_.batch_capacity();
    RET_CHECK_EQ(frame.NumberOfChannels(), C)
        << "model expects " << C << " input channels but the frame has "
        << frame.NumberOfChannels();
    cv::Mat src = formats::MatView(&frame);
    const int fw = frame.Width();
    const int fh = frame.Height();

    const int T = static_cast<int>(plan.tiles.size());
    // Single-batch-per-frame only for now: emitting >1 batch per Process() at
    // one input timestamp would violate output-stream timestamp monotonicity.
    // Fail cleanly here rather than crash deeper in the framework. Multi-batch
    // (T > batch_capacity) emission is deferred.
    RET_CHECK_LE(T, cap)
        << "tile count " << T << " exceeds batch_capacity " << cap
        << "; multi-batch emission (T > batch_capacity) is not yet supported";
    const int total_batches = (T + cap - 1) / cap;  // 0 when T == 0
    const int64_t ts = cc->InputTimestamp().Value();

    int emitted = 0;
    for (int start = 0; start < T; start += cap) {
      const int rows = std::min(cap, T - start);
      const int N = dynamic_batch_ ? rows : cap;
      Tensor tensor(Tensor::ElementType::kFloat32,
                    Tensor::Shape{N, H, W, C});
      auto write = tensor.GetCpuWriteView();
      float* buf = write.buffer<float>();
      std::memset(buf, 0, sizeof(float) * N * H * W * C);

      TensorBatchInfo info;
      info.source_frame_timestamp = ts;
      info.batch_index = emitted;
      info.total_batches = total_batches;
      info.batch_capacity = N;
      info.valid_count = rows;

      for (int r = 0; r < rows; ++r) {
        const TileGeometry& g = plan.tiles[start + r];
        int rx = static_cast<int>(std::lround(g.x0() * fw));
        int ry = static_cast<int>(std::lround(g.y0() * fh));
        int rw = static_cast<int>(std::lround(g.width * fw));
        int rh = static_cast<int>(std::lround(g.height * fh));
        rx = std::clamp(rx, 0, fw - 1);
        ry = std::clamp(ry, 0, fh - 1);
        rw = std::clamp(rw, 1, fw - rx);
        rh = std::clamp(rh, 1, fh - ry);
        cv::Mat roi = src(cv::Rect(rx, ry, rw, rh));
        cv::Mat resized;
        cv::resize(roi, resized, cv::Size(W, H));
        cv::Mat f32;
        resized.convertTo(f32, CV_32FC(C), 1.0 / 255.0);
        std::memcpy(buf + static_cast<size_t>(r) * H * W * C,
                    f32.ptr<float>(0), sizeof(float) * H * W * C);
        info.tile_indices.push_back(g.tile_index);
      }

      std::vector<Tensor> tensors;
      tensors.push_back(std::move(tensor));
      kOutTensors(cc).Send(std::move(tensors));
      kOutInfo(cc).Send(std::move(info));
      ++emitted;
    }
    return absl::OkStatus();
  }

 private:
  mediapipe::StreamingTilesToTensorBatchCalculatorOptions options_;
  InferenceMetadata meta_;
  bool dynamic_batch_ = false;
};

MEDIAPIPE_REGISTER_NODE(StreamingTilesToTensorBatchCalculator);

}  // namespace api2
}  // namespace mediapipe
