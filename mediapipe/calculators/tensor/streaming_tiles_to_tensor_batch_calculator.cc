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
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.pb.h"
#include "mediapipe/calculators/tensor/tiling_cache_stats.h"
#include "mediapipe/calculators/tensor/tiling_cache_utils.h"
#include "mediapipe/calculators/tensor/tiling_matrix_utils.h"
#include "mediapipe/calculators/tensor/tiling_types.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/timestamp.h"

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
  static constexpr Output<TilingCacheStats>::Optional kOutStats{"CACHE_STATS"};
  MEDIAPIPE_NODE_CONTRACT(kInImage, kInPlan, kSideMeta, kOutTensors, kOutInfo,
                          kOutStats,
                          ::mediapipe::api2::TimestampChange::Arbitrary());

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<
        mediapipe::StreamingTilesToTensorBatchCalculatorOptions>();
    meta_ = kSideMeta(cc).Get();
    RET_CHECK_GT(meta_.input_height(), 0);
    RET_CHECK_GT(meta_.input_width(), 0);
    RET_CHECK_GT(meta_.input_channels(), 0);
    RET_CHECK_GT(meta_.batch_capacity(), 0);
    dynamic_batch_ = meta_.is_dynamic_batch() || options_.dynamic_batch();
    RET_CHECK_GE(options_.max_cached_tile_matrices(), 0);
    matrix_cache_ = BoundedLruCache<std::shared_ptr<const TileBatchGeometry>>(
        static_cast<size_t>(options_.max_cached_tile_matrices()));
    RET_CHECK_GE(options_.max_cpu_tensor_workspaces(), 0);
    if (options_.max_cpu_tensor_workspaces() > 0) {
      memory_manager_ = std::make_shared<MemoryManager>(
          static_cast<size_t>(options_.max_cpu_tensor_workspaces()));
    }
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
    // Multi-batch emission is supported: when T > batch_capacity, each batch
    // is emitted at a distinct synthetic timestamp (batch_ts_), which is a
    // calculator-local monotonic counter incremented after each emit. This
    // follows the BeginLoopCalculator pattern and guarantees output-stream
    // timestamp monotonicity even when one Process() call emits multiple batches.
    const int total_batches = (T + cap - 1) / cap;  // 0 when T == 0
    const int64_t ts = cc->InputTimestamp().Value();

    int emitted = 0;
    for (int start = 0; start < T; start += cap) {
      const int rows = std::min(cap, T - start);
      const int N = dynamic_batch_ ? rows : cap;
      Tensor tensor(Tensor::ElementType::kFloat32,
                    Tensor::Shape{N, H, W, C},
                    memory_manager_.get());
      auto write = tensor.GetCpuWriteView();
      float* buf = write.buffer<float>();
      std::memset(buf, 0, sizeof(float) * N * H * W * C);

      // Geometry (matrices + ROIs) depends only on frame size + tile set, not
      // pixels. Cache it; the pixel crop/resize below always runs.
      std::shared_ptr<const TileBatchGeometry> geom;
      StableCacheKey key;
      if (matrix_cache_.enabled()) {
        StableKeyBuilder kb;
        kb.AddInt(fw).AddInt(fh).AddInt(W).AddInt(H).AddInt(C);
        for (int r = 0; r < rows; ++r) {
          const TileGeometry& g = plan.tiles[start + r];
          kb.AddInt(g.tile_index).AddFloat(g.x_center).AddFloat(g.y_center)
            .AddFloat(g.width).AddFloat(g.height);
        }
        key = kb.Build();
        if (auto* hit = matrix_cache_.Get(key)) geom = *hit;
      }
      if (geom == nullptr) {
        geom = BuildBatchGeometry(plan, start, rows, fw, fh);
        if (matrix_cache_.enabled()) matrix_cache_.Put(key, geom);
      }

      TensorBatchInfo info;
      info.source_frame_timestamp = ts;
      info.batch_timestamp = batch_ts_.Value();
      info.batch_index = emitted;
      info.total_batches = total_batches;
      info.batch_capacity = N;
      info.batch_size = N;
      info.valid_count = rows;
      info.tile_indices = geom->tile_indices;  // back-compat mirror
      info.geometry = geom;

      // Pixel crop/resize/normalize loop — ALWAYS runs; uses cached ROIs.
      for (int r = 0; r < rows; ++r) {
        const TilePixelRoi& proi = geom->effective_pixel_rois[r];
        cv::Mat roi = src(cv::Rect(proi.x, proi.y, proi.width, proi.height));
        cv::resize(roi, resized_workspace_, cv::Size(W, H));
        resized_workspace_.convertTo(f32_workspace_, CV_32FC(C), 1.0 / 255.0);
        std::memcpy(buf + static_cast<size_t>(r) * H * W * C,
                    f32_workspace_.ptr<float>(0), sizeof(float) * H * W * C);
      }

      std::vector<Tensor> tensors;
      tensors.push_back(std::move(tensor));
      kOutTensors(cc).Send(
          mediapipe::api2::MakePacket<std::vector<Tensor>>(std::move(tensors))
              .At(batch_ts_));
      kOutInfo(cc).Send(
          mediapipe::api2::MakePacket<TensorBatchInfo>(std::move(info))
              .At(batch_ts_));
      ++batch_ts_;
      ++emitted;
    }
    // If T == 0, emit one empty BATCH_INFO at batch_ts_ so merge sees the frame.
    if (T == 0) {
      TensorBatchInfo info;
      info.source_frame_timestamp = ts;
      info.batch_timestamp = batch_ts_.Value();
      info.total_batches = 0;
      info.valid_count = 0;
      kOutInfo(cc).Send(
          mediapipe::api2::MakePacket<TensorBatchInfo>(std::move(info))
              .At(batch_ts_));
      ++batch_ts_;
    }
    if (options_.emit_cache_stats() && kOutStats(cc).IsConnected()) {
      TilingCacheStats stats;
      stats.tile_matrix = matrix_cache_.stats();
      if (memory_manager_ && memory_manager_->GetCpuBufferPool()) {
        stats.cpu_tensor_pool = memory_manager_->GetCpuBufferPool()->stats();
      }
      kOutStats(cc).Send(
          mediapipe::api2::MakePacket<TilingCacheStats>(stats).At(Timestamp(ts)));
    }
    return absl::OkStatus();
  }

 private:
  static std::shared_ptr<const TileBatchGeometry> BuildBatchGeometry(
      const TilePlan& plan, int start, int rows, int fw, int fh) {
    auto geom = std::make_shared<TileBatchGeometry>();
    geom->tile_indices.reserve(rows);
    geom->tile_geometries.reserve(rows);
    geom->effective_pixel_rois.reserve(rows);
    geom->tile_to_image_matrices.reserve(rows);
    geom->image_to_tile_matrices.reserve(rows);
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
      TilePixelRoi proi{rx, ry, rw, rh};
      const std::array<float, 16> t2i = TileToImageMatrix(proi, fw, fh);
      geom->tile_indices.push_back(g.tile_index);
      geom->tile_geometries.push_back(g);
      geom->effective_pixel_rois.push_back(proi);
      geom->tile_to_image_matrices.push_back(t2i);
      geom->image_to_tile_matrices.push_back(InvertAffine2d(t2i));
    }
    return geom;
  }

  mediapipe::StreamingTilesToTensorBatchCalculatorOptions options_;
  InferenceMetadata meta_;
  bool dynamic_batch_ = false;
  BoundedLruCache<std::shared_ptr<const TileBatchGeometry>> matrix_cache_{0};
  // Monotonic synthetic timestamp for output packets. Incremented after each
  // batch emit (following the BeginLoopCalculator pattern). Never reset —
  // guarantees output-stream timestamp monotonicity across all Process() calls.
  Timestamp batch_ts_ = Timestamp(0);
  std::shared_ptr<MemoryManager> memory_manager_;  // null unless pooling enabled
  cv::Mat resized_workspace_;  // reused across rows/batches when shape matches
  cv::Mat f32_workspace_;
};

MEDIAPIPE_REGISTER_NODE(StreamingTilesToTensorBatchCalculator);

}  // namespace api2
}  // namespace mediapipe
