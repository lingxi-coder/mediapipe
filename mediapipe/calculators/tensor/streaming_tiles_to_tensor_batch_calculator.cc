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

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.pb.h"
#include "mediapipe/framework/formats/tiling_cache_stats.h"
#include "mediapipe/util/tiling_cache_utils.h"
#include "mediapipe/util/tiling_matrix_utils.h"
#include "mediapipe/framework/formats/tiling_types.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/gpu/tiling_gpu_resource.h"

// GPU zero-copy path (Plan 4 OpenGL / Phase 5 Metal). All GPU code is compiled
// out under MEDIAPIPE_DISABLE_GPU=1; the GLES branch additionally requires GLES
// 3.1 (Android/Linux) and the Metal branch requires MEDIAPIPE_METAL_ENABLED
// (Apple). When no GPU path compiles, the CPU path below is the only code and
// stays byte-identical to Plan 3.
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"  // BorderMode
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"      // RotatedRect
#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_gl.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "tflite/delegates/gpu/common/types.h"
#include "tflite/delegates/gpu/gl/command_queue.h"
#include "tflite/delegates/gpu/gl/gl_buffer.h"
#include "tflite/delegates/gpu/gl/gl_texture.h"
#include "tflite/delegates/gpu/gl/request_gpu_info.h"
#elif MEDIAPIPE_METAL_ENABLED
#import <Metal/Metal.h>

#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"  // BorderMode
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"      // RotatedRect
#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_metal.h"
#include "mediapipe/framework/formats/tensor_mtl_buffer_view.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/MPPMetalHelper.h"
#include "mediapipe/gpu/gpu_service.h"
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {
namespace api2 {

// GLES 3.1 SSBO zero-copy branch (Android/Linux GL).
#define MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY \
  (!MEDIAPIPE_DISABLE_GPU &&                     \
   MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31)
// Metal zero-copy branch (Apple). GLES takes precedence where both could apply.
#define MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY \
  (!MEDIAPIPE_DISABLE_GPU && !MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY && \
   MEDIAPIPE_METAL_ENABLED)
// Either GPU zero-copy branch (shared helpers: RoiToRotatedRect, kInImageGpu).
#define MEDIAPIPE_STREAMING_TILES_ANY_GPU_ZERO_COPY \
  (MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY ||       \
   MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY)

// Streams externally-supplied tiles into batched [N,H,W,C] float32 tensors.
// CPU: crop -> resize -> normalize into batch rows.
// GPU (GLES 3.1, optional, default off): crop/resize/normalize each tile in a
// compute shader directly into its batch row of a GPU-backed tensor, no CPU
// readback.
class StreamingTilesToTensorBatchCalculator : public Node {
 public:
  ~StreamingTilesToTensorBatchCalculator() override {
#if MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY
    // Open() may fail after creating GL objects, before Close() can run.
    const absl::Status status = ReleaseGlResources();
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed releasing tiled GL resources: " << status;
    }
#endif
  }

  // IMAGE is the CPU input. It is optional so a GPU-only graph can wire
  // IMAGE_GPU instead; exactly one image input must be present per frame.
  static constexpr Input<ImageFrame>::Optional kInImage{"IMAGE"};
  static constexpr Input<TilePlan> kInPlan{"TILE_PLAN"};
  static constexpr SideInput<InferenceMetadata>::Optional kSideMeta{"METADATA"};
  static constexpr Output<std::vector<Tensor>> kOutTensors{"TENSORS"};
  static constexpr Output<TensorBatchInfo> kOutInfo{"BATCH_INFO"};
  static constexpr Output<TilingCacheStats>::Optional kOutStats{"CACHE_STATS"};
#if !MEDIAPIPE_DISABLE_GPU
  static constexpr Input<mediapipe::GpuBuffer>::Optional kInImageGpu{
      "IMAGE_GPU"};
  MEDIAPIPE_NODE_CONTRACT(kInImage, kInImageGpu, kInPlan, kSideMeta,
                          kOutTensors, kOutInfo, kOutStats,
                          ::mediapipe::api2::TimestampChange::Arbitrary());
#else
  MEDIAPIPE_NODE_CONTRACT(kInImage, kInPlan, kSideMeta, kOutTensors, kOutInfo,
                          kOutStats,
                          ::mediapipe::api2::TimestampChange::Arbitrary());
#endif  // !MEDIAPIPE_DISABLE_GPU

#if MEDIAPIPE_STREAMING_TILES_ANY_GPU_ZERO_COPY
  // Declare the (optional) GPU service so it is injected when the graph provides
  // GpuResources. Keeping it OPTIONAL means a CPU-only graph on a GPU-enabled
  // build (no GpuResources) still runs the CPU path unaffected.
  static absl::Status UpdateContract(CalculatorContract* cc) {
#if MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY
    return [MPPMetalHelper updateContract:cc requestGpuAsOptional:true];
#else
    return mediapipe::GlCalculatorHelper::UpdateContract(
        cc, /*request_gpu_as_optional=*/true);
#endif
  }
#endif  // MEDIAPIPE_STREAMING_TILES_ANY_GPU_ZERO_COPY

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<
        mediapipe::StreamingTilesToTensorBatchCalculatorOptions>();
    if (kSideMeta(cc).IsConnected()) {
      meta_ = kSideMeta(cc).Get();
      if (options_.has_metadata_batch_capacity() ||
          options_.has_metadata_input_height() ||
          options_.has_metadata_input_width() ||
          options_.has_metadata_input_channels() ||
          options_.has_metadata_is_dynamic_batch()) {
        ABSL_LOG(WARNING) << "Both METADATA side packet and options-borne "
                             "metadata set; side packet wins.";
      }
    } else {
      meta_.set_batch_capacity(options_.metadata_batch_capacity());
      meta_.set_input_height(options_.metadata_input_height());
      meta_.set_input_width(options_.metadata_input_width());
      meta_.set_input_channels(options_.metadata_input_channels());
      meta_.set_is_dynamic_batch(options_.metadata_is_dynamic_batch());
    }
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
    // GPU zero-copy option validation: reject invalid capacities up front so
    // a bad config fails at Open() rather than mid-stream.
    RET_CHECK_GE(options_.max_gpu_tensor_buffers(), 0);
    // Bound retained AHWB buffers. Buffers still referenced downstream may
    // exceed this cache budget and must not be reclaimed early.
    if (options_.enable_gpu_zero_copy()) {
      RET_CHECK_GT(options_.max_gpu_tensor_buffers(), 0)
          << "enable_gpu_zero_copy requires a finite GPU buffer capacity "
             "(set max_gpu_tensor_buffers > 0)";
    }
#if MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY
    if (options_.enable_gpu_zero_copy() && kInImageGpu(cc).IsConnected()) {
      RET_CHECK_EQ(meta_.input_channels(), 3)
          << "GPU zero-copy path supports RGB (input_channels == 3) only";
      ABSL_RETURN_IF_ERROR(gl_helper_.Open(cc));
      const int H = meta_.input_height();
      const int W = meta_.input_width();
      const int C = meta_.input_channels();
      ABSL_RETURN_IF_ERROR(gl_helper_.RunInGlContext([&]() -> absl::Status {
        tflite::gpu::GpuInfo gpu_info;
        ABSL_RETURN_IF_ERROR(tflite::gpu::gl::RequestGpuInfo(&gpu_info));
        RET_CHECK(gpu_info.IsApiOpenGl31OrAbove())
            << "enable_gpu_zero_copy requires OpenGL ES 3.1.";
        command_queue_ = tflite::gpu::gl::NewCommandQueue(gpu_info);
        ABSL_ASSIGN_OR_RETURN(
            gl_writer_,
            TiledBatchGlWriter::Create(gl_helper_.GetGlContext(), W, H, C,
                                       BorderMode::kReplicate,
                                       /*input_starts_at_bottom=*/false));
        return absl::OkStatus();
      }));
      // Cache 5 (GPU tensor-buffer pool): on AHardwareBuffer-capable platforms a
      // MemoryManager pools the batch tensors' AHWB storage and reclaims it on
      // packet release + read-finished fence (framework-managed) — true
      // release+fence ownership for the AHWB case. On non-AHWB GL (e.g. Linux
      // EGL) it allocates per batch (no SSBO pool exists in the framework). The
      // explicit GpuResourceLedger state machine (tiling_gpu_resource.h, unit
      // tested) models the SSBO in-flight/reclaim accounting; wiring it to real
      // SSBO reuse requires a packet-release hook and is verified on a GLES
      // device (it is intentionally NOT gating this hot path, since a reclaim
      // signal that never fires would wedge the pool after `capacity` frames).
      if (options_.max_gpu_tensor_buffers() > 0) {
        gpu_memory_manager_ = std::make_shared<MemoryManager>(
            TilingGpuPoolOptions(options_.max_gpu_tensor_buffers()));
      }
      gpu_zero_copy_active_ = true;
    }
#endif  // MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY
#if MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY
    if (options_.enable_gpu_zero_copy() && kInImageGpu(cc).IsConnected()) {
      RET_CHECK_EQ(meta_.input_channels(), 3)
          << "Metal zero-copy path supports RGB (input_channels == 3) only";
      RET_CHECK(cc->Service(mediapipe::kGpuService).IsAvailable())
          << "GPU service not available for Metal tiled preprocessing";
      metal_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
      RET_CHECK(metal_helper_ != nil) << "failed creating MPPMetalHelper";
      ABSL_ASSIGN_OR_RETURN(
          metal_writer_,
          TiledBatchMetalWriter::Create(metal_helper_.mtlDevice,
                                        meta_.input_width(),
                                        meta_.input_height(),
                                        meta_.input_channels(),
                                        BorderMode::kReplicate,
                                        options_.metal_direct_delegate_input()));
      // Metal has no tensor buffer pool; use per-batch allocation without
      // creating an unused CPU pool from max_gpu_tensor_buffers.
      metal_zero_copy_active_ = true;
    }
#endif  // MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // A missing TILE_PLAN packet (bound advanced past this timestamp without
    // a packet) is treated as an empty plan: emit the empty-frame BATCH_INFO
    // so the downstream merge still sees the frame (api2 Get() on an empty
    // packet is fatal).
    if (kInPlan(cc).IsEmpty()) {
      const int64_t ts = cc->InputTimestamp().Value();
      EmitEmptyFrameIfNeeded(cc, /*T=*/0, ts);
      MaybeEmitStats(cc, ts);
      return absl::OkStatus();
    }
#if MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY
    if (metal_zero_copy_active_ && kInImageGpu(cc).IsConnected() &&
        !kInImageGpu(cc).IsEmpty()) {
      return ProcessMetal(cc);
    }
#endif  // MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY
#if MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY
    if (gpu_zero_copy_active_ && kInImageGpu(cc).IsConnected() &&
        !kInImageGpu(cc).IsEmpty()) {
      return ProcessGpu(cc);
    }
#endif  // MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY
#if MEDIAPIPE_STREAMING_TILES_ANY_GPU_ZERO_COPY
    // Zero-copy was requested (the GPU writer initialized) but this frame had
    // no usable GPU input: it is served on the CPU path below. Count it so
    // CACHE_STATS.gpu_to_cpu_fallbacks reports the silent fallback honestly.
#if MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY
    if (metal_zero_copy_active_) ++gpu_to_cpu_fallbacks_;
#else
    if (gpu_zero_copy_active_) ++gpu_to_cpu_fallbacks_;
#endif
#endif  // MEDIAPIPE_STREAMING_TILES_ANY_GPU_ZERO_COPY
    return ProcessCpu(cc);
  }

  absl::Status Close(CalculatorContext* cc) override {
#if MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY
    return ReleaseGlResources();
#else
    return absl::OkStatus();
#endif
  }

 private:
#if MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY
  absl::Status ReleaseGlResources() {
    if (!gl_writer_ && !command_queue_) return absl::OkStatus();
    return gl_helper_.RunInGlContext([this]() -> absl::Status {
      gl_writer_.reset();
      command_queue_.reset();
      gpu_zero_copy_active_ = false;
      return absl::OkStatus();
    });
  }
#endif

  absl::Status ProcessCpu(CalculatorContext* cc) {
    RET_CHECK(kInImage(cc).IsConnected() && !kInImage(cc).IsEmpty())
        << "no IMAGE (CPU) input present; either wire IMAGE or enable the GPU "
           "zero-copy path with an IMAGE_GPU input";
    const ImageFrame& frame = *kInImage(cc);
    RET_CHECK(frame.Format() == ImageFormat::SRGB ||
              frame.Format() == ImageFormat::SRGBA ||
              frame.Format() == ImageFormat::SBGRA ||
              frame.Format() == ImageFormat::GRAY8)
        << "CPU tiled preprocessing supports SRGB, SRGBA, SBGRA and GRAY8; got "
        << static_cast<int>(frame.Format());
    const TilePlan& plan = *kInPlan(cc);
    const int H = meta_.input_height();
    const int W = meta_.input_width();
    const int C = meta_.input_channels();
    const int cap = meta_.batch_capacity();
    bool physical_metal_input = false;
#if MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY
    physical_metal_input = options_.enable_gpu_zero_copy() &&
                           options_.metal_direct_delegate_input() &&
                           kInImageGpu(cc).IsConnected();
#endif
    const int tensor_channels = physical_metal_input ? 4 : C;
    const int frame_channels = frame.NumberOfChannels();
    int alpha_to_rgb_conversion = -1;
    if (C == 3 && frame.Format() == ImageFormat::SRGBA) {
      alpha_to_rgb_conversion = cv::COLOR_RGBA2RGB;
    } else if (C == 3 && frame.Format() == ImageFormat::SBGRA) {
      alpha_to_rgb_conversion = cv::COLOR_BGRA2RGB;
    }
    const bool drop_alpha = alpha_to_rgb_conversion >= 0;
    RET_CHECK(frame_channels == C || drop_alpha)
        << "model expects " << C << " input channels but the frame has "
        << frame_channels;
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
      if (physical_metal_input) {
        RET_CHECK_EQ(N, 1)
            << "metal_direct_delegate_input requires batch N==1 "
               "(set batch_capacity=1); got N=" << N;
      }
      Tensor tensor(Tensor::ElementType::kFloat32,
                    Tensor::Shape({N, H, W, tensor_channels},
                                  dynamic_batch_ && !physical_metal_input),
                    memory_manager_.get());
      auto write = tensor.GetCpuWriteView();
      float* buf = write.buffer<float>();
      std::memset(buf, 0, sizeof(float) * N * H * W * tensor_channels);

      std::shared_ptr<const TileBatchGeometry> geom =
          BuildOrGetGeometry(plan, start, rows, fw, fh, W, H, C);

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
        const cv::Mat* model_input = &resized_workspace_;
        if (drop_alpha) {
          cv::cvtColor(resized_workspace_, rgb_workspace_,
                       alpha_to_rgb_conversion);
          model_input = &rgb_workspace_;
        }
        model_input->convertTo(f32_workspace_, CV_32FC(C), 1.0 / 255.0);
        float* row = buf + static_cast<size_t>(r) * H * W * tensor_channels;
        const float* pixels = f32_workspace_.ptr<float>(0);
        if (physical_metal_input) {
          // The downstream delegate keeps its physical PHWC4 input contract
          // on a CPU fallback frame. The fourth channel remains zero-filled.
          for (int p = 0; p < H * W; ++p) {
            std::copy_n(pixels + p * C, C, row + p * tensor_channels);
          }
        } else {
          std::memcpy(row, pixels, sizeof(float) * H * W * C);
        }
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
    EmitEmptyFrameIfNeeded(cc, T, ts);
    MaybeEmitStats(cc, ts);
    return absl::OkStatus();
  }

#if MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY
  // GPU zero-copy path: crop/resize/normalize each tile into its batch row of a
  // GPU-backed tensor via the GLES 3.1 compute writer, no CPU readback. Mirrors
  // ProcessCpu's batch/timestamp/INFO semantics exactly.
  absl::Status ProcessGpu(CalculatorContext* cc) {
    const mediapipe::GpuBuffer& gpu = *kInImageGpu(cc);
    RET_CHECK(gpu.format() == GpuBufferFormat::kBGRA32 ||
              gpu.format() == GpuBufferFormat::kRGBA32 ||
              gpu.format() == GpuBufferFormat::kRGB24 ||
              gpu.format() == GpuBufferFormat::kRGBAHalf64 ||
              gpu.format() == GpuBufferFormat::kRGBAFloat128)
        << "GPU tiled preprocessing requires an RGB or RGBA input format; got "
        << static_cast<uint32_t>(gpu.format());
    const TilePlan& plan = *kInPlan(cc);
    const int H = meta_.input_height();
    const int W = meta_.input_width();
    const int C = meta_.input_channels();
    const int cap = meta_.batch_capacity();
    const int fw = gpu.width();
    const int fh = gpu.height();

    const int T = static_cast<int>(plan.tiles.size());
    const int total_batches = (T + cap - 1) / cap;
    const int64_t ts = cc->InputTimestamp().Value();

    ++gpu_frames_;  // one program use per frame (Cache 4 reuse accounting)
    int emitted = 0;
    for (int start = 0; start < T; start += cap) {
      const int rows = std::min(cap, T - start);
      const int N = dynamic_batch_ ? rows : cap;
      // gpu_memory_manager_ pools AHWB-backed storage where available (Cache 5).
      Tensor tensor(Tensor::ElementType::kFloat32,
                    Tensor::Shape({N, H, W, C}, dynamic_batch_),
                    gpu_memory_manager_.get());

      std::shared_ptr<const TileBatchGeometry> geom =
          BuildOrGetGeometry(plan, start, rows, fw, fh, W, H, C);

      ABSL_RETURN_IF_ERROR(gl_helper_.RunInGlContext([&]() -> absl::Status {
        auto src = gl_helper_.CreateSourceTexture(gpu);
        // GpuBuffer textures are 4-channel (e.g. BGRA32); GL sampling returns
        // normalized [0,1], so alpha=1 (NOT 1/255) matches the CPU uint8/255.
        tflite::gpu::gl::GlTexture input_texture(
            GL_TEXTURE_2D, src.name(), GL_RGBA,
            src.width() * src.height() * 4, /*layer=*/0, /*owned=*/false);
        const tflite::gpu::HW tex_size(src.height(), src.width());

        auto write_view = tensor.GetOpenGlBufferWriteView();
        tflite::gpu::gl::GlBuffer dest(GL_SHADER_STORAGE_BUFFER,
                                       write_view.name(), tensor.bytes(),
                                       /*offset=*/0, /*has_ownership=*/false);
        for (int r = 0; r < rows; ++r) {
          const RotatedRect rr = RoiToRotatedRect(geom->effective_pixel_rois[r]);
          ABSL_RETURN_IF_ERROR(gl_writer_->WriteTileRow(
              input_texture, tex_size, rr, r, /*alpha=*/1.0f, /*beta=*/0.0f,
              command_queue_.get(), &dest));
        }
        // Clear padding rows [rows, N) on the GPU: alpha=0,beta=0 writes zeros
        // via the same shader (no GLES buffer-clear extension needed).
        const RotatedRect full = RoiToRotatedRect(TilePixelRoi{0, 0, fw, fh});
        for (int r = rows; r < N; ++r) {
          ABSL_RETURN_IF_ERROR(gl_writer_->WriteTileRow(
              input_texture, tex_size, full, r, /*alpha=*/0.0f, /*beta=*/0.0f,
              command_queue_.get(), &dest));
        }
        return absl::OkStatus();
        // write_view destructs here -> GL fence created; downstream inference's
        // OpenGL read view waits on it on the GPU. No CPU readback.
      }));

      TensorBatchInfo info;
      info.source_frame_timestamp = ts;
      info.batch_timestamp = batch_ts_.Value();
      info.batch_index = emitted;
      info.total_batches = total_batches;
      info.batch_capacity = N;
      info.batch_size = N;
      info.valid_count = rows;
      info.tile_indices = geom->tile_indices;
      info.geometry = geom;

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
    last_frame_gpu_batches_ = emitted;  // batches emitted from this frame
    EmitEmptyFrameIfNeeded(cc, T, ts);
    MaybeEmitStats(cc, ts);
    return absl::OkStatus();
  }
#endif  // MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY

#if MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY
  // Metal zero-copy path (Apple): crop/resize/normalize each tile into its row
  // of a Metal-backed LOGICAL [N,H,W,C] tensor via a Metal compute writer, no
  // CPU readback. Mirrors ProcessCpu's batch/timestamp/INFO semantics. The
  // emitted tensor is the logical model input (contiguous BHWC); downstream
  // InferenceCalculatorMetal runs its normal BHWC->BPHWC4 conversion, which
  // handles batch correctly for any N. (Earlier this emitted physical PHWC4 and
  // bound it directly to the delegate, but that contiguous layout only matches
  // the delegate's batch-innermost SHWBC4 input at N=1; the logical-tensor
  // approach here works for all N at the cost of one cheap on-GPU conversion.)
  absl::Status ProcessMetal(CalculatorContext* cc) {
    const mediapipe::GpuBuffer& gpu = *kInImageGpu(cc);
    // ImageFrame-backed RGBA32 buffers are converted to BGRA CVPixelBuffers
    // by the storage registry. Validate the source format as well as the
    // directly supported CVPixelBuffer formats, without admitting YUV planes.
    RET_CHECK(gpu.format() == GpuBufferFormat::kBGRA32 ||
              gpu.format() == GpuBufferFormat::kRGBA32 ||
              gpu.format() == GpuBufferFormat::kRGBAHalf64)
        << "Metal tiled preprocessing requires an RGB or RGBA input format; got "
        << static_cast<uint32_t>(gpu.format());
    const TilePlan& plan = *kInPlan(cc);
    const int H = meta_.input_height();
    const int W = meta_.input_width();
    const int C = meta_.input_channels();
    const int cap = meta_.batch_capacity();
    const int fw = gpu.width();
    const int fh = gpu.height();

    const int T = static_cast<int>(plan.tiles.size());
    const int total_batches = (T + cap - 1) / cap;
    const int64_t ts = cc->InputTimestamp().Value();

    ++gpu_frames_;  // one program use per frame (Cache 4 reuse accounting)
    int emitted = 0;
    for (int start = 0; start < T; start += cap) {
      const int rows = std::min(cap, T - start);
      const int N = dynamic_batch_ ? rows : cap;

      std::shared_ptr<const TileBatchGeometry> geom =
          BuildOrGetGeometry(plan, start, rows, fw, fh, W, H, C);

      if (options_.metal_direct_delegate_input()) {
        // Direct delegate input: physical PHWC4, N==1 per batch (fixed-batch-1
        // model). Guard loudly — N>1 / C>4 cannot be direct-bound.
        RET_CHECK_EQ(N, 1) << "metal_direct_delegate_input requires batch N==1 "
                              "(set batch_capacity=1); got N="
                           << N;
        RET_CHECK_LE(C, 4) << "metal_direct_delegate_input requires C<=4; got C="
                           << C;
        Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{1, H, W, 4},
                      gpu_memory_manager_.get());

        @autoreleasepool {
          id<MTLTexture> texture = [metal_helper_ metalTextureWithGpuBuffer:gpu];
          RET_CHECK(texture != nil)
              << "failed creating MTLTexture from GpuBuffer";
          id<MTLCommandBuffer> command_buffer = [metal_helper_ commandBuffer];
          const auto& write_view =
              MtlBufferView::GetWriteView(tensor, command_buffer);
          // Physical PHWC4: write the single tile (the one batch row) at row 0.
          // The writer was created with physical_phwc4=true so it writes 4
          // floats/pixel (RGB + 0.0).
          const RotatedRect rr =
              RoiToRotatedRect(geom->effective_pixel_rois[0]);
          ABSL_RETURN_IF_ERROR(metal_writer_->WriteTileRow(
              texture, rr, /*tile_row=*/0, /*alpha=*/1.0f, /*beta=*/0.0f,
              command_buffer, write_view.buffer()));
          [command_buffer commit];
          // write_view destructs here -> Metal fence; downstream Metal
          // inference's read view waits on it. No CPU readback.
        }

        TensorBatchInfo info;
        info.source_frame_timestamp = ts;
        info.batch_timestamp = batch_ts_.Value();
        info.batch_index = emitted;
        info.total_batches = total_batches;
        info.batch_capacity = N;
        info.batch_size = N;
        info.valid_count = rows;
        info.tile_indices = geom->tile_indices;
        info.geometry = geom;

        std::vector<Tensor> tensors;
        tensors.push_back(std::move(tensor));
        kOutTensors(cc).Send(
            mediapipe::api2::MakePacket<std::vector<Tensor>>(std::move(tensors))
                .At(batch_ts_));
        kOutInfo(cc).Send(
            mediapipe::api2::MakePacket<TensorBatchInfo>(std::move(info))
                .At(batch_ts_));
      } else {
        // Logical [N,H,W,C] Metal path (default): each tile written to its row.
        Tensor tensor(Tensor::ElementType::kFloat32,
                      Tensor::Shape({N, H, W, C}, dynamic_batch_),
                      gpu_memory_manager_.get());

        @autoreleasepool {
          id<MTLTexture> texture = [metal_helper_ metalTextureWithGpuBuffer:gpu];
          RET_CHECK(texture != nil)
              << "failed creating MTLTexture from GpuBuffer";
          id<MTLCommandBuffer> command_buffer = [metal_helper_ commandBuffer];
          const auto& write_view =
              MtlBufferView::GetWriteView(tensor, command_buffer);
          // GpuBuffer textures are 4-channel; Metal samples them as [0,1], so
          // alpha=1 (NOT 1/255) matches the CPU uint8/255 normalization.
          for (int r = 0; r < rows; ++r) {
            const RotatedRect rr =
                RoiToRotatedRect(geom->effective_pixel_rois[r]);
            ABSL_RETURN_IF_ERROR(metal_writer_->WriteTileRow(
                texture, rr, r, /*alpha=*/1.0f, /*beta=*/0.0f, command_buffer,
                write_view.buffer()));
          }
          // Clear padding rows [rows, N): alpha=0,beta=0 writes zeros.
          const RotatedRect full = RoiToRotatedRect(TilePixelRoi{0, 0, fw, fh});
          for (int r = rows; r < N; ++r) {
            ABSL_RETURN_IF_ERROR(metal_writer_->WriteTileRow(
                texture, full, r, /*alpha=*/0.0f, /*beta=*/0.0f, command_buffer,
                write_view.buffer()));
          }
          [command_buffer commit];
          // write_view destructs at scope end -> Metal fence; downstream Metal
          // inference's read view waits on it. No CPU readback.
        }

        TensorBatchInfo info;
        info.source_frame_timestamp = ts;
        info.batch_timestamp = batch_ts_.Value();
        info.batch_index = emitted;
        info.total_batches = total_batches;
        info.batch_capacity = N;
        info.batch_size = N;
        info.valid_count = rows;
        info.tile_indices = geom->tile_indices;
        info.geometry = geom;

        std::vector<Tensor> tensors;
        tensors.push_back(std::move(tensor));
        kOutTensors(cc).Send(
            mediapipe::api2::MakePacket<std::vector<Tensor>>(std::move(tensors))
                .At(batch_ts_));
        kOutInfo(cc).Send(
            mediapipe::api2::MakePacket<TensorBatchInfo>(std::move(info))
                .At(batch_ts_));
      }
      ++batch_ts_;
      ++emitted;
    }
    last_frame_gpu_batches_ = emitted;
    EmitEmptyFrameIfNeeded(cc, T, ts);
    MaybeEmitStats(cc, ts);
    return absl::OkStatus();
  }
#endif  // MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY

#if MEDIAPIPE_STREAMING_TILES_ANY_GPU_ZERO_COPY
  static RotatedRect RoiToRotatedRect(const TilePixelRoi& roi) {
    RotatedRect rect;
    rect.center_x = roi.x + roi.width / 2.0f;
    rect.center_y = roi.y + roi.height / 2.0f;
    rect.width = roi.width;
    rect.height = roi.height;
    rect.rotation = 0.0f;
    return rect;
  }
#endif  // MEDIAPIPE_STREAMING_TILES_ANY_GPU_ZERO_COPY

  // Geometry (matrices + ROIs) depends only on frame size + tile set, not
  // pixels. Cache it; the pixel crop/resize always runs every frame.
  std::shared_ptr<const TileBatchGeometry> BuildOrGetGeometry(
      const TilePlan& plan, int start, int rows, int fw, int fh, int W, int H,
      int C) {
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
    return geom;
  }

  // If T == 0, emit one empty BATCH_INFO at batch_ts_ so merge sees the frame.
  void EmitEmptyFrameIfNeeded(CalculatorContext* cc, int T, int64_t ts) {
    if (T != 0) return;
    TensorBatchInfo info;
    info.source_frame_timestamp = ts;
    info.batch_timestamp = batch_ts_.Value();
    info.total_batches = 0;
    info.valid_count = 0;
    kOutInfo(cc).Send(
        mediapipe::api2::MakePacket<TensorBatchInfo>(std::move(info))
            .At(batch_ts_));
    // No TENSORS packet is produced for an empty frame, but the TENSORS stream
    // must still advance past this timestamp. Otherwise downstream inference
    // (which only emits detections for TENSORS timestamps) never advances its
    // output bound, and the synchronized merge — which pairs BATCH_INFO with
    // detections — would stall instead of emitting the empty source-frame
    // result. (EndLoopCalculator uses the same SetNextTimestampBound pattern.)
    kOutTensors(cc).SetNextTimestampBound(batch_ts_ + 1);
    ++batch_ts_;
  }

  void MaybeEmitStats(CalculatorContext* cc, int64_t ts) {
    if (!options_.emit_cache_stats() || !kOutStats(cc).IsConnected()) return;
    TilingCacheStats stats;
    stats.tile_matrix = matrix_cache_.stats();
    if (memory_manager_ && memory_manager_->GetCpuBufferPool()) {
      stats.cpu_tensor_pool = memory_manager_->GetCpuBufferPool()->stats();
    }
#if MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY
    if (gpu_zero_copy_active_) {
      // Cache 4 (program/tile-surface): the compute program is compiled once
      // (the single miss) and reused every subsequent frame (a hit per frame).
      stats.tile_surface.misses = (gl_writer_ != nullptr) ? 1 : 0;
      stats.tile_surface.hits = (gpu_frames_ > 0) ? gpu_frames_ - 1 : 0;
      stats.last_frame_gpu_batches = last_frame_gpu_batches_;
      stats.gpu_to_cpu_fallbacks = gpu_to_cpu_fallbacks_;
      // gpu_tensor_buffer hit/miss accounting is reported by the framework AHWB
      // pool (Cache 5) on AHWB platforms; the SSBO-pool counters are surfaced
      // once GpuResourceLedger is wired to real packet-release/fence signals on
      // a GLES device.
    }
#endif  // MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY
#if MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY
    if (metal_zero_copy_active_) {
      // Cache 4: the Metal render pipeline is compiled once and reused.
      stats.tile_surface.misses = (metal_writer_ != nullptr) ? 1 : 0;
      stats.tile_surface.hits = (gpu_frames_ > 0) ? gpu_frames_ - 1 : 0;
      stats.last_frame_gpu_batches = last_frame_gpu_batches_;
      stats.gpu_to_cpu_fallbacks = gpu_to_cpu_fallbacks_;
    }
#endif  // MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY
    kOutStats(cc).Send(
        mediapipe::api2::MakePacket<TilingCacheStats>(stats).At(Timestamp(ts)));
  }

  static std::shared_ptr<const TileBatchGeometry> BuildBatchGeometry(
      const TilePlan& plan, int start, int rows, int fw, int fh) {
    auto geom = std::make_shared<TileBatchGeometry>();
    geom->tile_indices.reserve(rows);
    geom->tile_geometries.reserve(rows);
    geom->effective_pixel_rois.reserve(rows);
    geom->tile_to_image_matrices.reserve(rows);
    geom->image_to_tile_matrices.reserve(rows);
    const auto round_pixel = [](float normalized, int extent) {
      // Preserve the existing float multiplication/rounding for ordinary
      // tiles (e.g. 0.35f * 10 rounds to 4). Use double only when the float
      // product overflows, then clip before converting the result to int.
      const float scaled = normalized * extent;
      return std::isfinite(scaled)
                 ? static_cast<double>(std::round(scaled))
                 : std::round(static_cast<double>(normalized) * extent);
    };
    for (int r = 0; r < rows; ++r) {
      const TileGeometry& g = plan.tiles[start + r];
      // Clip both endpoints of the rounded pixel crop. Clamping its origin
      // without shrinking the width/height shifts left/top boundary tiles
      // into the frame and samples pixels outside the requested region.
      // Keep intermediates wide until clipped, including for large but finite
      // normalized rectangles that overlap the frame.
      const double requested_x = round_pixel(g.x0(), fw);
      const double requested_y = round_pixel(g.y0(), fh);
      const double requested_w = round_pixel(g.width, fw);
      const double requested_h = round_pixel(g.height, fh);
      const int rx = static_cast<int>(
          std::clamp(requested_x, 0.0, static_cast<double>(fw - 1)));
      const int ry = static_cast<int>(
          std::clamp(requested_y, 0.0, static_cast<double>(fh - 1)));
      const int right = static_cast<int>(std::clamp(
          requested_x + requested_w, static_cast<double>(rx + 1),
          static_cast<double>(fw)));
      const int bottom = static_cast<int>(std::clamp(
          requested_y + requested_h, static_cast<double>(ry + 1),
          static_cast<double>(fh)));
      const int rw = right - rx;
      const int rh = bottom - ry;
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
  cv::Mat rgb_workspace_;      // used only when CPU input has an alpha channel
  cv::Mat f32_workspace_;
#if MEDIAPIPE_STREAMING_TILES_ANY_GPU_ZERO_COPY
  // Shared by both GPU branches (Cache 5 AHWB pool + diagnostic counters).
  std::shared_ptr<MemoryManager> gpu_memory_manager_;
  int64_t gpu_frames_ = 0;             // ProcessGpu/Metal() calls = program uses
  int64_t last_frame_gpu_batches_ = 0;  // gauge: batches emitted from the last GPU frame
  int64_t gpu_to_cpu_fallbacks_ = 0;   // GPU-requested frames served on CPU
#endif  // MEDIAPIPE_STREAMING_TILES_ANY_GPU_ZERO_COPY
#if MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY
  mediapipe::GlCalculatorHelper gl_helper_;
  std::unique_ptr<tflite::gpu::gl::CommandQueue> command_queue_;
  std::unique_ptr<TiledBatchGlWriter> gl_writer_;
  bool gpu_zero_copy_active_ = false;
#endif  // MEDIAPIPE_STREAMING_TILES_GPU_ZERO_COPY
#if MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY
  MPPMetalHelper* metal_helper_ = nil;
  std::unique_ptr<TiledBatchMetalWriter> metal_writer_;
  bool metal_zero_copy_active_ = false;
#endif  // MEDIAPIPE_STREAMING_TILES_METAL_ZERO_COPY
};

MEDIAPIPE_REGISTER_NODE(StreamingTilesToTensorBatchCalculator);

}  // namespace api2
}  // namespace mediapipe
