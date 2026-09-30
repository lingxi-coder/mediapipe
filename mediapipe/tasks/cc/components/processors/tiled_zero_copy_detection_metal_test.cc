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
// END-TO-END proof of TRUE Metal zero-copy through a real tiled OBB detection
// pipeline (runs on this Mac, Apple M2 Max):
//
//   GPU frame -> StreamingTilesToTensorBatch (physical PHWC4, direct delegate
//   input) -> InferenceCalculatorMetal (binds the packet MTLBuffer directly,
//   converter skipped) -> YoloObb decode -> tile->global merge -> global
//   rotated NMS.
//
// The zero-copy path (variant 2) is validated against a CPU-front control
// (variant 1) that runs the SAME Metal delegate on the SAME frame + tile plan,
// differing ONLY in the front + inference input binding:
//   * Converter-skipped on every inference batch (ZERO_COPY_DEBUG != 0).
//   * Buffer IDENTITY: the delegate bound exactly the front-produced MTLBuffers.
//   * GPU residency (no CPU materialization of the front tensors).
//   * No silent GPU->CPU readback fallback (gpu_to_cpu_fallbacks == 0).
//   * Merge emits exactly once per source frame.
//   * Detection equivalence (variant 2 ~= variant 1).
//   * A ship (DOTA class 1) is detected.
//
// On non-Metal configs the whole TU compiles to nothing.

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_METAL_ENABLED

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.pb.h"
#include "mediapipe/framework/formats/tiling_cache_stats.h"
#include "mediapipe/framework/formats/tiling_types.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/formats/tensor_mtl_buffer_view.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/util/image_test_utils.h"

namespace mediapipe {
namespace {

// Path convention mirrors the sibling oriented_object_detector tests.
constexpr char kTestDataDir[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kOrientedModel[] = "yolov8n-obb.tflite";
constexpr char kTestImage[] = "boats.jpg";

// Model fixture geometry (YOLOv8n-OBB on DOTAv1: 15 classes, 640x640x3 input).
// Both decode nodes use num_classes:15, layout:CHANNELS_FIRST, and a
// conf_threshold of 0.30 -- a margin above the model's natural 0.25 so the kept
// set is unambiguous (set inline in the graph protos below).
constexpr int kInputSize = 640;

// Equivalence tolerances. The merge/NMS output is in original-image NORMALIZED
// [0,1] coordinates (OrientedDetection fields are normalized; the merge
// projects tile->global normalized), so positional tolerances are normalized.
// kPxTol ~= a few px on a 640 input (3/640 ~= 0.0047); 0.01 covers a few px.
constexpr float kPxTol = 0.01f;     // normalized; ~6.4 px on the 640 input
constexpr float kRotTol = 0.02f;    // radians
constexpr float kScoreTol = 0.02f;  // softmax-style confidence

std::string ModelPath() {
  return ::mediapipe::file::JoinPath("./", kTestDataDir, kOrientedModel);
}
std::string ImagePath() {
  return ::mediapipe::file::JoinPath("./", kTestDataDir, kTestImage);
}

// Builds a deterministic 2-tile TilePlan: left x in [0.0,0.6], right x in
// [0.4,1.0], full height (overlapping in x so a boat near the seam can be seen
// by both tiles and deduped by the global NMS).
TilePlan TwoTilePlan() {
  TilePlan plan;
  TileGeometry left;
  left.tile_index = 0;
  left.x_center = 0.3f;  // [0.0, 0.6]
  left.y_center = 0.5f;
  left.width = 0.6f;
  left.height = 1.0f;
  TileGeometry right;
  right.tile_index = 1;
  right.x_center = 0.7f;  // [0.4, 1.0]
  right.y_center = 0.5f;
  right.width = 0.6f;
  right.height = 1.0f;
  plan.tiles = {left, right};
  return plan;
}

// Decodes boats.jpg into an RGB ImageFrame.
std::unique_ptr<ImageFrame> DecodeRgbImageFrame(const std::string& path) {
  cv::Mat rgb = GetRgb(path);  // BGR-decoded then converted to RGB
  auto frame = std::make_unique<ImageFrame>(ImageFormat::SRGB, rgb.cols,
                                            rgb.rows);
  cv::Mat dst(frame->Height(), frame->Width(), CV_8UC3,
              frame->MutablePixelData(), frame->WidthStep());
  rgb.copyTo(dst);
  return frame;
}

// Builds an opaque-alpha RGBA GpuBuffer matching the RGB frame (alpha = 255).
GpuBuffer RgbFrameToOpaqueRgbaGpuBuffer(const ImageFrame& rgb) {
  GpuBuffer buffer(rgb.Width(), rgb.Height(), GpuBufferFormat::kRGBA32);
  std::shared_ptr<ImageFrame> view = buffer.GetWriteView<ImageFrame>();
  const uint8_t* src = rgb.PixelData();
  const int src_stride = rgb.WidthStep();
  uint8_t* dst = view->MutablePixelData();
  const int dst_stride = view->WidthStep();
  for (int y = 0; y < rgb.Height(); ++y) {
    const uint8_t* sp = src + y * src_stride;
    uint8_t* dp = dst + y * dst_stride;
    for (int x = 0; x < rgb.Width(); ++x) {
      dp[0] = sp[0];
      dp[1] = sp[1];
      dp[2] = sp[2];
      dp[3] = 255;  // opaque alpha
      sp += 3;
      dp += 4;
    }
  }
  return buffer;
}

InferenceMetadata MakeMetadata() {
  InferenceMetadata meta;
  meta.set_input_height(kInputSize);
  meta.set_input_width(kInputSize);
  meta.set_input_channels(3);
  meta.set_batch_capacity(1);
  meta.set_is_dynamic_batch(false);
  return meta;
}

// Sort key for stable detection comparison: score desc, then label_id, then cx.
void SortDetections(std::vector<OrientedDetection>* dets) {
  std::sort(dets->begin(), dets->end(),
            [](const OrientedDetection& a, const OrientedDetection& b) {
              if (a.score(0) != b.score(0)) return a.score(0) > b.score(0);
              if (a.label_id(0) != b.label_id(0))
                return a.label_id(0) < b.label_id(0);
              return a.cx() < b.cx();
            });
}

class TiledZeroCopyDetectionMetalTest : public testing::Test {
 protected:
  GpuSharedData gpu_shared_;
  std::shared_ptr<GpuResources> gpu_resources_ = gpu_shared_.gpu_resources;
};

// ---------------------------------------------------------------------------
// Variant 1 (Metal control): CPU front (IMAGE, no zero-copy) -> Metal delegate
// (no external-input zero-copy) -> OBB decode -> merge -> rotated NMS.
// ---------------------------------------------------------------------------
absl::Status RunControlVariant(std::shared_ptr<GpuResources> gpu_resources,
                               std::unique_ptr<ImageFrame> rgb,
                               const TilePlan& plan,
                               const InferenceMetadata& meta,
                               std::vector<Packet>* final_out) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    input_stream: "tile_plan"
    output_stream: "detections"
    node {
      calculator: "StreamingTilesToTensorBatchCalculator"
      input_stream: "IMAGE:image"
      input_stream: "TILE_PLAN:tile_plan"
      input_side_packet: "METADATA:meta"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:info"
      options {
        [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
          enable_gpu_zero_copy: false
        }
      }
    }
    node {
      calculator: "InferenceCalculator"
      input_stream: "TENSORS:tensors"
      output_stream: "TENSORS:inferred"
      options {
        [mediapipe.InferenceCalculatorOptions.ext] {
          model_path: "mediapipe/tasks/testdata/vision/yolov8n-obb.tflite"
          delegate {
            gpu {
              allow_precision_loss: false
              metal_external_input_zero_copy: false
            }
          }
        }
      }
    }
    node {
      calculator: "YoloObbTensorsToOrientedDetectionsCalculator"
      input_stream: "TENSORS:inferred"
      output_stream: "ORIENTED_DETECTIONS:batched_dets"
      options {
        [mediapipe.YoloObbTensorsToOrientedDetectionsCalculatorOptions.ext] {
          num_classes: 15
          layout: CHANNELS_FIRST
          conf_threshold: 0.30
        }
      }
    }
    node {
      calculator: "MergeTileDetectionsAccumulatorCalculator"
      input_stream: "ORIENTED_DETECTIONS:batched_dets"
      input_stream: "BATCH_INFO:info"
      output_stream: "ORIENTED_DETECTIONS:merged"
    }
    node {
      calculator: "RotatedNonMaxSuppressionCalculator"
      input_stream: "ORIENTED_DETECTIONS:merged"
      output_stream: "ORIENTED_DETECTIONS:detections"
      options {
        [mediapipe.RotatedNonMaxSuppressionCalculatorOptions.ext] {
          iou_threshold: 0.45
        }
      }
    }
  )pb");

  CalculatorGraph graph;
  ABSL_RETURN_IF_ERROR(graph.Initialize(config));
  ABSL_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  ABSL_RETURN_IF_ERROR(
      graph.ObserveOutputStream("detections", [final_out](const Packet& p) {
        final_out->push_back(p);
        return absl::OkStatus();
      }));
  ABSL_RETURN_IF_ERROR(
      graph.StartRun({{"meta", MakePacket<InferenceMetadata>(meta)}}));
  ABSL_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "image", Adopt(rgb.release()).At(Timestamp(0))));
  ABSL_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "tile_plan", MakePacket<TilePlan>(plan).At(Timestamp(0))));
  ABSL_RETURN_IF_ERROR(graph.WaitUntilIdle());
  ABSL_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  return graph.WaitUntilDone();
}

// Per-batch zero-copy observations captured from variant 2.
struct ZeroCopyVariantResult {
  std::vector<Packet> final_out;       // final std::vector<OrientedDetection>
  std::vector<Packet> zero_copy_debug; // int64 bound-buffer addresses (per call)
  std::vector<Packet> front_tensors;   // std::vector<Tensor> (per batch)
  std::vector<Packet> cache_stats;     // TilingCacheStats (per source frame)
};

// ---------------------------------------------------------------------------
// Variant 2 (true zero-copy): GPU front (IMAGE_GPU, enable_gpu_zero_copy,
// metal_direct_delegate_input, emit_cache_stats, no readback fallback) ->
// Metal delegate (metal_external_input_zero_copy, ZERO_COPY_DEBUG connected) ->
// OBB decode -> merge -> rotated NMS. The front's TENSORS and CACHE_STATS are
// ALSO fanned to graph outputs so the test can observe GPU residency + the
// produced buffer addresses (a stream may feed both inference and an output).
// ---------------------------------------------------------------------------
absl::Status RunZeroCopyVariant(std::shared_ptr<GpuResources> gpu_resources,
                                const GpuBuffer& gpu_frame,
                                const TilePlan& plan,
                                const InferenceMetadata& meta,
                                ZeroCopyVariantResult* out) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image_gpu"
    input_stream: "tile_plan"
    output_stream: "detections"
    output_stream: "zero_copy_debug"
    output_stream: "front_tensors"
    output_stream: "cache_stats"
    node {
      calculator: "StreamingTilesToTensorBatchCalculator"
      input_stream: "IMAGE_GPU:image_gpu"
      input_stream: "TILE_PLAN:tile_plan"
      input_side_packet: "METADATA:meta"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:info"
      output_stream: "CACHE_STATS:cache_stats"
      options {
        [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
          enable_gpu_zero_copy: true
          metal_direct_delegate_input: true
          emit_cache_stats: true
          # Required for the zero-copy path: a finite GPU buffer pool. 2 tiles ->
          # 2 single-tile batches; pool 2 so both can be in flight.
          max_gpu_tensor_buffers: 2
        }
      }
    }
    node {
      calculator: "InferenceCalculator"
      input_stream: "TENSORS:tensors"
      output_stream: "TENSORS:inferred"
      output_stream: "ZERO_COPY_DEBUG:zero_copy_debug"
      options {
        [mediapipe.InferenceCalculatorOptions.ext] {
          model_path: "mediapipe/tasks/testdata/vision/yolov8n-obb.tflite"
          delegate {
            gpu {
              allow_precision_loss: false
              metal_external_input_zero_copy: true
            }
          }
        }
      }
    }
    node {
      calculator: "YoloObbTensorsToOrientedDetectionsCalculator"
      input_stream: "TENSORS:inferred"
      output_stream: "ORIENTED_DETECTIONS:batched_dets"
      options {
        [mediapipe.YoloObbTensorsToOrientedDetectionsCalculatorOptions.ext] {
          num_classes: 15
          layout: CHANNELS_FIRST
          conf_threshold: 0.30
        }
      }
    }
    node {
      calculator: "MergeTileDetectionsAccumulatorCalculator"
      input_stream: "ORIENTED_DETECTIONS:batched_dets"
      input_stream: "BATCH_INFO:info"
      output_stream: "ORIENTED_DETECTIONS:merged"
    }
    node {
      calculator: "RotatedNonMaxSuppressionCalculator"
      input_stream: "ORIENTED_DETECTIONS:merged"
      output_stream: "ORIENTED_DETECTIONS:detections"
      options {
        [mediapipe.RotatedNonMaxSuppressionCalculatorOptions.ext] {
          iou_threshold: 0.45
        }
      }
    }
    # Fan the front TENSORS to a graph output too (feeds inference AND output).
    node {
      calculator: "PassThroughCalculator"
      input_stream: "tensors"
      output_stream: "front_tensors"
    }
  )pb");

  CalculatorGraph graph;
  ABSL_RETURN_IF_ERROR(graph.Initialize(config));
  ABSL_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  ABSL_RETURN_IF_ERROR(
      graph.ObserveOutputStream("detections", [out](const Packet& p) {
        out->final_out.push_back(p);
        return absl::OkStatus();
      }));
  ABSL_RETURN_IF_ERROR(
      graph.ObserveOutputStream("zero_copy_debug", [out](const Packet& p) {
        out->zero_copy_debug.push_back(p);
        return absl::OkStatus();
      }));
  ABSL_RETURN_IF_ERROR(
      graph.ObserveOutputStream("front_tensors", [out](const Packet& p) {
        out->front_tensors.push_back(p);
        return absl::OkStatus();
      }));
  ABSL_RETURN_IF_ERROR(
      graph.ObserveOutputStream("cache_stats", [out](const Packet& p) {
        out->cache_stats.push_back(p);
        return absl::OkStatus();
      }));
  ABSL_RETURN_IF_ERROR(
      graph.StartRun({{"meta", MakePacket<InferenceMetadata>(meta)}}));
  ABSL_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "image_gpu", MakePacket<GpuBuffer>(gpu_frame).At(Timestamp(0))));
  ABSL_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "tile_plan", MakePacket<TilePlan>(plan).At(Timestamp(0))));
  ABSL_RETURN_IF_ERROR(graph.WaitUntilIdle());
  ABSL_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  return graph.WaitUntilDone();
}

TEST_F(TiledZeroCopyDetectionMetalTest, EndToEndTrueZeroCopyMatchesControl) {
  // Fixture gate (the fixtures ARE present in this environment -> runs).
  if (!::mediapipe::file::Exists(ModelPath()).ok() ||
      !::mediapipe::file::Exists(ImagePath()).ok()) {
    GTEST_SKIP() << "OBB fixtures not available (" << ModelPath() << " / "
                 << ImagePath() << "); add //mediapipe/tasks/testdata/vision:"
                    "yolo_obb_test_model to run.";
  }

  std::unique_ptr<ImageFrame> rgb = DecodeRgbImageFrame(ImagePath());
  // Build the GPU frame from the decoded pixels (copies), then the control
  // variant can take ownership of the CPU ImageFrame.
  GpuBuffer gpu_frame = RgbFrameToOpaqueRgbaGpuBuffer(*rgb);
  const TilePlan plan = TwoTilePlan();
  const InferenceMetadata meta = MakeMetadata();

  // Model metadata sanity (640x640x3 input, batch_capacity=1).
  ASSERT_EQ(meta.input_height(), 640);
  ASSERT_EQ(meta.input_width(), 640);
  ASSERT_EQ(meta.input_channels(), 3);
  ASSERT_EQ(meta.batch_capacity(), 1);

  // --- Variant 1: Metal control (CPU front, normal delegate input). ---
  std::vector<Packet> control_out;
  MP_ASSERT_OK(RunControlVariant(gpu_resources_, std::move(rgb), plan, meta,
                                 &control_out));

  // --- Variant 2: true zero-copy (GPU front, direct delegate bind). ---
  ZeroCopyVariantResult zc;
  MP_ASSERT_OK(RunZeroCopyVariant(gpu_resources_, gpu_frame, plan, meta, &zc));

  // Collect per-batch zero-copy debug addresses (bound input MTLBuffer / 0).
  std::vector<int64_t> zero_copy_debug;
  for (const Packet& p : zc.zero_copy_debug) {
    zero_copy_debug.push_back(p.Get<int64_t>());
  }
  // Collect the front-produced tensors' Metal buffer addresses + residency.
  // A standalone command buffer (the smoke-test pattern) lets us read the
  // existing Metal-resident buffer without forcing a CPU readback.
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ASSERT_TRUE(device != nil);
  id<MTLCommandQueue> queue = [device newCommandQueue];
  std::vector<int64_t> front_addrs;
  std::vector<const Tensor*> front_tensor_ptrs;
  for (const Packet& p : zc.front_tensors) {
    const auto& tensors = p.Get<std::vector<Tensor>>();
    ASSERT_EQ(tensors.size(), 1u);
    const Tensor& t = tensors[0];
    front_tensor_ptrs.push_back(&t);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    int64_t addr = reinterpret_cast<int64_t>(
        (__bridge void*)MtlBufferView::GetReadView(t, cb).buffer());
    front_addrs.push_back(addr);
  }

  // (a) Converter skipped on every inference batch. total_batches == 2.
  ASSERT_EQ(zero_copy_debug.size(), 2u);
  for (int64_t a : zero_copy_debug) {
    EXPECT_NE(a, 0) << "converter ran / no direct bind";
  }

  // (a') Buffer IDENTITY: the delegate bound exactly the front-produced
  // buffers (compare as sorted multisets).
  std::sort(zero_copy_debug.begin(), zero_copy_debug.end());
  std::sort(front_addrs.begin(), front_addrs.end());
  ASSERT_EQ(front_addrs.size(), zero_copy_debug.size());
  EXPECT_EQ(front_addrs, zero_copy_debug)
      << "delegate bound a different buffer than the front produced";

  // (b) GPU residency: front tensors are Metal-resident and were not CPU
  // materialized (read BEFORE any CPU readback above; the standalone read
  // view does not pull them to the CPU).
  for (const Tensor* t : front_tensor_ptrs) {
    EXPECT_TRUE(t->ready_as_metal_buffer());
    EXPECT_FALSE(t->ready_on_cpu());
  }

  // (b') No silent GPU->CPU fallback in the zero-copy front.
  ASSERT_EQ(zc.cache_stats.size(), 1u);
  const TilingCacheStats& stats = zc.cache_stats[0].Get<TilingCacheStats>();
  EXPECT_EQ(stats.gpu_to_cpu_fallbacks, 0);

  // (c) Merge emits exactly once per source frame (both variants).
  ASSERT_EQ(zc.final_out.size(), 1u);
  ASSERT_EQ(control_out.size(), 1u);

  // (d) Equivalence: variant 2 ~= variant 1.
  std::vector<OrientedDetection> dets1 =
      control_out[0].Get<std::vector<OrientedDetection>>();
  std::vector<OrientedDetection> dets2 =
      zc.final_out[0].Get<std::vector<OrientedDetection>>();
  SortDetections(&dets1);
  SortDetections(&dets2);

  // COUNT GUARD (documented): the two fronts feed the SAME Metal delegate, so
  // the decoded oriented boxes are near-bit-identical (observed: cx/cy/score
  // agree to ~1e-4 between the CPU front and the GPU direct-PHWC4 front).
  // BUT the final count can differ by a single box: in this dense marina scene
  // (~135 ships) some box pairs sit right at the rotated-NMS IoU boundary
  // (iou_threshold=0.45), and a sub-1e-3 coordinate difference flips whether one
  // of a near-duplicate pair is suppressed. That is an NMS tie-break, NOT a
  // pipeline divergence, so we allow |count| to differ by at most 1 and require
  // an exact 1:1 nearest-neighbour match of every detection in the SMALLER set
  // to a distinct detection in the larger set (same label, all box params within
  // tolerance). This is strictly stronger than index-aligned EXPECT_NEAR (it
  // tolerates exactly the one unmatched NMS-tie box and nothing else).
  const size_t n1 = dets1.size(), n2 = dets2.size();
  ASSERT_LE(n1 > n2 ? n1 - n2 : n2 - n1, 1u)
      << "detection counts differ by more than one NMS tie-break: v1=" << n1
      << " v2=" << n2;

  const std::vector<OrientedDetection>& smaller = (n2 <= n1) ? dets2 : dets1;
  const std::vector<OrientedDetection>& larger = (n2 <= n1) ? dets1 : dets2;
  std::vector<bool> used(larger.size(), false);
  auto matches = [](const OrientedDetection& a, const OrientedDetection& b) {
    return a.label_id(0) == b.label_id(0) &&
           std::abs(a.cx() - b.cx()) <= kPxTol &&
           std::abs(a.cy() - b.cy()) <= kPxTol &&
           std::abs(a.width() - b.width()) <= kPxTol &&
           std::abs(a.height() - b.height()) <= kPxTol &&
           std::abs(a.rotation() - b.rotation()) <= kRotTol &&
           std::abs(a.score(0) - b.score(0)) <= kScoreTol;
  };
  size_t matched = 0;
  for (const OrientedDetection& s : smaller) {
    for (size_t j = 0; j < larger.size(); ++j) {
      if (!used[j] && matches(s, larger[j])) {
        used[j] = true;
        ++matched;
        break;
      }
    }
  }
  EXPECT_EQ(matched, smaller.size())
      << "only " << matched << "/" << smaller.size()
      << " detections matched 1:1 within tolerance between the two fronts";

  // (e) Sanity: a ship (DOTA class 1) detected on boats.jpg.
  EXPECT_TRUE(std::any_of(dets2.begin(), dets2.end(),
                          [](const OrientedDetection& d) {
                            return d.label_id(0) == 1;
                          }))
      << "expected a ship (class 1) on boats.jpg";
}

}  // namespace
}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
