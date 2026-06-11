/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector.h"

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/detection_result.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.pb.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace yolo_object_detector {
namespace {

constexpr char kDetectionsOutStreamName[] = "detections_out";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectName[] = "norm_rect_in";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kTaskName[] = "YoloObjectDetector";
constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.vision.yolo_object_detector.YoloObjectDetectorGraph";
constexpr int kMicroSecondsPerMilliSecond = 1000;

using ::mediapipe::NormalizedRect;
using ::mediapipe::tasks::components::containers::ConvertToDetectionResult;
using ::mediapipe::tasks::vision::core::GetCoreRunningMode;
using YoloObjectDetectorOptionsProto =
    proto::YoloObjectDetectorOptions;

// Returns true when the options enable the tiled path (grid larger than one
// tile, or explicit tiles provided).
// Must match yolo_object_detector_graph.cc's tiling-enabled predicate.
bool TilingEnabled(const YoloObjectDetectorOptionsProto& options_proto) {
  const auto& tiling = options_proto.tiling();
  return tiling.tile_rows() * tiling.tile_cols() > 1 ||
         tiling.explicit_tiles_size() > 0;
}

// In tiled mode the task graph has no NORM_RECT input, so per-call
// ImageProcessingOptions cannot be honored: ROI is mutually exclusive with
// tiling, and rotation is unsupported (it would be silently ignored
// otherwise).
absl::Status CheckTiledImageProcessingOptions(
    const std::optional<core::ImageProcessingOptions>&
        image_processing_options) {
  if (!image_processing_options.has_value()) {
    return absl::OkStatus();
  }
  if (image_processing_options->region_of_interest.has_value()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiling and ROI are mutually exclusive",
        MediaPipeTasksStatus::kImageProcessingInvalidArgumentError);
  }
  if (image_processing_options->rotation_degrees != 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiling does not support rotation_degrees",
        MediaPipeTasksStatus::kImageProcessingInvalidArgumentError);
  }
  return absl::OkStatus();
}

// Creates a MediaPipe graph config that contains a subgraph node of
// "mediapipe.tasks.vision.yolo_object_detector.YoloObjectDetectorGraph". If
// the task is running in the live stream mode, a "FlowLimiterCalculator" will
// be added to limit the number of frames in flight.
// When `tiling_enabled` is true, the NORM_RECT graph input is neither declared
// nor connected (the tiled task subgraph has no such input); only IMAGE is
// fed. The outputs are identical in both modes.
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<YoloObjectDetectorOptionsProto> options_proto,
    bool enable_flow_limiting, bool tiling_enabled) {
  api2::builder::Graph graph;
  graph.In(kImageTag).SetName(kImageInStreamName);
  if (!tiling_enabled) {
    graph.In(kNormRectTag).SetName(kNormRectName);
  }
  auto& task_subgraph = graph.AddNode(kSubgraphTypeName);
  task_subgraph.GetOptions<YoloObjectDetectorOptionsProto>().Swap(
      options_proto.get());
  task_subgraph.Out(kDetectionsTag).SetName(kDetectionsOutStreamName) >>
      graph.Out(kDetectionsTag);
  task_subgraph.Out(kImageTag).SetName(kImageOutStreamName) >>
      graph.Out(kImageTag);
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(
        graph, task_subgraph,
        tiling_enabled ? std::vector<std::string>{kImageTag}
                       : std::vector<std::string>{kImageTag, kNormRectTag},
        kDetectionsTag);
  }
  graph.In(kImageTag) >> task_subgraph.In(kImageTag);
  if (!tiling_enabled) {
    graph.In(kNormRectTag) >> task_subgraph.In(kNormRectTag);
  }
  return graph.GetConfig();
}

}  // namespace

// Converts the user-facing YoloObjectDetectorOptions struct to the internal
// YoloObjectDetectorOptions proto.  Defined outside the anonymous namespace
// so that unit tests can call it directly to verify options→proto mapping.
std::unique_ptr<YoloObjectDetectorOptionsProto>
ConvertYoloObjectDetectorOptionsToProto(YoloObjectDetectorOptions* options) {
  auto options_proto = std::make_unique<YoloObjectDetectorOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode != core::RunningMode::IMAGE);
  options_proto->set_display_names_locale(options->display_names_locale);
  options_proto->set_max_results(options->max_results);
  options_proto->set_score_threshold(options->score_threshold);
  for (const std::string& category : options->category_allowlist) {
    options_proto->add_category_allowlist(category);
  }
  for (const std::string& category : options->category_denylist) {
    options_proto->add_category_denylist(category);
  }
  options_proto->set_iou_threshold(options->iou_threshold);
  options_proto->set_layout(
      static_cast<YoloObjectDetectorOptionsProto::Layout>(options->layout));
  options_proto->set_num_classes(options->num_classes);
  auto* tiling = options_proto->mutable_tiling();
  tiling->set_tile_rows(options->tiling.tile_rows);
  tiling->set_tile_cols(options->tiling.tile_cols);
  tiling->set_tile_overlap_fraction(options->tiling.tile_overlap_fraction);
  for (const auto& e : options->tiling.explicit_tiles) {
    auto* t = tiling->add_explicit_tiles();
    t->set_x_center(e.x_center);
    t->set_y_center(e.y_center);
    t->set_width(e.width);
    t->set_height(e.height);
  }
  tiling->set_tile_local_nms_iou_threshold(
      options->tiling.tile_local_nms_iou_threshold);
  tiling->set_max_detections_after_tile_nms(
      options->tiling.max_detections_after_tile_nms);
  return options_proto;
}

absl::StatusOr<std::unique_ptr<YoloObjectDetector>> YoloObjectDetector::Create(
    std::unique_ptr<YoloObjectDetectorOptions> options) {
  auto options_proto =
      ConvertYoloObjectDetectorOptionsToProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  if (options->result_callback) {
    auto result_callback = options->result_callback;
    packets_callback =
        [=](absl::StatusOr<tasks::core::PacketMap> status_or_packets) {
          if (!status_or_packets.ok()) {
            Image image;
            result_callback(status_or_packets.status(), image,
                            Timestamp::Unset().Value());
            return;
          }
          if (status_or_packets.value()[kImageOutStreamName].IsEmpty()) {
            return;
          }
          Packet image_packet = status_or_packets.value()[kImageOutStreamName];
          Packet detections_packet =
              status_or_packets.value()[kDetectionsOutStreamName];
          if (detections_packet.IsEmpty()) {
            Packet empty_packet =
                status_or_packets.value()[kDetectionsOutStreamName];
            result_callback(
                {ConvertToDetectionResult({})}, image_packet.Get<Image>(),
                empty_packet.Timestamp().Value() / kMicroSecondsPerMilliSecond);
            return;
          }
          result_callback(ConvertToDetectionResult(
                              detections_packet.Get<std::vector<Detection>>()),
                          image_packet.Get<Image>(),
                          detections_packet.Timestamp().Value() /
                              kMicroSecondsPerMilliSecond);
        };
  }
  const bool tiling_enabled = TilingEnabled(*options_proto);
  auto detector =
      core::VisionTaskApiFactory::Create<YoloObjectDetector,
                                         YoloObjectDetectorOptionsProto>(
          {.config = CreateGraphConfig(
               std::move(options_proto),
               options->running_mode == core::RunningMode::LIVE_STREAM,
               tiling_enabled),
           .task_name = kTaskName,
           .task_running_mode = GetCoreRunningMode(options->running_mode),
           .op_resolver = std::move(options->base_options.op_resolver),
           .packets_callback = std::move(packets_callback),
           .disable_default_service =
               options->base_options.disable_default_service,
           .host_environment = options->base_options.host_environment,
           .host_system = options->base_options.host_system,
           .host_version = options->base_options.host_version});
  if (detector.ok()) {
    (*detector)->tiling_enabled_ = tiling_enabled;
  }
  return detector;
}

absl::StatusOr<YoloObjectDetectorResult> YoloObjectDetector::Detect(
    mediapipe::Image image,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  tasks::core::PacketMap input_packets;
  if (tiling_enabled_) {
    MP_RETURN_IF_ERROR(
        CheckTiledImageProcessingOptions(image_processing_options));
    input_packets = {{kImageInStreamName, MakePacket<Image>(std::move(image))}};
  } else {
    MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                        ConvertToNormalizedRect(image_processing_options, image,
                                                /*roi_allowed=*/false));
    input_packets = {
        {kImageInStreamName, MakePacket<Image>(std::move(image))},
        {kNormRectName, MakePacket<NormalizedRect>(std::move(norm_rect))}};
  }
  MP_ASSIGN_OR_RETURN(auto output_packets,
                      ProcessImageData(std::move(input_packets)));
  if (output_packets[kDetectionsOutStreamName].IsEmpty()) {
    return {ConvertToDetectionResult({})};
  }
  return ConvertToDetectionResult(
      output_packets[kDetectionsOutStreamName].Get<std::vector<Detection>>());
}

absl::StatusOr<YoloObjectDetectorResult> YoloObjectDetector::DetectForVideo(
    mediapipe::Image image, int64_t timestamp_ms,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  tasks::core::PacketMap input_packets;
  if (tiling_enabled_) {
    MP_RETURN_IF_ERROR(
        CheckTiledImageProcessingOptions(image_processing_options));
    input_packets = {
        {kImageInStreamName,
         MakePacket<Image>(std::move(image))
             .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}};
  } else {
    MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                        ConvertToNormalizedRect(image_processing_options, image,
                                                /*roi_allowed=*/false));
    input_packets = {
        {kImageInStreamName,
         MakePacket<Image>(std::move(image))
             .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))},
        {kNormRectName,
         MakePacket<NormalizedRect>(std::move(norm_rect))
             .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}};
  }
  MP_ASSIGN_OR_RETURN(auto output_packets,
                      ProcessVideoData(std::move(input_packets)));
  if (output_packets[kDetectionsOutStreamName].IsEmpty()) {
    return {ConvertToDetectionResult({})};
  }
  return ConvertToDetectionResult(
      output_packets[kDetectionsOutStreamName].Get<std::vector<Detection>>());
}

absl::Status YoloObjectDetector::DetectAsync(
    Image image, int64_t timestamp_ms,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  if (tiling_enabled_) {
    MP_RETURN_IF_ERROR(
        CheckTiledImageProcessingOptions(image_processing_options));
    return SendLiveStreamData(
        {{kImageInStreamName,
          MakePacket<Image>(std::move(image))
              .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}});
  }
  MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                      ConvertToNormalizedRect(image_processing_options, image,
                                              /*roi_allowed=*/false));
  return SendLiveStreamData(
      {{kImageInStreamName,
        MakePacket<Image>(std::move(image))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))},
       {kNormRectName,
        MakePacket<NormalizedRect>(std::move(norm_rect))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}});
}

}  // namespace yolo_object_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
