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

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.pb.h"
#include "mediapipe/calculators/util/detection_label_id_to_text_calculator.pb.h"
#include "mediapipe/calculators/util/non_max_suppression_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/vision/utils/detection_label_resolution.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.pb.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace yolo_object_detector {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using YoloObjectDetectorOptionsProto =
    proto::YoloObjectDetectorOptions;
using TensorsSource =
    mediapipe::api2::builder::Source<std::vector<mediapipe::Tensor>>;

constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kImageTag[] = "IMAGE";
constexpr char kMatrixTag[] = "MATRIX";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kPixelDetectionsTag[] = "PIXEL_DETECTIONS";
constexpr char kProjectionMatrixTag[] = "PROJECTION_MATRIX";
constexpr char kTensorTag[] = "TENSORS";

// Struct holding the different output streams produced by the YOLO object
// detection subgraph.
struct YoloObjectDetectionOutputStreams {
  Source<std::vector<Detection>> detections;
  Source<Image> image;
};

absl::Status SanityCheckOptions(
    const YoloObjectDetectorOptionsProto& options) {
  if (options.max_results() == 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Invalid `max_results` option: value must be != 0",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (options.category_allowlist_size() > 0 &&
      options.category_denylist_size() > 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "`category_allowlist` and `category_denylist` are mutually "
        "exclusive options.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

}  // namespace

// A "mediapipe.tasks.vision.yolo_object_detector.YoloObjectDetectorGraph"
// performs YOLO object detection.
// - Accepts CPU input images and outputs detections on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform detection on.
//   NORM_RECT - NormalizedRect @Optional
//     Describes image rotation and region of image to perform detection on.
//     @Optional: rect covering the whole image is used if not specified.
//
// Outputs:
//   DETECTIONS - std::vector<Detection>
//     Detected objects with bounding box in pixel units.
//   IMAGE - mediapipe::Image
//     The image that object detection runs on.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.yolo_object_detector.YoloObjectDetectorGraph"
//   input_stream: "IMAGE:image_in"
//   output_stream: "DETECTIONS:detections_out"
//   output_stream: "IMAGE:image_out"
//   options {
//     [mediapipe.tasks.vision.yolo_object_detector.proto.YoloObjectDetectorOptions.ext]
//     {
//       base_options {
//         model_asset {
//           file_name: "/path/to/yolo_model.tflite"
//         }
//       }
//       num_classes: 80
//       max_results: 100
//       score_threshold: 0.25
//       iou_threshold: 0.45
//     }
//   }
// }
class YoloObjectDetectorGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    MP_ASSIGN_OR_RETURN(
        const auto* model_resources,
        CreateModelResources<YoloObjectDetectorOptionsProto>(sc));
    Graph graph;
    MP_ASSIGN_OR_RETURN(
        auto output_streams,
        BuildYoloObjectDetectionTask(
            sc->Options<YoloObjectDetectorOptionsProto>(), *model_resources,
            graph[Input<Image>(kImageTag)],
            graph[Input<NormalizedRect>::Optional(kNormRectTag)], graph));
    output_streams.detections >>
        graph[Output<std::vector<Detection>>(kDetectionsTag)];
    output_streams.image >> graph[Output<Image>(kImageTag)];
    return graph.GetConfig();
  }

 private:
  // Adds a YOLO object detection task graph into the provided builder::Graph
  // instance. The detection task takes images (mediapipe::Image) as the input
  // and returns two output streams:
  //   - the detection results (std::vector<Detection>),
  //   - the processed image that has pixel data stored on the target storage
  //     (mediapipe::Image).
  //
  // task_options: the mediapipe tasks YoloObjectDetectorOptions proto.
  // model_resources: the ModelSources object initialized from a YOLO model
  //                  file with model metadata.
  // image_in: (mediapipe::Image) stream to run object detection on.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<YoloObjectDetectionOutputStreams> BuildYoloObjectDetectionTask(
      const YoloObjectDetectorOptionsProto& task_options,
      const core::ModelResources& model_resources, Source<Image> image_in,
      Source<NormalizedRect> norm_rect_in, Graph& graph) {
    MP_RETURN_IF_ERROR(SanityCheckOptions(task_options));
    auto& model = *model_resources.GetTfLiteModel();
    if (model.subgraphs()->size() != 1) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Expected a model with a single subgraph, found %d.",
                          model.subgraphs()->size()),
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    // Checks that metadata is available.
    auto* metadata_extractor = model_resources.GetMetadataExtractor();
    if (metadata_extractor->GetModelMetadata() == nullptr ||
        metadata_extractor->GetModelMetadata()->subgraph_metadata() ==
            nullptr) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "YOLO object detection models require TFLite Model Metadata but "
          "none was found",
          MediaPipeTasksStatus::kMetadataNotFoundError);
    }

    // Resolve label items from model metadata (empty map if the model has no
    // label file). Used for both category-name mapping and for resolving
    // category_allowlist/category_denylist names to class indices.
    MP_ASSIGN_OR_RETURN(
        auto label_items,
        GetLabelItemsFromMetadata(model_resources,
                                  task_options.display_names_locale()));

    // Adds preprocessing calculators and connects them to the graph input image
    // stream.
    auto& preprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.ImagePreprocessingGraph");
    bool use_gpu =
        components::processors::DetermineImagePreprocessingGpuBackend(
            task_options.base_options().acceleration());
    MP_RETURN_IF_ERROR(components::processors::ConfigureImagePreprocessingGraph(
        model_resources, use_gpu, task_options.base_options().gpu_origin(),
        &preprocessing.GetOptions<tasks::components::processors::proto::
                                      ImagePreprocessingGraphOptions>()));
    image_in >> preprocessing.In(kImageTag);
    norm_rect_in >> preprocessing.In(kNormRectTag);

    // Adds inference subgraph and connects its input stream to the output
    // tensors produced by the ImageToTensorCalculator.
    auto& inference = AddInference(
        model_resources, task_options.base_options().acceleration(), graph);
    preprocessing.Out(kTensorTag) >> inference.In(kTensorTag);
    TensorsSource model_output_tensors =
        inference.Out(kTensorTag).Cast<std::vector<Tensor>>();

    // YOLO decode: raw tensors -> batched axis-aligned Detections.
    auto& yolo_decode = graph.AddNode("YoloTensorsToDetectionsCalculator");
    {
      auto& opts = yolo_decode.GetOptions<
          ::mediapipe::YoloTensorsToDetectionsCalculatorOptions>();
      int num_classes = task_options.num_classes();
      RET_CHECK_GT(num_classes, 0)
          << "num_classes must be set in YoloObjectDetectorOptions "
             "(metadata-derived num_classes is a future enhancement)";
      opts.set_num_classes(num_classes);
      opts.set_conf_threshold(task_options.score_threshold());
      opts.set_layout(
          task_options.layout() ==
                  YoloObjectDetectorOptionsProto::CHANNELS_LAST
              ? ::mediapipe::YoloTensorsToDetectionsCalculatorOptions::
                    CHANNELS_LAST
              : ::mediapipe::YoloTensorsToDetectionsCalculatorOptions::
                    CHANNELS_FIRST);
      // Resolve category allow/deny names -> class indices and apply them as
      // decoder-level filters (applied before the score/NMS cap, matching
      // upstream semantics). Empty lists => no filtering; an all-unknown
      // allowlist resolves to an empty set (a no-op, not "drop all").
      MP_ASSIGN_OR_RETURN(
          auto allow_idx,
          ResolveCategoryIndices(label_items, task_options.category_allowlist(),
                                 task_options.category_denylist()));
      if (!task_options.category_allowlist().empty()) {
        for (int c : allow_idx) opts.add_allow_classes(c);
      } else {
        for (int c : allow_idx) opts.add_ignore_classes(c);
      }
    }
    model_output_tensors >> yolo_decode.In(kTensorTag);

    // Flatten batch (single-image Task: N==1) -> std::vector<Detection>.
    auto& batch_to_single =
        graph.AddNode("YoloBatchDetectionsToSingleCalculator");
    yolo_decode.Out(kDetectionsTag) >> batch_to_single.In(kDetectionsTag);

    // Axis-aligned NMS.
    auto& nms = graph.AddNode("NonMaxSuppressionCalculator");
    {
      auto& nms_opts =
          nms.GetOptions<::mediapipe::NonMaxSuppressionCalculatorOptions>();
      nms_opts.set_min_suppression_threshold(task_options.iou_threshold());
      nms_opts.set_max_num_detections(task_options.max_results());
      nms_opts.set_overlap_type(
          ::mediapipe::NonMaxSuppressionCalculatorOptions::
              INTERSECTION_OVER_UNION);
      nms_opts.set_return_empty_detections(true);
    }
    batch_to_single.Out(kDetectionsTag) >> nms.In("");

    // Map integer class ids -> category-name strings from the model metadata.
    // keep_label_id=true preserves label_id so Category.index survives
    // (ConvertToDetectionResult would otherwise emit index = -1). With an empty
    // label_items map this is a safe pass-through: no labels are added and
    // label_id is left untouched, so a model without metadata labels behaves
    // exactly as before.
    auto& label_id_to_text =
        graph.AddNode("DetectionLabelIdToTextCalculator");
    {
      auto& label_opts = label_id_to_text.GetOptions<
          ::mediapipe::DetectionLabelIdToTextCalculatorOptions>();
      label_opts.set_keep_label_id(true);
      *label_opts.mutable_label_items() = label_items;
    }
    nms.Out("") >> label_id_to_text.In("");
    auto detections = label_id_to_text.Out("");

    // Calculator to project detections back to the original coordinate system.
    auto& detection_projection = graph.AddNode("DetectionProjectionCalculator");
    detections >> detection_projection.In(kDetectionsTag);
    preprocessing.Out(kMatrixTag) >>
        detection_projection.In(kProjectionMatrixTag);

    // Calculator to convert relative detection bounding boxes to pixel
    // detection bounding boxes.
    auto& detection_transformation =
        graph.AddNode("DetectionTransformationCalculator");
    detection_projection.Out(kDetectionsTag) >>
        detection_transformation.In(kDetectionsTag);
    preprocessing.Out(kImageSizeTag) >>
        detection_transformation.In(kImageSizeTag);
    auto detections_in_pixel =
        detection_transformation.Out(kPixelDetectionsTag);

    // Deduplicate Detections with same bounding box coordinates.
    auto& detections_deduplicate =
        graph.AddNode("DetectionsDeduplicateCalculator");
    detections_in_pixel >> detections_deduplicate.In("");

    // Outputs the labeled detections and the processed image as the subgraph
    // output streams.
    return {{
        /* detections= */
        detections_deduplicate[Output<std::vector<Detection>>("")],
        /* image= */ preprocessing[Output<Image>(kImageTag)],
    }};
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::yolo_object_detector::YoloObjectDetectorGraph);

}  // namespace yolo_object_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
