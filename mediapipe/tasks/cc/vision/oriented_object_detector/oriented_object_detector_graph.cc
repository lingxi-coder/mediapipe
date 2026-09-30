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

#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/yolo_obb_tensors_to_oriented_detections_calculator.pb.h"
#include "mediapipe/calculators/util/oriented_detection_label_id_to_text_calculator.pb.h"
#include "mediapipe/calculators/util/rotated_non_max_suppression_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/components/processors/proto/tiled_detection_graph_options.pb.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/vision/oriented_object_detector/proto/oriented_object_detector_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/detection_label_resolution.h"
#include "mediapipe/tasks/cc/vision/utils/tiled_detection_utils.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace oriented_object_detector {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using OrientedObjectDetectorOptionsProto =
    proto::OrientedObjectDetectorOptions;
using TensorsSource =
    mediapipe::api2::builder::Source<std::vector<mediapipe::Tensor>>;

constexpr char kBatchInfoTag[] = "BATCH_INFO";
constexpr char kImageTag[] = "IMAGE";
constexpr char kImageCpuTag[] = "IMAGE_CPU";
constexpr char kMatrixTag[] = "MATRIX";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kOrientedDetectionsTag[] = "ORIENTED_DETECTIONS";
constexpr char kProjectionMatrixTag[] = "PROJECTION_MATRIX";
constexpr char kTensorTag[] = "TENSORS";

// Struct holding the different output streams produced by the oriented object
// detection subgraph.
struct OrientedObjectDetectionOutputStreams {
  Source<std::vector<OrientedDetection>> oriented_detections;
  Source<Image> image;
};

absl::Status SanityCheckOptions(
    const OrientedObjectDetectorOptionsProto& options) {
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
  if (options.tiling().explicit_tiles_size() > 0 &&
      options.tiling().tile_overlap_fraction() != 0.0f) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiling.tile_overlap_fraction is ignored with tiling.explicit_tiles; "
        "do not set both",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (options.tiling().tile_rows() < 0 || options.tiling().tile_cols() < 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiling.tile_rows and tiling.tile_cols must be >= 0.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (options.tiling().explicit_tiles_size() > 0 &&
      (options.tiling().tile_rows() > 1 || options.tiling().tile_cols() > 1)) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiling.explicit_tiles is mutually exclusive with a tiling.tile_rows / "
        "tiling.tile_cols grid (> 1).",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (options.tiling().tile_overlap_fraction() < 0.0f ||
      options.tiling().tile_overlap_fraction() >= 1.0f) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tiling.tile_overlap_fraction must be in [0.0, 1.0).",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

}  // namespace

// A "mediapipe.tasks.vision.oriented_object_detector.OrientedObjectDetectorGraph"
// performs OBB object detection.
// - Accepts CPU input images and outputs oriented detections on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform detection on.
//   NORM_RECT - NormalizedRect @Optional
//     Describes image rotation and region of image to perform detection on.
//     @Optional: rect covering the whole image is used if not specified.
//     NOTE: when tiling is enabled in the options (tile_rows * tile_cols > 1
//     or explicit_tiles non-empty), this input is NOT declared by the graph
//     and must not be connected.
//
// Outputs:
//   ORIENTED_DETECTIONS - std::vector<OrientedDetection>
//     Detected oriented bounding boxes in normalized original-image coordinates.
//   IMAGE - mediapipe::Image
//     The image that object detection runs on.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.oriented_object_detector.OrientedObjectDetectorGraph"
//   input_stream: "IMAGE:image_in"
//   output_stream: "ORIENTED_DETECTIONS:oriented_detections_out"
//   output_stream: "IMAGE:image_out"
//   options {
//     [mediapipe.tasks.vision.oriented_object_detector.proto.OrientedObjectDetectorOptions.ext]
//     {
//       base_options {
//         model_asset {
//           file_name: "/path/to/yolo_obb_model.tflite"
//         }
//       }
//       num_classes: 15
//       max_results: 100
//       score_threshold: 0.25
//       iou_threshold: 0.45
//     }
//   }
// }
class OrientedObjectDetectorGraph : public tasks::core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    ABSL_ASSIGN_OR_RETURN(
        const auto* model_resources,
        CreateModelResources<OrientedObjectDetectorOptionsProto>(sc));
    Graph graph;
    ABSL_ASSIGN_OR_RETURN(
        auto output_streams,
        BuildOrientedObjectDetectionTask(
            sc->Options<OrientedObjectDetectorOptionsProto>(), *model_resources,
            graph[Input<Image>(kImageTag)], graph));
    output_streams.oriented_detections >>
        graph[Output<std::vector<OrientedDetection>>(kOrientedDetectionsTag)];
    output_streams.image >> graph[Output<Image>(kImageTag)];
    return graph.GetConfig();
  }

 private:
  // Adds an oriented object detection task graph into the provided builder::Graph
  // instance. The detection task takes images (mediapipe::Image) as the input
  // and returns two output streams:
  //   - the oriented detection results (std::vector<OrientedDetection>),
  //   - the processed image that has pixel data stored on the target storage
  //     (mediapipe::Image).
  //
  // task_options: the mediapipe tasks OrientedObjectDetectorOptions proto.
  // model_resources: the ModelSources object initialized from an OBB model
  //                  file with model metadata.
  // image_in: (mediapipe::Image) stream to run object detection on.
  // graph: the mediapipe builder::Graph instance to be updated. The optional
  //        NORM_RECT graph input is only declared (accessed) on the
  //        single-image path; the tiled path never references it.
  absl::StatusOr<OrientedObjectDetectionOutputStreams>
  BuildOrientedObjectDetectionTask(
      const OrientedObjectDetectorOptionsProto& task_options,
      const tasks::core::ModelResources& model_resources,
      Source<Image> image_in,
      Graph& graph) {
    ABSL_RETURN_IF_ERROR(SanityCheckOptions(task_options));
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
          "Oriented object detection models require TFLite Model Metadata but "
          "none was found",
          MediaPipeTasksStatus::kMetadataNotFoundError);
    }

    // Resolve label items from model metadata (empty map if the model has no
    // label file). Used for both category-name mapping and for resolving
    // category_allowlist/category_denylist names to class indices.
    ABSL_ASSIGN_OR_RETURN(
        auto label_items,
        GetLabelItemsFromMetadata(model_resources,
                                  task_options.display_names_locale()));

    const auto& tiling = task_options.tiling();
    // Shared predicate (//mediapipe/tasks/cc/vision/utils:
    // tiled_detection_utils TilingEnabled), also used by the
    // oriented_object_detector.cc wrapper.
    const bool tiling_enabled = TilingEnabled(tiling);

    // Configures the OBB decode node (raw tensors -> batched oriented
    // detections) IDENTICALLY for both branches; the tiled branch additionally
    // sets the tile-local NMS options afterwards.
    auto configure_obb_decode =
        [&](mediapipe::api2::builder::GenericNode& obb_decode) -> absl::Status {
      auto& opts = obb_decode.GetOptions<
          ::mediapipe::YoloObbTensorsToOrientedDetectionsCalculatorOptions>();
      const int num_classes = task_options.num_classes();
      RET_CHECK_GT(num_classes, 0)
          << "num_classes must be set in OrientedObjectDetectorOptions";
      opts.set_num_classes(num_classes);
      opts.set_conf_threshold(task_options.score_threshold());
      opts.set_layout(
          task_options.layout() ==
                  OrientedObjectDetectorOptionsProto::CHANNELS_LAST
              ? ::mediapipe::YoloObbTensorsToOrientedDetectionsCalculatorOptions::
                    CHANNELS_LAST
              : ::mediapipe::YoloObbTensorsToOrientedDetectionsCalculatorOptions::
                    CHANNELS_FIRST);
      // Resolve category allow/deny names -> class indices and apply them as
      // decoder-level filters (before the score/NMS cap, matching upstream
      // semantics). Empty lists => no filtering; an all-unknown allowlist
      // resolves to an empty set (a no-op, not "drop all").
      ABSL_ASSIGN_OR_RETURN(
          auto allow_idx,
          ResolveCategoryIndices(label_items, task_options.category_allowlist(),
                                 task_options.category_denylist()));
      if (!task_options.category_allowlist().empty()) {
        for (int c : allow_idx) opts.add_allow_classes(c);
      } else {
        for (int c : allow_idx) opts.add_ignore_classes(c);
      }
      return absl::OkStatus();
    };

    // Filled by exactly one of the two branches below, then fed into the
    // shared label_id_to_text tail. Both are original-image-normalized.
    std::optional<Source<std::vector<OrientedDetection>>> detections_pre_label;
    std::optional<Source<Image>> image_out;

    if (!tiling_enabled) {
      // ======================= Single-image path (UNCHANGED wiring:
      // preprocessing -> inference -> decode -> flatten -> RotatedNMS ->
      // projection) =======================

      // Adds preprocessing calculators and connects them to the graph input
      // image stream.
      auto& preprocessing = graph.AddNode(
          "mediapipe.tasks.components.processors.ImagePreprocessingGraph");
      bool use_gpu =
          components::processors::DetermineImagePreprocessingGpuBackend(
              task_options.base_options().acceleration());
      ABSL_RETURN_IF_ERROR(
          components::processors::ConfigureImagePreprocessingGraph(
              model_resources, use_gpu, task_options.base_options().gpu_origin(),
              &preprocessing.GetOptions<tasks::components::processors::proto::
                                            ImagePreprocessingGraphOptions>()));
      image_in >> preprocessing.In(kImageTag);
      graph[Input<NormalizedRect>::Optional(kNormRectTag)] >>
          preprocessing.In(kNormRectTag);

      // Adds inference subgraph and connects its input stream to the output
      // tensors produced by the ImageToTensorCalculator.
      auto& inference = AddInference(
          model_resources, task_options.base_options().acceleration(), graph);
      preprocessing.Out(kTensorTag) >> inference.In(kTensorTag);
      TensorsSource model_output_tensors =
          inference.Out(kTensorTag).Cast<std::vector<Tensor>>();

      // OBB decode: raw tensors -> batched oriented detections.
      auto& obb_decode =
          graph.AddNode("YoloObbTensorsToOrientedDetectionsCalculator");
      ABSL_RETURN_IF_ERROR(configure_obb_decode(obb_decode));
      model_output_tensors >> obb_decode.In(kTensorTag);

      // Flatten batch (single-image Task: N==1) ->
      // std::vector<OrientedDetection>.
      auto& flatten = graph.AddNode("YoloObbBatchDetectionsToSingleCalculator");
      obb_decode.Out(kOrientedDetectionsTag) >>
          flatten.In(kOrientedDetectionsTag);

      // Rotated NMS in model-input-normalized space (Group-1 calc, UNCHANGED).
      auto& nms = graph.AddNode("RotatedNonMaxSuppressionCalculator");
      {
        auto& nms_opts =
            nms.GetOptions<
                ::mediapipe::RotatedNonMaxSuppressionCalculatorOptions>();
        nms_opts.set_iou_threshold(task_options.iou_threshold());
        nms_opts.set_max_detections(task_options.max_results());
        nms_opts.set_class_agnostic(task_options.class_agnostic_nms());
      }
      flatten.Out(kOrientedDetectionsTag) >> nms.In(kOrientedDetectionsTag);

      // Project to original-image-normalized coords (using preprocessing
      // matrix).
      auto& projection = graph.AddNode("OrientedDetectionProjectionCalculator");
      nms.Out(kOrientedDetectionsTag) >> projection.In(kOrientedDetectionsTag);
      preprocessing.Out(kMatrixTag) >> projection.In(kProjectionMatrixTag);

      detections_pre_label = projection.Out(kOrientedDetectionsTag)
                                 .Cast<std::vector<OrientedDetection>>();
      image_out = preprocessing[Output<Image>(kImageTag)];
    } else {
      // ======================= Tiled path: FromImage -> TiledDetectionFront
      // -> inference -> decode (+ tile-local NMS) -> TiledObbMerge
      // =======================

      // Model input dims [N,H,W,C]; validation (float32/4D/normalization)
      // lives in the shared //mediapipe/tasks/cc/vision/utils:
      // tiled_detection_utils ValidateTiledModelInputAndGetDims.
      ABSL_ASSIGN_OR_RETURN(const TiledModelInputDims dims,
                          ValidateTiledModelInputAndGetDims(model_resources));

      // mediapipe::Image -> ImageFrame (the tiled front consumes ImageFrame).
      auto& to_frame = graph.AddNode("FromImageCalculator");
      image_in >> to_frame.In(kImageTag);

      // Tile + batch front: IMAGE -> TENSORS (one packet per batch, synthetic
      // timestamps) + BATCH_INFO (per-batch tile geometry / source timestamp).
      auto& front =
          graph.AddNode("mediapipe.tiled_detection.TiledDetectionFrontGraph");
      auto& fo = front.GetOptions<::mediapipe::TiledDetectionFrontGraphOptions>();
      auto* tg = fo.mutable_tile_grid();
      tg->set_rows(tiling.tile_rows());
      tg->set_cols(tiling.tile_cols());
      if (tiling.explicit_tiles_size() > 0) {
        // TileGridCalculator rejects a SET overlap_fraction (presence check)
        // alongside explicit tiles, so only forward it in grid mode.
        for (const auto& e : tiling.explicit_tiles()) {
          auto* t = tg->add_explicit_tiles();
          t->set_x_center(e.x_center());
          t->set_y_center(e.y_center());
          t->set_width(e.width());
          t->set_height(e.height());
        }
      } else {
        tg->set_overlap_fraction(tiling.tile_overlap_fraction());
      }
      fo.set_batch_capacity(dims.batch);
      fo.set_is_dynamic_batch(dims.is_dynamic_batch);
      fo.set_input_height(dims.height);
      fo.set_input_width(dims.width);
      fo.set_input_channels(dims.channels);
      to_frame.Out(kImageCpuTag) >> front.In(kImageTag);

      auto& inference = AddInference(
          model_resources, task_options.base_options().acceleration(), graph);
      front.Out(kTensorTag) >> inference.In(kTensorTag);

      // OBB decode configured identically to the single-image branch PLUS the
      // tile-local (in-decoder, per batch row) rotated NMS options. The merge
      // graph consumes the BATCHED decode output directly (no flatten).
      auto& obb_decode =
          graph.AddNode("YoloObbTensorsToOrientedDetectionsCalculator");
      ABSL_RETURN_IF_ERROR(configure_obb_decode(obb_decode));
      {
        auto& opts = obb_decode.GetOptions<
            ::mediapipe::YoloObbTensorsToOrientedDetectionsCalculatorOptions>();
        opts.set_tile_local_nms_iou_threshold(
            tiling.tile_local_nms_iou_threshold());
        opts.set_max_detections_after_tile_nms(
            tiling.max_detections_after_tile_nms());
      }
      inference.Out(kTensorTag) >> obb_decode.In(kTensorTag);

      // Merge tile-local detections back to frame space + global rotated NMS;
      // emits one packet per source frame at the source frame timestamp.
      // When stream mode + tiling + BOTSORT is selected, route through the
      // track-merge subgraph (which additionally consumes the source IMAGE for
      // GMC); otherwise use the plain merge subgraph (default path, byte-
      // identical to before).
      const bool obb_tracking =
          task_options.base_options().use_stream_mode() &&
          TilingEnabled(tiling) &&
          task_options.tracking().tracker_type() ==
              proto::OrientedObjectDetectorOptions::TrackingOptions::BOTSORT;
      if (obb_tracking) {
        auto& merge = graph.AddNode(
            "mediapipe.tiled_detection.TiledObbTrackMergeGraph");
        auto& mo = merge.GetOptions<::mediapipe::TiledObbMergeGraphOptions>();
        mo.set_iou_threshold(task_options.iou_threshold());
        mo.set_class_agnostic(task_options.class_agnostic_nms());
        mo.set_max_detections(task_options.max_results());
        auto* mt = mo.mutable_tracking();
        // Pin the OBB-proto <-> TiledTrackingGraph TrackerType values so an enum
        // reorder is a build error, not a silent tracker miswire.
        static_assert(static_cast<int>(OrientedObjectDetectorOptionsProto::TrackingOptions::TRACKER_UNSPECIFIED) ==
                          static_cast<int>(::mediapipe::TiledTrackingGraphOptions::TRACKER_UNSPECIFIED));
        static_assert(static_cast<int>(OrientedObjectDetectorOptionsProto::TrackingOptions::BOX_TRACKER) ==
                          static_cast<int>(::mediapipe::TiledTrackingGraphOptions::BOX_TRACKER));
        static_assert(static_cast<int>(OrientedObjectDetectorOptionsProto::TrackingOptions::BOTSORT) ==
                          static_cast<int>(::mediapipe::TiledTrackingGraphOptions::BOTSORT));
        mt->set_tracker_type(
            static_cast<::mediapipe::TiledTrackingGraphOptions::TrackerType>(
                task_options.tracking().tracker_type()));
        mt->set_track_high_threshold(
            task_options.tracking().track_high_threshold());
        mt->set_track_low_threshold(
            task_options.tracking().track_low_threshold());
        mt->set_new_track_threshold(
            task_options.tracking().new_track_threshold());
        mt->set_track_buffer(task_options.tracking().track_buffer());
        mt->set_match_threshold(task_options.tracking().match_threshold());
        mt->set_enable_gmc(task_options.tracking().enable_gmc());
        mt->set_nominal_frame_rate(
            task_options.tracking().nominal_frame_rate());
        obb_decode.Out(kOrientedDetectionsTag) >>
            merge.In(kOrientedDetectionsTag);
        front.Out(kBatchInfoTag) >> merge.In(kBatchInfoTag);
        to_frame.Out(kImageCpuTag) >> merge.In(kImageTag);
        detections_pre_label = merge.Out(kOrientedDetectionsTag)
                                   .Cast<std::vector<OrientedDetection>>();
      } else {
        auto& merge =
            graph.AddNode("mediapipe.tiled_detection.TiledObbMergeGraph");
        auto& mo = merge.GetOptions<::mediapipe::TiledObbMergeGraphOptions>();
        mo.set_iou_threshold(task_options.iou_threshold());
        mo.set_class_agnostic(task_options.class_agnostic_nms());
        mo.set_max_detections(task_options.max_results());
        obb_decode.Out(kOrientedDetectionsTag) >>
            merge.In(kOrientedDetectionsTag);
        front.Out(kBatchInfoTag) >> merge.In(kBatchInfoTag);
        detections_pre_label = merge.Out(kOrientedDetectionsTag)
                                   .Cast<std::vector<OrientedDetection>>();
      }

      // The tiled path has no preprocessing node to forward the input image,
      // so pass it through explicitly as the IMAGE output.
      auto& pass = graph.AddNode("PassThroughCalculator");
      image_in >> pass.In("");
      image_out = pass.Out("").Cast<Image>();
    }

    // ======================= Shared tail =======================
    // Map integer class ids -> category-name strings from the model metadata.
    // keep_label_id=true preserves label_id so Category.index survives
    // (ConvertToOrientedObjectDetectionResult would otherwise emit index = -1).
    // With an empty label_items map this is a safe pass-through: no labels are
    // added and label_id is untouched, so a model without metadata labels
    // behaves exactly as before. Label mapping is geometry-independent, so
    // running it after projection/merge is equivalent.
    auto& label_id_to_text =
        graph.AddNode("OrientedDetectionLabelIdToTextCalculator");
    {
      auto& label_opts = label_id_to_text.GetOptions<
          ::mediapipe::OrientedDetectionLabelIdToTextCalculatorOptions>();
      label_opts.set_keep_label_id(true);
      *label_opts.mutable_label_items() = label_items;
    }
    *detections_pre_label >> label_id_to_text.In("");

    // Outputs the oriented detections and the processed image as the subgraph
    // output streams.
    return {{
        /* oriented_detections= */
        label_id_to_text.Out("").Cast<std::vector<OrientedDetection>>(),
        /* image= */ *image_out,
    }};
  }
};

// NOTE: keep the fully-qualified type name on a single line. The
// REGISTER_MEDIAPIPE_GRAPH macro stringifies its argument with `#name`, so a
// line break here would inject a stray space into the registered name (e.g.
// "oriented_object_detector:: OrientedObjectDetectorGraph") and the graph would
// never be found by lookup.
// clang-format off
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tasks::vision::oriented_object_detector::OrientedObjectDetectorGraph);  // NOLINT(whitespace/line_length)
// clang-format on

}  // namespace oriented_object_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
