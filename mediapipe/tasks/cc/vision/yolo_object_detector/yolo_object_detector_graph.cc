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
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/graphs/tiled_detection/tiled_detection_graphs.pb.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/vision/utils/detection_label_resolution.h"
#include "mediapipe/tasks/cc/vision/utils/tiled_detection_utils.h"
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

constexpr char kBatchInfoTag[] = "BATCH_INFO";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kImageCpuTag[] = "IMAGE_CPU";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kImageTag[] = "IMAGE";
constexpr char kMatrixTag[] = "MATRIX";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kPixelDetectionsTag[] = "PIXEL_DETECTIONS";
constexpr char kProjectionMatrixTag[] = "PROJECTION_MATRIX";
constexpr char kSizeTag[] = "SIZE";
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
//     NOTE: when tiling is enabled in the options (tile_rows * tile_cols > 1
//     or explicit_tiles non-empty), this input is NOT declared by the graph
//     and must not be connected.
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
class YoloObjectDetectorGraph : public tasks::core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    ABSL_ASSIGN_OR_RETURN(
        const auto* model_resources,
        CreateModelResources<YoloObjectDetectorOptionsProto>(sc));
    Graph graph;
    ABSL_ASSIGN_OR_RETURN(
        auto output_streams,
        BuildYoloObjectDetectionTask(
            sc->Options<YoloObjectDetectorOptionsProto>(), *model_resources,
            graph[Input<Image>(kImageTag)], graph));
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
  // graph: the mediapipe builder::Graph instance to be updated. The optional
  //        NORM_RECT graph input is only declared (accessed) on the
  //        single-image path; the tiled path never references it.
  absl::StatusOr<YoloObjectDetectionOutputStreams> BuildYoloObjectDetectionTask(
      const YoloObjectDetectorOptionsProto& task_options,
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
          "YOLO object detection models require TFLite Model Metadata but "
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
    // yolo_object_detector.cc wrapper.
    const bool tiling_enabled = TilingEnabled(tiling);

    // Configures the YOLO decode node (raw tensors -> batched axis-aligned
    // Detections) IDENTICALLY for both branches; the tiled branch additionally
    // sets the tile-local NMS options afterwards.
    auto configure_yolo_decode =
        [&](mediapipe::api2::builder::GenericNode& yolo_decode)
        -> absl::Status {
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

    // Configures the label-mapping node identically for both branches.
    // Map integer class ids -> category-name strings from the model metadata.
    // keep_label_id=true preserves label_id so Category.index survives
    // (ConvertToDetectionResult would otherwise emit index = -1). With an empty
    // label_items map this is a safe pass-through: no labels are added and
    // label_id is left untouched, so a model without metadata labels behaves
    // exactly as before.
    auto configure_label_id_to_text =
        [&](mediapipe::api2::builder::GenericNode& label_id_to_text) {
          auto& label_opts = label_id_to_text.GetOptions<
              ::mediapipe::DetectionLabelIdToTextCalculatorOptions>();
          label_opts.set_keep_label_id(true);
          *label_opts.mutable_label_items() = label_items;
        };

    // Filled by exactly one of the two branches below. Both branches emit
    // labeled, deduplicated detections with PIXEL bounding boxes.
    std::optional<Source<std::vector<Detection>>> detections_out;
    std::optional<Source<Image>> image_out;

    if (!tiling_enabled) {
      // ======================= Single-image path (UNCHANGED wiring:
      // preprocessing -> inference -> decode -> flatten -> NMS -> label ->
      // projection -> transformation -> dedup) =======================

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

      // YOLO decode: raw tensors -> batched axis-aligned Detections.
      auto& yolo_decode = graph.AddNode("YoloTensorsToDetectionsCalculator");
      ABSL_RETURN_IF_ERROR(configure_yolo_decode(yolo_decode));
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

      auto& label_id_to_text =
          graph.AddNode("DetectionLabelIdToTextCalculator");
      configure_label_id_to_text(label_id_to_text);
      nms.Out("") >> label_id_to_text.In("");
      auto detections = label_id_to_text.Out("");

      // Calculator to project detections back to the original coordinate
      // system.
      auto& detection_projection =
          graph.AddNode("DetectionProjectionCalculator");
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

      detections_out =
          detections_deduplicate.Out("").Cast<std::vector<Detection>>();
      image_out = preprocessing[Output<Image>(kImageTag)];
    } else {
      // ======================= Tiled path: FromImage -> TiledDetectionFront
      // -> inference -> decode (+ tile-local NMS) -> TiledBoxMerge -> label ->
      // pixel transformation -> dedup =======================

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
      // Stream mode + opted-in motion scheduling -> the scheduler-bearing
      // stream front (it runs its own optical-flow pass and consumes a
      // PRIOR_DETECTIONS loopback); otherwise the plain front. Both read the
      // same TiledDetectionFrontGraphOptions.
      const bool scheduling_enabled =
          task_options.base_options().use_stream_mode() &&
          ::mediapipe::tasks::vision::SchedulingEnabled(tiling);
      auto& front = graph.AddNode(
          scheduling_enabled
              ? "mediapipe.tiled_detection.TiledDetectionStreamFrontGraph"
              : "mediapipe.tiled_detection.TiledDetectionFrontGraph");
      auto& fo =
          front.GetOptions<::mediapipe::TiledDetectionFrontGraphOptions>();
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

      // The scheduler needs the previous frame's merged detections as
      // PRIOR_DETECTIONS; PreviousLoopbackCalculator emits an empty packet on
      // frame 0 (-> first frame DETECTs). The LOOP back edge is closed after
      // merged_dets is produced, below.
      ::mediapipe::api2::builder::GenericNode* scheduler_loopback = nullptr;
      if (scheduling_enabled) {
        fo.set_max_scheduled_tiles(tiling.max_scheduled_tiles());
        auto& lb = graph.AddNode("PreviousLoopbackCalculator");
        image_in >> lb.In("MAIN");
        lb.Out("PREV_LOOP").Cast<std::vector<Detection>>() >>
            front.In("PRIOR_DETECTIONS");
        scheduler_loopback = &lb;
      }

      auto& inference = AddInference(
          model_resources, task_options.base_options().acceleration(), graph);
      front.Out(kTensorTag) >> inference.In(kTensorTag);

      // YOLO decode configured identically to the single-image branch PLUS the
      // tile-local (in-decoder, per batch row) NMS options. The merge graph
      // consumes the BATCHED decode output directly (no flatten).
      auto& yolo_decode = graph.AddNode("YoloTensorsToDetectionsCalculator");
      ABSL_RETURN_IF_ERROR(configure_yolo_decode(yolo_decode));
      {
        auto& opts = yolo_decode.GetOptions<
            ::mediapipe::YoloTensorsToDetectionsCalculatorOptions>();
        opts.set_tile_local_nms_iou_threshold(
            tiling.tile_local_nms_iou_threshold());
        opts.set_max_detections_after_tile_nms(
            tiling.max_detections_after_tile_nms());
      }
      inference.Out(kTensorTag) >> yolo_decode.In(kTensorTag);

      // Merge tile-local detections back to frame space + global NMS; emits
      // one packet per source frame at the source frame timestamp. The
      // single-image path's NonMaxSuppressionCalculator (multiclass_nms left
      // default false) suppresses jointly across ALL classes, so mirror that
      // semantic with class_agnostic=true (the task options expose no
      // class_agnostic knob).
      //
      // In stream mode (VIDEO / LIVE_STREAM) fuse the merged tile detections
      // with optical-flow tracker-propagated boxes via the TRACKER_DETECTIONS
      // seam; in IMAGE mode keep the stateless merge. Both emit one packet per
      // source frame on DETECTIONS.
      std::optional<Source<std::vector<Detection>>> merged_dets;
      if (task_options.base_options().use_stream_mode()) {
        auto& merge = graph.AddNode(
            "mediapipe.tiled_detection.TiledBoxTrackMergeGraph");
        auto& mo = merge.GetOptions<::mediapipe::TiledBoxMergeGraphOptions>();
        mo.set_iou_threshold(task_options.iou_threshold());
        mo.set_class_agnostic(true);
        mo.set_max_detections(task_options.max_results());
        // Forward the user's tracker selection into the tracking subgraph.
        // Both enums share numeric values (UNSPECIFIED=0/BOX_TRACKER=1/
        // BOTSORT=2), so the static_cast across enum types is valid.
        auto* mtracking = mo.mutable_tracking();
        // Pin the YOLO-proto <-> TiledTrackingGraph TrackerType values.
        static_assert(static_cast<int>(YoloObjectDetectorOptionsProto::TrackingOptions::TRACKER_UNSPECIFIED) ==
                          static_cast<int>(::mediapipe::TiledTrackingGraphOptions::TRACKER_UNSPECIFIED));
        static_assert(static_cast<int>(YoloObjectDetectorOptionsProto::TrackingOptions::BOX_TRACKER) ==
                          static_cast<int>(::mediapipe::TiledTrackingGraphOptions::BOX_TRACKER));
        static_assert(static_cast<int>(YoloObjectDetectorOptionsProto::TrackingOptions::BOTSORT) ==
                          static_cast<int>(::mediapipe::TiledTrackingGraphOptions::BOTSORT));
        mtracking->set_tracker_type(
            static_cast<::mediapipe::TiledTrackingGraphOptions::TrackerType>(
                task_options.tracking().tracker_type()));
        mtracking->set_track_high_threshold(
            task_options.tracking().track_high_threshold());
        mtracking->set_track_low_threshold(
            task_options.tracking().track_low_threshold());
        mtracking->set_new_track_threshold(
            task_options.tracking().new_track_threshold());
        mtracking->set_track_buffer(task_options.tracking().track_buffer());
        mtracking->set_match_threshold(
            task_options.tracking().match_threshold());
        mtracking->set_enable_gmc(task_options.tracking().enable_gmc());
        yolo_decode.Out(kDetectionsTag) >> merge.In(kDetectionsTag);
        front.Out(kBatchInfoTag) >> merge.In(kBatchInfoTag);
        // The tracker needs the source video frame; reuse the ImageFrame the
        // tiled front already consumes (to_frame's IMAGE_CPU output).
        to_frame.Out(kImageCpuTag) >> merge.In(kImageTag);
        merged_dets = merge.Out(kDetectionsTag).Cast<std::vector<Detection>>();
      } else {
        auto& merge =
            graph.AddNode("mediapipe.tiled_detection.TiledBoxMergeGraph");
        auto& mo = merge.GetOptions<::mediapipe::TiledBoxMergeGraphOptions>();
        mo.set_iou_threshold(task_options.iou_threshold());
        mo.set_class_agnostic(true);
        mo.set_max_detections(task_options.max_results());
        yolo_decode.Out(kDetectionsTag) >> merge.In(kDetectionsTag);
        front.Out(kBatchInfoTag) >> merge.In(kBatchInfoTag);
        merged_dets = merge.Out(kDetectionsTag).Cast<std::vector<Detection>>();
      }

      // Merge output is frame-normalized RELATIVE_BOUNDING_BOX (already in
      // original-image space — no projection needed): label mapping, then
      // pixel-unit transformation, then dedup, mirroring the single path's
      // public output contract.
      // Close the scheduler's PRIOR_DETECTIONS loopback with this frame's
      // merged (frame-normalized, post-fusion) detections.
      if (scheduler_loopback != nullptr) {
        *merged_dets >> scheduler_loopback->In("LOOP").AsBackEdge();
      }

      auto& label_id_to_text =
          graph.AddNode("DetectionLabelIdToTextCalculator");
      configure_label_id_to_text(label_id_to_text);
      *merged_dets >> label_id_to_text.In("");

      // The tiled path has no preprocessing node to provide IMAGE_SIZE, so
      // derive it from the ORIGINAL input image.
      auto& image_properties = graph.AddNode("ImagePropertiesCalculator");
      image_in >> image_properties.In(kImageTag);

      // Calculator to convert relative detection bounding boxes to pixel
      // detection bounding boxes.
      auto& detection_transformation =
          graph.AddNode("DetectionTransformationCalculator");
      label_id_to_text.Out("") >> detection_transformation.In(kDetectionsTag);
      image_properties.Out(kSizeTag) >>
          detection_transformation.In(kImageSizeTag);
      auto detections_in_pixel =
          detection_transformation.Out(kPixelDetectionsTag);

      // Deduplicate Detections with same bounding box coordinates.
      auto& detections_deduplicate =
          graph.AddNode("DetectionsDeduplicateCalculator");
      detections_in_pixel >> detections_deduplicate.In("");

      detections_out =
          detections_deduplicate.Out("").Cast<std::vector<Detection>>();

      // The tiled path has no preprocessing node to forward the input image,
      // so pass it through explicitly as the IMAGE output.
      auto& pass = graph.AddNode("PassThroughCalculator");
      image_in >> pass.In("");
      image_out = pass.Out("").Cast<Image>();
    }

    // Outputs the labeled detections and the processed image as the subgraph
    // output streams.
    return {{
        /* detections= */ *detections_out,
        /* image= */ *image_out,
    }};
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::yolo_object_detector::YoloObjectDetectorGraph);

}  // namespace yolo_object_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
