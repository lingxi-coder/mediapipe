# OBB BoTSORT Tracking (C++ core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional BoTSORT track_id assignment to the OBB tiled-livestream path — ID-association on axis-aligned bounding boxes (AABBs), output geometry always the fresh rotated detections; OBB default behavior unchanged.

**Architecture:** A self-contained `OrientedBotsortTrackingCalculator` (oriented→AABB→BoTSORT→id-match-back) feeds a new `TiledObbTrackMergeGraph` (rotated-NMS merge + the calculator); the OBB graph branches to it when stream+tiling+BOTSORT; a BoTSORT-only `TrackingOptions` + Create validation gate it; the `OrientedObjectDetection` result gains a `track_id` read from the proto's existing field.

**Tech Stack:** C++20, Bazel, api2 calculators/subgraphs, the vendored motion-only BoTSORT (`//third_party/botsort`), googletest. Build/test only via `bazel ... --define MEDIAPIPE_DISABLE_GPU=1` (authoritative; IGNORE clangd noise). No Python in this sub-project.

**Reference spec:** `docs/superpowers/specs/2026-06-16-obb-botsort-tracking-design.md`

**Standing constraints (every task):** stay on branch `dev`; never branch/merge; all comments English; TDD; commit after each task; commit-message trailer MUST be exactly:
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `mediapipe/calculators/tensor/oriented_botsort_tracking_calculator.cc` (+ test) | oriented→AABB→BoTSORT→id-back | 1 |
| `mediapipe/graphs/tiled_detection/tiled_detection_graphs.proto` | `tracking` field on `TiledObbMergeGraphOptions` | 2 |
| `mediapipe/graphs/tiled_detection/tiled_obb_track_merge_graph.cc` (+ test) | stream merge subgraph | 2 |
| `mediapipe/tasks/cc/vision/oriented_object_detector/proto/oriented_object_detector_options.proto` | `TrackingOptions` + field | 3 |
| `oriented_object_detector.h` / `.cc` | public struct + converter | 3 |
| `oriented_object_detector_graph.cc` | stream branch | 4 |
| `oriented_object_detector.cc` (Create) (+ test) | BOTSORT validation | 5 |
| `mediapipe/tasks/cc/components/containers/oriented_object_detection_result.{h,cc}` (+ test) | result `track_id` | 6 |
| `oriented_object_detector_test.cc` | e2e + regression | 7 |

---

## Task 1: `OrientedBotsortTrackingCalculator`

**Files:**
- Create: `mediapipe/calculators/tensor/oriented_botsort_tracking_calculator.cc`
- Create: `mediapipe/calculators/tensor/oriented_botsort_tracking_calculator_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

Reuses the existing `BotsortTrackingCalculatorOptions` proto (`:botsort_tracking_calculator_cc_proto`) — no new options message. Read `botsort_tracking_calculator.cc` first for the BoTSORT `Open`/conversion idiom to mirror. The oriented type is `mediapipe::OrientedDetection` (`mediapipe/framework/formats/oriented_detection.pb.h`), with normalized `cx,cy,width,height,rotation` + `label_id`/`score` + `track_id`.

- [ ] **Step 1: Write the failing unit test**

Create `oriented_botsort_tracking_calculator_test.cc` (Apache header). Use `CalculatorRunner` (mirror `botsort_tracking_calculator_test.cc`). Helper to build an `OrientedDetection` (set cx,cy,width,height,rotation,label_id,score). Node config:
```
calculator: "OrientedBotsortTrackingCalculator"
input_stream: "IMAGE:image"
input_stream: "ORIENTED_DETECTIONS:dets"
output_stream: "ORIENTED_DETECTIONS:tracked"
node_options { [type.googleapis.com/mediapipe.BotsortTrackingCalculatorOptions] {
  track_high_threshold: 0.05 track_low_threshold: 0.02 new_track_threshold: 0.05
} }
```
Tests:
```cpp
TEST(OrientedBotsortTrackingCalculatorTest, SetsStableTrackIdAndPreservesRotation) {
  // 3 frames: same oriented box (rotation 0.5 rad) drifting cx by 0.01/frame,
  // label 1, score 0.9. After Run: 3 output packets; the last frame's first
  // detection has a non-empty track_id; rotation == 0.5 (unchanged); the same
  // track_id appears on the last two output frames (stable).
}

TEST(OrientedBotsortTrackingCalculatorTest, EmptyDetectionsProducesEmptyPacket) {
  // 1 frame, empty ORIENTED_DETECTIONS vector -> exactly 1 output packet, empty.
}
```
Push a 200x200 SRGB `ImageFrame` per frame (like the BoTSORT calc test). Read the output `std::vector<OrientedDetection>`; assert `last[0].rotation()` == input, `last[0].has_track_id()`, and last-two-frame id stability (collect ids per frame, intersect).

Add the `cc_test` target to `mediapipe/calculators/tensor/BUILD` (deps: `:oriented_botsort_tracking_calculator`, calculator_framework, calculator_runner, `//mediapipe/framework/formats:oriented_detection_cc_proto`, image_frame, `//mediapipe/framework/port:gtest_main`, parse_text_proto, status_matchers).

- [ ] **Step 2: Run to confirm RED**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:oriented_botsort_tracking_calculator_test --test_output=errors
```
Expected: FAIL to build (calculator not registered yet).

- [ ] **Step 3: Write the calculator**

Create `oriented_botsort_tracking_calculator.cc` (Apache header):

```cpp
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "BoTSORT.h"
#include "DataType.h"
#include "GmcParams.h"
#include "ReIDParams.h"
#include "TrackerParams.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/botsort_tracking_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "opencv2/core.hpp"

namespace mediapipe {
namespace api2 {

namespace {
// Axis-aligned bounding box (pixel tlwh) enclosing a rotated rect given in
// normalized space (cx,cy,width,height in [0,1], rotation in radians CCW).
cv::Rect_<float> OrientedToAabbTlwh(const OrientedDetection& d, float img_w,
                                    float img_h) {
  const float hw = 0.5f * d.width();
  const float hh = 0.5f * d.height();
  const float c = std::abs(std::cos(d.rotation()));
  const float s = std::abs(std::sin(d.rotation()));
  const float ax = hw * c + hh * s;  // normalized half-extent x
  const float ay = hw * s + hh * c;  // normalized half-extent y
  return cv::Rect_<float>((d.cx() - ax) * img_w, (d.cy() - ay) * img_h,
                          2.0f * ax * img_w, 2.0f * ay * img_h);
}

float AabbIoU(const cv::Rect_<float>& a, const cv::Rect_<float>& b) {
  const float inter = (a & b).area();
  const float uni = a.area() + b.area() - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}
}  // namespace

// Assigns stable BoTSORT track ids to oriented detections via ID-association on
// their axis-aligned bounding boxes. Output geometry is always the fresh rotated
// detection; rotation is never tracked; there is no gap-fill (a Kalman/AABB box
// has no angle). track_id is the proto string form of BoTSORT's int track id.
//
// NOTE: BoTSORT only emits a track_id for CONFIRMED tracks; the confirmation
// thresholds must be at/below the detector's score_threshold or no id is set.
//
// Inputs:  IMAGE (ImageFrame), ORIENTED_DETECTIONS (vector<OrientedDetection>).
// Output:  ORIENTED_DETECTIONS (same fresh detections, track_id set on matches).
class OrientedBotsortTrackingCalculator : public Node {
 public:
  static constexpr Input<ImageFrame> kImage{"IMAGE"};
  static constexpr Input<std::vector<OrientedDetection>> kDets{
      "ORIENTED_DETECTIONS"};
  static constexpr Output<std::vector<OrientedDetection>> kOut{
      "ORIENTED_DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kImage, kDets, kOut);

  absl::Status Open(CalculatorContext* cc) override {
    const auto& opts = cc->Options<BotsortTrackingCalculatorOptions>();
    TrackerParams params;
    params.track_high_thresh = opts.track_high_threshold();
    params.track_low_thresh = opts.track_low_threshold();
    params.new_track_thresh = opts.new_track_threshold();
    params.track_buffer = opts.track_buffer();
    params.match_thresh = opts.match_threshold();
    params.gmc_enabled = opts.enable_gmc();
    params.reid_enabled = false;
    Config<GMC_Params> gmc_config = std::monostate{};
    if (opts.enable_gmc()) {
      GMC_Params gmc;
      gmc.method_ = GMC_Method::SparseOptFlow;
      gmc.method_params_ = SparseOptFlow_Params{};
      gmc_config = gmc;
    }
    tracker_ = std::make_unique<BoTSORT>(
        Config<TrackerParams>(params), gmc_config,
        Config<ReIDParams>(std::monostate{}), "");
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const ImageFrame& image = *kImage(cc);
    cv::Mat frame = formats::MatView(&image);
    const float w = static_cast<float>(image.Width());
    const float h = static_cast<float>(image.Height());

    std::vector<OrientedDetection> out;
    std::vector<cv::Rect_<float>> aabbs;  // parallel to `out`
    if (kDets(cc).IsConnected() && !kDets(cc).IsEmpty()) {
      out = *kDets(cc);
      aabbs.reserve(out.size());
      for (const OrientedDetection& d : out) {
        aabbs.push_back(OrientedToAabbTlwh(d, w, h));
      }
    }

    std::vector<::Detection> bs_dets;
    bs_dets.reserve(out.size());
    for (size_t i = 0; i < out.size(); ++i) {
      ::Detection bd;
      bd.bbox_tlwh = aabbs[i];
      bd.class_id = out[i].label_id_size() > 0 ? out[i].label_id(0) : 0;
      bd.confidence = out[i].score_size() > 0 ? out[i].score(0) : 0.0f;
      bs_dets.push_back(bd);
    }

    std::vector<std::shared_ptr<Track>> tracks = tracker_->track(bs_dets, frame);
    for (const std::shared_ptr<Track>& t : tracks) {
      const std::vector<float> tlwh = t->get_tlwh();  // pixel
      const cv::Rect_<float> track_box(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);
      int best_idx = -1;
      float best_iou = 0.1f;  // minimum to accept a match
      for (size_t i = 0; i < out.size(); ++i) {
        if (out[i].has_track_id()) continue;  // one id per detection
        const float iou = AabbIoU(track_box, aabbs[i]);
        if (iou >= best_iou) {
          best_iou = iou;
          best_idx = static_cast<int>(i);
        }
      }
      if (best_idx >= 0) {
        out[best_idx].set_track_id(std::to_string(t->track_id));
      }
    }
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<BoTSORT> tracker_;
};

MEDIAPIPE_REGISTER_NODE(OrientedBotsortTrackingCalculator);

}  // namespace api2
}  // namespace mediapipe
```

Add to `mediapipe/calculators/tensor/BUILD` (mirror the `botsort_tracking_calculator` cc_library):
```python
cc_library(
    name = "oriented_botsort_tracking_calculator",
    srcs = ["oriented_botsort_tracking_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":botsort_tracking_calculator_cc_proto",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:oriented_detection_cc_proto",
        "//third_party:opencv",
        "//third_party/botsort",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)
```
(Confirm the `oriented_detection_cc_proto` target label by grepping how other targets depend on `oriented_detection.proto`; adjust if it differs.)

- [ ] **Step 4: Run to confirm GREEN**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:oriented_botsort_tracking_calculator_test --test_output=errors
```
Expected: PASS (2 cases). If the track isn't confirmed within 3 frames at these low thresholds, add a 4th frame (do not weaken the id/stability assertion).

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/oriented_botsort_tracking_calculator.cc \
        mediapipe/calculators/tensor/oriented_botsort_tracking_calculator_test.cc \
        mediapipe/calculators/tensor/BUILD
git commit -m "feat(obb-tracking): OrientedBotsortTrackingCalculator

Assigns stable BoTSORT track ids to oriented detections via AABB ID-association
(oriented->AABB->BoTSORT->best-IoU id-match-back). Output geometry is the fresh
rotated detection; rotation untracked; no gap-fill. Reuses
BotsortTrackingCalculatorOptions.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `TiledObbTrackMergeGraph` + options

**Files:**
- Modify: `mediapipe/graphs/tiled_detection/tiled_detection_graphs.proto`
- Create: `mediapipe/graphs/tiled_detection/tiled_obb_track_merge_graph.cc`
- Modify: `mediapipe/graphs/tiled_detection/tiled_tracking_graphs_test.cc` (or the obb merge test file)
- Modify: `mediapipe/graphs/tiled_detection/BUILD`

- [ ] **Step 1: Add the `tracking` field to `TiledObbMergeGraphOptions`**

In `tiled_detection_graphs.proto`, add to the existing `TiledObbMergeGraphOptions` message (after `class_agnostic = 3`), reusing the existing `TiledTrackingGraphOptions` (already defined for YOLO):
```proto
  // Tracker selection forwarded to the OBB stream tracking calculator. Honored
  // only by TiledObbTrackMergeGraph (stream mode). UNSPECIFIED/BOX_TRACKER are
  // ignored here (OBB supports BOTSORT only; the detector Create() validates).
  optional TiledTrackingGraphOptions tracking = 4;
```
(Field 4 is free in that message.)

- [ ] **Step 2: Write the subgraph**

Create `tiled_obb_track_merge_graph.cc` (Apache header), mirroring `tiled_obb_merge_graph.cc` + adding the tracker:

```cpp
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/botsort_tracking_calculator.pb.h"
#include "mediapipe/calculators/tensor/rotated_non_max_suppression_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/graphs/tiled_detection/tiled_detection_graphs.pb.h"

namespace mediapipe {
namespace tiled_detection {

// Stream-mode sibling of TiledObbMergeGraph: merges per-batch tile-local
// oriented detections into frame space + global rotated NMS, then assigns
// BoTSORT track ids via OrientedBotsortTrackingCalculator (ID-only; geometry
// unchanged). One packet per source frame.
//
// Inputs:  ORIENTED_DETECTIONS (vector<vector<OrientedDetection>>),
//          BATCH_INFO (TensorBatchInfo), IMAGE (ImageFrame).
// Outputs: ORIENTED_DETECTIONS (vector<OrientedDetection>, track_id set).
class TiledObbTrackMergeGraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    const auto& options = sc->Options<TiledObbMergeGraphOptions>();
    api2::builder::Graph graph;

    auto& merge = graph.AddNode("MergeTileDetectionsAccumulatorCalculator");
    graph.In("ORIENTED_DETECTIONS") >> merge.In("ORIENTED_DETECTIONS");
    graph.In("BATCH_INFO") >> merge.In("BATCH_INFO");

    auto& nms = graph.AddNode("RotatedNonMaxSuppressionCalculator");
    auto& no = nms.GetOptions<RotatedNonMaxSuppressionCalculatorOptions>();
    no.set_iou_threshold(options.iou_threshold());
    no.set_max_detections(options.max_detections());
    no.set_class_agnostic(options.class_agnostic());
    merge.Out("ORIENTED_DETECTIONS") >> nms.In("ORIENTED_DETECTIONS");

    auto& track = graph.AddNode("OrientedBotsortTrackingCalculator");
    auto& to = track.GetOptions<BotsortTrackingCalculatorOptions>();
    const auto& tr = options.tracking();
    to.set_track_high_threshold(tr.track_high_threshold());
    to.set_track_low_threshold(tr.track_low_threshold());
    to.set_new_track_threshold(tr.new_track_threshold());
    to.set_track_buffer(tr.track_buffer());
    to.set_match_threshold(tr.match_threshold());
    to.set_enable_gmc(tr.enable_gmc());
    graph.In("IMAGE") >> track.In("IMAGE");
    nms.Out("ORIENTED_DETECTIONS") >> track.In("ORIENTED_DETECTIONS");

    track.Out("ORIENTED_DETECTIONS") >> graph.Out("ORIENTED_DETECTIONS");
    return graph.GetConfig();
  }
};

// clang-format off
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tiled_detection::TiledObbTrackMergeGraph);  // NOLINT(whitespace/line_length)
// clang-format on

}  // namespace tiled_detection
}  // namespace mediapipe
```
(Confirm the `rotated_non_max_suppression_calculator.pb.h` include path + the `RotatedNonMaxSuppressionCalculatorOptions` name by checking `tiled_obb_merge_graph.cc`'s includes — match exactly. Keep the `REGISTER_MEDIAPIPE_GRAPH` name on ONE line.)

- [ ] **Step 3: BUILD — add the subgraph to the tiled_detection graph library + deps**

In `mediapipe/graphs/tiled_detection/BUILD`, add `tiled_obb_track_merge_graph.cc` to the srcs of the cc_library that registers these subgraphs (the same lib as `tiled_obb_merge_graph.cc`), and add deps `//mediapipe/calculators/tensor:oriented_botsort_tracking_calculator` and `//mediapipe/calculators/tensor:botsort_tracking_calculator_cc_proto`. READ the BUILD to find the exact target.

- [ ] **Step 4: Write a subgraph test**

In the obb merge graph test file (find it: `ls mediapipe/graphs/tiled_detection/*obb*test* ` or it may be in `tiled_detection_graphs_test.cc`), add a test that builds a graph with a single `mediapipe.tiled_detection.TiledObbTrackMergeGraph` node, `node_options { [type.googleapis.com/mediapipe.TiledObbMergeGraphOptions] { iou_threshold: 0.5 class_agnostic: true tracking { tracker_type: BOTSORT track_high_threshold: 0.05 new_track_threshold: 0.05 } } }`, feeds 2 frames of (IMAGE 200x200 + ORIENTED_DETECTIONS as vector<vector<OrientedDetection>> single-tile + a matching BATCH_INFO), and asserts 2 output ORIENTED_DETECTIONS packets and that the output carries a track_id by the last frame. Match the existing obb merge test's harness for building BATCH_INFO / the vector<vector> input. Add the test target deps for `oriented_botsort_tracking_calculator` + `oriented_detection_cc_proto`.

- [ ] **Step 5: Run + build**

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/graphs/tiled_detection/...
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/graphs/tiled_detection:tiled_detection_graphs_test --test_output=errors
```
Expected: builds; the new subgraph test passes; pre-existing tests unchanged. (Use the actual test target name for the file you edited.)

- [ ] **Step 6: Commit**

```bash
git add mediapipe/graphs/tiled_detection/
git commit -m "feat(obb-tracking): TiledObbTrackMergeGraph stream subgraph

Stream sibling of TiledObbMergeGraph: rotated-NMS merge then
OrientedBotsortTrackingCalculator for BoTSORT track ids (ID-only, no fusion).
Adds a tracking field (reusing TiledTrackingGraphOptions) to
TiledObbMergeGraphOptions.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: OBB `TrackingOptions` (proto + struct + converter)

**Files:**
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/proto/oriented_object_detector_options.proto`
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h`
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.cc`
- Test: `oriented_object_detector_test.cc`

- [ ] **Step 1: Write the failing converter test**

In `oriented_object_detector_test.cc`, add (confirm/define the proto alias the file uses for the generated options proto — mirror the existing tiling converter test if present):
```cpp
TEST(OrientedObjectDetectorOptionsConverterTest, MapsTrackingOptions) {
  auto options = std::make_unique<OrientedObjectDetectorOptions>();
  options->tracking.tracker_type =
      OrientedObjectDetectorOptions::TrackingOptions::kBotsort;
  options->tracking.track_high_threshold = 0.55f;
  options->tracking.match_threshold = 0.75f;
  options->tracking.enable_gmc = true;
  auto proto = ConvertOrientedObjectDetectorOptionsToProto(options.get());
  EXPECT_EQ(proto->tracking().tracker_type(),
            OrientedObjectDetectorOptionsProto::TrackingOptions::BOTSORT);
  EXPECT_FLOAT_EQ(proto->tracking().track_high_threshold(), 0.55f);
  EXPECT_FLOAT_EQ(proto->tracking().match_threshold(), 0.75f);
  EXPECT_TRUE(proto->tracking().enable_gmc());
}
```
Run → FAIL (no `tracking` member).

- [ ] **Step 2: Add the proto message + field**

In `oriented_object_detector_options.proto`, add (after `TilingOptions` + `tiling = 11`):
```proto
  // Tracker selection for the tiled VIDEO/LIVE_STREAM path. OBB supports only
  // BOTSORT (axis-aligned ID-association; oriented geometry preserved); default
  // is no tracking. Honored only when tiling is enabled and running mode is not
  // IMAGE (validated at Create()).
  // NOTE: BoTSORT only emits a track_id for CONFIRMED tracks, so
  // track_high_threshold / new_track_threshold must be at or below the detector's
  // score_threshold, otherwise low-confidence detections never confirm and no
  // track_id is produced.
  message TrackingOptions {
    enum TrackerType {
      TRACKER_UNSPECIFIED = 0;  // no tracking (stateless per-frame; default)
      BOX_TRACKER = 1;          // NOT supported for OBB (rejected at Create)
      BOTSORT = 2;              // tracking-by-detection, motion-only
    }
    optional TrackerType tracker_type = 1 [default = TRACKER_UNSPECIFIED];
    optional float track_high_threshold = 2 [default = 0.6];
    optional float track_low_threshold = 3 [default = 0.1];
    optional float new_track_threshold = 4 [default = 0.7];
    optional int32 track_buffer = 5 [default = 30];
    optional float match_threshold = 6 [default = 0.7];
    optional bool enable_gmc = 7 [default = false];
  }
  optional TrackingOptions tracking = 12;
```

- [ ] **Step 3: Add the public C++ struct**

In `oriented_object_detector.h`, after the `TilingOptions` struct + `tiling` member, add (mirror the YOLO struct; values pinned to the proto enum):
```cpp
  // Tracker selection for the tiled VIDEO/LIVE_STREAM path. OBB supports only
  // BOTSORT; default is no tracking. Honored only when tiling is enabled and
  // running mode is not IMAGE; validated at Create(). NOTE: BoTSORT only emits a
  // track_id for confirmed tracks, so set track_high_threshold /
  // new_track_threshold at or below your score_threshold.
  struct TrackingOptions {
    // Values must stay numerically equal to the proto enum (UNSPECIFIED=0,
    // BOX_TRACKER=1, BOTSORT=2) -- the converter static_casts between them.
    enum TrackerType {
      kTrackerUnspecified = 0,  // no tracking (default)
      kBoxTracker = 1,          // not supported for OBB
      kBotsort = 2,             // tracking-by-detection, motion-only
    };
    TrackerType tracker_type = kTrackerUnspecified;
    float track_high_threshold = 0.6f;
    float track_low_threshold = 0.1f;
    float new_track_threshold = 0.7f;
    int track_buffer = 30;
    float match_threshold = 0.7f;
    bool enable_gmc = false;
  };
  TrackingOptions tracking;
```

- [ ] **Step 4: Map it in the converter + static_assert the enum**

In `oriented_object_detector.cc` `ConvertOrientedObjectDetectorOptionsToProto`, after the tiling mapping and before `return`, add (use the file's proto alias):
```cpp
  static_assert(static_cast<int>(OrientedObjectDetectorOptions::TrackingOptions::kBotsort) ==
                    static_cast<int>(OrientedObjectDetectorOptionsProto::TrackingOptions::BOTSORT));
  static_assert(static_cast<int>(OrientedObjectDetectorOptions::TrackingOptions::kBoxTracker) ==
                    static_cast<int>(OrientedObjectDetectorOptionsProto::TrackingOptions::BOX_TRACKER));
  auto* tracking = options_proto->mutable_tracking();
  tracking->set_tracker_type(
      static_cast<OrientedObjectDetectorOptionsProto::TrackingOptions::TrackerType>(
          options->tracking.tracker_type));
  tracking->set_track_high_threshold(options->tracking.track_high_threshold);
  tracking->set_track_low_threshold(options->tracking.track_low_threshold);
  tracking->set_new_track_threshold(options->tracking.new_track_threshold);
  tracking->set_track_buffer(options->tracking.track_buffer);
  tracking->set_match_threshold(options->tracking.match_threshold);
  tracking->set_enable_gmc(options->tracking.enable_gmc);
```

- [ ] **Step 5: Run the converter test (GREEN)**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test --test_output=errors --test_filter='*MapsTrackingOptions*'
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mediapipe/tasks/cc/vision/oriented_object_detector/
git commit -m "feat(obb-tracking): OBB TrackingOptions proto + struct + converter

BoTSORT-only TrackingOptions (default no tracking) mirroring YOLO for binding
parity; converter maps it with static_assert-pinned enum agreement.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: OBB graph stream branch

**Files:**
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_graph.cc`

- [ ] **Step 1: Branch to the track-merge subgraph**

In the tiled path (around line 408 where `TiledObbMergeGraph` is added), replace the single merge node with a branch. `task_options.tracking().tracker_type()` selects BOTSORT; `to_frame.Out(kImageCpuTag)` is the source ImageFrame:
```cpp
      const bool obb_tracking =
          task_options.base_options().use_stream_mode() &&
          TilingEnabled(tiling) &&
          task_options.tracking().tracker_type() ==
              OrientedObjectDetectorOptionsProto::TrackingOptions::BOTSORT;
      if (obb_tracking) {
        auto& merge = graph.AddNode(
            "mediapipe.tiled_detection.TiledObbTrackMergeGraph");
        auto& mo = merge.GetOptions<::mediapipe::TiledObbMergeGraphOptions>();
        mo.set_iou_threshold(task_options.iou_threshold());
        mo.set_class_agnostic(task_options.class_agnostic_nms());
        mo.set_max_detections(task_options.max_results());
        auto* mt = mo.mutable_tracking();
        mt->set_tracker_type(static_cast<::mediapipe::TiledTrackingGraphOptions::
                                 TrackerType>(
            task_options.tracking().tracker_type()));
        mt->set_track_high_threshold(task_options.tracking().track_high_threshold());
        mt->set_track_low_threshold(task_options.tracking().track_low_threshold());
        mt->set_new_track_threshold(task_options.tracking().new_track_threshold());
        mt->set_track_buffer(task_options.tracking().track_buffer());
        mt->set_match_threshold(task_options.tracking().match_threshold());
        mt->set_enable_gmc(task_options.tracking().enable_gmc());
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
```
This replaces the existing merge block (lines ~406-415). The `else` branch is byte-identical to today. Add the graph cc_library deps `//mediapipe/graphs/tiled_detection:tiled_obb_track_merge_graph` (the registered subgraph lib) and `//mediapipe/graphs/tiled_detection:tiled_detection_graphs_cc_proto` if not already present — READ the OBB graph BUILD and add the subgraph-registration dep so `TiledObbTrackMergeGraph` links. Confirm `kImageTag`/`kImageCpuTag`/`OrientedObjectDetectorOptionsProto` are in scope.

- [ ] **Step 2: Build the graph**

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_graph
```
Expected: builds. (Behavioral verification is in Task 7's e2e; default path unchanged.)

- [ ] **Step 3: Commit**

```bash
git add mediapipe/tasks/cc/vision/oriented_object_detector/
git commit -m "feat(obb-tracking): OBB graph stream branch to TiledObbTrackMergeGraph

When stream_mode + tiling + BOTSORT, route the tiled OBB path through
TiledObbTrackMergeGraph (IMAGE wired from to_frame); otherwise the existing
TiledObbMergeGraph (default path byte-identical).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Create-time validation

**Files:**
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.cc`
- Test: `oriented_object_detector_test.cc`

- [ ] **Step 1: Write failing model-free validation tests**

In `oriented_object_detector_test.cc`, mirror the YOLO validation tests' idiom (build options, call Create with no model, assert kInvalidArgument + a distinctive message substring). Add:
```cpp
TEST(OrientedTrackingValidationTest, BotsortInImageModeRejected) {
  auto o = std::make_unique<OrientedObjectDetectorOptions>();
  o->running_mode = core::RunningMode::IMAGE;
  o->num_classes = 15; o->tiling.tile_rows = 2; o->tiling.tile_cols = 2;
  o->tracking.tracker_type = OrientedObjectDetectorOptions::TrackingOptions::kBotsort;
  auto r = OrientedObjectDetector::Create(std::move(o));
  EXPECT_EQ(r.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(r.status().message()), testing::HasSubstr("IMAGE"));
}
TEST(OrientedTrackingValidationTest, BotsortWithoutTilingRejected) {
  auto o = std::make_unique<OrientedObjectDetectorOptions>();
  o->running_mode = core::RunningMode::VIDEO;
  o->num_classes = 15;  // tiling default 1x1 (disabled)
  o->tracking.tracker_type = OrientedObjectDetectorOptions::TrackingOptions::kBotsort;
  auto r = OrientedObjectDetector::Create(std::move(o));
  EXPECT_EQ(r.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(r.status().message()), testing::HasSubstr("tiling"));
}
TEST(OrientedTrackingValidationTest, BoxTrackerRejected) {
  auto o = std::make_unique<OrientedObjectDetectorOptions>();
  o->running_mode = core::RunningMode::VIDEO;
  o->num_classes = 15; o->tiling.tile_rows = 2; o->tiling.tile_cols = 2;
  o->tracking.tracker_type = OrientedObjectDetectorOptions::TrackingOptions::kBoxTracker;
  auto r = OrientedObjectDetector::Create(std::move(o));
  EXPECT_EQ(r.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(r.status().message()), testing::HasSubstr("BoxTracker"));
}
```
Confirm the OBB stream tests' result_callback requirement (LIVE_STREAM needs one; VIDEO may not). Match how the file's existing tests construct stream-mode options. `<gmock>`/`HasSubstr` should already be available (mirror YOLO).

- [ ] **Step 2: Add the validation block in Create()**

In `oriented_object_detector.cc` `Create()`, before the `VisionTaskApiFactory::Create` call, add (mirror YOLO; `tiling_enabled` should be computed as in YOLO — `TilingEnabled(options_proto->tiling())`):
```cpp
  const auto tracker = options_proto->tracking().tracker_type();
  if (tracker == OrientedObjectDetectorOptionsProto::TrackingOptions::BOX_TRACKER) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "tracking.tracker_type=BOX_TRACKER: BoxTracker is not supported for "
        "oriented detection; use BOTSORT.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (tracker == OrientedObjectDetectorOptionsProto::TrackingOptions::BOTSORT) {
    if (options->running_mode == core::RunningMode::IMAGE) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "tracking.tracker_type=BOTSORT requires VIDEO or LIVE_STREAM running "
          "mode; tracking is not available in IMAGE mode.",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    if (!TilingEnabled(options_proto->tiling())) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "tracking.tracker_type=BOTSORT requires tiling to be enabled; the "
          "non-tiled path has no tracker stage.",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    if (options_proto->tiling().enable_motion_scheduling()) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "tracking.tracker_type=BOTSORT is incompatible with "
          "tiling.enable_motion_scheduling.",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    if (options_proto->num_classes() < 1 || options_proto->num_classes() > 256) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "tracking.tracker_type=BOTSORT requires num_classes in [1, 256] "
          "(BoTSORT stores the class id as uint8).",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
  }
```
NOTE: the OBB `TilingOptions` proto has no `enable_motion_scheduling` field (OBB never had scheduling). If that field does not exist on the OBB tiling proto, DROP the motion_scheduling check (confirm by reading the OBB options proto's TilingOptions). Keep the other three gates. Confirm `TilingEnabled`, `CreateStatusWithPayload`, `MediaPipeTasksStatus`, `core::RunningMode` are in scope (the graph builder uses `TilingEnabled`; the wrapper may need the include).

- [ ] **Step 3: Run the validation tests (GREEN) + full suite**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test --test_output=errors --test_filter='*TrackingValidation*'
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test --test_output=errors
```
Expected: 3 validation tests pass; full suite green.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/cc/vision/oriented_object_detector/
git commit -m "feat(obb-tracking): reject invalid BOTSORT selections at Create()

BOTSORT requires VIDEO/LIVE_STREAM + tiling + num_classes in [1,256]; BOX_TRACKER
is rejected for OBB. Each a distinct InvalidArgument. Default (no tracking)
unaffected.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Surface `track_id` on `OrientedObjectDetection`

**Files:**
- Modify: `mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h`
- Modify: `mediapipe/tasks/cc/components/containers/oriented_object_detection_result.cc`
- Test: `mediapipe/tasks/cc/components/containers/oriented_object_detection_result_test.cc`

- [ ] **Step 1: Write the failing conversion test**

In `oriented_object_detection_result_test.cc` (exists — has the OBB result tests), add:
```cpp
TEST(OrientedObjectDetectionResultTest, ConvertMapsTrackId) {
  mediapipe::OrientedDetection proto;
  proto.add_score(0.9f); proto.add_label_id(1);
  proto.set_cx(0.5f); proto.set_cy(0.5f); proto.set_width(0.2f);
  proto.set_height(0.1f); proto.set_rotation(0.3f);
  proto.set_track_id("42");
  auto result = ConvertToOrientedObjectDetectionResult({proto}, {100, 100});
  ASSERT_EQ(result.detections.size(), 1u);
  ASSERT_TRUE(result.detections[0].track_id.has_value());
  EXPECT_EQ(*result.detections[0].track_id, "42");
}
TEST(OrientedObjectDetectionResultTest, ConvertNoTrackId) {
  mediapipe::OrientedDetection proto;
  proto.add_score(0.9f); proto.add_label_id(1);
  proto.set_cx(0.5f); proto.set_cy(0.5f); proto.set_width(0.2f);
  proto.set_height(0.1f); proto.set_rotation(0.3f);
  auto result = ConvertToOrientedObjectDetectionResult({proto}, {100, 100});
  EXPECT_FALSE(result.detections[0].track_id.has_value());
}
```
Confirm the exact `ConvertToOrientedObjectDetectionResult` signature (image_size arg type — `{width,height}`) by reading the header; adapt the call. Run → FAIL (no `track_id` member).

- [ ] **Step 2: Add the field + conversion**

In `oriented_object_detection_result.h`, ensure `#include <optional>` and `#include <string>`; add to `struct OrientedObjectDetection` (after `rotation`):
```cpp
  // Optional persistent track ID (set by BoTSORT tracking; later group). nullopt
  // when the detection is not part of a track.
  std::optional<std::string> track_id = std::nullopt;
```
In `oriented_object_detection_result.cc` `ConvertToOrientedObjectDetectionResult`, where each `OrientedObjectDetection od` is built (after `od.rotation = d.rotation();`), add:
```cpp
    if (d.has_track_id()) od.track_id = d.track_id();
```

- [ ] **Step 3: Run the test (GREEN)**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/components/containers:oriented_object_detection_result_test --test_output=errors
```
Expected: PASS (incl. the 2 new cases). (Confirm the target name from the BUILD.)

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/cc/components/containers/
git commit -m "feat(obb-tracking): optional track_id on OrientedObjectDetection

ConvertToOrientedObjectDetectionResult now reads the OrientedDetection proto's
track_id into a new std::optional<std::string> field (nullopt when absent).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Real OBB e2e + regression

**Files:**
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_test.cc`

- [ ] **Step 1: Add the BOTSORT e2e test**

Find the existing real-fixture OBB tiled test (`TiledImageMode`, uses `yolo_obb_test_model` + boats.jpg, finds a ship). READ it for the model-path constant, image loader, and tiling setup. Add a VIDEO test mirroring it but stream + tracking, feeding the same image across frames via `DetectForVideo` and asserting stable track_ids:
```cpp
TEST(OrientedObjectDetectorTest, TiledVideoTracksShipWithBotsort) {
  const std::string model_path = /* same fixture path constant as TiledImageMode */;
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "OBB model fixture not present.";
  }
  auto options = std::make_unique<OrientedObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::VIDEO;
  options->num_classes = /* same as TiledImageMode */;
  options->max_results = 10;
  options->tiling.tile_rows = 2;
  options->tiling.tile_cols = 2;
  options->tiling.tile_overlap_fraction = 0.2f;
  // Tuned to this low-confidence fixture so BoTSORT confirms tracks (these are
  // NOT representative production defaults; see the TrackingOptions doc note).
  options->tracking.tracker_type =
      OrientedObjectDetectorOptions::TrackingOptions::kBotsort;
  options->tracking.track_high_threshold = 0.05f;
  options->tracking.track_low_threshold = 0.02f;
  options->tracking.new_track_threshold = 0.05f;
  options->tracking.match_threshold = 0.95f;
  options->tracking.track_buffer = 60;
  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          OrientedObjectDetector::Create(std::move(options)));
  Image image = /* load boats.jpg as in TiledImageMode */;
  std::vector<std::set<std::string>> per_frame_ids;
  for (int ts = 0; ts < 5; ++ts) {
    MP_ASSERT_OK_AND_ASSIGN(auto result, detector->DetectForVideo(image, ts));
    std::set<std::string> ids;
    for (const auto& d : result.detections) {
      if (d.track_id.has_value()) ids.insert(*d.track_id);
    }
    per_frame_ids.push_back(ids);
  }
  MP_ASSERT_OK(detector->Close());
  ASSERT_GE(per_frame_ids.size(), 2u);
  EXPECT_FALSE(per_frame_ids.back().empty()) << "final frame had no track ids";
  bool stable = false;
  for (size_t f = 1; f < per_frame_ids.size() && !stable; ++f) {
    for (const auto& id : per_frame_ids[f]) {
      if (per_frame_ids[f - 1].count(id)) { stable = true; break; }
    }
  }
  EXPECT_TRUE(stable) << "expected a track id stable across consecutive frames";
}
```
Adapt the model path / image load / num_classes to match `TiledImageMode` exactly (same fixture). Add `#include <set>`/`<string>` if needed. If the same input image yields no inter-frame motion, that's fine — BoTSORT still confirms + keeps a stable id; if ids don't appear, lower thresholds further / add frames, and report (do not weaken the assertion).

- [ ] **Step 2: Run the e2e**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test --test_output=all --test_filter='*TracksShipWithBotsort*' --nocache_test_results
```
Expected: RUNS (fixture present) and PASSES — oriented detections carry stable track ids. If it SKIPs, report (fixture missing). If ids never appear at these thresholds, investigate + report (do not weaken).

- [ ] **Step 3: Full regression**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/calculators/tensor:oriented_botsort_tracking_calculator_test \
  //mediapipe/calculators/tensor:botsort_tracking_calculator_test \
  //mediapipe/graphs/tiled_detection:tiled_detection_graphs_test \
  //mediapipe/tasks/cc/components/containers:oriented_object_detection_result_test \
  //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test \
  --test_output=errors
```
Expected: all PASS (the existing OBB IMAGE/tiled tests confirm the default path is unchanged).

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_test.cc
git commit -m "test(obb-tracking): e2e asserts oriented detections carry stable track ids

Tiled VIDEO OBB e2e selecting BOTSORT on the vendored fixture; asserts oriented
detections carry track ids stable across consecutive frames. Thresholds tuned to
the low-confidence fixture (test-only, not production defaults).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- `OrientedBotsortTrackingCalculator` (oriented→AABB→BoTSORT→id-back, no gap-fill) → T1 ✓
- `TiledObbTrackMergeGraph` + `tracking` on `TiledObbMergeGraphOptions` → T2 ✓
- OBB `TrackingOptions` proto+struct+converter (BoTSORT-only, default none) → T3 ✓
- OBB graph stream branch (default byte-identical) → T4 ✓
- Create validation (BOTSORT gates + BOX_TRACKER rejected) → T5 ✓
- `track_id` on `OrientedObjectDetection` + conversion → T6 ✓
- Real e2e stable-across-frames + regression → T7 ✓
- Foot-gun doc note → in the proto + struct comments (T3) ✓

**Placeholder scan:** No TBD/TODO; complete code for the new calculator + subgraph + validation + conversion; tests have real assertions; commands have expected output. Two spots intentionally instruct the implementer to confirm a local detail and adapt (the `oriented_detection_cc_proto` target label; whether the OBB tiling proto has `enable_motion_scheduling` — drop that one gate if absent; the obb-merge-test harness for vector<vector> input; the `TiledImageMode` fixture constants) — these are file-local facts to match, not logic gaps.

**Type consistency:** `track_id` is proto `string` (set via `set_track_id(std::to_string(t->track_id))` T1) → container `std::optional<std::string>` (T6). `TrackerType` enum values 0/1/2 identical across the OBB struct (kTrackerUnspecified/kBoxTracker/kBotsort), OBB proto (TRACKER_UNSPECIFIED/BOX_TRACKER/BOTSORT), and the reused `TiledTrackingGraphOptions` (T2/T3/T4), pinned by static_assert (T3). `OrientedBotsortTrackingCalculator` consumes/produces `std::vector<OrientedDetection>`; `TiledObbTrackMergeGraph` wires it with `BotsortTrackingCalculatorOptions` from `TiledObbMergeGraphOptions.tracking`. `kImageCpuTag`/`kImageTag`/`kOrientedDetectionsTag`/`kBatchInfoTag` reused from the OBB graph.

**Known verification-time notes flagged in-plan:** `oriented_detection_cc_proto` label (T1); RotatedNMS options include (T2); the obb-merge-test harness (T2); the OBB tiling proto's `enable_motion_scheduling` presence (T5); the `TiledImageMode` fixture constants/num_classes (T7); LIVE_STREAM callback requirement (T5).
