# YOLO Tiled Detection × BoxTracker (Video) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In VIDEO/LIVE_STREAM mode, feed the optical-flow BoxTracker's propagated detections into `YoloObjectDetector`'s tiled `TRACKER_DETECTIONS` seam so temporally-missed objects are filled in; IMAGE mode unchanged.

**Architecture:** Two tiny new calculators (`DetectionLabelIdCodecCalculator` carries the int `label_id` through the string-only tracker round-trip; `DetectionsTickGateCalculator` materializes one tracker packet per source frame) + one amended calculator (`TiledFrameSuppressionCalculator` gains a `tracker_is_gap_fill_only` opt-in) + two new builder subgraphs (`TiledTrackingGraph` wraps `ObjectTrackingSubgraphCpu` + the codec + the gate; `TiledBoxTrackMergeGraph` fans merged-fresh detections to the tracker and the suppressor) + a `use_stream_mode` branch in `YoloObjectDetectorGraph`.

**Tech Stack:** MediaPipe Bazel (C++20, `--define MEDIAPIPE_DISABLE_GPU=1`), api2 Node/Subgraph framework, proto2, the legacy `mediapipe/util/tracking` optical-flow stack via `ObjectTrackingSubgraphCpu`, GoogleTest.

**Spec:** `docs/superpowers/specs/2026-06-14-yolo-tiled-boxtracker-design.md`

**Conventions for every task:** code comments in English; build/test with `--define MEDIAPIPE_DISABLE_GPU=1`; commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Work on `dev`. Tests are written first and must fail (red) before implementation.

---

## File Structure

**New files**
- `mediapipe/calculators/tensor/detection_label_id_codec_calculator.proto` — options (`Direction` enum).
- `mediapipe/calculators/tensor/detection_label_id_codec_calculator.cc` — the codec calculator.
- `mediapipe/calculators/tensor/detection_label_id_codec_calculator_test.cc` — unit test.
- `mediapipe/calculators/tensor/detections_tick_gate_calculator.cc` — the per-frame tracker-packet materializer (no options).
- `mediapipe/calculators/tensor/detections_tick_gate_calculator_test.cc` — unit test.
- `mediapipe/graphs/tiled_detection/tiled_tracking_graph.cc` — `TiledTrackingGraph` subgraph.
- `mediapipe/graphs/tiled_detection/tiled_box_track_merge_graph.cc` — `TiledBoxTrackMergeGraph` subgraph.
- `mediapipe/graphs/tiled_detection/tiled_tracking_graphs_test.cc` — graph tests for both new subgraphs (kept separate from `tiled_detection_graphs_test.cc` so the heavy optical-flow/OpenCV deps stay isolated).

**Modified files**
- `mediapipe/calculators/tensor/tiled_frame_suppression_calculator.proto` — add field 4 `tracker_is_gap_fill_only`.
- `mediapipe/calculators/tensor/tiled_frame_suppression_calculator.cc` — gap-fill drop.
- `mediapipe/calculators/tensor/tiled_frame_suppression_calculator_test.cc` — fresh-wins + default-unchanged tests.
- `mediapipe/calculators/tensor/BUILD` — targets for the two new calculators.
- `mediapipe/graphs/tiled_detection/BUILD` — targets for the two new subgraphs + the new graph test.
- `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc` — `use_stream_mode` branch in the tiled path.
- `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD` — dep on the new merge subgraph.
- `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc` — public `DetectForVideo` tiled-tracking e2e + panning-video helper.

---

## Task 1: `DetectionLabelIdCodecCalculator` (label_id ↔ string round-trip)

**Files:**
- Create: `mediapipe/calculators/tensor/detection_label_id_codec_calculator.proto`
- Create: `mediapipe/calculators/tensor/detection_label_id_codec_calculator.cc`
- Test: `mediapipe/calculators/tensor/detection_label_id_codec_calculator_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

- [ ] **Step 1: Write the proto**

Create `mediapipe/calculators/tensor/detection_label_id_codec_calculator.proto` (Apache header omitted here for brevity — copy the 13-line header from `tiled_frame_suppression_calculator.proto`):

```proto
syntax = "proto2";

package mediapipe;

import "mediapipe/framework/calculator.proto";

message DetectionLabelIdCodecCalculatorOptions {
  extend mediapipe.CalculatorOptions {
    optional DetectionLabelIdCodecCalculatorOptions ext = 471230016;
  }

  // ENCODE: copy each int label_id(i) into the string label[i] (parallel to
  // score) so the optical-flow tracker, which only carries string labels in
  // TrackedDetection::label_to_score_map, preserves the class. DECODE: parse
  // each string label(i) back into label_id[i] and clear the synthetic label.
  enum Direction {
    ENCODE = 0;
    DECODE = 1;
  }
  optional Direction direction = 1 [default = ENCODE];
}
```

- [ ] **Step 2: Write the failing test**

Create `mediapipe/calculators/tensor/detection_label_id_codec_calculator_test.cc` (Apache header omitted):

```cpp
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

Detection Det(float score, int label_id) {
  Detection d;
  d.add_score(score);
  d.add_label_id(label_id);
  auto* ld = d.mutable_location_data();
  ld->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* bb = ld->mutable_relative_bounding_box();
  bb->set_xmin(0.1f);
  bb->set_ymin(0.2f);
  bb->set_width(0.3f);
  bb->set_height(0.4f);
  return d;
}

CalculatorRunner MakeRunner(const std::string& direction) {
  return CalculatorRunner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      "calculator: \"DetectionLabelIdCodecCalculator\"\n"
      "input_stream: \"DETECTIONS:in\"\n"
      "output_stream: \"DETECTIONS:out\"\n"
      "options { [mediapipe.DetectionLabelIdCodecCalculatorOptions.ext] {"
      "  direction: " + direction + " } }"));
}

const std::vector<Detection>& Out(const CalculatorRunner& r) {
  return r.Outputs().Tag("DETECTIONS").packets[0].Get<std::vector<Detection>>();
}

TEST(DetectionLabelIdCodecCalculatorTest, EncodeWritesStringLabel) {
  CalculatorRunner runner = MakeRunner("ENCODE");
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{Det(0.7f, 8)})
          .At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = Out(runner);
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].label_size(), 1);
  EXPECT_EQ(out[0].label(0), "8");
  ASSERT_EQ(out[0].score_size(), 1);
  EXPECT_NEAR(out[0].score(0), 0.7f, 1e-6);
}

TEST(DetectionLabelIdCodecCalculatorTest, DecodeRoundTripsLabelId) {
  // A detection as the tracker would emit it: string label, score, no label_id.
  Detection tracked;
  tracked.add_label("8");
  tracked.add_score(0.5f);
  CalculatorRunner runner = MakeRunner("DECODE");
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{tracked})
          .At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = Out(runner);
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].label_id_size(), 1);
  EXPECT_EQ(out[0].label_id(0), 8);
  EXPECT_EQ(out[0].label_size(), 0);  // synthetic label cleared
}

TEST(DetectionLabelIdCodecCalculatorTest, DecodeDropsUnclassifiable) {
  Detection bad;       // non-numeric label
  bad.add_label("boat");
  bad.add_score(0.5f);
  Detection empty;     // no label at all
  empty.add_score(0.4f);
  CalculatorRunner runner = MakeRunner("DECODE");
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{bad, empty})
          .At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  EXPECT_TRUE(Out(runner).empty());  // both dropped, none reach the output
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:detection_label_id_codec_calculator_test`
Expected: build failure (calculator/proto/BUILD target don't exist yet).

- [ ] **Step 4: Write the calculator**

Create `mediapipe/calculators/tensor/detection_label_id_codec_calculator.cc` (Apache header omitted):

```cpp
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensor/detection_label_id_codec_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {
namespace api2 {

// Carries the integer label_id through the optical-flow tracker, which only
// preserves string labels (TrackedDetection::label_to_score_map). ENCODE
// stringifies each label_id into the parallel string `label`; DECODE parses it
// back and drops any detection it cannot classify (so no category=-1 box
// reaches the public API).
class DetectionLabelIdCodecCalculator : public Node {
 public:
  static constexpr Input<std::vector<Detection>> kIn{"DETECTIONS"};
  static constexpr Output<std::vector<Detection>> kOut{"DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<mediapipe::DetectionLabelIdCodecCalculatorOptions>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    std::vector<Detection> out;
    if (kIn(cc).IsEmpty()) {
      kOut(cc).Send(std::move(out));
      return absl::OkStatus();
    }
    const bool encode =
        options_.direction() ==
        mediapipe::DetectionLabelIdCodecCalculatorOptions::ENCODE;
    for (const Detection& d : *kIn(cc)) {
      Detection nd = d;
      if (encode) {
        nd.clear_label();
        for (int i = 0; i < nd.label_id_size(); ++i) {
          nd.add_label(absl::StrCat(nd.label_id(i)));
        }
        out.push_back(std::move(nd));
      } else {
        nd.clear_label_id();
        bool ok = nd.label_size() > 0;
        for (int i = 0; i < nd.label_size() && ok; ++i) {
          int id = 0;
          if (absl::SimpleAtoi(nd.label(i), &id)) {
            nd.add_label_id(id);
          } else {
            ok = false;
          }
        }
        if (!ok || nd.label_id_size() == 0) {
          ABSL_LOG_FIRST_N(WARNING, 1)
              << "DetectionLabelIdCodecCalculator: dropping a tracker "
                 "detection with no parseable label_id.";
          continue;
        }
        nd.clear_label();
        out.push_back(std::move(nd));
      }
    }
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }

 private:
  mediapipe::DetectionLabelIdCodecCalculatorOptions options_;
};

MEDIAPIPE_REGISTER_NODE(DetectionLabelIdCodecCalculator);

}  // namespace api2
}  // namespace mediapipe
```

- [ ] **Step 5: Add BUILD targets**

In `mediapipe/calculators/tensor/BUILD`, add (mirroring the `tiled_frame_suppression_calculator` stanzas):

```python
mediapipe_proto_library(
    name = "detection_label_id_codec_calculator_proto",
    srcs = ["detection_label_id_codec_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "detection_label_id_codec_calculator",
    srcs = ["detection_label_id_codec_calculator.cc"],
    deps = [
        ":detection_label_id_codec_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_test(
    name = "detection_label_id_codec_calculator_test",
    srcs = ["detection_label_id_codec_calculator_test.cc"],
    deps = [
        ":detection_label_id_codec_calculator",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
    ],
)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:detection_label_id_codec_calculator_test`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add mediapipe/calculators/tensor/detection_label_id_codec_calculator.proto \
        mediapipe/calculators/tensor/detection_label_id_codec_calculator.cc \
        mediapipe/calculators/tensor/detection_label_id_codec_calculator_test.cc \
        mediapipe/calculators/tensor/BUILD
git commit -m "feat(tiling): DetectionLabelIdCodecCalculator — label_id<->string round-trip for tracker

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `DetectionsTickGateCalculator` (one tracker packet per source frame)

**Files:**
- Create: `mediapipe/calculators/tensor/detections_tick_gate_calculator.cc`
- Test: `mediapipe/calculators/tensor/detections_tick_gate_calculator_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

- [ ] **Step 1: Write the failing test**

Create `mediapipe/calculators/tensor/detections_tick_gate_calculator_test.cc` (Apache header omitted):

```cpp
#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

Detection MakeDet() {
  Detection d;
  d.add_score(0.9f);
  d.add_label_id(0);
  return d;
}

// Two ticks; DATA only at the first. The gate must emit a packet at BOTH tick
// timestamps (the second one empty), so the downstream synchronized consumer
// never stalls on a gap.
TEST(DetectionsTickGateCalculatorTest, EmitsOnePacketPerTickEmptyOnGap) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsTickGateCalculator"
    input_stream: "TICK:tick"
    input_stream: "DATA:data"
    output_stream: "DETECTIONS:out"
  )pb"));

  runner.MutableInputs()->Tag("TICK").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{}).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TICK").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{}).At(Timestamp(1)));
  runner.MutableInputs()->Tag("DATA").packets.push_back(
      MakePacket<std::vector<Detection>>(std::vector<Detection>{MakeDet()})
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& outs = runner.Outputs().Tag("DETECTIONS").packets;
  ASSERT_EQ(outs.size(), 2u);
  EXPECT_EQ(outs[0].Timestamp(), Timestamp(0));
  EXPECT_EQ(outs[0].Get<std::vector<Detection>>().size(), 1u);
  EXPECT_EQ(outs[1].Timestamp(), Timestamp(1));
  EXPECT_TRUE(outs[1].Get<std::vector<Detection>>().empty());
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:detections_tick_gate_calculator_test`
Expected: build failure (target/calculator don't exist).

- [ ] **Step 3: Write the calculator**

Create `mediapipe/calculators/tensor/detections_tick_gate_calculator.cc` (Apache header omitted):

```cpp
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"

namespace mediapipe {
namespace api2 {

// Driven by the TICK stream (one packet per source frame). Emits the DATA
// vector at the tick timestamp if present, otherwise an empty vector. This
// converts the tracker's gappy DETECTIONS stream (the manager only emits on
// frames that have tracking boxes) into a dense one-packet-per-frame stream,
// so the downstream synchronized TRACKER_DETECTIONS input never stalls.
class DetectionsTickGateCalculator : public Node {
 public:
  static constexpr Input<std::vector<Detection>> kTick{"TICK"};
  static constexpr Input<std::vector<Detection>>::Optional kData{"DATA"};
  static constexpr Output<std::vector<Detection>> kOut{"DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kTick, kData, kOut);

  absl::Status Process(CalculatorContext* cc) override {
    std::vector<Detection> out;
    if (kData(cc).IsConnected() && !kData(cc).IsEmpty()) {
      out = *kData(cc);
    }
    kOut(cc).Send(std::move(out));
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(DetectionsTickGateCalculator);

}  // namespace api2
}  // namespace mediapipe
```

- [ ] **Step 4: Add BUILD targets**

In `mediapipe/calculators/tensor/BUILD`, add:

```python
cc_library(
    name = "detections_tick_gate_calculator",
    srcs = ["detections_tick_gate_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:detection_cc_proto",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "detections_tick_gate_calculator_test",
    srcs = ["detections_tick_gate_calculator_test.cc"],
    deps = [
        ":detections_tick_gate_calculator",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
    ],
)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:detections_tick_gate_calculator_test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mediapipe/calculators/tensor/detections_tick_gate_calculator.cc \
        mediapipe/calculators/tensor/detections_tick_gate_calculator_test.cc \
        mediapipe/calculators/tensor/BUILD
git commit -m "feat(tiling): DetectionsTickGateCalculator — one tracker packet per source frame

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `TiledFrameSuppressionCalculator` — `tracker_is_gap_fill_only`

**Files:**
- Modify: `mediapipe/calculators/tensor/tiled_frame_suppression_calculator.proto`
- Modify: `mediapipe/calculators/tensor/tiled_frame_suppression_calculator.cc`
- Test: `mediapipe/calculators/tensor/tiled_frame_suppression_calculator_test.cc`

- [ ] **Step 1: Write the failing tests**

Append to `mediapipe/calculators/tensor/tiled_frame_suppression_calculator_test.cc` (uses the existing `Det`, `PushDets`, `GetOutput` helpers in that file):

```cpp
// gap-fill: a high-score tracker box overlapping a lower-score fresh box is
// dropped before NMS, so the FRESH box (its geometry/score) is what survives.
TEST(TiledFrameSuppressionCalculatorTest, GapFillDropsTrackerBoxOverlappingFresh) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "TRACKER_DETECTIONS:tracker"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        class_agnostic: true
        tracker_is_gap_fill_only: true
      }
    }
  )pb"));
  // Fresh box (score 0.4) and a heavily-overlapping tracker box with a HIGHER
  // (stale historical-max) score 0.95. Gap-fill drops the tracker box, so the
  // surviving box is the fresh 0.4 one.
  PushDets(&runner, "DETECTIONS", {Det(0.4f, 0, 0.10f, 0.10f, 0.40f, 0.40f)});
  PushDets(&runner, "TRACKER_DETECTIONS",
           {Det(0.95f, 0, 0.11f, 0.11f, 0.40f, 0.40f)});
  MP_ASSERT_OK(runner.Run());
  const auto& out = GetOutput(runner);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NEAR(out[0].score(0), 0.4f, 1e-5);  // fresh won, not the stale tracker
}

// gap-fill: a tracker box that does NOT overlap any fresh box is kept (the
// "fill the gap" case).
TEST(TiledFrameSuppressionCalculatorTest, GapFillKeepsNonOverlappingTrackerBox) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "TRACKER_DETECTIONS:tracker"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        class_agnostic: true
        tracker_is_gap_fill_only: true
      }
    }
  )pb"));
  PushDets(&runner, "DETECTIONS", {Det(0.4f, 0, 0.05f, 0.05f, 0.10f, 0.10f)});
  PushDets(&runner, "TRACKER_DETECTIONS",
           {Det(0.8f, 0, 0.70f, 0.70f, 0.10f, 0.10f)});
  MP_ASSERT_OK(runner.Run());
  EXPECT_EQ(GetOutput(runner).size(), 2u);  // fresh + non-overlapping tracker
}

// Default (option absent) keeps today's blind-concatenate behavior: both
// overlapping boxes go into one NMS and the HIGHER score wins.
TEST(TiledFrameSuppressionCalculatorTest, DefaultConcatenatesTrackerIntoNms) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "TRACKER_DETECTIONS:tracker"
    output_stream: "DETECTIONS:out"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.5
        class_agnostic: true
      }
    }
  )pb"));
  PushDets(&runner, "DETECTIONS", {Det(0.4f, 0, 0.10f, 0.10f, 0.40f, 0.40f)});
  PushDets(&runner, "TRACKER_DETECTIONS",
           {Det(0.95f, 0, 0.11f, 0.11f, 0.40f, 0.40f)});
  MP_ASSERT_OK(runner.Run());
  const auto& out = GetOutput(runner);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NEAR(out[0].score(0), 0.95f, 1e-5);  // tracker won (default behavior)
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tiled_frame_suppression_calculator_test`
Expected: FAIL — `GapFillDrops...` and `GapFillKeeps...` fail (no `tracker_is_gap_fill_only` field / option unhandled); `DefaultConcatenates...` passes already.

- [ ] **Step 3: Add the proto field**

In `mediapipe/calculators/tensor/tiled_frame_suppression_calculator.proto`, after the `bypass_single_tile` field (field 3), add:

```proto
  // When true, before the global NMS, drop every TRACKER_DETECTIONS box that
  // overlaps any fresh DETECTIONS box (IoU >= iou_threshold, honoring
  // class_agnostic) so fresh detections always win and the tracker only fills
  // gaps. Default false = today's blind concatenate-then-NMS behavior.
  optional bool tracker_is_gap_fill_only = 4 [default = false];
```

- [ ] **Step 4: Implement the gap-fill drop**

In `mediapipe/calculators/tensor/tiled_frame_suppression_calculator.cc`, replace the body from `std::vector<Detection> combined = std::move(fresh);` through the closing of the `if (tracker_present)` block with:

```cpp
    std::vector<Detection> combined = std::move(fresh);
    if (tracker_present) {
      const auto& tr = *kInTracker(cc);
      if (options_.tracker_is_gap_fill_only()) {
        // Keep only tracker boxes that don't overlap any fresh box. Inner loop
        // is bounded by the original fresh count so appended tracker boxes are
        // never treated as "fresh".
        const size_t fresh_count = combined.size();
        for (const Detection& t : tr) {
          bool overlaps = false;
          for (size_t i = 0; i < fresh_count; ++i) {
            const Detection& f = combined[i];
            if (!options_.class_agnostic() && t.label_id_size() > 0 &&
                f.label_id_size() > 0 && t.label_id(0) != f.label_id(0)) {
              continue;
            }
            if (DetectionRelativeIoU(t, f) >= options_.iou_threshold()) {
              overlaps = true;
              break;
            }
          }
          if (!overlaps) combined.push_back(t);
        }
      } else {
        combined.insert(combined.end(), tr.begin(), tr.end());
      }
    }
```

Add `#include <cstddef>` (for `size_t`) to the include block if not already transitively available. `DetectionRelativeIoU` is already declared in the already-included `detection_nms_util.h`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tiled_frame_suppression_calculator_test`
Expected: PASS (all tests, including the three new ones and all pre-existing ones).

- [ ] **Step 6: Commit**

```bash
git add mediapipe/calculators/tensor/tiled_frame_suppression_calculator.proto \
        mediapipe/calculators/tensor/tiled_frame_suppression_calculator.cc \
        mediapipe/calculators/tensor/tiled_frame_suppression_calculator_test.cc
git commit -m "feat(tiling): TiledFrameSuppression tracker_is_gap_fill_only (fresh-wins fusion)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `TiledTrackingGraph` subgraph

**Files:**
- Create: `mediapipe/graphs/tiled_detection/tiled_tracking_graph.cc`
- Modify: `mediapipe/graphs/tiled_detection/BUILD`
- Test: deferred to Task 5's shared test file (`tiled_tracking_graphs_test.cc`), written in Step 1 below and run after the BUILD target exists.

- [ ] **Step 1: Write the failing graph test**

Create `mediapipe/graphs/tiled_detection/tiled_tracking_graphs_test.cc` (Apache header omitted). This file holds tests for BOTH new subgraphs (Task 4 + Task 5):

```cpp
#include <cstring>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

std::unique_ptr<ImageFrame> WhiteFrame(int w, int h) {
  auto f = std::make_unique<ImageFrame>(ImageFormat::SRGB, w, h);
  std::memset(f->MutablePixelData(), 255, f->Height() * f->WidthStep());
  return f;
}

Detection Box(float score, int label_id, float xmin, float ymin, float w,
              float h) {
  Detection d;
  d.add_score(score);
  d.add_label_id(label_id);
  auto* ld = d.mutable_location_data();
  ld->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* bb = ld->mutable_relative_bounding_box();
  bb->set_xmin(xmin);
  bb->set_ymin(ymin);
  bb->set_width(w);
  bb->set_height(h);
  return d;
}

// TiledTrackingGraph: IMAGE + DETECTIONS -> TRACKER_DETECTIONS. Feeds 3 frames
// of fresh detections through the real optical-flow tracker and asserts the
// tick gate materializes EXACTLY one TRACKER_DETECTIONS packet per source
// frame (no stall — including the first frame, before any flow exists).
TEST(TiledTrackingGraphTest, EmitsOnePacketPerFrameNoStall) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    input_stream: "dets"
    output_stream: "tracker"
    node {
      calculator: "mediapipe.tiled_detection.TiledTrackingGraph"
      input_stream: "IMAGE:image"
      input_stream: "DETECTIONS:dets"
      output_stream: "TRACKER_DETECTIONS:tracker"
    }
  )pb");

  std::vector<Packet> out;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("tracker", [&](const Packet& p) {
    out.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < 3; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image", Adopt(WhiteFrame(64, 64).release()).At(Timestamp(i))));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "dets",
        MakePacket<std::vector<Detection>>(
            std::vector<Detection>{Box(0.9f, 8, 0.3f, 0.3f, 0.2f, 0.2f)})
            .At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(out.size(), 3u);  // one TRACKER_DETECTIONS packet per source frame
  for (const Packet& p : out) {
    // Type is correct; any tracked box carries label_id (not a string label).
    for (const Detection& d : p.Get<std::vector<Detection>>()) {
      EXPECT_GT(d.label_id_size(), 0);
      EXPECT_EQ(d.label_size(), 0);
    }
  }
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 2: Write the subgraph**

Create `mediapipe/graphs/tiled_detection/tiled_tracking_graph.cc` (Apache header omitted):

```cpp
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/detection_label_id_codec_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/subgraph.h"

namespace mediapipe {
namespace tiled_detection {

// A "mediapipe.tiled_detection.TiledTrackingGraph" produces class-carrying
// tracker-propagated detections from video frames + per-frame fresh
// detections, one packet per source frame:
// DETECTIONS -[ENCODE label_id->string]-> ObjectTrackingSubgraphCpu(VIDEO,
// DETECTIONS) -[DECODE string->label_id, drop unclassifiable]->
// DetectionsTickGate(TICK=DETECTIONS) -> TRACKER_DETECTIONS.
//
// Inputs:
//   IMAGE       - ImageFrame (the source video frame; fed to optical flow).
//   DETECTIONS  - std::vector<Detection> (merged-fresh, frame-normalized,
//                 carries label_id + score), one packet per source frame.
// Outputs:
//   TRACKER_DETECTIONS - std::vector<Detection> (carries label_id), EXACTLY
//                 one packet per source frame (empty when nothing tracked).
//
// Example:
// node {
//   calculator: "mediapipe.tiled_detection.TiledTrackingGraph"
//   input_stream: "IMAGE:image"
//   input_stream: "DETECTIONS:merged_fresh"
//   output_stream: "TRACKER_DETECTIONS:tracker_dets"
// }
class TiledTrackingGraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    api2::builder::Graph graph;
    auto image = graph.In("IMAGE").Cast<ImageFrame>();
    auto fresh = graph.In("DETECTIONS").Cast<std::vector<Detection>>();

    auto& encode = graph.AddNode("DetectionLabelIdCodecCalculator");
    encode.GetOptions<DetectionLabelIdCodecCalculatorOptions>().set_direction(
        DetectionLabelIdCodecCalculatorOptions::ENCODE);
    fresh >> encode.In("DETECTIONS");

    auto& tracker = graph.AddNode("ObjectTrackingSubgraphCpu");
    image >> tracker.In("VIDEO");
    encode.Out("DETECTIONS") >> tracker.In("DETECTIONS");

    auto& decode = graph.AddNode("DetectionLabelIdCodecCalculator");
    decode.GetOptions<DetectionLabelIdCodecCalculatorOptions>().set_direction(
        DetectionLabelIdCodecCalculatorOptions::DECODE);
    tracker.Out("DETECTIONS") >> decode.In("DETECTIONS");

    auto& gate = graph.AddNode("DetectionsTickGateCalculator");
    fresh >> gate.In("TICK");
    decode.Out("DETECTIONS") >> gate.In("DATA");

    gate.Out("DETECTIONS") >> graph.Out("TRACKER_DETECTIONS");
    return graph.GetConfig();
  }
};

// NOTE: keep the fully-qualified type name on a single line. The
// REGISTER_MEDIAPIPE_GRAPH macro stringifies its argument with `#name`, so a
// line break here would inject a stray space into the registered name and the
// graph would never be found by lookup.
// clang-format off
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tiled_detection::TiledTrackingGraph);  // NOLINT(whitespace/line_length)
// clang-format on

}  // namespace tiled_detection
}  // namespace mediapipe
```

- [ ] **Step 3: Add BUILD targets**

In `mediapipe/graphs/tiled_detection/BUILD`, add the subgraph library and the new test:

```python
cc_library(
    name = "tiled_tracking_graph",
    srcs = ["tiled_tracking_graph.cc"],
    deps = [
        "//mediapipe/calculators/tensor:detection_label_id_codec_calculator",
        "//mediapipe/calculators/tensor:detection_label_id_codec_calculator_cc_proto",
        "//mediapipe/calculators/tensor:detections_tick_gate_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/graphs/tracking/subgraphs:object_tracking_cpu",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_test(
    name = "tiled_tracking_graphs_test",
    size = "medium",
    srcs = ["tiled_tracking_graphs_test.cc"],
    deps = [
        ":tiled_box_track_merge_graph",
        ":tiled_tracking_graph",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
```

(The `:tiled_box_track_merge_graph` dep is added in Task 5; the test file compiles after Task 5's library exists. Run this test at the end of Task 5.)

- [ ] **Step 4: Build the subgraph library**

Run: `bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/graphs/tiled_detection:tiled_tracking_graph`
Expected: builds (links `ObjectTrackingSubgraphCpu` + the two new calculators). If the link fails with undefined OpenCV symbols (e.g. `cv::video`/`cv::features2d`), add the missing dylib to `third_party/opencv_macos.BUILD` `srcs` glob (calib3d/core/imgproc/imgcodecs are already present) and rebuild.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/graphs/tiled_detection/tiled_tracking_graph.cc \
        mediapipe/graphs/tiled_detection/tiled_tracking_graphs_test.cc \
        mediapipe/graphs/tiled_detection/BUILD
git commit -m "feat(tiling): TiledTrackingGraph — BoxTracker + label_id codec + tick gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `TiledBoxTrackMergeGraph` subgraph

**Files:**
- Create: `mediapipe/graphs/tiled_detection/tiled_box_track_merge_graph.cc`
- Modify: `mediapipe/graphs/tiled_detection/BUILD`
- Test: `mediapipe/graphs/tiled_detection/tiled_tracking_graphs_test.cc` (add a test)

- [ ] **Step 1: Add the failing graph test**

Append to `mediapipe/graphs/tiled_detection/tiled_tracking_graphs_test.cc` (before the closing `}  // namespace`). It reuses `WhiteFrame`/`Box` and adds the tiling-types includes it needs:

```cpp
// Add near the top includes of the file:
//   #include <cstdint>
//   #include <utility>
//   #include "mediapipe/calculators/tensor/tiling_matrix_utils.h"
//   #include "mediapipe/calculators/tensor/tiling_types.h"

// Helper: one full-frame tile geometry + batch info (mirrors
// tiled_detection_graphs_test.cc).
TileGeometry FullFrameTile2() {
  TileGeometry g;
  g.tile_index = 0;
  g.x_center = 0.5f;
  g.y_center = 0.5f;
  g.width = 1.0f;
  g.height = 1.0f;
  return g;
}

std::shared_ptr<TileBatchGeometry> Geom1(int fw, int fh) {
  auto geom = std::make_shared<TileBatchGeometry>();
  TileGeometry g = FullFrameTile2();
  TilePixelRoi roi;
  roi.x = 0;
  roi.y = 0;
  roi.width = fw;
  roi.height = fh;
  geom->tile_indices.push_back(g.tile_index);
  geom->tile_geometries.push_back(g);
  geom->effective_pixel_rois.push_back(roi);
  geom->tile_to_image_matrices.push_back(TileToImageMatrix(roi, fw, fh));
  return geom;
}

TensorBatchInfo Info1(int64_t source_ts, std::shared_ptr<TileBatchGeometry> g) {
  TensorBatchInfo info;
  info.source_frame_timestamp = source_ts;
  info.batch_index = 0;
  info.total_batches = 1;
  info.valid_count = 1;
  info.tile_indices = g->tile_indices;
  info.geometry = std::move(g);
  return info;
}

// TiledBoxTrackMergeGraph: per-batch DETECTIONS + BATCH_INFO + IMAGE -> merged
// DETECTIONS. On a white frame the tracker emits nothing, so the merged output
// is exactly the fresh detection (tracker gap-fill drops nothing), at the
// source timestamp — proving the merge+suppression+IMAGE wiring runs without
// stalling.
TEST(TiledBoxTrackMergeGraphTest, FreshDetectionFlowsThroughWithTracker) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets"
    input_stream: "info"
    input_stream: "image"
    output_stream: "merged"
    node {
      calculator: "mediapipe.tiled_detection.TiledBoxTrackMergeGraph"
      input_stream: "DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
      input_stream: "IMAGE:image"
      output_stream: "DETECTIONS:merged"
      options {
        [mediapipe.TiledBoxMergeGraphOptions.ext] {
          iou_threshold: 0.5
          class_agnostic: true
        }
      }
    }
  )pb");

  std::vector<Packet> merged;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("merged", [&](const Packet& p) {
    merged.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));

  auto geom = Geom1(100, 100);
  // One batch for source frame ts=50; per-batch DETECTIONS at synthetic ts=0.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "dets",
      MakePacket<std::vector<std::vector<Detection>>>(
          std::vector<std::vector<Detection>>{
              {Box(0.9f, 8, 0.3f, 0.3f, 0.2f, 0.2f)}})
          .At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "info", MakePacket<TensorBatchInfo>(Info1(50, geom)).At(Timestamp(0))));
  // IMAGE on the source timeline (ts=50), where the merged output lands.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image", Adopt(WhiteFrame(100, 100).release()).At(Timestamp(50))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(merged.size(), 1u);
  EXPECT_EQ(merged[0].Timestamp(), Timestamp(50));
  const auto& dets = merged[0].Get<std::vector<Detection>>();
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_NEAR(dets[0].score(0), 0.9f, 1e-5);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/graphs/tiled_detection:tiled_tracking_graphs_test`
Expected: build failure (`tiled_box_track_merge_graph` target / subgraph don't exist).

- [ ] **Step 3: Write the subgraph**

Create `mediapipe/graphs/tiled_detection/tiled_box_track_merge_graph.cc` (Apache header omitted):

```cpp
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/core/clip_vector_size_calculator.pb.h"
#include "mediapipe/calculators/tensor/tiled_frame_suppression_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/graphs/tiled_detection/tiled_detection_graphs.pb.h"

namespace mediapipe {
namespace tiled_detection {

// A "mediapipe.tiled_detection.TiledBoxTrackMergeGraph" is the stream-mode
// sibling of TiledBoxMergeGraph: it merges per-batch tile-local axis-aligned
// detections back into source-frame space, then FUSES them with optical-flow
// tracker-propagated detections via a single global NMS (fresh-wins):
//   DETECTIONS + BATCH_INFO -> MergeTileBoxDetectionsAccumulator -> merged_fresh
//   merged_fresh + IMAGE -> TiledTrackingGraph -> tracker_dets
//   TiledFrameSuppression(DETECTIONS=merged_fresh,
//                         TRACKER_DETECTIONS=tracker_dets,
//                         tracker_is_gap_fill_only=true)
//   [-> ClipDetectionVectorSize if max_detections >= 1].
//
// Inputs:
//   DETECTIONS - std::vector<std::vector<Detection>> (per-batch tile-local).
//   BATCH_INFO - TensorBatchInfo.
//   IMAGE      - ImageFrame (source video frame; drives the tracker's flow).
// Outputs:
//   DETECTIONS - std::vector<Detection> (merged + tracker-fused, one packet
//                per source frame, frame-normalized).
//
// Example:
// node {
//   calculator: "mediapipe.tiled_detection.TiledBoxTrackMergeGraph"
//   input_stream: "DETECTIONS:tile_detections"
//   input_stream: "BATCH_INFO:batch_info"
//   input_stream: "IMAGE:image"
//   output_stream: "DETECTIONS:detections"
//   options {
//     [mediapipe.TiledBoxMergeGraphOptions.ext] { iou_threshold: 0.5 }
//   }
// }
class TiledBoxTrackMergeGraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    const auto& options = sc->Options<TiledBoxMergeGraphOptions>();
    if (options.max_detections() == 0) {
      return absl::InvalidArgumentError(
          "TiledBoxMergeGraphOptions.max_detections must be -1 (uncapped) or "
          ">= 1; got 0.");
    }
    api2::builder::Graph graph;

    auto& merge = graph.AddNode("MergeTileBoxDetectionsAccumulatorCalculator");
    graph.In("DETECTIONS") >> merge.In("DETECTIONS");
    graph.In("BATCH_INFO") >> merge.In("BATCH_INFO");
    auto merged_fresh = merge.Out("DETECTIONS").Cast<std::vector<Detection>>();

    auto& track = graph.AddNode("mediapipe.tiled_detection.TiledTrackingGraph");
    graph.In("IMAGE") >> track.In("IMAGE");
    merged_fresh >> track.In("DETECTIONS");

    auto& nms = graph.AddNode("TiledFrameSuppressionCalculator");
    auto& no = nms.GetOptions<TiledFrameSuppressionCalculatorOptions>();
    no.set_iou_threshold(options.iou_threshold());
    no.set_class_agnostic(options.class_agnostic());
    no.set_tracker_is_gap_fill_only(true);
    merged_fresh >> nms.In("DETECTIONS");
    track.Out("TRACKER_DETECTIONS") >> nms.In("TRACKER_DETECTIONS");

    if (options.max_detections() >= 1) {
      auto& clip = graph.AddNode("ClipDetectionVectorSizeCalculator");
      clip.GetOptions<ClipVectorSizeCalculatorOptions>().set_max_vec_size(
          options.max_detections());
      nms.Out("DETECTIONS") >> clip.In("");
      clip.Out("") >> graph.Out("DETECTIONS");
    } else {
      nms.Out("DETECTIONS") >> graph.Out("DETECTIONS");
    }
    return graph.GetConfig();
  }
};

// NOTE: keep the fully-qualified type name on a single line. The
// REGISTER_MEDIAPIPE_GRAPH macro stringifies its argument with `#name`, so a
// line break here would inject a stray space into the registered name and the
// graph would never be found by lookup.
// clang-format off
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tiled_detection::TiledBoxTrackMergeGraph);  // NOLINT(whitespace/line_length)
// clang-format on

}  // namespace tiled_detection
}  // namespace mediapipe
```

- [ ] **Step 4: Add the BUILD library target**

In `mediapipe/graphs/tiled_detection/BUILD`, add:

```python
cc_library(
    name = "tiled_box_track_merge_graph",
    srcs = ["tiled_box_track_merge_graph.cc"],
    deps = [
        ":tiled_detection_graphs_cc_proto",
        ":tiled_tracking_graph",
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator_cc_proto",
        "//mediapipe/calculators/tensor:merge_tile_box_detections_accumulator_calculator",
        "//mediapipe/calculators/tensor:tiled_frame_suppression_calculator",
        "//mediapipe/calculators/tensor:tiled_frame_suppression_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)
```

The `tiled_tracking_graphs_test` cc_test stanza (added in Task 4 Step 3) already depends on both `:tiled_tracking_graph` and `:tiled_box_track_merge_graph`, plus the tiling-types libs needed by the new helpers — confirm these are present in its deps, adding if missing:

```python
        "//mediapipe/calculators/tensor:tiling_matrix_utils",
        "//mediapipe/calculators/tensor:tiling_types",
        "//mediapipe/framework/formats:tensor",
```

- [ ] **Step 5: Run the full graph test to verify it passes**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/graphs/tiled_detection:tiled_tracking_graphs_test`
Expected: PASS (both `TiledTrackingGraphTest.EmitsOnePacketPerFrameNoStall` and `TiledBoxTrackMergeGraphTest.FreshDetectionFlowsThroughWithTracker`). If either hangs (timeout), the tracker chain is not advancing its output timestamp bound on gap frames — verify the `DetectionsTickGateCalculator` TICK input is wired to `merged_fresh` (always present per frame) and that the manager output (`DATA`) is connected; the gate, driven by the always-present TICK, must emit per frame.

- [ ] **Step 6: Commit**

```bash
git add mediapipe/graphs/tiled_detection/tiled_box_track_merge_graph.cc \
        mediapipe/graphs/tiled_detection/tiled_tracking_graphs_test.cc \
        mediapipe/graphs/tiled_detection/BUILD
git commit -m "feat(tiling): TiledBoxTrackMergeGraph — fresh+tracker fusion subgraph

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `YoloObjectDetectorGraph` — branch the tiled path on `use_stream_mode`

**Files:**
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc`
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`

There is no new unit test here; correctness in IMAGE mode is guarded by the existing `yolo_object_detector_test` (which must keep passing), and stream-mode behavior is covered by the e2e in Task 7.

- [ ] **Step 1: Replace the single `merge` node with a `use_stream_mode` branch**

In `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc`, inside the tiled `else` block, find the current block:

```cpp
      auto& merge =
          graph.AddNode("mediapipe.tiled_detection.TiledBoxMergeGraph");
      auto& mo = merge.GetOptions<::mediapipe::TiledBoxMergeGraphOptions>();
      mo.set_iou_threshold(task_options.iou_threshold());
      mo.set_class_agnostic(true);
      mo.set_max_detections(task_options.max_results());
      yolo_decode.Out(kDetectionsTag) >> merge.In(kDetectionsTag);
      front.Out(kBatchInfoTag) >> merge.In(kBatchInfoTag);

      auto& label_id_to_text =
          graph.AddNode("DetectionLabelIdToTextCalculator");
      configure_label_id_to_text(label_id_to_text);
      merge.Out(kDetectionsTag) >> label_id_to_text.In("");
```

Replace it with (the only behavioral change is choosing the track-merge subgraph and teeing the ImageFrame into it when in stream mode; the label tail is unchanged but now consumes a captured `merged_dets`):

```cpp
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

      auto& label_id_to_text =
          graph.AddNode("DetectionLabelIdToTextCalculator");
      configure_label_id_to_text(label_id_to_text);
      *merged_dets >> label_id_to_text.In("");
```

`std::optional` and `Source` are already in scope (`<optional>` is included; `Source` is the `using` alias). `kImageCpuTag` and `kImageTag` constants already exist; `to_frame` is the `FromImageCalculator` node already in this block.

- [ ] **Step 2: Add the BUILD dep**

In `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`, in `cc_library(name = "yolo_object_detector_graph")` deps, add alongside the existing `//mediapipe/graphs/tiled_detection:*` entries:

```python
        "//mediapipe/graphs/tiled_detection:tiled_box_track_merge_graph",
```

- [ ] **Step 3: Build the graph and run the existing detector test (IMAGE path must still pass)**

Run: `bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_graph`
Then: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test`
Expected: builds; existing tests pass or GTEST_SKIP (fixtures gated). The IMAGE-mode `TiledGridDetectsBoatsOnBoats` behavior is unchanged.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc \
        mediapipe/tasks/cc/vision/yolo_object_detector/BUILD
git commit -m "feat(tiling): YoloObjectDetectorGraph routes stream mode to tracked merge graph

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Public `DetectForVideo` tiled-tracking e2e (panning boats)

**Files:**
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc`
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`

- [ ] **Step 1: Add the failing e2e test + panning helper**

Add these includes to the top of `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc`:

```cpp
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
```

Add, inside the anonymous namespace (after `ImagePath()`):

```cpp
// Translates `src` by (dx, dy) pixels (border replicated), returning a new
// SRGB Image. Produces real, trackable motion for the optical-flow tracker.
Image TranslateImage(const Image& src, int dx, int dy) {
  cv::Mat in = mediapipe::formats::MatView(src.GetImageFrameSharedPtr().get());
  cv::Mat shifted;
  cv::Mat m = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
  cv::warpAffine(in, shifted, m, in.size(), cv::INTER_LINEAR,
                 cv::BORDER_REPLICATE);
  cv::Mat out = shifted.clone();  // own the buffer
  mediapipe::ImageFrame frame(mediapipe::ImageFormat::SRGB, out.cols, out.rows,
                              out.step, out.data, [out](uint8_t[]) {});
  return Image(std::make_shared<mediapipe::ImageFrame>(std::move(frame)));
}
```

Add the test:

```cpp
// Tiled VIDEO mode with the BoxTracker: pan boats.jpg a few px/frame so the
// optical-flow tracker has real motion to follow, and assert boat detections
// persist across frames (temporal recall), not just on the first frame.
TEST(YoloObjectDetectorTest, TiledVideoTracksBoatsWhilePanning) {
  const std::string model_path = ModelPath();
  const std::string image_path = JoinPath("./", kTestDataDirectory, kBoatsImage);
  if (!mediapipe::file::Exists(model_path).ok() ||
      !mediapipe::file::Exists(image_path).ok()) {
    GTEST_SKIP() << "YOLO model or boats.jpg fixture not available.";
  }

  MP_ASSERT_OK_AND_ASSIGN(Image base_image, DecodeImageFromFile(image_path));

  auto options = std::make_unique<YoloObjectDetectorOptions>();
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::VIDEO;
  options->max_results = 10;
  options->num_classes = 80;
  options->score_threshold = 0.09f;
  options->iou_threshold = 0.45f;
  options->tiling.tile_cols = 2;
  options->tiling.tile_overlap_fraction = 0.2f;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));

  // Count boat (COCO index 8) detections over a short panning sequence.
  int frames_with_boat = 0;
  const int kFrames = 8;
  for (int i = 0; i < kFrames; ++i) {
    Image frame = TranslateImage(base_image, /*dx=*/2 * i, /*dy=*/0);
    MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result,
                            detector->DetectForVideo(frame, /*timestamp_ms=*/i));
    for (const auto& d : result.detections) {
      ASSERT_EQ(d.categories.size(), 1u);
      if (d.categories[0].index == 8) {
        ++frames_with_boat;
        break;
      }
    }
  }
  MP_ASSERT_OK(detector->Close());

  // With tracking, boats should be present on a majority of frames, not just
  // the occasional frame the single-shot tiled detector happens to fire on.
  EXPECT_GE(frames_with_boat, kFrames / 2)
      << "expected boats tracked across most panning frames";
}
```

- [ ] **Step 2: Run the test to verify it fails (red) before the deps exist**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_filter=*TiledVideoTracksBoatsWhilePanning*`
Expected: build failure (missing opencv/image_frame_opencv deps) — fix in Step 3. (If fixtures are absent the test would otherwise GTEST_SKIP; the build failure is the red signal here.)

- [ ] **Step 3: Add the test deps**

In `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`, in `cc_test(name = "yolo_object_detector_test")` deps, add:

```python
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
```

- [ ] **Step 4: Run the test to verify it passes (or skips if fixtures absent)**

Run: `bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_filter=*TiledVideoTracksBoatsWhilePanning*`
Expected: PASS if `yolov8n.tflite` + `boats.jpg` are present; otherwise GTEST_SKIP. If it fails on the boat-count assertion, the panning may be too weak/strong for the flow — adjust `dx` per frame (try 3–4 px) before changing the assertion threshold.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc \
        mediapipe/tasks/cc/vision/yolo_object_detector/BUILD
git commit -m "test(tiling): YOLO tiled VIDEO BoxTracker e2e — boats tracked while panning

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Full regression

**Files:** none (verification only).

- [ ] **Step 1: Run the new and adjacent calculator/graph tests**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/calculators/tensor:detection_label_id_codec_calculator_test \
  //mediapipe/calculators/tensor:detections_tick_gate_calculator_test \
  //mediapipe/calculators/tensor:tiled_frame_suppression_calculator_test \
  //mediapipe/graphs/tiled_detection:tiled_detection_graphs_test \
  //mediapipe/graphs/tiled_detection:tiled_tracking_graphs_test
```
Expected: all PASS.

- [ ] **Step 2: Run both detector Tasks tests (IMAGE path regression + new VIDEO e2e)**

Run:
```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test \
  //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test
```
Expected: all PASS or GTEST_SKIP (fixtures gated). The OBB test must be unaffected (this sub-project does not touch the OBB path).

- [ ] **Step 3: Confirm the working tree is clean and the commit chain is intact**

Run: `git status --short && git log --oneline -8`
Expected: clean tree; the 7 feature/test commits from Tasks 1–7 on top of the two spec commits.

---

## Self-Review notes (author checklist, completed)

- **Spec coverage:** codec (Task 1) ↔ spec §1; tick gate (Tasks 2,4) ↔ spec §2 / [P1#2]; gap-fill (Task 3) ↔ spec §3b / [P1#1]; `TiledTrackingGraph` (Task 4) ↔ spec §2; `TiledBoxTrackMergeGraph` (Task 5) ↔ spec §3; YOLO `use_stream_mode` branch (Task 6) ↔ spec §4; DECODE drop (Task 1) ↔ spec [P2]; public VIDEO e2e (Task 7) ↔ spec testing §4; regression (Task 8) ↔ spec testing. Wrapper (spec §5) needs no change — covered by the Task 6 build + Task 7 e2e exercising `DetectForVideo`.
- **Type consistency:** `TRACKER_DETECTIONS`/`DETECTIONS`/`TICK`/`DATA` tags and `std::vector<Detection>` payloads are consistent across calculators and subgraphs; `ObjectTrackingSubgraphCpu` IO tags are `VIDEO`/`DETECTIONS`→`DETECTIONS` (verified). Options reuse: `TiledBoxTrackMergeGraph` shares `TiledBoxMergeGraphOptions`; suppression option is field 4 `tracker_is_gap_fill_only`; codec ext is `471230016`.
- **Known residual risk:** the tick gate's no-stall behavior depends on the tracker chain advancing timestamp bounds on gap frames — Task 5 Step 5 is the explicit arbiter (a hang there means revisiting the gate's stream-handler config).
