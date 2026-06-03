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

// End-to-end integration test: PreviousLoopbackCalculator +
// VideoTileSchedulerCalculator + GatedDetectionEmitterCalculator (stub) +
// TiledFrameSuppressionCalculator wired as a cyclic graph.
//
// Proves:
//  - Exactly one final-detection packet per input timestamp (4 total).
//  - SKIP frames (FLAG_DUPLICATED + priors present) → sched empty, REFRESH=false,
//    final detections come from the tracker path (not from stale scheduler priors).
//  - DETECT frames → fresh detections survive global NMS through suppression.
//  - No PROPAGATED_DETECTIONS stream: the tracker stub routes through REFRESH.
//  - Detection bounding boxes use RELATIVE_BOUNDING_BOX.
//
// Graph structure (single cyclic graph):
//
//   [tick, base_tiles, tracking] → PreviousLoopbackCalculator
//                                → VideoTileSchedulerCalculator
//                                → GatedDetectionEmitter (detect stub, REFRESH=T)
//                                → GatedDetectionEmitter (tracker stub, REFRESH=F)
//                                → TiledFrameSuppressionCalculator
//                                  ↑_______(loopback: final_dets)____________|
//
// The bypass integration test is a separate smaller test using CalculatorRunner
// to avoid the loopback complexity (per-frame graph vs. single-shot runner).

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/tracking/flow_packager.pb.h"

namespace mediapipe {
namespace {

// ---------------------------------------------------------------------------
// Test-only stub calculators (anonymous namespace — never shipped)
// ---------------------------------------------------------------------------

// GatedDetectionEmitterCalculator
//
// Inputs:
//   REFRESH (bool): the scheduler's REFRESH output.
// Side packet:
//   DETECTIONS (std::vector<Detection>): canned detections to emit.
// Options (via node-level options proto, NOT used here):
//   We pass behaviour via a bool side packet EMIT_WHEN_REFRESH so we avoid
//   a custom proto entirely.
// Side packet:
//   EMIT_WHEN_REFRESH (bool): emit the canned detections when
//     REFRESH == emit_when_refresh; otherwise emit an empty vector.
//
// Output:
//   DETECTIONS (std::vector<Detection>)
//
// NOTE: Using legacy CalculatorBase (not api2) to keep the side-packet
// interface simple without a custom proto.
class GatedDetectionEmitterCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("REFRESH").Set<bool>();
    cc->InputSidePackets().Tag("DETECTIONS").Set<std::vector<Detection>>();
    cc->InputSidePackets().Tag("EMIT_WHEN_REFRESH").Set<bool>();
    cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    emit_when_refresh_ =
        cc->InputSidePackets().Tag("EMIT_WHEN_REFRESH").Get<bool>();
    canned_ =
        cc->InputSidePackets().Tag("DETECTIONS").Get<std::vector<Detection>>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    const bool refresh = cc->Inputs().Tag("REFRESH").Get<bool>();
    if (refresh == emit_when_refresh_) {
      cc->Outputs()
          .Tag("DETECTIONS")
          .AddPacket(MakePacket<std::vector<Detection>>(canned_).At(
              cc->InputTimestamp()));
    } else {
      cc->Outputs()
          .Tag("DETECTIONS")
          .AddPacket(MakePacket<std::vector<Detection>>(
                         std::vector<Detection>{})
                         .At(cc->InputTimestamp()));
    }
    return absl::OkStatus();
  }

 private:
  bool emit_when_refresh_ = true;
  std::vector<Detection> canned_;
};
REGISTER_CALCULATOR(GatedDetectionEmitterCalculator);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

Detection MakeDet(float score, float xmin, float ymin, float w, float h) {
  Detection d;
  d.add_score(score);
  auto* ld = d.mutable_location_data();
  ld->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* rbb = ld->mutable_relative_bounding_box();
  rbb->set_xmin(xmin);
  rbb->set_ymin(ymin);
  rbb->set_width(w);
  rbb->set_height(h);
  return d;
}

NormalizedRect MakeRect(float xc, float yc, float w, float h) {
  NormalizedRect r;
  r.set_x_center(xc);
  r.set_y_center(yc);
  r.set_width(w);
  r.set_height(h);
  return r;
}

// Build a minimal valid TrackingData with the given frame_flags.
// domain_width/height/frame_aspect must be valid for
// MotionVectorFrameFromTrackingData (avoids div-by-zero).
TrackingData MakeTrackingData(int frame_flags) {
  TrackingData td;
  td.set_frame_flags(frame_flags);
  td.set_domain_width(100.0f);
  td.set_domain_height(100.0f);
  td.set_frame_aspect(1.0f);
  return td;
}

// Two non-overlapping detections for the "fresh detect" stub.
std::vector<Detection> FreshDets() {
  return {
      MakeDet(0.9f, 0.10f, 0.10f, 0.15f, 0.15f),
      MakeDet(0.8f, 0.50f, 0.50f, 0.15f, 0.15f),
  };
}

// One tracker detection (on a different region) for the tracker stub.
std::vector<Detection> TrackerDets() {
  return {
      MakeDet(0.75f, 0.70f, 0.70f, 0.15f, 0.15f),
  };
}

// Two base tiles covering the full frame.
std::vector<NormalizedRect> BaseTiles() {
  return {MakeRect(0.25f, 0.5f, 0.5f, 1.0f),
          MakeRect(0.75f, 0.5f, 0.5f, 1.0f)};
}

// Graph config for the full cyclic e2e test.
//
// Scheduler options:
//   refresh_on_uncertain_tracking: false  — disables the uncertain-tracking
//     fallback so that FLAG_DUPLICATED with priors present reliably gives SKIP.
//   skip_on_duplicated: true (default)
//   refresh_on_chunk_boundary: true (default)
//   detect_without_tracking: true (default)
//
// TiledFrameSuppressionCalculator:
//   bypass_single_tile: false (default) — NMS always runs.
//   iou_threshold: 0.3 — tight enough to dedup real overlaps, loose enough for
//     well-separated test boxes.
constexpr char kE2EGraphConfig[] = R"pb(
  input_stream: "tick"
  input_stream: "base_tiles"
  input_stream: "tracking"
  input_side_packet: "fresh_dets"
  input_side_packet: "tracker_dets"
  node {
    calculator: "PreviousLoopbackCalculator"
    input_stream: "MAIN:tick"
    input_stream: "LOOP:final_dets"
    input_stream_info { tag_index: "LOOP" back_edge: true }
    output_stream: "PREV_LOOP:priors"
  }
  node {
    calculator: "VideoTileSchedulerCalculator"
    input_stream: "TILES:base_tiles"
    input_stream: "PRIOR_DETECTIONS:priors"
    input_stream: "TRACKING:tracking"
    output_stream: "TILES:sched"
    output_stream: "REFRESH:refresh"
    options {
      [mediapipe.VideoTileSchedulerCalculatorOptions.ext] {
        refresh_on_uncertain_tracking: false
      }
    }
  }
  node {
    calculator: "GatedDetectionEmitterCalculator"
    input_stream: "REFRESH:refresh"
    input_side_packet: "DETECTIONS:fresh_dets"
    input_side_packet: "EMIT_WHEN_REFRESH:emit_true"
    output_stream: "DETECTIONS:fresh"
  }
  node {
    calculator: "GatedDetectionEmitterCalculator"
    input_stream: "REFRESH:refresh"
    input_side_packet: "DETECTIONS:tracker_dets"
    input_side_packet: "EMIT_WHEN_REFRESH:emit_false"
    output_stream: "DETECTIONS:tracker"
  }
  node {
    calculator: "TiledFrameSuppressionCalculator"
    input_stream: "DETECTIONS:fresh"
    input_stream: "TRACKER_DETECTIONS:tracker"
    output_stream: "DETECTIONS:final_dets"
    options {
      [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
        iou_threshold: 0.3
      }
    }
  }
)pb";

// ---------------------------------------------------------------------------
// Main e2e test: 4 frames, cyclic loopback graph
// ---------------------------------------------------------------------------
TEST(VideoTileSchedulerE2ETest, CyclicLoopbackFourFrames) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(kE2EGraphConfig);

  std::vector<Packet> final_dets_packets;
  std::vector<Packet> sched_packets;
  std::vector<Packet> refresh_packets;

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));

  MP_ASSERT_OK(graph.ObserveOutputStream(
      "final_dets", [&](const Packet& p) -> absl::Status {
        final_dets_packets.push_back(p);
        return absl::OkStatus();
      }));
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "sched", [&](const Packet& p) -> absl::Status {
        sched_packets.push_back(p);
        return absl::OkStatus();
      }));
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "refresh", [&](const Packet& p) -> absl::Status {
        refresh_packets.push_back(p);
        return absl::OkStatus();
      }));

  const std::vector<Detection> kFreshDets = FreshDets();
  const std::vector<Detection> kTrackerDets = TrackerDets();

  // Side packets: canned detection sets + the two EMIT_WHEN_REFRESH booleans.
  MP_ASSERT_OK(graph.StartRun({
      {"fresh_dets",
       MakePacket<std::vector<Detection>>(kFreshDets)},
      {"tracker_dets",
       MakePacket<std::vector<Detection>>(kTrackerDets)},
      {"emit_true", MakePacket<bool>(true)},
      {"emit_false", MakePacket<bool>(false)},
  }));

  // -----------------------------------------------------------------------
  // Frame 0 (ts=0): no priors (first frame), no tracking → DETECT.
  //   priors are absent/empty from loopback → priors_empty=true → DETECT
  //   regardless of tracking.
  // -----------------------------------------------------------------------
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tick", MakePacket<int>(0).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "base_tiles",
      MakePacket<std::vector<NormalizedRect>>(BaseTiles()).At(Timestamp(0))));
  // No tracking for frame 0 — use detect_without_tracking=true default.
  // (We feed an empty TrackingData slot by simply not sending on "tracking".)
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // -----------------------------------------------------------------------
  // Frame 1 (ts=1): FLAG_DUPLICATED, priors exist from frame 0 → SKIP.
  //   skip_on_duplicated=true, priors_empty=false → REFRESH=false.
  // -----------------------------------------------------------------------
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tick", MakePacket<int>(1).At(Timestamp(1))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "base_tiles",
      MakePacket<std::vector<NormalizedRect>>(BaseTiles()).At(Timestamp(1))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tracking",
      MakePacket<TrackingData>(MakeTrackingData(TrackingData::FLAG_DUPLICATED))
          .At(Timestamp(1))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // -----------------------------------------------------------------------
  // Frame 2 (ts=2): FLAG_CHUNK_BOUNDARY → DETECT even with priors.
  //   refresh_on_chunk_boundary=true dominates.
  // -----------------------------------------------------------------------
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tick", MakePacket<int>(2).At(Timestamp(2))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "base_tiles",
      MakePacket<std::vector<NormalizedRect>>(BaseTiles()).At(Timestamp(2))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tracking",
      MakePacket<TrackingData>(
          MakeTrackingData(TrackingData::FLAG_CHUNK_BOUNDARY))
          .At(Timestamp(2))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // -----------------------------------------------------------------------
  // Frame 3 (ts=3): plain (no tracking) → detect_without_tracking=true → DETECT.
  // -----------------------------------------------------------------------
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tick", MakePacket<int>(3).At(Timestamp(3))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "base_tiles",
      MakePacket<std::vector<NormalizedRect>>(BaseTiles()).At(Timestamp(3))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  // -----------------------------------------------------------------------
  // Assertion 1: exactly one final_dets packet per timestamp (4 total).
  // -----------------------------------------------------------------------
  ASSERT_EQ(final_dets_packets.size(), 4u);
  ASSERT_EQ(refresh_packets.size(), 4u);
  ASSERT_EQ(sched_packets.size(), 4u);

  // Timestamps must be 0, 1, 2, 3 in order.
  for (int t = 0; t < 4; ++t) {
    EXPECT_EQ(final_dets_packets[t].Timestamp(), Timestamp(t))
        << "final_dets timestamp mismatch at index " << t;
    EXPECT_EQ(refresh_packets[t].Timestamp(), Timestamp(t))
        << "refresh timestamp mismatch at index " << t;
    EXPECT_EQ(sched_packets[t].Timestamp(), Timestamp(t))
        << "sched timestamp mismatch at index " << t;
  }

  // -----------------------------------------------------------------------
  // Assertion 2: REFRESH values match expected DETECT/SKIP per frame.
  //   Frame 0: DETECT (no priors), Frame 1: SKIP (duplicated + priors),
  //   Frame 2: DETECT (chunk_boundary), Frame 3: DETECT (no tracking).
  // -----------------------------------------------------------------------
  const bool kExpectedRefresh[] = {true, false, true, true};
  for (int t = 0; t < 4; ++t) {
    EXPECT_EQ(refresh_packets[t].Get<bool>(), kExpectedRefresh[t])
        << "REFRESH mismatch at ts=" << t;
  }

  // -----------------------------------------------------------------------
  // Assertion 3: SKIP frame (ts=1): sched is empty.
  // -----------------------------------------------------------------------
  EXPECT_TRUE(
      sched_packets[1].Get<std::vector<NormalizedRect>>().empty())
      << "sched must be empty on SKIP frame (ts=1)";

  // -----------------------------------------------------------------------
  // Assertion 4: DETECT frames (0,2,3): sched is non-empty (all base tiles).
  // -----------------------------------------------------------------------
  for (int t : {0, 2, 3}) {
    EXPECT_FALSE(
        sched_packets[t].Get<std::vector<NormalizedRect>>().empty())
        << "sched must be non-empty on DETECT frame (ts=" << t << ")";
  }

  // -----------------------------------------------------------------------
  // Assertion 5: SKIP frame (ts=1): final_dets comes from the TRACKER path.
  //   The tracker stub emits TrackerDets() when REFRESH=false, and the
  //   TiledFrameSuppressionCalculator passes these through NMS.
  //   Since TrackerDets has one disjoint detection, NMS preserves it.
  //   The fresh stub emits empty on SKIP, so only tracker dets survive.
  // -----------------------------------------------------------------------
  const auto& skip_final =
      final_dets_packets[1].Get<std::vector<Detection>>();
  ASSERT_EQ(skip_final.size(), 1u)
      << "SKIP frame (ts=1): final_dets must have exactly 1 tracker detection";
  // Verify it matches the tracker detection score (0.75).
  EXPECT_NEAR(skip_final[0].score(0), 0.75f, 1e-4f)
      << "SKIP frame final detection must come from tracker stub";

  // -----------------------------------------------------------------------
  // Assertion 6: DETECT frames: final_dets comes from the FRESH path.
  //   The fresh stub emits 2 non-overlapping detections; the tracker stub
  //   emits empty on DETECT. After NMS over 2 non-overlapping boxes → 2 kept.
  // -----------------------------------------------------------------------
  for (int t : {0, 2, 3}) {
    const auto& detect_final =
        final_dets_packets[t].Get<std::vector<Detection>>();
    EXPECT_EQ(detect_final.size(), 2u)
        << "DETECT frame (ts=" << t
        << "): expected 2 fresh detections after NMS";
  }

  // -----------------------------------------------------------------------
  // Assertion 7: All final detections use RELATIVE_BOUNDING_BOX.
  // -----------------------------------------------------------------------
  for (int t = 0; t < 4; ++t) {
    const auto& dets = final_dets_packets[t].Get<std::vector<Detection>>();
    for (const Detection& d : dets) {
      EXPECT_EQ(d.location_data().format(),
                LocationData::RELATIVE_BOUNDING_BOX)
          << "Detection at ts=" << t << " must use RELATIVE_BOUNDING_BOX";
    }
  }
}

// ---------------------------------------------------------------------------
// Focused bypass integration test (CalculatorRunner-based, no loopback).
// ---------------------------------------------------------------------------
// This reuses CalculatorRunner (single-shot) to exercise the bypass contract
// independently of the cyclic graph complexity.

// Include the runner header:
// (CalculatorRunner is already pulled in transitively via gtest_main, but we
//  need to list it in deps.)

// Test A: bypass_single_tile=true, NUM_TILES=1, no tracker, 2 overlapping →
//         both survive (NMS bypassed).
TEST(TiledFrameSuppressionBypassE2ETest, SingleTileNoTrackerBothSurvive) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "TiledFrameSuppressionCalculator"
        input_stream: "DETECTIONS:fresh"
        input_stream: "NUM_TILES:num_tiles"
        output_stream: "DETECTIONS:out"
        options {
          [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
            iou_threshold: 0.5
            bypass_single_tile: true
          }
        }
      )pb"));

  // Two heavily-overlapping boxes (IoU > 0.5).
  std::vector<Detection> fresh = {
      MakeDet(0.9f, 0.0f, 0.0f, 0.6f, 0.6f),
      MakeDet(0.8f, 0.05f, 0.05f, 0.6f, 0.6f),
  };
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(fresh).At(Timestamp(0)));
  runner.MutableInputs()->Tag("NUM_TILES").packets.push_back(
      MakePacket<int>(1).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& out =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<std::vector<Detection>>();
  // Bypass active → NMS skipped → both boxes survive.
  EXPECT_EQ(out.size(), 2u) << "bypass=true with NUM_TILES=1: both boxes must survive";
}

// Test B: NUM_TILES=2 (control) — NMS runs → 1 survives.
TEST(TiledFrameSuppressionBypassE2ETest, MultiTileNmsRunsOneSurvives) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "TiledFrameSuppressionCalculator"
        input_stream: "DETECTIONS:fresh"
        input_stream: "NUM_TILES:num_tiles"
        output_stream: "DETECTIONS:out"
        options {
          [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
            iou_threshold: 0.5
            bypass_single_tile: true
          }
        }
      )pb"));

  std::vector<Detection> fresh = {
      MakeDet(0.9f, 0.0f, 0.0f, 0.6f, 0.6f),
      MakeDet(0.8f, 0.05f, 0.05f, 0.6f, 0.6f),
  };
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(fresh).At(Timestamp(0)));
  runner.MutableInputs()->Tag("NUM_TILES").packets.push_back(
      MakePacket<int>(2).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& out =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<std::vector<Detection>>();
  // NMS runs (NUM_TILES=2) → high IoU → 1 kept.
  EXPECT_EQ(out.size(), 1u) << "NUM_TILES=2: NMS must run, 1 box survives";
}

// Test C: fresh+tracker combined through NMS in an integration-ish setup.
// Ensures the "one global NMS over fresh+tracker" contract holds when both
// paths feed the suppressor.
TEST(TiledFrameSuppressionBypassE2ETest, FreshPlusTrackerOneGlobalNms) {
  CalculatorRunner runner(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "TiledFrameSuppressionCalculator"
        input_stream: "DETECTIONS:fresh"
        input_stream: "TRACKER_DETECTIONS:tracker"
        input_stream: "NUM_TILES:num_tiles"
        output_stream: "DETECTIONS:out"
        options {
          [mediapipe.TiledFrameSuppressionCalculatorOptions.ext] {
            iou_threshold: 0.5
          }
        }
      )pb"));

  // Fresh: 2 non-overlapping boxes. Tracker: 1 box heavily overlapping fresh[0].
  std::vector<Detection> fresh = {
      MakeDet(0.9f, 0.0f, 0.0f, 0.3f, 0.3f),   // box A
      MakeDet(0.7f, 0.6f, 0.6f, 0.3f, 0.3f),   // box B (non-overlapping)
  };
  std::vector<Detection> tracker = {
      MakeDet(0.85f, 0.05f, 0.05f, 0.3f, 0.3f),  // ≈ box A (high IoU with A)
  };
  runner.MutableInputs()->Tag("DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(fresh).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TRACKER_DETECTIONS").packets.push_back(
      MakePacket<std::vector<Detection>>(tracker).At(Timestamp(0)));
  runner.MutableInputs()->Tag("NUM_TILES").packets.push_back(
      MakePacket<int>(2).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& out =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<std::vector<Detection>>();
  // Combined = {A(0.9), tracker(0.85), B(0.7)}.
  // NMS: keep A(0.9); suppress tracker (high IoU with A); keep B(0.7, disjoint).
  // → 2 boxes survive.
  EXPECT_EQ(out.size(), 2u)
      << "fresh+tracker NMS: A kept, tracker suppressed by A, B kept";
}

}  // namespace
}  // namespace mediapipe
