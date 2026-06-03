# Video Tile Scheduler (Phase 4 / M9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `VideoTileSchedulerCalculator` — a CPU, per-frame video-mode scheduler that uses `FlowPackagerCalculator`'s `TrackingData` (motion) plus the previous frame's final detections to decide DETECT vs SKIP, prioritize/cap the tile list, and on SKIP propagate the prior axis-aligned `Detection`s through the single global NMS.

**Architecture:** A new api2 calculator emits scheduled tile **rects** (consumed by the shipped `TileSpecToTilePlanCalculator`) plus a `PROPAGATED_DETECTIONS` stream. Decisions follow a deterministic 4-rule precedence; `TrackingData` is decoded with the existing `MotionVectorFrameFromTrackingData` and is optional (cadence/confidence fallback). The decision logic is built up incrementally: image-mode parity first, then cadence/confidence/propagation, then prioritization, then motion. A final integration test wires `PreviousLoopbackCalculator` + `ConcatenateDetectionVectorCalculator` + one NMS.

**Tech Stack:** C++20, MediaPipe api2, `mediapipe/util/tracking` (`MotionVectorFrameFromTrackingData`), `Detection`/`NormalizedRect` protos, `PreviousLoopbackCalculator`, `ConcatenateDetectionVectorCalculator`, `NonMaxSuppressionCalculator`, Bazel (`--define MEDIAPIPE_DISABLE_GPU=1`), GoogleTest.

**Spec:** `docs/superpowers/specs/2026-06-03-video-tile-scheduler-design.md`.

---

## Reference facts (verified against the codebase)

- New calculator lives in `mediapipe/calculators/tensor/` with the tiling family (it feeds `TileSpecToTilePlanCalculator`). It depends on `//mediapipe/util/tracking:tracking` for decoding.
- `MotionVectorFrameFromTrackingData(const TrackingData&, MotionVectorFrame*)` — `mediapipe/util/tracking/tracking.h:233`, lib `//mediapipe/util/tracking:tracking`. Yields `MotionVectorFrame`:
  - `std::vector<MotionVector> motion_vectors` — each `MotionVector` has `Vector2_f pos` (normalized, longest-side-normalized domain), `Vector2_f object` (foreground motion), `Vector2_f background` (camera motion), `int track_id`; helpers `Location()`, `Motion()`.
  - `bool valid_background_model`, `bool is_duplicated`, `bool is_chunk_boundary`, `float aspect_ratio` (w/h).
  - `Vector2_f` has `.x()`, `.y()`; magnitude via `.Norm()` (if `.Norm()` is unavailable, use `std::hypot(v.x(), v.y())` — confirm during Task 4).
- `TrackingData` proto: `//mediapipe/util/tracking:flow_packager_cc_proto`, header `mediapipe/util/tracking/flow_packager.pb.h`.
- `NormalizedRect` (`mediapipe/framework/formats/rect.proto`, `//mediapipe/framework/formats:rect_cc_proto`): float `x_center`, `y_center`, `width`, `height`, `rotation`. Tile bounds: `x0 = x_center - width/2`, `y0 = y_center - height/2`.
- `Detection` (`mediapipe/framework/formats/detection.proto`, `//mediapipe/framework/formats:detection_cc_proto`): `repeated float score`, `optional LocationData location_data`. `location_data().relative_bounding_box()` → `xmin`, `ymin`, `width`, `height` (normalized). Prior-detection center: `cx = xmin + width/2`, `cy = ymin + height/2`. "Prior max score" = `max over priors of (score_size()>0 ? score(0) : 0)`.
- `PreviousLoopbackCalculator` (`mediapipe/calculators/core/`): tags `MAIN`, `LOOP`, `PREV_LOOP`; on the first MAIN packet, `PREV_LOOP` is an empty/timestamp-bound packet ⇒ the scheduler must treat an empty `PRIOR_DETECTIONS` as a cache miss.
- `ConcatenateDetectionVectorCalculator` = `ConcatenateVectorCalculator<::mediapipe::Detection>` (`mediapipe/calculators/core/concatenate_detection_vector_calculator.cc`): N input streams of `std::vector<Detection>` → one concatenated `std::vector<Detection>`.
- `NonMaxSuppressionCalculator` (`mediapipe/calculators/util/non_max_suppression_calculator.cc`): consumes `std::vector<Detection>`, emits a `std::vector<Detection>`. `typedef std::vector<Detection> Detections;`.
- Calculator-options extension id `471230006` is unused (verified). Tiling protos use `471230004`/`471230005`.
- BUILD pattern: see the shipped `tile_spec_to_tile_plan_calculator_proto` / `_calculator` / `_calculator_test` targets in `mediapipe/calculators/tensor/BUILD` (~lines 2506–2545) — mirror them.

## Cross-cutting conventions
- Build/test: `bazel {build,test} -c opt --define MEDIAPIPE_DISABLE_GPU=1 <target> --test_output=errors`.
- In-editor clang errors are FALSE POSITIVES; only bazel is authoritative.
- All options default so that **defaults + no TRACKING = image mode** (every frame DETECT, all tiles, empty propagation).
- Calculator is api2, `namespace mediapipe::api2`, default timestamp offset (NO `TimestampChange::Arbitrary`) — both outputs emitted every frame at the input timestamp.
- Each task commits separately; co-author trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Branch `dev` (commit on `dev`, do NOT branch).

## File Structure
- **Create** `mediapipe/calculators/tensor/video_tile_scheduler_calculator.proto` — `VideoTileSchedulerCalculatorOptions`.
- **Create** `mediapipe/calculators/tensor/video_tile_scheduler_calculator.cc` — the calculator: contract, decision precedence, prioritization, propagation, motion decode.
- **Create** `mediapipe/calculators/tensor/video_tile_scheduler_calculator_test.cc` — unit tests (`CalculatorRunner`).
- **Create** `mediapipe/calculators/tensor/video_tile_scheduler_pipeline_test.cc` — loopback + concat + single-NMS integration test.
- **Modify** `mediapipe/calculators/tensor/BUILD` — proto, calculator, two test targets.

---

### Task 1: Proto + calculator skeleton (image-mode default)

**Files:** Create the `.proto`, `.cc`, `_test.cc`; modify `BUILD`.

Establishes a working calculator that, with default options and no decision logic yet, always DETECTs (emits all base tiles) and emits empty `PROPAGATED_DETECTIONS` — image-mode parity.

- [ ] **Step 1: Write the proto** `video_tile_scheduler_calculator.proto` (full Apache-2.0 header):
```proto
syntax = "proto2";

package mediapipe;

import "mediapipe/framework/calculator.proto";

message VideoTileSchedulerCalculatorOptions {
  extend mediapipe.CalculatorOptions {
    optional VideoTileSchedulerCalculatorOptions ext = 471230006;
  }
  // Run a full DETECT at least every N frames (staleness bound). 1 = always
  // detect (image-mode default). 0 = never on cadence alone.
  optional int32 detect_every_n_frames = 1 [default = 1];
  // Force DETECT when the prior frame's strongest score is below this. 0 = off.
  optional float min_confidence = 2 [default = 0.0];
  // Cap scheduled tiles on a DETECT frame (keep top-K by priority). 0 = no cap.
  optional int32 max_scheduled_tiles = 3 [default = 0];
  // Force DETECT when aggregate foreground motion energy exceeds this. 0 = off.
  optional float motion_refresh_threshold = 4 [default = 0.0];
  // Allow SKIP (override cadence) when foreground motion is below this. 0 = off.
  optional float motion_skip_threshold = 5 [default = 0.0];
  // Force DETECT when decoded feature count is below this. 0 = off.
  optional int32 min_features = 6 [default = 0];
}
```

- [ ] **Step 2: Write the failing test** `video_tile_scheduler_calculator_test.cc` (full Apache header) — defaults ⇒ always DETECT all tiles, empty propagation:
```cpp
#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

NormalizedRect Rect(float xc, float yc, float w, float h) {
  NormalizedRect r;
  r.set_x_center(xc); r.set_y_center(yc); r.set_width(w); r.set_height(h);
  return r;
}

// Default options: every frame is a DETECT of all tiles, no propagation.
TEST(VideoTileSchedulerTest, DefaultsAlwaysDetectAllTiles) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "VideoTileSchedulerCalculator"
    input_stream: "TILES:tiles"
    input_stream: "PRIOR_DETECTIONS:priors"
    output_stream: "TILES:sched"
    output_stream: "PROPAGATED_DETECTIONS:prop"
  )pb"));
  auto tiles = std::make_unique<std::vector<NormalizedRect>>();
  tiles->push_back(Rect(.25, .5, .5, 1.0));
  tiles->push_back(Rect(.75, .5, .5, 1.0));
  runner.MutableInputs()->Tag("TILES").packets.push_back(
      Adopt(tiles.release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("PRIOR_DETECTIONS").packets.push_back(
      Adopt(new std::vector<Detection>()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& sched = runner.Outputs().Tag("TILES").packets[0]
                          .Get<std::vector<NormalizedRect>>();
  const auto& prop = runner.Outputs().Tag("PROPAGATED_DETECTIONS").packets[0]
                         .Get<std::vector<Detection>>();
  EXPECT_EQ(sched.size(), 2);   // all tiles emitted
  EXPECT_EQ(prop.size(), 0);    // nothing propagated on a DETECT frame
}

}  // namespace
}  // namespace mediapipe
```

- [ ] **Step 3: Write the calculator** `video_tile_scheduler_calculator.cc` (full Apache header). Image-mode-only logic for now (always DETECT all tiles):
```cpp
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/video_tile_scheduler_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/tracking/flow_packager.pb.h"

namespace mediapipe {
namespace api2 {

// Per-frame video-mode tile scheduler. See
// docs/superpowers/specs/2026-06-03-video-tile-scheduler-design.md.
class VideoTileSchedulerCalculator : public Node {
 public:
  static constexpr Input<std::vector<NormalizedRect>> kInTiles{"TILES"};
  static constexpr Input<std::vector<Detection>> kInPriorDets{
      "PRIOR_DETECTIONS"};
  static constexpr Input<TrackingData>::Optional kInTracking{"TRACKING"};
  static constexpr Output<std::vector<NormalizedRect>> kOutTiles{"TILES"};
  static constexpr Output<std::vector<Detection>> kOutProp{
      "PROPAGATED_DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInTiles, kInPriorDets, kInTracking, kOutTiles,
                          kOutProp);

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<mediapipe::VideoTileSchedulerCalculatorOptions>();
    RET_CHECK_GE(options_.detect_every_n_frames(), 0);
    RET_CHECK_GE(options_.max_scheduled_tiles(), 0);
    RET_CHECK_GE(options_.min_features(), 0);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const std::vector<NormalizedRect>& base = *kInTiles(cc);
    // Image-mode only for now: always DETECT, emit all base tiles, no prop.
    kOutTiles(cc).Send(std::vector<NormalizedRect>(base));
    kOutProp(cc).Send(std::vector<Detection>{});
    ++frame_index_;
    return absl::OkStatus();
  }

 private:
  mediapipe::VideoTileSchedulerCalculatorOptions options_;
  int frame_index_ = 0;
};

MEDIAPIPE_REGISTER_NODE(VideoTileSchedulerCalculator);

}  // namespace api2
}  // namespace mediapipe
```

- [ ] **Step 4: BUILD** — add to `mediapipe/calculators/tensor/BUILD` (mirror the `tile_spec_to_tile_plan_calculator` targets):
```python
mediapipe_proto_library(
    name = "video_tile_scheduler_calculator_proto",
    srcs = ["video_tile_scheduler_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "video_tile_scheduler_calculator",
    srcs = ["video_tile_scheduler_calculator.cc"],
    deps = [
        ":video_tile_scheduler_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util/tracking:tracking",
        "//mediapipe/util/tracking:flow_packager_cc_proto",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "video_tile_scheduler_calculator_test",
    srcs = ["video_tile_scheduler_calculator_test.cc"],
    size = "small",
    deps = [
        ":video_tile_scheduler_calculator",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
        "//mediapipe/util/tracking:flow_packager_cc_proto",
    ],
)
```
> Confirm `//mediapipe/util/tracking:tracking` and `:flow_packager_cc_proto` are the right labels (`grep -n 'name = "tracking"\|flow_packager' mediapipe/util/tracking/BUILD`). The proto's `cc_proto` is named `<proto_name>_cc_proto` by `mediapipe_proto_library`.

- [ ] **Step 5: Build + test, commit**
Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:video_tile_scheduler_calculator_test --test_output=all` → PASS.
```bash
git add mediapipe/calculators/tensor/video_tile_scheduler_calculator.proto mediapipe/calculators/tensor/video_tile_scheduler_calculator.cc mediapipe/calculators/tensor/video_tile_scheduler_calculator_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'feat(video-scheduler): VideoTileSchedulerCalculator skeleton (image-mode default)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: Cadence + cache-miss + confidence-drop + SKIP propagation (no TRACKING)

**Files:** Modify `.cc` + `_test.cc`.

Implements decision rules 1 (first/cache-miss/confidence), 3 (cadence), 4 (skip), and SKIP-frame propagation. No TRACKING yet (rule-2 motion skip and the motion parts of rule 1 come in Task 4).

- [ ] **Step 1: Add failing tests** to `..._test.cc`. Add a helper to set options + a Detection builder:
```cpp
Detection Det(float score, float xmin, float ymin, float w, float h) {
  Detection d;
  d.add_score(score);
  auto* bb = d.mutable_location_data()->mutable_relative_bounding_box();
  d.mutable_location_data()->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  bb->set_xmin(xmin); bb->set_ymin(ymin); bb->set_width(w); bb->set_height(h);
  return d;
}

CalculatorRunner MakeRunner(const std::string& options_pb) {
  return CalculatorRunner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      absl::StrCat(R"pb(
        calculator: "VideoTileSchedulerCalculator"
        input_stream: "TILES:tiles"
        input_stream: "PRIOR_DETECTIONS:priors"
        output_stream: "TILES:sched"
        output_stream: "PROPAGATED_DETECTIONS:prop"
        options { [mediapipe.VideoTileSchedulerCalculatorOptions.ext] { )pb",
        options_pb, " } }")));
}
```
Add tests (feed N frames at increasing timestamps through one `runner.Run()`):
```cpp
// detect_every_n_frames=3 -> DETECT on frames 0,3; SKIP on 1,2 (empty tiles +
// propagated priors).
TEST(VideoTileSchedulerTest, CadenceSkipsBetweenDetects) { /* feed 4 frames;
  frame 0 DETECT(all tiles, empty prop); frames 1,2 SKIP(empty tiles, prop ==
  priors fed that frame); frame 3 DETECT. Assert sched/prop sizes per frame. */ }

// Empty priors on an off-cadence frame -> DETECT (cache miss).
TEST(VideoTileSchedulerTest, CacheMissForcesDetect) { /* detect_every_n_frames:5,
  frame 1 with empty priors -> sched == all tiles, prop empty. */ }

// Prior max score below min_confidence on an off-cadence frame -> DETECT.
TEST(VideoTileSchedulerTest, ConfidenceDropForcesDetect) { /* min_confidence:0.5
  detect_every_n_frames:5; frame 1 priors=[Det(0.2,...)] -> DETECT. */ }
```
(For SKIP frames the test feeds non-empty `priors` and asserts `prop` equals them and `sched` is empty.)

- [ ] **Step 2: Implement the decision** in `Process()`. Replace the image-mode body:
```cpp
  absl::Status Process(CalculatorContext* cc) override {
    const std::vector<NormalizedRect>& base = *kInTiles(cc);
    static const std::vector<Detection> kEmptyDets;
    const std::vector<Detection>& priors =
        kInPriorDets(cc).IsEmpty() ? kEmptyDets : *kInPriorDets(cc);

    const bool detect = DecideDetect(priors);
    if (detect) {
      kOutTiles(cc).Send(std::vector<NormalizedRect>(base));
      kOutProp(cc).Send(std::vector<Detection>{});
    } else {
      kOutTiles(cc).Send(std::vector<NormalizedRect>{});       // no inference
      kOutProp(cc).Send(std::vector<Detection>(priors));        // hold
    }
    ++frame_index_;
    return absl::OkStatus();
  }

 private:
  bool DecideDetect(const std::vector<Detection>& priors) const {
    // Rule 1: hard refresh.
    if (frame_index_ == 0) return true;                 // first frame
    if (priors.empty()) return true;                    // cache miss
    if (options_.min_confidence() > 0.0f &&
        MaxScore(priors) < options_.min_confidence()) {
      return true;                                      // confidence drop
    }
    // Rule 2 (motion skip) added in Task 4.
    // Rule 3: cadence/staleness.
    if (options_.detect_every_n_frames() > 0 &&
        frame_index_ % options_.detect_every_n_frames() == 0) {
      return true;
    }
    // Rule 4.
    return false;
  }

  static float MaxScore(const std::vector<Detection>& dets) {
    float m = 0.0f;
    for (const Detection& d : dets) {
      if (d.score_size() > 0) m = std::max(m, d.score(0));
    }
    return m;
  }
```
Add includes `#include <algorithm>` and ensure `absl/strings/str_cat.h` is available to the test.

- [ ] **Step 3: Build + test, commit.**
Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:video_tile_scheduler_calculator_test --test_output=all`
```bash
git add mediapipe/calculators/tensor/video_tile_scheduler_calculator.cc mediapipe/calculators/tensor/video_tile_scheduler_calculator_test.cc
git commit -m "$(printf 'feat(video-scheduler): cadence/cache-miss/confidence DETECT + SKIP propagation\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 3: Tile prioritization + `max_scheduled_tiles` cap (prior-detection fallback)

**Files:** Modify `.cc` + `_test.cc`.

On a DETECT frame, when `max_scheduled_tiles > 0` and tile count exceeds it, keep the top-K by priority. This task implements the **prior-detection-overlap** priority (the no-TRACKING fallback); motion priority comes in Task 4.

- [ ] **Step 1: Add failing test** — 4 tiles, `max_scheduled_tiles=2`, priors whose centers fall in 2 specific tiles ⇒ exactly those 2 tiles kept (deterministic order = ascending original index):
```cpp
TEST(VideoTileSchedulerTest, CapKeepsTilesOverlappingPriorDetections) {
  // tiles 0..3 across x; priors centered in tiles 1 and 3.
  // detect_every_n_frames=1 (DETECT), max_scheduled_tiles=2.
  // Expect sched == {tile1, tile3} (by original index order).
}
TEST(VideoTileSchedulerTest, NoCapEmitsAllTiles) {
  // max_scheduled_tiles=0 -> all tiles regardless of priors.
}
```

- [ ] **Step 2: Implement** prioritization. Add a `ScheduleTiles` helper and call it on the DETECT branch:
```cpp
  std::vector<NormalizedRect> ScheduleTiles(
      const std::vector<NormalizedRect>& base,
      const std::vector<Detection>& priors) const {
    const int cap = options_.max_scheduled_tiles();
    if (cap <= 0 || static_cast<int>(base.size()) <= cap) {
      return base;  // full refresh, original order preserved
    }
    // Priority = number of prior-detection centers inside the tile.
    std::vector<std::pair<int, int>> scored;  // (-priority, index) for sort
    scored.reserve(base.size());
    for (int i = 0; i < static_cast<int>(base.size()); ++i) {
      int hits = 0;
      for (const Detection& d : priors) {
        if (!d.has_location_data() ||
            !d.location_data().has_relative_bounding_box()) continue;
        const auto& b = d.location_data().relative_bounding_box();
        const float cx = b.xmin() + b.width() / 2.0f;
        const float cy = b.ymin() + b.height() / 2.0f;
        if (PointInTile(cx, cy, base[i])) ++hits;
      }
      scored.push_back({-hits, i});
    }
    std::stable_sort(scored.begin(), scored.end());  // high priority, then index
    std::vector<NormalizedRect> out;
    out.reserve(cap);
    std::vector<int> kept_idx;
    for (int k = 0; k < cap; ++k) kept_idx.push_back(scored[k].second);
    std::sort(kept_idx.begin(), kept_idx.end());     // restore original order
    for (int idx : kept_idx) out.push_back(base[idx]);
    return out;
  }

  static bool PointInTile(float x, float y, const NormalizedRect& t) {
    const float x0 = t.x_center() - t.width() / 2.0f;
    const float y0 = t.y_center() - t.height() / 2.0f;
    return x >= x0 && x <= x0 + t.width() && y >= y0 && y <= y0 + t.height();
  }
```
In the DETECT branch replace `std::vector<NormalizedRect>(base)` with `ScheduleTiles(base, priors)`.

- [ ] **Step 3: Build + test, commit.**
```bash
git add mediapipe/calculators/tensor/video_tile_scheduler_calculator.cc mediapipe/calculators/tensor/video_tile_scheduler_calculator_test.cc
git commit -m "$(printf 'feat(video-scheduler): tile prioritization + max_scheduled_tiles cap (prior-detection fallback)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 4: TRACKING motion — decode, triggers, motion-priority, aspect mapping

**Files:** Modify `.cc` + `_test.cc`.

Wire `TrackingData`: rule-1 motion refresh triggers, rule-2 motion skip, and motion-energy tile priority (overriding the prior-detection fallback when features land in a tile).

- [ ] **Step 1: Confirm the decode API** by reading `mediapipe/util/tracking/tracking.h` (`MotionVectorFrameFromTrackingData`, `MotionVector`, `MotionVectorFrame`) and how positions are normalized in `tracking.cc`. Verify `Vector2_f` exposes `.Norm()` (else use `std::hypot`). Note the exact `MotionVector` accessor for foreground motion (`object`).

- [ ] **Step 2: Add failing tests** with hand-built `TrackingData`. Build a small helper that constructs a `TrackingData` whose decoded `MotionVectorFrame` has known features — OR (simpler and robust to the CSC encoding) test the *decision/priority helpers directly* by refactoring them to accept a decoded `MotionVectorFrame` (see Step 3) and feeding a hand-built `MotionVectorFrame`. Tests:
```cpp
// is_duplicated -> SKIP on a non-first frame even when cadence says detect.
// !valid_background_model -> DETECT.
// is_chunk_boundary -> DETECT.
// mean object motion > motion_refresh_threshold -> DETECT.
// mean object motion < motion_skip_threshold -> SKIP overriding cadence.
// feature_count < min_features -> DETECT.
// prioritization: features (foreground motion) in tiles 0 and 2 -> those kept
//   under max_scheduled_tiles=2.
// aspect mapping: a feature at domain-center maps to frame (0.5,0.5)
//   for landscape (aspect 16/9) and portrait (aspect 9/16).
```
Prefer driving these through the public calculator with a `TRACKING` input built via a tiny `MakeTrackingData(features, flags, aspect)` helper that you implement by populating `TrackingData` and round-tripping through `MotionVectorFrameFromTrackingData` in the test to assert your construction decodes as intended. If hand-encoding `motion_data` CSC is impractical, split the motion logic into free functions over `MotionVectorFrame` and unit-test those directly (this keeps the calculator thin and the math testable without the proto encoding).

- [ ] **Step 3: Implement.** Add includes:
```cpp
#include <cmath>
#include "mediapipe/util/tracking/tracking.h"
```
Decode once per frame when TRACKING is connected and present, and extend the decision + scheduling. Decision becomes:
```cpp
  bool DecideDetect(const std::vector<Detection>& priors,
                    const MotionVectorFrame* mvf) const {
    if (frame_index_ == 0) return true;
    if (priors.empty()) return true;
    if (options_.min_confidence() > 0.0f &&
        MaxScore(priors) < options_.min_confidence()) return true;
    if (mvf != nullptr) {
      if (!mvf->valid_background_model || mvf->is_chunk_boundary) return true;
      const float e = MeanForegroundMotion(*mvf);
      if (options_.motion_refresh_threshold() > 0.0f &&
          e > options_.motion_refresh_threshold()) return true;
      if (options_.min_features() > 0 &&
          static_cast<int>(mvf->motion_vectors.size()) <
              options_.min_features()) return true;
      // Rule 2: motion skip overrides cadence.
      if (mvf->is_duplicated ||
          (options_.motion_skip_threshold() > 0.0f &&
           e < options_.motion_skip_threshold())) return false;
    }
    if (options_.detect_every_n_frames() > 0 &&
        frame_index_ % options_.detect_every_n_frames() == 0) return true;
    return false;
  }

  static float Magnitude(const Vector2_f& v) { return std::hypot(v.x(), v.y()); }
  static float MeanForegroundMotion(const MotionVectorFrame& mvf) {
    if (mvf.motion_vectors.empty()) return 0.0f;
    float s = 0.0f;
    for (const auto& m : mvf.motion_vectors) s += Magnitude(m.object);
    return s / mvf.motion_vectors.size();
  }
  // longest-side-normalized -> frame-normalized [0,1]^2.
  static void FeatureFramePos(const Vector2_f& pos, float aspect,
                              float* fx, float* fy) {
    if (aspect >= 1.0f) { *fx = pos.x(); *fy = pos.y() * aspect; }
    else { *fx = pos.x() / aspect; *fy = pos.y(); }
  }
```
Extend `ScheduleTiles` to accept `const MotionVectorFrame* mvf`: when `mvf != nullptr`, tile priority = Σ `Magnitude(object)` over features whose `FeatureFramePos` lies in the tile; if that sum is 0 for all tiles (no features land anywhere), fall back to the prior-detection-center count. Use a `float` priority key (`std::pair<float,int>` with negated priority) instead of the int version. In `Process()`, decode:
```cpp
    MotionVectorFrame mvf;
    const MotionVectorFrame* mvf_ptr = nullptr;
    if (kInTracking(cc).IsConnected() && !kInTracking(cc).IsEmpty()) {
      MotionVectorFrameFromTrackingData(*kInTracking(cc), &mvf);
      mvf_ptr = &mvf;
    }
```
(`MotionVectorFrame`, `MotionVector`, `Vector2_f`, `MotionVectorFrameFromTrackingData` are in `namespace mediapipe`; the class is in `mediapipe::api2`, so reference them unqualified.)

- [ ] **Step 4: Build + test, commit.**
```bash
git add mediapipe/calculators/tensor/video_tile_scheduler_calculator.cc mediapipe/calculators/tensor/video_tile_scheduler_calculator_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'feat(video-scheduler): TrackingData motion triggers + motion tile-priority\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```
(If the test BUILD target needs `//mediapipe/util/tracking:tracking` for `MotionVectorFrame` in tests, add it.)

---

### Task 5: Integration test — loopback + concat + single global NMS

**Files:** Create `video_tile_scheduler_pipeline_test.cc`; modify `BUILD`.

Prove the scheduler composes into a video graph that emits one detection result per frame through exactly one NMS, with SKIP frames carried by propagation. The tiled-detect path is stubbed (canned merged detections fed at DETECT timestamps) so this stays CPU-only and inference-free.

- [ ] **Step 1: Write the integration test.** Build a `CalculatorGraph` wiring:
  - `PreviousLoopbackCalculator` (`MAIN:tick`, `LOOP:final_dets`, `PREV_LOOP:priors`).
  - `VideoTileSchedulerCalculator` (`TILES:base_tiles`, `PRIOR_DETECTIONS:priors`; outputs `TILES:sched`, `PROPAGATED_DETECTIONS:prop`).
  - A stub for the detect path: feed `merged_dets` (a `std::vector<Detection>`) directly at the same timestamps (no real inference); on SKIP frames feed an empty `merged_dets`.
  - `ConcatenateDetectionVectorCalculator` (inputs `prop` + `merged_dets`) → `combined`.
  - `NonMaxSuppressionCalculator` (`combined`) → `final_dets` (also looped back).
  Drive 3 frames: frame 0 DETECT (stub merged has 1 det), frame 1 SKIP (cadence with `detect_every_n_frames=2`; merged empty; prop carries frame-0 result), frame 2 DETECT. Assert: one `final_dets` packet per frame at its timestamp; frame 1's result equals the propagated prior; each result passed through NMS once. Confirm the exact `PreviousLoopbackCalculator`/`NonMaxSuppressionCalculator`/`ConcatenateDetectionVectorCalculator` stream tag names and any required options by reading their sources first.

- [ ] **Step 2: BUILD** — add:
```python
cc_test(
    name = "video_tile_scheduler_pipeline_test",
    srcs = ["video_tile_scheduler_pipeline_test.cc"],
    size = "small",
    deps = [
        ":video_tile_scheduler_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/core:concatenate_detection_vector_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
    ],
)
```
(Confirm those calculator target labels exist; adjust to the actual names.)

- [ ] **Step 3: Build + test, commit.**
Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:video_tile_scheduler_pipeline_test --test_output=all`
```bash
git add mediapipe/calculators/tensor/video_tile_scheduler_pipeline_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "$(printf 'test(video-scheduler): loopback + concat + single-NMS integration\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Self-review checklist (run before final review)
- Defaults + no TRACKING ⇒ every frame DETECT, all tiles, empty propagation (Task 1 test).
- Decision precedence matches the spec: hard-refresh → motion-skip → cadence → skip (Tasks 2+4).
- SKIP emits empty `TILES` + propagated priors; DETECT emits scheduled tiles + empty propagation.
- `max_scheduled_tiles` cap keeps top-K deterministically (stable order, original-index tie-break).
- Motion priority uses foreground (`object`) motion; falls back to prior-detection containment.
- Aspect mapping verified (center feature → (0.5,0.5)) for landscape and portrait.
- TRACKING optional: unconnected ⇒ cadence/confidence path; connected ⇒ motion path.
- One global NMS per frame; no track IDs; no cross-frame batches.

## Final verification
- [ ] `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:video_tile_scheduler_calculator_test //mediapipe/calculators/tensor:video_tile_scheduler_pipeline_test --test_output=errors` — all pass.
- [ ] `git status` clean.

## Done criteria
- `VideoTileSchedulerCalculator` implements the deterministic DETECT/SKIP precedence, prioritization+cap, and SKIP propagation.
- Motion-driven via decoded `TrackingData`; graceful cadence/confidence fallback when `TRACKING` unconnected.
- Defaults reproduce image mode. Integration test proves single-global-NMS + loopback + propagation.
- All tests pass under `--define MEDIAPIPE_DISABLE_GPU=1`; no track IDs, no cross-frame batches.
