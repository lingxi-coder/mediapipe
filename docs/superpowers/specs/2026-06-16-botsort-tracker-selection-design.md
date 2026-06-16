# BoTSORT Tracker Selection — Design

**Date:** 2026-06-16
**Branch:** `dev` (MediaPipe fork; unmerged, no upstream PR)
**Status:** Approved design — ready for implementation plan

## Goal

In the YOLO tiled livestream path, let the user choose the tracker: keep the
existing **BoxTracker** (optical-flow propagation, the default, fully
backward-compatible) or select **BoTSORT** (tracking-by-detection, motion-only).
The detector's public output contract is unchanged — BoTSORT's persistent track
IDs are used internally but not surfaced on output Detections (that is a
follow-up; see Scope).

## Background — what exists today

Tracking currently lives in exactly one place: the *tiled stream* path of the
YOLO object detector (`yolo_object_detector_graph.cc`, the
`use_stream_mode()` branch). It uses **BoxTracker**, an optical-flow
propagator, wrapped as `ObjectTrackingSubgraphCpu`, and fused back through a
`TRACKER_DETECTIONS` seam:

```
IMAGE ─┬───────────────────────────────────────────────┐
       │                                                 ▼
DETECTIONS ─▶ ENCODE label_id→str ─▶ ObjectTrackingSubgraphCpu (BoxTracker)
                                                         │
                              DECODE str→label_id ◀──────┘
                                      │
                              DetectionsTickGate (TICK = fresh DETECTIONS)
                                      │
                                      ▼
                              TRACKER_DETECTIONS ─▶ merge ─▶ output
```

The seam is assembled by `TiledTrackingGraph`
(`mediapipe/graphs/tiled_detection/tiled_tracking_graph.cc`), instantiated by
`TiledBoxTrackMergeGraph` (`tiled_box_track_merge_graph.cc`), which fuses
fresh + tracker detections with a single global NMS via
`TiledFrameSuppressionCalculator` (`tracker_is_gap_fill_only=true`,
fresh-wins). The ENCODE/DECODE codec exists because BoxTracker carries only
string ids and loses `label_id`.

## Background — BoTSORT-cpp (viplix3/BoTSORT-cpp)

Clean source layout: `botsort/include/*.h` + `botsort/src/*.cpp`. Public API:

```cpp
BoTSORT(tracker_cfg, gmc_cfg, reid_cfg, reid_onnx_model_path);
std::vector<std::shared_ptr<Track>> track(const std::vector<Detection>& dets,
                                          const cv::Mat& frame);
```

It is **tracking-by-detection**: Kalman motion prediction + Hungarian/lapjv
association + global motion compensation (GMC), producing **persistent track
IDs**. The `frame` argument is always required (GMC uses it).

**Important — verified empty-detection behavior:** `track()` returns only the
`is_activated` tracks remaining in `_tracked_tracks`
(`BoTSORT.cpp` "Update output tracks" loop). On a frame with **no** detections,
all previously-tracked tracks go unmatched and are `mark_lost()` → removed from
`_tracked_tracks` → the output is **empty**. BoTSORT does **not** emit
Kalman-predicted boxes on a detection-less frame (upstream issue #26 requests
exactly this and is open). Consequence: BoTSORT cannot serve as the SKIP-frame
gap-filler that the motion scheduler relies on, so this design **forbids**
combining BoTSORT with `enable_motion_scheduling` (see §3 validation). In the
non-scheduled tiled stream path every frame is fully inferred, so BoTSORT always
receives fresh detections — its natural operating mode.

- **Dependencies:** OpenCV (vendored ✓), Eigen3 (vendored ✓), Boost::filesystem
  (only in their `.ini` config loader — avoidable), CUDA/TensorRT/ONNXRuntime
  (only for ReID appearance embeddings; not built without CUDA).
- The interfaces line up (`frame + detections → detections`), so BoTSORT can be
  a drop-in alternative at the seam. Its signature difference is track IDs.
- **License:** MIT.

## Architecture

### 1. Vendoring — `third_party/botsort/`

Vendor upstream `botsort/include/*.h` + `botsort/src/*.cpp` **in-tree** (it is
source, not a fetched archive — no repository rule needed) with a **hand-written
`BUILD`**, following the `opencv_macos.BUILD` / `nlohmann.BUILD` precedent:

```python
cc_library(
    name = "botsort",
    srcs = [  # motion-only subset
        "botsort/src/BoTSORT.cpp",
        "botsort/src/track.cpp",
        "botsort/src/KalmanFilter.cpp",
        "botsort/src/KalmanFilterAccBased.cpp",
        "botsort/src/matching.cpp",
        "botsort/src/lapjv.cpp",
        "botsort/src/GlobalMotionCompensation.cpp",
        "botsort/src/GmcParams.cpp",
        "botsort/src/TrackerParams.cpp",
        "botsort/src/utils.cpp",
    ],
    # Vendor all include/*.h EXCEPT ReID.h; do not vendor TRT_InferenceEngine/.
    # (ReIDParams.h is kept — plain struct needed by BoTSORT.h's signature.)
    hdrs = glob(["botsort/include/*.h"], exclude = ["botsort/include/ReID.h"]),
    includes = ["botsort/include"],
    deps = ["@eigen//:eigen"] + select({opencv per platform}),
    licenses = ["notice"],  # MIT
)
```

- **Excluded sources:** `ReID.cpp`, `ReIDParams.cpp`, `TRT_InferenceEngine/*`.
  Result: motion-only (Kalman + IoU/lapjv + GMC).
- **Mandatory vendoring patch** (`third_party/botsort/botsort_motion_only.diff`,
  following `third_party`'s established `.diff` convention). This is **not** a
  "verify and maybe patch" — the upstream headers make it a deterministic
  compile break without it (verified against source):
  - `BoTSORT.h` *unconditionally* `#include`s `ReID.h`, and `ReID.h`
    *unconditionally* `#include`s `TRT_InferenceEngine/TensorRT_InferenceEngine.h`
    (TensorRT). **Patch:** in `BoTSORT.h` remove `#include "ReID.h"`, forward-
    declare `class ReIDModel;`, and keep `#include "ReIDParams.h"` (a plain
    struct — the constructor signature `Config<ReIDParams>` needs the complete
    type; it pulls in no TensorRT).
  - `~BoTSORT() = default;` is declared **inline** in the header while the class
    holds `std::unique_ptr<ReIDModel>` (an incomplete type after the patch
    above). **Patch:** move the destructor out-of-line — declare `~BoTSORT();`
    in the header, define `BoTSORT::~BoTSORT() = default;` in `BoTSORT.cpp`
    (where `ReIDModel` need not be complete because we never compile `ReID.cpp`;
    the `_reid_model` member is left null — `reid_enabled` defaults false and the
    only user, `_extract_features`, is never reached on the motion-only path).
  - `TrackerParams.cpp` and `GmcParams.cpp` both `#include "INIReader.h"` and use
    it solely in their `static load_config(path)` methods. We configure
    programmatically (pass the `TrackerParams` / `GMC_Params` value alternative
    of the `Config<T>` variant, never a string path), so `load_config` is dead.
    **Patch:** drop the `#include "INIReader.h"` and the `load_config` bodies
    from both `.cpp` files. This removes the INIReader/Boost dependency entirely
    (no need to vendor inih or resolve its license).
- Add upstream `LICENSE` (MIT) under `third_party/botsort/` and reference it in
  the BUILD.
- **Verifiability:** deps (Eigen, OpenCV) are already vendored; no CUDA/ONNX →
  satisfies the desktop-C++-only build constraint on this machine.

### 2. The BoTSORT calculator

New api2 calculator
`mediapipe/calculators/tracking/botsort_tracking_calculator.cc`:

- **Inputs:** `IMAGE` (ImageFrame → `cv::Mat` for GMC), `DETECTIONS`
  (`std::vector<Detection>`, frame-normalized `RELATIVE_BOUNDING_BOX`, carrying
  `label_id` + `score`).
- **Output:** `DETECTIONS` (`std::vector<Detection>`) — tracked, the **same
  contract** as the BoxTracker subgraph output (one packet per source frame).
- Wraps `botsort::BoTSORT`. Converts MediaPipe normalized
  `RELATIVE_BOUNDING_BOX` ↔ BoTSORT pixel `tlwh` (using image size). BoTSORT's
  input `Detection.class_id` is `int`, so `label_id` is carried in directly;
  this path **does not need** the ENCODE/DECODE label-id codec that BoxTracker
  requires.
- **`label_id` fidelity / >255-class limit (verified):** although the *input*
  `Detection.class_id` is `int`, the internal `Track` narrows it to `uint8_t`
  (`Track(..., uint8_t class_id, ...)`, `get_class_id() → uint8_t`), so reading a
  track's class back round-trips through `uint8_t`. This is exact for
  `label_id ≤ 255` (covers COCO-80 and typical YOLO models) but truncates above
  that. The graph builder therefore **rejects models with > 256 classes on the
  BOTSORT path at `Create()`** (clear `InvalidArgument`); BoxTracker is
  unaffected. (Internal mapping for >255 classes is a follow-up.)
- **Track IDs:** `Track::track_id` is computed but **not** written to the output
  `Detection.track_id` / `detection_id` (decision A — parity output). A test
  asserts the output carries neither field.
- Options proto `BotsortTrackingCalculatorOptions`, defaulted to **upstream
  `TrackerParams` values** (verified): `track_high_threshold = 0.6`,
  `track_low_threshold = 0.1`, `new_track_threshold = 0.7`, `track_buffer = 30`,
  `match_threshold = 0.7`, `enable_gmc = false` (upstream `gmc_enabled` default;
  GMC method fixed to `sparseOptFlow` when enabled). Enabling GMC is recommended
  for moving-camera streams but defaults off for upstream parity.

### 3. Selection plumbing

Branch inside `TiledTrackingGraph` on a tracker type; the downstream
merge/suppression path is untouched.

- **Detector proto** (`yolo_object_detector/proto/yolo_object_detector_options.proto`):
  ```proto
  message TrackingOptions {
    enum TrackerType {
      TRACKER_UNSPECIFIED = 0;  // -> BOX_TRACKER (default; backward compatible)
      BOX_TRACKER = 1;          // optical-flow propagation (existing)
      BOTSORT = 2;              // tracking-by-detection, motion-only
    }
    optional TrackerType tracker_type = 1 [default = BOX_TRACKER];
    // BoTSORT motion-only knobs (defaults = upstream TrackerParams values):
    optional float track_high_threshold = 2 [default = 0.6];
    optional float track_low_threshold = 3 [default = 0.1];
    optional float new_track_threshold = 4 [default = 0.7];
    optional int32 track_buffer = 5 [default = 30];
    optional float match_threshold = 6 [default = 0.7];
    optional bool enable_gmc = 7 [default = false];
  }
  optional TrackingOptions tracking = 11;
  ```
  `UNSPECIFIED`/default → `BOX_TRACKER` ⇒ existing behavior byte-identical.
- **Public C++ options struct** (`yolo_object_detector.h`): the public entry is
  the hand-written `YoloObjectDetectorOptions` struct →
  `ConvertYoloObjectDetectorOptionsToProto()` (`yolo_object_detector.cc`), which
  today has a `tiling` field but **no** `tracking`. Without this the proto field
  exists but users cannot set it. Add a nested
  `YoloObjectDetectorOptions::TrackingOptions` C++ struct (mirroring `tiling`'s
  struct+proto pairing: a `TrackerType` enum + the six knobs, same defaults) and
  map it in the converter. Round-trip covered by a converter unit test.
- **Subgraph proto** (`tiled_detection/tiled_detection_graphs.proto`): add a
  `TiledTrackingGraphOptions` message (carrying `tracker_type` + the BoTSORT
  params) and extend `TiledBoxMergeGraphOptions` to forward them.
- **Wiring:** the detector graph builder (`yolo_object_detector_graph.cc`)
  reads `tracking`, sets it on the merge-graph options;
  `TiledBoxTrackMergeGraph` forwards them to `TiledTrackingGraph`, which
  branches:
  - `BOX_TRACKER` → existing ENCODE + `ObjectTrackingSubgraphCpu` + DECODE +
    `DetectionsTickGate` (unchanged).
  - `BOTSORT` → `BotsortTrackingCalculator` + a `DetectionsTickGate` (one
    packet per source frame).
  Both emit `TRACKER_DETECTIONS`; `TiledFrameSuppressionCalculator` fuses
  fresh-wins exactly as today.
- **Validation** (at `Create()`, mirroring how `enable_motion_scheduling` is
  gated). Because only the *tiled stream* branch instantiates the tracking
  graph, `tracker_type == BOTSORT` requires **all** of the following, else a
  clean `InvalidArgument` (so a selected BoTSORT can never be silently ignored):
  - `base_options.use_stream_mode() == true` (reject IMAGE mode), and
  - `TilingEnabled(tiling) == true` (reject non-tiled stream — that path has no
    tracker node), and
  - `tiling.enable_motion_scheduling() == false` (BoTSORT can't gap-fill SKIP
    frames — see §"BoTSORT-cpp" empty-detection behavior), and
  - the model has `≤ 256` classes (the `uint8_t` class-id limit, §2).

  Each is a distinct, message-specific error. `BOX_TRACKER` / default is exempt
  from all four (fully backward-compatible).

## Data flow (BoTSORT branch)

```
merged_fresh (vector<Detection>, normalized, label_id+score)
   │                    IMAGE (ImageFrame)
   ▼                       │
BotsortTrackingCalculator ◀┘   (norm→tlwh, BoTSORT.track(), tlwh→norm)
   │
DetectionsTickGate (TICK = merged_fresh)
   │
   ▼
TRACKER_DETECTIONS ─▶ TiledFrameSuppressionCalculator (fresh-wins) ─▶ output
```

## Error handling

- `BOTSORT` selected without `stream_mode` + tiling + (`!motion_scheduling`) +
  (`≤256` classes) → distinct `InvalidArgument` at `Create()` (see §3
  validation).
- Empty detections on a frame → BoTSORT returns only active tracks (so an
  all-unmatched frame yields an empty output); the gate still emits one packet
  per source frame (possibly empty). This is acceptable because the scheduler
  SKIP combo is rejected, so empty-detection frames only occur when the frame
  genuinely had zero detections.
- ReID/TensorRT/INIReader compile breaks → eliminated up front by the mandatory
  vendoring patch (§1), not left to build-time discovery.

## Testing (TDD)

- **Vendored lib:** `botsort_smoke_test` — construct the tracker, feed 2
  synthetic frames of detections, assert stable association across frames.
  Proves the vendored lib builds + links + runs on this machine.
- **Calculator:** unit test — Detection↔tlwh round-trip preserves
  `label_id` (≤255) + `score`; a track persists across consecutive frames; an
  empty-detections frame produces a valid (possibly empty) output packet with no
  crash; the output Detection carries **neither** `track_id` nor `detection_id`.
- **Converter:** round-trip test for the `TrackingOptions` C++ struct →
  proto (`tracker_type` + all six knobs), mirroring the tiling converter test.
- **Graph:** extend `tiled_tracking_graphs_test.cc` — the `BOTSORT` branch
  produces `TRACKER_DETECTIONS`; the default / `BOX_TRACKER` path is
  **byte-identical** to current output.
- **Detector validation (model-free):** distinct rejection tests for
  `BOTSORT` + IMAGE mode, `BOTSORT` + stream-without-tiling, and `BOTSORT` +
  `enable_motion_scheduling` (each fires at `Create()` before model load, like
  the existing `MotionSchedulingInImageModeRejected` test). The >256-class
  rejection is gated on a model.
- **Detector e2e:** a real stream e2e (boats.jpg-style panning sequence)
  selecting BoTSORT, gated on the YOLO fixture like existing YOLO stream tests.

## Scope boundaries (YAGNI)

**In scope:** YOLO detector only · tiled stream path only · motion-only BoTSORT ·
C++ layer end-to-end (vendored lib + calculator + proto + **public options
struct + converter + Create-time validation**, so a user can actually select it).

**Follow-ups (not this spec):**
- BoTSORT SKIP-frame gap-fill: patch the vendored tracker to emit Kalman-
  predicted (still-alive) tracks on detection-less frames (upstream issue #26),
  which would let BoTSORT combine with `enable_motion_scheduling`.
- C-API + Python bindings for `TrackingOptions` (mirrors the tiling bindings
  arc: Phase 1 C-API, Phase 2 Python).
- Surfacing BoTSORT track IDs on output Detections (the deferred "decision B" —
  the public `Detection.track_id`/`detection_id` fields already exist).
- Internal class-id mapping to lift the ≤256-class limit; GMC method selection.
- OBB detector tracking.
- Non-tiled livestream tracking.
- ReID / appearance embeddings (needs CUDA/ONNX — not verifiable here).

## Verifiability

Everything in this spec compiles and runs under desktop C++
(`--define MEDIAPIPE_DISABLE_GPU=1`) on this machine: no CUDA, no ONNX, deps
already vendored. The detector stream e2e is gated on the YOLO tflite fixture
(skips when absent, like existing YOLO stream tests).
