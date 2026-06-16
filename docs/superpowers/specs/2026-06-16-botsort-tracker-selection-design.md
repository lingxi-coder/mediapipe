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
IDs**. On a detection-less frame it Kalman-predicts existing tracks forward (a
valid gap-fill). The `frame` argument is always required (GMC uses it).

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
    hdrs = glob(["botsort/include/*.h"]),  # minus ReID/TRT if they break compile
    includes = ["botsort/include"],
    deps = ["@eigen//:eigen"] + select({opencv per platform}),
    licenses = ["notice"],  # MIT
)
```

- **Excluded:** `ReID.*`, `ReIDParams.*`, `TRT_InferenceEngine/*`,
  `INIReader`/Boost path. Result: motion-only (Kalman + IoU/lapjv + GMC). We
  configure from proto, not `.ini`, so Boost::filesystem is dropped.
- **Key vendoring risk (must verify in plan):** `BoTSORT.h` includes `ReID.h`
  and holds a `std::unique_ptr<ReIDModel>` member. Upstream's no-CUDA CMake
  build already excludes `ReID.cpp`, so ReID usage *should* already be
  compile/link-clean without CUDA. The plan verifies this empirically by
  building the `cc_library`. If a guard is missing, add a minimal vendoring
  patch following `third_party`'s established `.diff` convention
  (e.g. `botsort_motion_only.diff`), adjusting `BoTSORT.h`/`BoTSORT.cpp` to
  compile without the ReID member.
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
  `RELATIVE_BOUNDING_BOX` ↔ BoTSORT pixel `tlwh` (using image size), and
  **preserves `label_id` + `score` natively** across `track()`. Because BoTSORT
  carries class/score directly, this path **does not need** the ENCODE/DECODE
  label-id codec that BoxTracker requires.
- Options proto `BotsortTrackingCalculatorOptions`: `track_buffer`,
  `track_high_threshold`, `track_low_threshold`, `new_track_threshold`,
  `match_threshold`, `enable_gmc` — each defaulted to BoTSORT's upstream
  defaults. (Track IDs are computed but not written to the output Detection in
  this spec — see Scope.)

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
    // BoTSORT motion-only knobs (defaulted to upstream values):
    optional float track_high_threshold = 2 [default = 0.6];
    optional float track_low_threshold = 3 [default = 0.1];
    optional float new_track_threshold = 4 [default = 0.7];
    optional int32 track_buffer = 5 [default = 30];
    optional float match_threshold = 6 [default = 0.8];
    optional bool enable_gmc = 7 [default = true];
  }
  optional TrackingOptions tracking = 11;
  ```
  `UNSPECIFIED`/default → `BOX_TRACKER` ⇒ existing behavior byte-identical.
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
- **Validation:** selecting `BOTSORT` outside stream mode (IMAGE mode) →
  clean `InvalidArgument` at `Create()`, mirroring how `enable_motion_scheduling`
  is gated today.

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

- `BOTSORT` + IMAGE running mode → `InvalidArgument` at `Create()`.
- Empty detections on a frame → BoTSORT Kalman-predicts existing tracks
  forward; gate still emits one packet (possibly empty).
- Vendored-lib build/link failure for ReID → resolved by the motion-only
  vendoring patch (Section 1).

## Testing (TDD)

- **Vendored lib:** `botsort_smoke_test` — construct the tracker, feed 2
  synthetic frames of detections, assert stable association across frames.
  Proves the vendored lib builds + links + runs on this machine.
- **Calculator:** unit test — Detection↔tlwh round-trip preserves
  `label_id`/`score`; a track persists across consecutive frames; an
  empty-detections frame Kalman-predicts (no crash, valid output packet).
- **Graph:** extend `tiled_tracking_graphs_test.cc` — the `BOTSORT` branch
  produces `TRACKER_DETECTIONS`; the default / `BOX_TRACKER` path is
  **byte-identical** to current output.
- **Detector:** a real stream e2e (boats.jpg-style panning sequence) selecting
  BoTSORT, gated on the YOLO fixture like existing YOLO stream tests; plus a
  model-free `BotsortInImageModeRejected` wiring test.

## Scope boundaries (YAGNI)

**In scope:** YOLO detector only · tiled stream path only · motion-only BoTSORT ·
C++ proto layer only.

**Follow-ups (not this spec):**
- C-API + Python bindings for `TrackingOptions` (mirrors the tiling bindings
  arc: Phase 1 C-API, Phase 2 Python).
- Surfacing BoTSORT track IDs on output Detections (the deferred "decision B").
- OBB detector tracking.
- Non-tiled livestream tracking.
- ReID / appearance embeddings (needs CUDA/ONNX — not verifiable here).

## Verifiability

Everything in this spec compiles and runs under desktop C++
(`--define MEDIAPIPE_DISABLE_GPU=1`) on this machine: no CUDA, no ONNX, deps
already vendored. The detector stream e2e is gated on the YOLO tflite fixture
(skips when absent, like existing YOLO stream tests).
