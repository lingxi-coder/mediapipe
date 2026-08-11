# BoTSORT Tracker Selection — Design

**Date:** 2026-06-16
**Branch:** `dev` (MediaPipe fork; unmerged, no upstream PR)
**Status:** Revised design — ready for implementation-plan update

## Goal

In the YOLO tiled `VIDEO` / `LIVE_STREAM` path, let the user choose the tracker:
keep the existing **BoxTracker** (optical-flow propagation, the default, fully
backward-compatible) or select **BoTSORT** (tracking-by-detection, motion-only).
Both trackers must work with the tiled motion scheduler: complete or partial
DETECT frames update the tracker from fresh detections only in the regions that
were actually inferred; SKIP frames advance the tracker without running
association and emit predicted boxes for that source-frame timestamp. BoTSORT's
persistent track IDs are surfaced through the existing optional
`Detection.track_id` field and preserved through fresh-wins suppression. Box
geometry, scores, labels, packet density, and the default BoxTracker behavior
remain backward-compatible.

## Background — what exists today

The axis-aligned YOLO *tiled stream* path owns the tracker-selection seam
(`yolo_object_detector_graph.cc`, the `use_stream_mode()` branch). Its default
branch uses **BoxTracker**, an optical-flow propagator wrapped as
`ObjectTrackingSubgraphCpu`, and fuses it back through
`TRACKER_DETECTIONS`:

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

**Important — verified upstream empty-detection behavior:** `track()` returns
only the `is_activated` tracks remaining in `_tracked_tracks` (`BoTSORT.cpp`
"Update output tracks" loop). On a frame with **no** detections, all
previously-tracked tracks go unmatched and are `mark_lost()` → removed from
`_tracked_tracks` → the output is **empty**. Upstream BoTSORT therefore cannot
directly fill a scheduler SKIP frame (upstream issue #26 requests predicted
output and is open).

This design closes that gap in the motion-only vendored adapter rather than
removing scheduler support. It adds a separate `predict_only(frame)` operation
and drives the frame decision with the scheduler's existing, dense `REFRESH`
stream. `REFRESH` alone is not an observation-completeness signal: when
`max_scheduled_tiles` is smaller than the configured tile count, a DETECT frame
contains detections from only a subset of the frame. The tracker therefore also
receives `OBSERVED_ROIS`, the exact `TilePixelRoi` values accumulated from
`TensorBatchInfo.geometry.effective_pixel_rois` for that source frame:

- `REFRESH=true` (DETECT): call
  `track_observed(detections, frame, observed_rois)`. The vendored API takes
  `std::vector<cv::Rect>` so `third_party/botsort` does not depend on a
  MediaPipe type. An empty detection vector
  is a real negative result only inside `observed_rois`. An unmatched active or
  unconfirmed track whose predicted box center lies inside the union of the
  observed ROIs follows normal BoTSORT lost/removal semantics. Outside that
  union, an unmatched active track remains active and is emitted at its
  predicted position; an unmatched unconfirmed track remains unconfirmed and
  is not emitted. All tracks still participate in association, so a detection
  may match a track whose prediction crossed an ROI boundary.
- `REFRESH=false` (SKIP): call `predict_only(frame)`. Advance Kalman/GMC state,
  do not run association or mark active tracks lost, and emit activated
  predicted tracks for this source-frame timestamp. `OBSERVED_ROIS` must be
  empty.

The explicit decision and coverage are both required: `merged_fresh.empty()`
cannot distinguish "inference was skipped" from "inference ran and found no
objects," while `REFRESH=true` cannot distinguish complete from partial tile
coverage. The center-in-union rule is intentionally conservative at tile
boundaries: a partially visible track whose center was not observed is retained
rather than falsely marked lost.

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
licenses(["notice"])  # MIT

cc_library(
    name = "botsort",
    srcs = [  # motion-only subset
        "src/BoTSORT.cpp",
        "src/track.cpp",
        "src/KalmanFilter.cpp",
        "src/KalmanFilterAccBased.cpp",
        "src/matching.cpp",
        "src/lapjv.cpp",
        "src/GlobalMotionCompensation.cpp",
        "src/GmcParams.cpp",
        "src/ReIDParams.cpp",
        "src/TrackerParams.cpp",
        "src/utils.cpp",
    ],
    # ReID.h is a local motion-only stub; do not vendor TRT_InferenceEngine/.
    hdrs = glob(["include/*.h"]),
    includes = ["include"],
    deps = [
        "//third_party:opencv",
        "@eigen//:eigen3",
    ],
)
```

- **Excluded sources:** `ReID.cpp`, `TRT_InferenceEngine/*`. `ReIDParams.cpp`
  stays in the target because `BoTSORT.cpp` references
  `ReIDParams::load_config`; its loader is replaced by the same never-executed
  programmatic-config stub as the tracker/GMC loaders. Result: motion-only
  (Kalman + IoU/lapjv + GMC).
- **Mandatory vendoring patch** (`third_party/botsort/botsort_motion_only.diff`,
  following `third_party`'s established `.diff` convention). This is **not** a
  "verify and maybe patch" — the upstream headers make it a deterministic
  compile break without it (verified against source):
  - `BoTSORT.h` *unconditionally* `#include`s `ReID.h`, and `ReID.h`
    *unconditionally* includes TensorRT. A forward declaration alone is
    insufficient: `BoTSORT.cpp` destroys `std::unique_ptr<ReIDModel>` and
    compiles calls to `ReIDModel::extract_features()`, both of which require a
    complete type even when ReID is disabled at runtime. **Patch:** replace
    upstream `ReID.h` with a complete local no-op `ReIDModel` stub implementing
    the constructor, destructor, `extract_features()`, and
    `get_distance_metric()` signatures used by `BoTSORT.cpp`. The calculator
    pins `reid_enabled=false`, so the stub is never used for inference.
  - `TrackerParams.cpp`, `GmcParams.cpp`, and `ReIDParams.cpp` include
    `INIReader.h` for their `static load_config(path)` methods. We configure
    programmatically (pass value alternatives of `Config<T>`, never a string
    path), but `BoTSORT.cpp` still references the loader symbols. **Patch:**
    remove the INIReader includes and replace all three loader bodies with
    deterministic default-returning stubs. This removes the INIReader/Boost
    dependency without leaving unresolved symbols.
  - Add `BoTSORT::predict_only(const cv::Mat& frame)`. It increments `_frame_id`,
    Kalman-predicts the union of `_tracked_tracks` and `_lost_tracks`, and then
    applies GMC to both lists when enabled. GMC receives the predicted boxes of
    active, unconfirmed, and not-yet-expired lost tracks as its
    foreground-exclusion mask; passing an empty or active-only mask would let
    moving-object features from retained tracks contaminate camera-motion
    estimation on SKIP frames. Finally it expires lost tracks using the existing
    `_max_time_lost` rule and returns only activated tracks still in
    `_tracked_tracks`. It does **not** associate, create tracks, or mark active
    tracks lost. It updates `frame_id` on tracks that remain active so a later
    DETECT miss starts the full lost-track retention window from the last
    propagated source frame; it does not update `frame_id` on tracks already
    lost, which must continue aging. Keeping lost tracks' Kalman state advancing
    is necessary so a later DETECT frame can re-associate against their current
    predicted position.
  - Add `BoTSORT::track_observed(detections, frame,
    const std::vector<cv::Rect>& observed_rois)` while keeping upstream
    `track(detections, frame)` as the full-frame convenience entry (it delegates
    with `{cv::Rect(0, 0, frame.cols, frame.rows)}`).
    `track_observed` performs prediction, GMC, and association against the full
    tracked/lost pool, but gates unmatched-track transitions by observation:
    unmatched active tracks are marked lost, and unmatched unconfirmed tracks
    are removed, only when their predicted box center is inside the union of
    `observed_rois`. Unmatched tracks outside coverage retain their state and
    predicted geometry. Lost-track aging remains frame-global, and matched/new
    tracks follow the normal upstream transitions. This is one association pass,
    not one tracker invocation per tile.
  - Upstream `SparseOptFlow_GMC::apply()` ignores its `detections` argument even
    though this is the only GMC method exposed here. Patch it to pass a
    background mask to `cv::goodFeaturesToTrack`: initialize the valid image
    area to 255; combine fresh detections with the active, unconfirmed, and
    retained-lost predicted boxes; divide every pixel-space rectangle by the
    configured `_downscale` using floor for the top/left and ceil for the
    bottom/right; clip the result to the **downscaled mask** bounds; and set
    those foreground regions to zero. SparseOptFlow defaults to
    `_downscale=2.0`, so clipping original-frame rectangles directly against the
    resized mask is incorrect and may throw. DETECT and SKIP paths use this same
    mask construction.
  - Preserve upstream's fixed-step Kalman model and configure its existing
    integer `frame_rate` from a public `nominal_frame_rate` option (default 30).
    A partial timestamp-driven patch is intentionally excluded: changing only
    the state-transition matrix leaves process-noise covariance and association
    gating scaled incorrectly. Every source packet, DETECT or SKIP, advances
    exactly one tracker step at `dt=1/nominal_frame_rate`; irregular streams use
    the configured nominal rate. `track_buffer` remains upstream's
    30-FPS-normalized retention parameter; the effective lost window is
    `floor(nominal_frame_rate / 30.0 * track_buffer)` source frames. SKIP does
    not mark an active track lost, while tracks already in `_lost_tracks`
    continue aging by one source frame per SKIP.
  - Replace `Track::next_id()`'s function-local non-atomic `static int` with an
    atomic counter. Separate detector instances may execute concurrently, and
    the upstream counter otherwise has a C++ data race. This also prevents
    duplicate IDs from being surfaced by concurrently running detector
    instances in the same process.
  - Rewrite upstream `eigen3/Eigen/...` includes to `Eigen/...`, matching the
    header root exported by `@eigen//:eigen3`. Remove the debug-only
    `cv::imshow` call from GMC so the CPU target does not acquire an undeclared
    OpenCV highgui / macOS Cocoa link dependency.
  - Keep the patch reproducible with both the machine-applicable `.diff` and a
    short `PATCHES.md` recording the exact upstream commit and omitted files.
- Add upstream `LICENSE` (MIT) under `third_party/botsort/` and reference it in
  the BUILD.
- **Verifiability:** deps (Eigen, OpenCV) are already vendored; no CUDA/ONNX is
  required for the motion-only build on this machine.

### 2. The BoTSORT calculator

New api2 calculator
`mediapipe/calculators/tensor/botsort_tracking_calculator.cc`:

- **Inputs:** `IMAGE` (ImageFrame → `cv::Mat` for GMC), `DETECTIONS`
  (`std::vector<Detection>`, frame-normalized `RELATIVE_BOUNDING_BOX`, carrying
  `label_id` + `score`), `REFRESH` (`bool`, dense at the source-frame timestamp;
  `true`=DETECT, `false`=SKIP), and `OBSERVED_ROIS`
  (`std::vector<TilePixelRoi>`, exact source-pixel regions actually submitted to
  inference for this source frame).
- **Output:** `DETECTIONS` (`std::vector<Detection>`) — tracked, with the same
  geometry/label/score and one-packet-per-source-frame contract as BoxTracker;
  BoTSORT additionally populates the optional `track_id`.
- Wraps `botsort::BoTSORT`. On `REFRESH=true`, validates non-empty observed
  coverage, converts MediaPipe normalized `RELATIVE_BOUNDING_BOX` ↔ BoTSORT
  pixel `tlwh` (using image size), converts `TilePixelRoi` to `cv::Rect`, and
  calls `track_observed()`. On
  `REFRESH=false`, requires both an empty fresh vector and empty observed
  coverage, then calls `predict_only()`; inconsistent packets are graph-contract
  errors rather than silently discarded data. BoTSORT's input
  `Detection.class_id` is `int`, so `label_id` is carried in directly; this path
  **does not need** the ENCODE/DECODE label-id codec that BoxTracker requires.
- The calculator validates its direct input contract instead of silently
  dropping malformed detections: positive image dimensions; exactly one usable
  `label_id` in `[0,255]`; exactly one finite `score` in `[0,1]`; finite
  normalized box coordinates with non-negative width/height; and
  `RELATIVE_BOUNDING_BOX` format. Invalid inputs return `InvalidArgument`.
- The calculator passes `nominal_frame_rate` into upstream `TrackerParams` and
  advances the tracker once per source packet. MediaPipe timestamps still own
  stream ordering/alignment; they are not reinterpreted as a second Kalman
  clock.
- **`label_id` fidelity / >255-class limit (verified):** although the *input*
  `Detection.class_id` is `int`, the internal `Track` narrows it to `uint8_t`
  (`Track(..., uint8_t class_id, ...)`, `get_class_id() → uint8_t`), so reading a
  track's class back round-trips through `uint8_t`. This is exact for
  `label_id ≤ 255` (covers COCO-80 and typical YOLO models) but truncates above
  that. The graph builder therefore **rejects models with > 256 classes on the
  BOTSORT path at `Create()`** (clear `InvalidArgument`); BoxTracker is
  unaffected. (Internal mapping for >255 classes is a follow-up.)
- **Track IDs:** write `std::to_string(Track::track_id)` to the existing optional
  `Detection.track_id`; do not populate the unrelated `detection_id`. The
  downstream fresh-wins suppression step transfers the best-IoU tracker ID onto
  the surviving fresh Detection, so most DETECT-frame results do not lose their
  IDs merely because fresh geometry wins. IDs remain absent on BoxTracker and
  no-tracking outputs.
- Options proto `BotsortTrackingCalculatorOptions`, defaulted to **upstream
  `TrackerParams` values** (verified): `track_high_threshold = 0.6`,
  `track_low_threshold = 0.1`, `new_track_threshold = 0.7`, `track_buffer = 30`,
  `match_threshold = 0.7`, `enable_gmc = false`,
  `nominal_frame_rate = 30` (upstream defaults; GMC method fixed to
  `sparseOptFlow` when enabled). Enabling GMC is recommended for moving-camera
  streams but defaults off for upstream parity.

### 3. Selection plumbing

Branch inside `TiledTrackingGraph` on a tracker type. The graph interface gains
the scheduler decision and exact inference coverage only on the BoTSORT branch;
the downstream suppression retains fresh-wins geometry but propagates BoTSORT
track IDs from the matched tracker Detection.

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
    optional int32 nominal_frame_rate = 8 [default = 30];
  }
  optional TrackingOptions tracking = 11;
  ```
  `UNSPECIFIED`/default → `BOX_TRACKER` ⇒ existing behavior byte-identical.
  For `BOTSORT`, validate all confidence/match thresholds as finite values in
  `[0,1]`, require `track_low_threshold <= track_high_threshold`, require
  `track_buffer` in `[0,255]`, and require `nominal_frame_rate` in `[1,255]`.
  Also require
  `floor(nominal_frame_rate / 30.0 * track_buffer) <= 255`, because upstream
  stores both the inputs and derived `_buffer_size` / `_max_time_lost` in
  `uint8_t`. Invalid options fail at `Create()` before graph construction and
  are checked again in the calculator `Open()` for direct-graph users.
- **Public C++ options struct** (`yolo_object_detector.h`): keep the nested
  `YoloObjectDetectorOptions::TrackingOptions` struct and add
  `nominal_frame_rate` beside the existing selector + six knobs, with the same
  default. Map every field in `ConvertYoloObjectDetectorOptionsToProto()` and
  cover the round-trip with a converter unit test.
- **C API and Python:** append `int nominal_frame_rate` to
  `MpTrackingOptions`, update `CppConvertToTrackingOptions`, and update the
  pinned C ABI size/offset test. The C converter interprets zero as the default
  30 so source callers rebuilt with the new header but leaving the trailing
  field zero do not fail the new `[1,255]` validation. Appending the nested field
  changes the containing options struct's binary layout; that is acceptable on
  this unmerged branch and is made explicit by updating the ABI pin rather than
  claiming old precompiled clients remain compatible. Add the matching
  `ctypes.c_int` field and `TrackingOptions.nominal_frame_rate = 30` dataclass
  member in Python, including conversion/default tests. The C header,
  converter, Python ctypes layout, and ABI test land atomically.
- **OBB parameter parity:** the oriented detector has separate C++, C, and
  Python TrackingOptions but uses the same BoTSORT implementation and
  `TiledTrackingGraphOptions`. Add field 8/default 30 to the oriented detector
  proto, add the field to its C++ struct/converter, append it to
  `MpOrientedTrackingOptions` with the same zero-to-30 C conversion rule, and
  update its Python dataclass/ctypes and ABI tests. Apply the same `[1,255]` and
  derived-buffer-overflow validation when OBB selects BoTSORT. This does not add
  BoxTracker or motion scheduling to OBB; it prevents the same tracker from
  having language- and detector-dependent time semantics.
- **Subgraph proto** (`tiled_detection/tiled_detection_graphs.proto`): carry
  `tracker_type` + all BoTSORT params, including `nominal_frame_rate`, in
  `TiledTrackingGraphOptions`; both box and OBB merge options forward it.
- **Wiring:** the detector graph builder (`yolo_object_detector_graph.cc`)
  reads `tracking`, sets it on the merge-graph options;
  `TiledBoxTrackMergeGraph` forwards them to `TiledTrackingGraph`, which
  branches:
  - `BOX_TRACKER` → existing ENCODE + `ObjectTrackingSubgraphCpu` + DECODE +
    `DetectionsTickGate` (unchanged).
  - `BOTSORT` →
    `BotsortTrackingCalculator(IMAGE, DETECTIONS, REFRESH, OBSERVED_ROIS)` + a
    `DetectionsTickGate` (one packet per source frame).
  Both emit `TRACKER_DETECTIONS`; `TiledFrameSuppressionCalculator` fuses
  fresh-wins exactly as today, except that it copies a matched tracker
  `track_id` onto the best-IoU fresh winner before global NMS.
- **Dense `REFRESH` wiring:** `TiledDetectionStreamFrontGraph` exposes the
  `VideoTileSchedulerCalculator.REFRESH` output. When motion scheduling is
  enabled, the detector graph forwards that stream through
  `TiledBoxTrackMergeGraph` → `TiledTrackingGraph` →
  `BotsortTrackingCalculator`. When scheduling is disabled, a
  `PacketPresenceCalculator` driven by the source `ImageFrame` emits
  `REFRESH=true` at every source-frame timestamp.
- **Exact observation coverage:** extend
  `MergeTileBoxDetectionsAccumulatorCalculator` with an `OBSERVED_ROIS` output.
  It deduplicates and emits the union of
  `TensorBatchInfo.geometry.effective_pixel_rois` only after all batches for the
  source frame have arrived; for `total_batches==0` it emits an empty vector.
  `TiledBoxTrackMergeGraph` forwards the source-timestamped vector to
  `TiledTrackingGraph`. This uses the exact rounded/clamped pixel ROIs sampled
  by preprocessing, works for one or many inference batches, and does not
  reconstruct coverage independently from configured normalized tiles. The
  detector connects `REFRESH` and `OBSERVED_ROIS` only when
  `tracker_type=BOTSORT`; the default BoxTracker graph contract and data path
  remain unchanged.
- **Validation** (at `Create()`, mirroring how `enable_motion_scheduling` is
  gated). Because only the *tiled stream* branch instantiates the tracking
  graph, `tracker_type == BOTSORT` requires **all** of the following, else a
  clean `InvalidArgument` (so a selected BoTSORT can never be silently ignored):
  - `base_options.use_stream_mode() == true` (reject IMAGE mode), and
  - `TilingEnabled(tiling) == true` (reject non-tiled stream — that path has no
    tracker node), and
  - configured `num_classes` is in `[1,256]` (the `uint8_t` class-id limit,
    §2), and
  - all BoTSORT thresholds, `track_buffer`, and `nominal_frame_rate` satisfy the
    ranges above.

  Each is a distinct, message-specific error. `BOTSORT` with
  `enable_motion_scheduling=true` is valid and exercises the REFRESH-driven
  DETECT/SKIP paths. `BOX_TRACKER` / default is exempt from these
  BoTSORT-specific checks (fully backward-compatible).

## Data flow (BoTSORT branch)

```
TiledDetectionStreamFrontGraph.REFRESH ─────────────────────────────┐
  (or dense true when scheduling is off)                            │
                                                                   ▼
BATCH_INFO ─▶ MergeTileBoxDetectionsAccumulator ─▶ OBSERVED_ROIS ──┤
                         │                                         │
                         └─▶ merged_fresh ──────────────────────────┤
                                                                   ▼
IMAGE (ImageFrame) ───────────────────────────────▶ BotsortTrackingCalculator
                      REFRESH=true  → track_observed(fresh, frame,
                                                     observed_rois)
                      REFRESH=false → predict_only(frame)
                                        │
                                        ▼
                         DetectionsTickGate (TICK = merged_fresh)
                                        │
                                        ▼
TRACKER_DETECTIONS ─▶ TiledFrameSuppressionCalculator (fresh-wins) ─▶ output
                                      └─ best-IoU track_id transfer
```

All four BoTSORT inputs are aligned to the original source-frame timestamp.
Per-batch synthetic inference timestamps terminate at the merge accumulator and
never enter the tracker.

## Error handling

- `BOTSORT` selected in IMAGE mode, without tiling, or with configured
  `num_classes` outside `[1,256]` → distinct `InvalidArgument` at `Create()`
  (see §3 validation).
- `REFRESH=true` + empty detections + non-empty `OBSERVED_ROIS` → a real
  negative detector result in those ROIs. Unmatched tracks centered inside the
  observed union follow normal lost/removal semantics; unmatched tracks outside
  it remain active and are emitted as predictions.
- `REFRESH=true` + empty `OBSERVED_ROIS` → `FailedPrecondition`; a DETECT result
  without any inferred coverage is internally inconsistent.
- `REFRESH=false` requires empty detections and empty `OBSERVED_ROIS`, predicts
  active tracks, and emits them at the current source-frame timestamp. Any
  non-empty detections or coverage on a SKIP → `FailedPrecondition`.
- Missing `REFRESH` or `OBSERVED_ROIS` on a BoTSORT graph is a graph-
  initialization error; neither may be inferred from detection-vector
  emptiness.
- An observed ROI with negative origin/extent, zero size, or bounds outside the
  source image is `InvalidArgument`; the accumulator emits the already-clamped
  `TilePixelRoi`, so this indicates corrupt batch metadata.
- Non-finite threshold values, invalid threshold ordering, out-of-range
  `track_buffer` / `nominal_frame_rate`, and an overflowing derived buffer fail
  explicitly; none may reach an upstream narrowing cast.
- ReID/TensorRT/INIReader compile breaks → eliminated up front by the mandatory
  vendoring patch (§1), not left to build-time discovery.

## Testing (TDD)

- **Vendored lib:** `botsort_smoke_test` — construct the tracker, establish a
  confirmed track, run at least two `predict_only()` frames, then re-detect the
  object. Assert non-empty predicted output, geometry advancing in the expected
  motion direction, and the same stable association after re-detection. Also
  establish a lost track before a SKIP and prove its state is predicted for
  later re-association but is not emitted. After a long active SKIP run, inject
  one missed DETECT and prove the track still receives the full configured
  lost-track window (the active `frame_id` was refreshed during prediction).
  Establish tracks in left and right ROIs, run an empty
  `track_observed(..., left_roi)`, and prove only the left track becomes lost
  while the unobserved right track remains active. Then detect the right object
  in the left ROI after motion and prove all tracks were eligible for matching,
  rather than partitioning association by tile.
  Add a concurrent two-tracker test for unique, race-free `Track::next_id()`
  allocation. This proves the patched lib builds + links + runs.
- **Calculator:** unit test — Detection↔tlwh round-trip preserves
  `label_id` (≤255) + `score`; a track persists across consecutive DETECT
  frames; `REFRESH=false` produces a predicted non-empty packet; `REFRESH=true`
  with empty detections loses only tracks centered in `OBSERVED_ROIS`; invalid
  REFRESH/coverage combinations fail; nominal 24/30/60-FPS options reach
  upstream `TrackerParams` and change the fixed Kalman step as specified; the
  GMC-enabled DETECT and predict-only paths mask active/unconfirmed/lost boxes
  correctly with `_downscale=2.0`, including clipped/out-of-frame boxes; and the
  output Detection carries a stable, non-empty `track_id` but no
  `detection_id`.
- **Converter:** round-trip test for the `TrackingOptions` C++ struct →
  proto (`tracker_type` + all seven knobs), mirroring the tiling converter test.
- **Bindings:** axis-aligned and OBB C converter tests cover
  `nominal_frame_rate` and the zero-to-30 default; both ABI tests pin the new
  trailing field and struct size; both Python option/ctypes tests verify default
  30 and an explicit 24/60 value.
- **Graph:** extend `tiled_tracking_graphs_test.cc` — the `BOTSORT` branch
  produces `TRACKER_DETECTIONS` on DETECT and SKIP timestamps, and preserves
  one-packet-per-source-frame density across a `DETECT → SKIP → SKIP → DETECT`
  sequence. Add a `T=5, B=2, max_scheduled_tiles=2` case with tracks in selected
  and unselected tiles: only the selected-region miss becomes lost, the other
  track remains active, and `OBSERVED_ROIS` equals the two exact effective pixel
  ROIs. Add a no-cap `T=5, B=2` case proving all three inference batches are
  accumulated before one tracker update. The default / `BOX_TRACKER` path is
  **byte-identical** to current output and has no new REFRESH/OBSERVED_ROIS
  inputs.
- **Detector validation (model-free):** distinct rejection tests for
  `BOTSORT` + IMAGE mode and `BOTSORT` + stream-without-tiling (each fires at
  `Create()` before model load, like the existing
  `MotionSchedulingInImageModeRejected` test), plus configured `num_classes`
  outside `[1,256]`. Replace the former
  `BotsortWithMotionSchedulingRejected` model-free test with an assertion that
  `Create()` proceeds past the compatibility guard and fails only for the
  intentionally missing model asset; the actual success case belongs in the
  fixture-backed e2e below. Add table-driven tests for
  NaN/infinite/out-of-range thresholds, low>high, and invalid/overflowing
  `track_buffer` × `nominal_frame_rate` combinations.
- **Detector e2e:** real stream e2es selecting BoTSORT, gated on the YOLO
  fixture like existing YOLO stream tests: (a) scheduling off, panning frames
  retain detections; (b) scheduling on, a confirmed DETECT frame followed by
  duplicate/static SKIP frames remains non-empty on every source timestamp;
  and (c) stable associations expose the same non-empty `track_id` across
  DETECT and SKIP frames.
  The graph-level assertion must also show `REFRESH=false` /
  `BATCH_INFO.total_batches==0` on the SKIP frames so non-empty results cannot
  be explained by inference. Use a deterministic graph integration test with a
  fake detector output for `REFRESH=true` + zero detections: tracks centered in
  full observed coverage must disappear, while tracks outside partial observed
  coverage must remain predicted. This proves REFRESH, emptiness, and coverage
  are not conflated without relying on a model to produce a specific negative.

## Scope boundaries

**In scope:** tiled YOLO stream tracking · motion-only BoTSORT · scheduled and
non-scheduled stream execution · REFRESH + exact-observation-driven partial
DETECT semantics · predict-only SKIP gap-fill · stable track ID propagation ·
C++/C/Python selector and parameter surfaces · Create-time validation · OBB
`nominal_frame_rate` parity across its existing C++/C/Python BoTSORT surfaces.

**Follow-ups (not this spec):**
- Internal class-id mapping to lift the ≤256-class limit; GMC method selection.
- Non-tiled livestream tracking.
- ReID / appearance embeddings (needs CUDA/ONNX — not verifiable here).

## Verifiability

The vendored tracker, calculators, graphs, task converters, and C APIs compile
and run under desktop C++ (`--define MEDIAPIPE_DISABLE_GPU=1`) on this machine:
no CUDA, no ONNX, deps already vendored. Python binding tests run through the
repository's Bazel Python targets. Detector stream e2es are gated on the YOLO
tflite fixture (skip when absent, like existing YOLO stream tests); model-free
graph tests cover partial observation deterministically.
