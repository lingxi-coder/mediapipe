# Sub-project B — YOLO Tiled Detection × Motion Scheduling (Video) Design

**Status:** Approved for planning (2026-06-14)
**Branch:** `dev` (fork; no upstream PR)
**Predecessors:** Sub-project A (static tiling, shipped) `[[tasks-tiled-detection-shipped]]`; Sub-project C (BoxTracker, shipped) `[[yolo-tiled-boxtracker-shipped]]`. This is the last of the A→C→B decomposition.

## Goal

In `VIDEO`/`LIVE_STREAM` mode, skip the expensive **tiled TFLite inference** on frames that don't need fresh detection; the BoxTracker (C) fills those frames from history. IMAGE mode is unchanged. OBB out of scope (YOLO only).

## Scope

**In scope**
- Wire the already-built-but-unwired `VideoTileSchedulerCalculator` into the tiled `YoloObjectDetector` stream path.
- The scheduler runs its **own** optical-flow pass (an `IMAGE → TRACKING` producer encapsulated in the stream front). **Sub-project C is kept frozen** — its tracker subgraphs and their tests are untouched and keep their existing internal optical flow. Two optical-flow passes total (the extra one is cheap relative to the tiled inference saved).
- The `PRIOR_DETECTIONS` loopback (previous frame's merged detections) the scheduler needs.
- Full vertical slice (L2): scheduling reachable from the public `DetectForVideo`/`DetectAsync` API + a real-video e2e.
- Minimal public config: `enable_motion_scheduling` + `max_scheduled_tiles`.

**Out of scope (explicit)**
- Any change to C's `TiledTrackingGraph` / `TiledBoxTrackMergeGraph` / suppression / codec / tick gate (frozen).
- Sharing one optical-flow pass between scheduler and tracker (considered and declined — would require refactoring C; a documented follow-up if the second pass ever matters).
- `OrientedObjectDetector` (OBB) — YOLO only.
- Exposing the scheduler's motion-threshold / policy knobs publicly (left at internal defaults; see "Default behavior").
- IMAGE mode (scheduling is meaningless without temporal continuity).

## Background / current state (verified)

- `VideoTileSchedulerCalculator` (`mediapipe/calculators/tensor/video_tile_scheduler_calculator.{cc,proto}`, options ext `471230006`) is fully implemented and unit/e2e-tested but referenced by **no** Tasks graph. Inputs `TILES: vector<NormalizedRect>` (required), `PRIOR_DETECTIONS: vector<Detection>` (required, empty-tolerant via a static empty fallback), `TRACKING: TrackingData` (**Optional**). Outputs `TILES` (scheduled subset; empty vector on SKIP) + `REFRESH: bool`. Decision core `ShouldRefreshFrame` + `ScheduleTiles` (`video_tile_scheduler_util.{h,cc}`).
- **SKIP is already wired end-to-end (C handles it):** empty `TILES` → `TileSpecToTilePlan` empty `TilePlan` → `StreamingTilesToTensorBatch::EmitEmptyFrameIfNeeded` emits a `BATCH_INFO`-only frame (`total_batches=0`, `SetNextTimestampBound` on TENSORS) → `MergeTileBoxDetectionsAccumulator` (comment: "An empty frame (T==0, e.g. scheduler SKIP)…") emits empty `merged_fresh` → C's `TiledTrackingGraph` produces `TRACKER_DETECTIONS` → `TiledFrameSuppression(tracker_is_gap_fill_only=true)` appends all tracker boxes (fresh is empty) → tracker fills the frame. **So `REFRESH` does not need to be consumed; SKIP rides the empty-TILES path.**
- The `PRIOR_DETECTIONS` loopback is expressible in the **api2 builder** (no pbtxt): `PreviousLoopbackCalculator` (`MAIN`, `LOOP` back-edge, `PREV_LOOP`; `ImmediateInputStreamHandler` is intrinsic to the calculator) + `Destination::AsBackEdge()`. Template: `builder_test.cc` `TEST(BuilderTest, CanUseBackEdges)` (deferred-wire idiom: capture the node, wire `detections >> node.In("LOOP").AsBackEdge()` after the forward path is built). First frame → no `LOOP` packet → it advances the `PREV_LOOP` bound → scheduler sees empty priors → `ShouldRefreshFrame` returns DETECT immediately.
- Optical-flow producer chain (for the scheduler's own pass): `ImageTransformation(320×240) → MotionAnalysis → FlowPackager → TRACKING`, mirroring `box_tracking_cpu.pbtxt` node options. All these calculators are independently registered and link on macOS via the opencv static-archive fix from C.
- `TilingOptions` (`yolo_object_detector_options.proto`, field 10 of `YoloObjectDetectorOptions`) has 6 fields (`tile_rows`, `tile_cols`, `tile_overlap_fraction`, `explicit_tiles`, `tile_local_nms_iou_threshold`, `max_detections_after_tile_nms`); next free field number 7. The C++ struct `YoloObjectDetectorOptions::TilingOptions` mirrors it 1:1; `ConvertYoloObjectDetectorOptionsToProto` copies field-by-field. `TilingEnabled()` in `tiled_detection_utils.h` is the shared gate predicate. The C stream branch in `yolo_object_detector_graph.cc` already selects `TiledBoxTrackMergeGraph` (IMAGE input) under `use_stream_mode()`.

## Key decisions

1. **Independent second optical-flow pass; C frozen.** The scheduler's `TRACKING` comes from its own `OpticalFlowTrackingGraph` encapsulated inside the stream front. C's tracker keeps its internal flow. B touches none of C's subgraph files or tests.
2. **`REFRESH` left unconnected** — SKIP is fully handled by the empty-TILES path.
3. **Only `PRIOR_DETECTIONS` loops back** (frame N-1's `merged_dets`, frame-normalized, post-fusion). The scheduler's `TRACKING` is computed forward from the current frame, internal to the stream front.

## Default behavior (documented)

With the minimal public config, the 7 unexposed scheduler knobs keep their proto defaults, under which `ShouldRefreshFrame` SKIPs essentially **only on near-duplicate frames** (`is_duplicated`) — any real motion (`mean_motion > motion_refresh_threshold=0`) forces DETECT. A deliberately safe default (no recall loss on moving content; meaningful savings on static-camera / low-change footage). Aggressive motion-threshold skipping is a follow-up.

## TRACKING density contract (Codex [P1])

The scheduler's `TRACKING` is `Optional`, but with the default input-stream handler a *connected* `TRACKING` is still synchronized with the required `TILES`/`PRIOR_DETECTIONS` inputs — so the scheduler can stall if `TRACKING`'s timestamp bound does not advance every source frame. `MotionAnalysisCalculator` **buffers** frames and on a no-result `Process` returns with no output (`motion_analysis_calculator.cc:627`); it declares `SetOffset(TimestampDiff(0))` (`:369`). The existing scheduler tests fed *synthetic* per-frame `TrackingData`, never real MotionAnalysis — so this is **new, unlocked behavior** that B must pin.

**Contract:** `OpticalFlowTrackingGraph` must, for every source-frame timestamp, either emit a `TRACKING` packet or advance the `TRACKING` stream's timestamp bound, so the downstream scheduler never stalls. Frame 0 (and any warm-up frame with no flow yet) legitimately carries no packet — that is fine **iff** the bound advances (scheduler then sees `TRACKING` absent → `detect_without_tracking` + `priors_empty` → DETECT).

**Implementation + fallback:** first rely on `MotionAnalysis`'s `SetOffset(0)` bound propagation (verified by test #1/#2 below — they drive the scheduler through *real* MotionAnalysis and assert no stall). If a stall or sync mismatch is observed, `OpticalFlowTrackingGraph` terminates in a small per-frame materializer driven by the `IMAGE` tick (analogous to C's `DetectionsTickGateCalculator`, but emitting `TrackingData`-or-bound) to guarantee one aligned `TRACKING` slot per frame. (C's tracker is unaffected — it is frozen and already guarantees one packet per frame via its own tick gate; `box_tracker_calculator.cc:903`'s "boxes only when tracking present" is C's internal concern, already solved.)

## Data flow (stream branch)

```
image ─┬─► TiledDetectionStreamFrontGraph(IMAGE, PRIOR_DETECTIONS=priors)
       │      IMAGE ─► OpticalFlowTrackingGraph (ImgXform 320×240→MotionAnalysis→FlowPackager) ─► TRACKING ┐  (internal)
       │      IMAGE ─► TileGrid ─► VideoTileScheduler(TILES, PRIOR_DETECTIONS, TRACKING) ─► TileSpecToTilePlan ─► batcher ─► TENSORS + BATCH_INFO
       │            (scheduler REFRESH output left unconnected)
       │                                                       ↓ AddInference → YOLO decode (per-tile NMS)
       ├─► (image also feeds C's TiledBoxTrackMergeGraph IMAGE input — UNCHANGED from C)
       │
       └─► PreviousLoopback(MAIN=image, LOOP=merged_dets ⟲back-edge) ─► priors ─► (front PRIOR_DETECTIONS)
                                                                                                    ▼
              TiledBoxTrackMergeGraph(DETECTIONS, BATCH_INFO, IMAGE)  [C, frozen] ─► merged_dets ─► [label tail] ─► pixel results
                   └ merge → TiledTrackingGraph(IMAGE, merged_fresh) → suppression(gap-fill)   [C's own internal flow]
                          merged_dets ──────────────────────────────────────────────────────────⟲ loops to LOOP
```

## Components

### 1. `OpticalFlowTrackingGraph` (new subgraph, `mediapipe.tiled_detection`)
- `mediapipe/graphs/tiled_detection/optical_flow_tracking_graph.cc`. IO: `IMAGE: ImageFrame` → `TRACKING: TrackingData`.
- Internals: `ImageTransformationCalculator` (downscale 320×240) → `MotionAnalysisCalculator` → `FlowPackagerCalculator`, mirroring the node options in `box_tracking_cpu.pbtxt` (`MotionAnalysisCalculatorOptions` ANALYSIS_POLICY_CAMERA_MOBILE; `FlowPackagerCalculatorOptions` `binary_tracking_data_support: false`). Single-purpose, independently testable.

### 2. `TiledDetectionStreamFrontGraph` (new subgraph) — stream-mode sibling of `TiledDetectionFrontGraph`
- `mediapipe/graphs/tiled_detection/tiled_detection_stream_front_graph.cc`. IO: `IMAGE: ImageFrame` + `PRIOR_DETECTIONS: vector<Detection>` → `TENSORS` + `BATCH_INFO`.
- Internals: same as the IMAGE front PLUS its own flow + the scheduler:
  - `IMAGE → OpticalFlowTrackingGraph → TRACKING` (internal);
  - `IMAGE → TileGrid → VideoTileScheduler(TILES, PRIOR_DETECTIONS, TRACKING) → TileSpecToTilePlan → StreamingTilesToTensorBatch(IMAGE, TILE_PLAN)`;
  - `scheduler.Out("REFRESH")` left unconnected.
- The IMAGE-mode `TiledDetectionFrontGraph` is **untouched**.
- Options: reuse `TiledDetectionFrontGraphOptions` (it carries `tile_grid` + batch metadata); add `optional int32 max_scheduled_tiles = 7 [default = 0]` to that message — the stream front forwards it to the scheduler; the image front ignores it. The other scheduler knobs stay at their proto defaults.

### 3. `YoloObjectDetectorGraph` stream branch (edit the C branch; do not change C's subgraphs)

**Two independent gates (Codex [P1/P2] — keep them separate):**
- `tracking_enabled = use_stream_mode && TilingEnabled(tiling)` — **C's existing behavior, UNCHANGED.** When true, the stream path runs `TiledBoxTrackMergeGraph` (IMAGE input, C's own internal flow). This holds whether or not scheduling is on.
- `scheduling_enabled = tracking_enabled && tiling.enable_motion_scheduling()` — **B's addition.** It controls *only* whether the scheduler front + loopback are inserted. It must never gate tracking, and tracking must never depend on the scheduler's optical flow (the two flows are independent; C never receives an external TRACKING).

So: `scheduling_enabled=false` in stream mode → exactly C's current path (no regression). `scheduling_enabled=true` → swap the front + add the loopback; the merge/tracking half is byte-identical to C either way.

- In the existing `if (task_options.base_options().use_stream_mode())` tiled branch, when `scheduling_enabled`:
  - add a `PreviousLoopbackCalculator` (`MAIN`=the source image stream; `PREV_LOOP`→`priors`; `LOOP`=`merged_dets` wired `.AsBackEdge()` via the deferred idiom);
  - swap the front node from `TiledDetectionFrontGraph` to `TiledDetectionStreamFrontGraph`, feeding it `IMAGE` + `priors`; forward `max_scheduled_tiles` into its options;
  - everything downstream (AddInference → YOLO decode → `TiledBoxTrackMergeGraph(IMAGE)` → label tail) stays exactly as C wired it; capture the existing `merged_dets` Source and close the loopback with it.
- When stream mode but scheduling NOT enabled, keep C's current stream wiring (stateless-tiled + tracker, no scheduler). The IMAGE branch is unchanged.

### 4. Public config + gating
- `TilingOptions` (proto): add `optional bool enable_motion_scheduling = 7 [default = false];` and `optional int32 max_scheduled_tiles = 8 [default = 0];`. Mirror both in the C++ struct `YoloObjectDetectorOptions::TilingOptions` and in `ConvertYoloObjectDetectorOptionsToProto`.
- New predicate `SchedulingEnabled(tiling, running_mode)` in `tiled_detection_utils.h`: returns `tiling.enable_motion_scheduling() && TilingEnabled(tiling) && running_mode != IMAGE`. The graph builder uses it to choose the scheduler front.
- `enable_motion_scheduling=true` with `running_mode==IMAGE` (or without tiling enabled) is rejected at `Create`/`Detect` time with `kImageProcessingInvalidArgumentError` (mirrors how tiled mode rejects ROI/rotation).

## Timestamps / error handling
- The stream front's `TRACKING(N)` is produced by its own `MotionAnalysis` on frame N (internally-buffered N-1); frame 0 → empty/chunk-boundary `TrackingData` → scheduler hits `detect_without_tracking` + `priors_empty` → DETECT. `TRACKING` is Optional on the scheduler, so an absent/lagging packet degrades to DETECT, never crashes.
- Loopback frame 0 → empty priors → DETECT (PreviousLoopback advances the bound; no stall).
- SKIP frame → empty TILES → empty merged_fresh → C's tracker (gap-fill) supplies the frame (already proven in C).

## Testing (TDD, red→green, English)

1. **`OpticalFlowTrackingGraph` graph test:** feed ≥3 textured frames with motion through the REAL chain → assert it emits `TrackingData` densely (one packet per frame after warm-up) and that a `CloseAllPacketSources`/`WaitUntilDone` run terminates (no hang) — locking the density contract end of the pipeline.
2. **`TiledDetectionStreamFrontGraph` integration graph test** driving the scheduler through the REAL `OpticalFlowTrackingGraph` (not synthetic `TrackingData`) + a small loopback so priors flow: feed **moving** frames → DETECT (non-empty TENSORS / `BATCH_INFO.total_batches>0`); feed **duplicate** frames (after warm-up, non-empty priors) → SKIP (no TENSORS packet, `BATCH_INFO.total_batches==0`). Asserts both DETECT-vs-SKIP mechanics AND that the scheduler does not stall on real (buffered/latent) MotionAnalysis output — the [P1] no-stall arbiter. The `WaitUntilDone` completing is the no-deadlock proof.
3. **Public `DetectForVideo` e2e — two scenarios** (`yolo_object_detector_test.cc`):
   - **(a) Motion/panning:** panning boats with `enable_motion_scheduling=true` → boats persist across frames; the cyclic graph runs to completion without deadlock (end-to-end + loopback no-stall).
   - **(b) SKIP → tracker fill (the [P2] gap):** after a first DETECT frame, feed several **identical/static** frames at strictly increasing timestamps → assert each call returns a non-empty result **for that frame** (boats still present), proving the full public `empty-TILES → tracker fill → current-frame output` path, not just frame-0 DETECT. Because `TaskRunner::Process` waits-idle and returns the latest output, each `DetectForVideo` call is checked per-frame (the synchronous API returns that frame's result); pair with the graph-level `total_batches==0` assertion in test #2, which proves inference was actually skipped on those frames.
   - **(c) Negative:** `enable_motion_scheduling=true` + `RunningMode::IMAGE` (or without tiling) → `Create`/`Detect` returns InvalidArgument.

All of the above must fail first (red) before implementation. **No edits to C's existing tests** — they must keep passing untouched (a regression check, not a rewrite).

## Build / dependency notes
- New BUILD targets: `optical_flow_tracking_graph` (deps on `image_transformation_calculator`, `motion_analysis_calculator`, `flow_packager_calculator`), `tiled_detection_stream_front_graph` (deps on `:optical_flow_tracking_graph`, `video_tile_scheduler_calculator` + its `_cc_proto`, and the existing tiling front calculators). The YOLO graph gains a dep on `previous_loopback_calculator` and the new stream-front subgraph. The opencv static-archive fix from C already links the optical-flow stack on macOS.
- Verifiable on desktop C++ only (`--define MEDIAPIPE_DISABLE_GPU=1`).

## Risks / open items
- **The cyclic loopback in the full Tasks graph is the highest risk** (deadlock/stall if mis-wired). `PreviousLoopbackCalculator` + `.AsBackEdge()` is the proven breaker; e2e #3 + the `WaitUntilDone` in #2 are the arbiters.
- **Scheduler ↔ real MotionAnalysis density/latency (Codex [P1]):** the scheduler is, for the first time, fed real (buffered) MotionAnalysis flow rather than synthetic per-frame `TrackingData`. The "TRACKING density contract" section pins the requirement; tests #1/#2 drive the real chain and assert no stall; the per-frame `TrackingData` materializer is the spec'd fallback if MotionAnalysis's bound propagation proves insufficient.
- **Gate separation (Codex [P1/P2]):** tracking (C, unchanged) and scheduling (B, additive) are independent gates; an implementer must not make C depend on the scheduler's flow. The C-frozen design makes the dangerous coupling structurally impossible (C never takes an external TRACKING), and the `scheduling_enabled=false` regression path is exactly C's current stream behavior.
- Conservative default skipping (near-duplicates only) is by design; documented above so it isn't mistaken for a bug on moving content. Test #3(b) uses static frames precisely to exercise the SKIP path.
- Two optical-flow passes (scheduler's + C's tracker's) by the user's choice to keep C frozen; cheap relative to the tiled inference saved. Sharing one pass remains a documented follow-up.

## Sub-project boundary
B completes the A→C→B arc: A = static tiling, C = always-on tracking, B = motion-gated tiling (skip inference, tracker fills). After B, the tiled detector does motion-adaptive tracked video detection end to end. Exposing the scheduler's motion thresholds, sharing one optical-flow pass, and an OBB stream path remain documented follow-ups.
