# Sub-project C — YOLO Tiled Detection × BoxTracker (Video) Design

**Status:** Approved for planning (2026-06-14)
**Branch:** `dev` (fork; no upstream PR)
**Predecessor:** Sub-project A (static tiled detection, shipped). See `[[tasks-tiled-detection-shipped]]`.
**Successor:** Sub-project B (VIDEO/LIVE_STREAM motion scheduling) builds on this.

## Goal

In `VIDEO` / `LIVE_STREAM` running mode, let `YoloObjectDetector`'s **tiled** path propagate
detections across frames with the existing optical-flow BoxTracker stack, fusing the
tracker-propagated boxes with the current frame's fresh tiled detections through the
already-built `TRACKER_DETECTIONS` seam of `TiledFrameSuppressionCalculator` (one global
NMS). The user-visible effect: objects momentarily missed by per-frame tiled detection are
filled in from recent history, improving temporal recall and identity stability.

`IMAGE` running mode is unchanged (stateless per-frame tiling).

## Scope

**In scope**
- Axis-aligned `YoloObjectDetector` only (it owns `TiledBoxMergeGraph`, which carries the
  `TRACKER_DETECTIONS` seam).
- The optical-flow BoxTracker stack already in the repo
  (`MotionAnalysis → FlowPackager → BoxTracker → TrackedDetectionManager`), reused via the
  registered `ObjectTrackingSubgraphCpu`.
- Full vertical slice (L2): the tracker is wired all the way into the public
  `DetectForVideo` / `DetectAsync` API and proven with a real-video e2e.

**Out of scope (explicit)**
- `OrientedObjectDetector` (OBB) tracking — `TiledObbMergeGraph` has no tracker seam and the
  converters are axis-aligned only; deferred to a follow-up.
- The base `ObjectDetector` Tasks class — it has no tiled branch at all.
- The motion-driven DETECT/SKIP scheduler (`VideoTileSchedulerCalculator`) — that is
  sub-project B. This sub-project runs the tracker on **every** frame (all frames DETECT).
- A manual `PRIOR_DETECTIONS` loopback — not needed here (see Key Insight); B introduces it.

## Background / current state (verified)

- `TiledFrameSuppressionCalculator`
  (`mediapipe/calculators/tensor/tiled_frame_suppression_calculator.cc`) already accepts an
  optional `TRACKER_DETECTIONS: std::vector<Detection>` input, concatenates it with the fresh
  `DETECTIONS`, and runs **one** global `GreedyDetectionNms`. Fully implemented and tested;
  in production (`TiledBoxMergeGraph`) the input is intentionally left unwired. **No
  calculator change is needed to accept tracker boxes** — the work is producing and wiring
  them.
- `TiledBoxMergeGraph` runs `class_agnostic = true`. In that mode
  `GreedyDetectionNms` reads only `score(0)` (it touches `label_id(0)` only when
  `!class_agnostic`, `detection_nms_util.cc:73`), and there is no `RET_CHECK` forcing
  `label_id`. So at the seam, tracker boxes need a valid `score` but not `label_id`.
- **However**, the YOLO tail after the merge graph is
  `DetectionLabelIdToText → ImageProperties + DetectionTransformation → DetectionsDeduplicate`,
  and `DetectionLabelIdToText` consumes the int `label_id`. The standard tracker output
  (`TrackedDetectionManagerCalculator::GetAxisAlignedDetectionFromTrackedDetection`,
  `tracked_detection_manager_calculator.cc:80-124`) writes back `detection_id` (from
  `previous_id`), the string `label`, and `score` from `label_to_score_map`, but **not** the
  int `label_id`. Therefore tracker boxes entering the seam (which is *before* the tail) must
  carry `label_id`. Restoring it is the single piece of net-new business logic.
- `ObjectTrackingSubgraphCpu` (`mediapipe/graphs/tracking/subgraphs/object_tracking_cpu.pbtxt`,
  `type: "ObjectTrackingSubgraphCpu"`) bundles
  `DetectionUniqueId → DetectionsToTimedBoxList → BoxTrackingSubgraphCpu (ImageTransformation
  320×240 → MotionAnalysis → FlowPackager → BoxTracker) → TrackedDetectionManager`. Inputs
  `VIDEO` (ImageFrame) + `DETECTIONS` (vec<Detection>); output `DETECTIONS` (tracked
  vec<Detection>). The `CANCEL_OBJECT_ID` back-edge is handled internally.
- `use_stream_mode` is a `base_options.proto` field (`base_options.proto:40`), already set by
  the wrapper (`yolo_object_detector.cc`). Graph builders read it via
  `tasks_options.base_options().use_stream_mode()` (pattern: `hand_landmarker_graph.cc:296`).
  The tiled YOLO graph currently does **not** read it; the topology is identical across
  running modes today.

## Key insight: no loopback, no cycle

BoxTracker and `TrackedDetectionManager` hold their cross-frame history **internally** —
each frame they propagate past detections to the current frame using current optical flow,
then associate the current fresh detections. So the integration is a pure forward DAG:

```
merged_fresh ──┬─────────────────────────────────────────────► TiledFrameSuppression ──► final
               └──► TiledTrackingGraph (+ current frame) ──────►        (DETECTIONS + TRACKER_DETECTIONS)
```

`merged_fresh` fans out to the tracker and to the suppressor; both converge at the
suppressor. There is **no** feedback edge between graph nodes (the only back-edge,
`CANCEL_OBJECT_ID`, is inside `ObjectTrackingSubgraphCpu`). A manual `PRIOR_DETECTIONS`
loopback is unnecessary and is deferred to sub-project B (the scheduler needs it).

## Data flow (stream-mode tiled branch)

```
Image(ImageFrame)
  ├─► [existing tiled inference chain] ─► MergeTileBoxDetectionsAccumulator
  │        merged_fresh: vector<Detection> (label_id+score, frame-normalized, @source_ts)
  │            ├──────────────────────────────────────────────────────────────────────────┐
  │            └──► TiledTrackingGraph(IMAGE=Image, DETECTIONS=merged_fresh)                │
  │                     └─► tracker_dets: vector<Detection> (label_id restored)             │
  └──► (Image also feeds TiledTrackingGraph as VIDEO)                                       │
                                                                                            ▼
                         TiledFrameSuppression(DETECTIONS=merged_fresh, TRACKER_DETECTIONS=tracker_dets)
                              └─► [optional ClipDetectionVectorSize] ─► YOLO label tail ─► pixel results
```

All boxes are frame-normalized `[0,1]` throughout; the seam is **before** the
pixel/label tail, so no pixel conversion happens at the tracker.

## Components

### 1. `DetectionLabelIdCodecCalculator` (new, net-new logic — stateless)

- **Location:** `mediapipe/calculators/tensor/detection_label_id_codec_calculator.{cc,proto}`
- **Why string-encoding, not a `detection_id` join:** `DetectionUniqueIdCalculator` (inside
  `ObjectTrackingSubgraphCpu`) **overwrites** `detection_id` with a fresh per-frame id
  (`detection_unique_id_calculator.cc:82,99` — no `has_detection_id` guard), so the current
  `merged_fresh` and the tracked output share **no** `detection_id`, and a propagated-only box
  has no current fresh peer to join against anyway. The robust carrier is the one the tracker
  is built around: `TrackedDetectionManager` seeds each track's `label_to_score_map` from the
  **string** `label` (`tracked_detection_manager_calculator.cc:73-74`,
  `tracked_detection.cc:86`) and writes it back on output (`:119-122`). So we stuff the int
  `label_id` into that string label going in, and parse it back out — the manager carries the
  class to propagated-only boxes for free, with no graph-side state.
- **API:** api2 `Node`. Single calculator with a direction option, instantiated twice.
  - `Input<std::vector<Detection>> kIn{"DETECTIONS"}` → `Output<std::vector<Detection>>
    kOut{"DETECTIONS"}`.
  - **ENCODE:** for each Detection, set `label[i] = absl::StrCat(label_id(i))` (parallel to the
    existing `score` array), so the manager carries the class as a string. Leaves geometry and
    `score` untouched.
  - **DECODE:** for each Detection, `SimpleAtoi(label(i)) → label_id[i]` and clear the
    synthetic `label` (so the downstream `DetectionLabelIdToText` tail sees uniform `label_id`
    and no stray text). **If a Detection ends up with no parseable `label_id` (parse failure or
    an empty `label` set), the whole Detection is dropped (logged once)** — a categoryless box
    must never reach the public API, where `ConvertToDetection` would emit category index `-1`
    / empty name (`detection_result.cc:29`). This is the "fail safe, not loud-junk" choice
    Codex [P2] asked for.
  - Emit at the input timestamp (default offset 0). Stateless — no map, no TTL.
- **Options proto:** `DetectionLabelIdCodecCalculatorOptions` with
  `extend mediapipe.CalculatorOptions { optional DetectionLabelIdCodecCalculatorOptions ext =
  471230016; }`, an `enum Direction { ENCODE = 0; DECODE = 1; }`, and
  `optional Direction direction = 1 [default = ENCODE]`. (471230016 is the next free fork
  extension number; in use today: 001-007, 010-015, 020.)

### 2. `TiledTrackingGraph` (new subgraph)

- **Registered name:** `mediapipe.tiled_detection.TiledTrackingGraph`
- **Location:** `mediapipe/graphs/tiled_detection/tiled_tracking_graph.cc`
- **IO:** `Input IMAGE: ImageFrame`, `Input DETECTIONS: std::vector<Detection>` (merged_fresh,
  carries `label_id`+`score`) → `Output TRACKER_DETECTIONS: std::vector<Detection>` (carries
  `label_id`, **exactly one packet per source frame, empty when the tracker produced nothing**).
  The `label_id` codec is fully internal, so the external contract is `label_id` in /
  `label_id` out.
- **Internals:** `DetectionLabelIdCodecCalculator[ENCODE]`(DETECTIONS) →
  `ObjectTrackingSubgraphCpu`(VIDEO=IMAGE, DETECTIONS=encoded) → tracked →
  `DetectionLabelIdCodecCalculator[DECODE]` → `DetectionsTickGateCalculator`(TICK=DETECTIONS,
  DATA=decoded) → TRACKER_DETECTIONS.
- **Why the tick gate (Codex [P1#2]):** `TrackedDetectionManagerCalculator` emits a
  `DETECTIONS` packet **only inside** the non-empty `TRACKING_BOXES` branch
  (`tracked_detection_manager_calculator.cc:208`), so the tracker stream has gaps (first
  frames before optical flow exists, no-track frames). `TRACKER_DETECTIONS` is a synchronized
  optional input of `TiledFrameSuppression`, so an un-materialized gap can stall or add
  unbounded latency to `DetectForVideo` / `DetectAsync`. The new
  `DetectionsTickGateCalculator` (small, stateless) is clocked by the `DETECTIONS` stream
  (1:1 with source frames) and emits the decoded tracker vector at each tick if present, else
  an **empty** `std::vector<Detection>` — guaranteeing one aligned packet per source frame.
  No off-the-shelf calculator does "empty-default on tick gap" (`PacketClonerCalculator`
  repeats the *previous* packet, which would resurrect stale boxes), hence a dedicated unit.
  Location `mediapipe/calculators/tensor/detections_tick_gate_calculator.cc`.
- **Responsibility:** "given frames + fresh detections, produce class-carrying
  tracker-propagated detections, one packet per source frame." Single purpose; independently
  testable; reusable by B.
- **Options:** none in v1 — tracker/manager knobs come from `ObjectTrackingSubgraphCpu`
  defaults (incl. its 320×240 motion-analysis downscale). Parameterizing is a follow-up.

### 3. `TiledBoxTrackMergeGraph` (new subgraph — stream-mode sibling of `TiledBoxMergeGraph`)

- **Registered name:** `mediapipe.tiled_detection.TiledBoxTrackMergeGraph`
- **Location:** `mediapipe/graphs/tiled_detection/tiled_box_track_merge_graph.cc`
- **IO:** `Input DETECTIONS: std::vector<std::vector<Detection>>` (per-batch tile-local) +
  `Input BATCH_INFO: TensorBatchInfo` + `Input IMAGE: ImageFrame` → `Output DETECTIONS:
  std::vector<Detection>` (final, frame-normalized).
- **Internals:** `MergeTileBoxDetectionsAccumulator` → `merged_fresh`, which fans out to
  `TiledTrackingGraph`(IMAGE, merged_fresh) → `tracker_dets` and to
  `TiledFrameSuppression(DETECTIONS=merged_fresh, TRACKER_DETECTIONS=tracker_dets,
  tracker_is_gap_fill_only=true)` → optional `ClipDetectionVectorSize`.
- **Options:** reuse the existing `TiledBoxMergeGraphOptions` message (`iou_threshold`,
  `class_agnostic`, `max_detections`) — same suppression knobs; `class_agnostic` stays
  `true` for the single-path YOLO NMS. `max_detections == 0` rejected at init (mirror
  `TiledBoxMergeGraph`).
- **`TiledBoxMergeGraph` (non-tracking / IMAGE path) is unchanged.**

### 3b. `TiledFrameSuppressionCalculator` change — fresh-wins fusion (Codex [P1#1])

- **Why:** `GreedyDetectionNms` sorts purely by `score(0)` (`detection_nms_util.cc:57`), and a
  tracker box's score is a **historical max** (`tracked_detection.cc:90`,
  `MergeLabelScore` `:131`). The manager prefers fresh geometry for *associated* objects
  (`tracked_detection_manager.cc:59`), but on an association miss a stale, high-historical-score
  tracker box could outrank — and thus suppress — a correct lower-score fresh box, yielding
  stale geometry or a duplicate. Today the calculator blindly concatenates fresh + tracker
  before one NMS.
- **Change:** add `optional bool tracker_is_gap_fill_only = 4 [default = false];` to
  `TiledFrameSuppressionCalculatorOptions` (field 4 is next free after 1-3; **default `false` ⇒ byte-identical
  to today**, so the IMAGE `TiledBoxMergeGraph` path and all existing tests are unaffected).
  When `true`, before the global NMS, **drop every `TRACKER_DETECTIONS` box that overlaps any
  `DETECTIONS` (fresh) box** (IoU ≥ `iou_threshold`, honoring `class_agnostic`). Fresh always
  wins where it exists; the tracker contributes only genuine gap-fills. `TiledBoxTrackMergeGraph`
  sets it `true`.
- This lives in the calculator whose sole job is fresh+tracker fusion — no new calculator, no
  score-decay magic number.

### 4. `YoloObjectDetectorGraph` change

- In the tiled branch (`yolo_object_detector_graph.cc`), read
  `tasks_options.base_options().use_stream_mode()`:
  - **stream mode:** use `TiledBoxTrackMergeGraph`, teeing the `FromImageCalculator`
    `IMAGE_CPU` (ImageFrame) stream into its `IMAGE` input.
  - **IMAGE mode:** use `TiledBoxMergeGraph` (current behavior).
- Forward the same suppression options to whichever merge graph is selected.

### 5. Wrapper (`yolo_object_detector.cc`)

- No change expected: `Detect` / `DetectForVideo` / `DetectAsync` already handle the tiled
  path in all running modes and `use_stream_mode` is already set. Verify the
  `DetectForVideo` / `DetectAsync` IMAGE-only packet mapping reaches the new branch correctly;
  add `FlowLimiterCalculator` for `LIVE_STREAM` exactly as the base detector does (it already
  does for the tiled path).

## Data formats, coordinates, timestamps

- Boxes are `LocationData::RELATIVE_BOUNDING_BOX`, frame-normalized `[0,1]`, origin top-left,
  end to end. No pixel conversion at the seam.
- `TimedBoxProto.time_msec` is milliseconds; graph timestamps are microseconds — the existing
  converters divide by 1000. Real-video frame spacing is ≥ 1 ms, so the truncation is safe.
- `merged_fresh` and the source `ImageFrame` share the source-frame timestamp; the merge
  accumulator emits `merged_fresh` at `info.source_frame_timestamp` and the original
  `ImageFrame` is at the same source timestamp. `ObjectTrackingSubgraphCpu`'s
  `SyncSetInputStreamHandler` aligns them.

## Error handling / degenerate cases

- First frame (no prior flow) / empty fresh detections / empty track set → the
  `DetectionsTickGateCalculator` still emits an **empty** `TRACKER_DETECTIONS` at the source
  timestamp, so `TiledFrameSuppression` runs on time and degrades to fresh-only. No stall.
- A tracked box that cannot be assigned a `label_id` under DECODE is **dropped entirely** (not
  passed on label-less), so no category `-1` reaches the public API.
- With `tracker_is_gap_fill_only=true`, a tracker box overlapping a fresh box is dropped pre-NMS;
  fresh geometry/score always wins, tracker fills only gaps.
- The codec and tick gate are effectively stateless w.r.t. detection identity; all cross-frame
  tracking state lives in `TrackedDetectionManager` (which already expires tracks after its own
  timeout and emits `CANCEL_OBJECT_ID`).

## Testing strategy (TDD, red→green, English test names/comments)

1. **`DetectionLabelIdCodecCalculator` unit test** (`CalculatorRunner`): ENCODE writes
   `label[i] = str(label_id[i])` and leaves `score`/geometry intact; DECODE round-trips
   `ENCODE`'d input back to the original `label_id` with the synthetic `label` cleared; a
   Detection whose `label` is non-numeric or empty is **dropped entirely** (logged once) — no
   label-less Detection is emitted (Codex [P2]).
1b. **`DetectionsTickGateCalculator` unit test:** a tick with a present DATA packet forwards it;
   a tick with no DATA at that timestamp emits an **empty** `std::vector<Detection>`; output
   timestamp equals the tick timestamp.
1c. **`TiledFrameSuppressionCalculator` fresh-wins test:** with `tracker_is_gap_fill_only=true`,
   a high-score tracker box overlapping a lower-score fresh box is dropped (fresh kept); a
   non-overlapping tracker box survives (gap fill). With the option `false` (default), behavior
   is byte-identical to today (regression guard for the IMAGE path).
2. **`TiledTrackingGraph` graph test:** synthetic ImageFrame sequence + fresh detections →
   asserts tracked output carries the round-tripped `label_id` (not a stray string `label`),
   persists an object across frames, and emits **exactly one packet per source frame including
   the first (empty) frame** (no stall).
3. **`TiledBoxTrackMergeGraph` graph test:** drive per-batch detections where a fresh
   detection is present at frame T and **absent** at frame T+1 → assert the object still
   appears at T+1 via the tracker path (the core "tracker fills the gap" behavior), the
   fresh+tracker fusion is deduped by the single global NMS, and a fresh box at T+1 overrides
   an overlapping stale tracker box (fresh-wins).
4. **Public `YoloObjectDetector.DetectForVideo` e2e:** generate a deterministic video by
   translating `boats.jpg` a few pixels per frame for ~10 frames (real, trackable motion);
   run with tiling + `RunningMode::VIDEO`; assert temporal persistence / track-stable boat
   detections across frames (output in pixel units, as the IMAGE-mode tiled e2e already
   asserts).

All of the above must fail first (red) before implementation.

## Build / dependency notes

- `MotionAnalysis` / `BoxTracker` / `FlowPackager` pull in additional OpenCV. `calib3d`
  (`libopencv_calib3d.dylib`) is already added to `third_party/opencv_macos.BUILD`; any
  further missing dylib will surface at the red-test link step and is added there.
- New BUILD targets: two calculators — `DetectionLabelIdCodecCalculator` (+ its `_cc_proto`)
  and `DetectionsTickGateCalculator` — the two subgraphs, and their tests; plus a one-field
  proto bump to `tiled_frame_suppression_calculator.proto` (`tracker_is_gap_fill_only`). The
  tracking subgraph depends on `ObjectTrackingSubgraphCpu`'s target and the constituent
  video-tracking calculators.
- Verifiable locally on desktop C++ only (`--define MEDIAPIPE_DISABLE_GPU=1`), per the fork's
  build constraints.

## Risks / open items

- `ObjectTrackingSubgraphCpu` carries a fixed 320×240 motion downscale and default tracker
  options; v1 inherits them. If boats-scale motion needs different tuning, parameterize in a
  follow-up (would add `TiledTrackingGraphOptions`).
- The deterministic panning-boats fixture must produce motion the sparse optical flow can
  lock onto; if flow is too weak at a few px/frame, increase the per-frame translation. The
  e2e assertion is temporal persistence, not an exact box, to stay robust to fixture/model
  drift.

## Sub-project boundary (C → B)

C delivers tracking that runs every frame (all DETECT) and is reachable from the public video
API. B inserts `VideoTileSchedulerCalculator` between `TileGrid` and `TileSpecToTilePlan`,
turning some frames into SKIP; on SKIP frames the tracker built here keeps propagating and is
the sole source of that frame's boxes. B also introduces the `PRIOR_DETECTIONS` loopback the
scheduler needs.
