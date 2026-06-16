# Expose BoTSORT Track IDs Through the Bindings — Design

**Date:** 2026-06-16
**Branch:** `dev` (MediaPipe fork; unmerged, no upstream PR)
**Status:** Approved design — ready for implementation plan

## Goal

Surface BoTSORT's persistent per-object **track ID** on output `Detection`s at every
layer — C++ task → C-API → Python — using the framework proto's existing
`optional string track_id` field. Output bounding-box geometry and recall are
unchanged; the BoxTracker path and all other detectors are unaffected (the change is
additive — the new field is empty when no track ID is present). This reverses the
original BoTSORT "IDs internal / parity output" decision (deferred decision B) for the
BOTSORT path only.

## Background

BoTSORT computes a persistent integer `track_id` per object but the
`BotsortTrackingCalculator` currently does **not** write it to the output Detection
(the shipped "parity output" contract). The framework `Detection` proto
(`mediapipe/framework/formats/detection.proto`) already has
`optional string track_id = 6` ("if detection is part of a track") and
`optional int64 detection_id = 7`. The Tasks-layer `Detection` container
(`components/containers/detection_result.h`) is **shared** across all detectors and
its `ConvertToDetection` currently ignores `track_id`. The C-API `MpDetection`
(`c/components/containers/detection_result.h`) and the Python `Detection` dataclass
(`python/components/containers/detections.py`) are likewise shared
(`MpYoloObjectDetectorResult` is `typedef MpDetectionResult`).

**Fusion subtlety (the crux):** the BOTSORT branch feeds
`TiledFrameSuppressionCalculator` with `tracker_is_gap_fill_only=true` — fresh-wins:
the per-frame fresh detections survive and BoTSORT's tracker detections only fill
gaps. The track IDs live on the *tracker* detections, so naively writing+plumbing the
ID would leave most output (fresh) detections with an empty ID — a half-feature.
**Approach A (chosen)** resolves this by transferring the matched tracker detection's
ID onto the surviving fresh detection during suppression.

**Field-survival (verified):** all three post-merge calculators copy the whole
Detection and mutate only `location_data`/labels, so `track_id` survives untouched to
the final protos that `ConvertToDetection` reads:
- `DetectionLabelIdToTextCalculator` — `push_back(input_detection)` then adds labels.
- `DetectionTransformationCalculator` — `Detection(kInDetection.Get())` copy-construct,
  mutates only `location_data`.
- `DetectionsDeduplicateCalculator` — `push_back(detection)`; when it folds
  same-bbox duplicates it keeps one Detection's `track_id` (same bbox = same object,
  acceptable).

## Architecture

### 1. Populate the ID — calculator + fusion

**`BotsortTrackingCalculator`** (`mediapipe/calculators/tensor/botsort_tracking_calculator.cc`):
write the track ID on each output Detection:
```cpp
det.set_track_id(std::to_string(t->track_id));
```
(reverses the current "intentionally NOT written" code + comment). The existing unit
test assertion `EXPECT_FALSE(last[0].has_track_id())` flips to expect a non-empty
`track_id`.

**`TiledFrameSuppressionCalculator`** (`mediapipe/calculators/tensor/tiled_frame_suppression_calculator.cc`)
— Approach A, ID propagation. In the `tracker_is_gap_fill_only` loop, currently each
tracker detection `t` is dropped if it IoU-overlaps any fresh detection (break on first
overlap) and appended otherwise. Change to:
- Track the **best-IoU** fresh match for `t` (not break-on-first).
- If `t` overlaps a fresh `f` (best match) AND `t` has a `track_id`: copy
  `t.track_id()` onto that fresh `f` (`combined[best_idx].set_track_id(t.track_id())`),
  only if `f` doesn't already carry one. Then drop `t` (fresh-wins geometry unchanged).
- If `t` overlaps nothing: append `t` as a gap-fill (keeps its own `track_id`).

This is additive and a **no-op for the BoxTracker path** (its tracker detections carry
no `track_id` — the BoxTracker subgraph strips to label via the codec). `GreedyDetectionNms`
returns the original Detection objects, so transferred/own `track_id`s survive the
final NMS (a test pins this).

### 2. Carry it up the binding chain (shared, additive)

- **Tasks container** (`components/containers/detection_result.{h,cc}`): add
  `std::optional<std::string> track_id = std::nullopt;` to `struct Detection`; in
  `ConvertToDetection`, set
  `detection.track_id = detection_proto.has_track_id() ? std::make_optional(detection_proto.track_id()) : std::nullopt;`.
- **C-API** (`c/components/containers/detection_result.h` + its converter + close/free):
  add `const char* track_id;` to `struct MpDetection` (nullptr when absent). The shared
  C converter (`CppConvertToDetectionResult`-equivalent) `strdup`s the string when the
  container's optional is set, else nullptr. The shared `MpDetection` free path frees it
  (only when non-null) — no leak. `MpYoloObjectDetectorResult` (typedef of
  `MpDetectionResult`) gets the field automatically.
- **Python** (`python/components/containers/detections.py`): add
  `track_id: Optional[str] = None` to the `Detection` dataclass; the ctypes
  `create_from_ctypes` reads the C `char*` (None when null, decode utf-8 otherwise).

### 3. Scope / blast radius

YOLO is where IDs originate, but the container / C struct / Python dataclass are
**shared** — so all detectors gain an optional `track_id` that is empty for them (their
protos never set it). Additive, natural home (the proto already had the field; the
container simply wasn't reading it). No behavioral change for any non-BoTSORT consumer.
The only shared *calculator* touched is `TiledFrameSuppressionCalculator` (ID transfer),
which is a no-op without tracker `track_id`s.

## Error handling / edge cases

- BoxTracker / no-tracking / IMAGE mode: no `track_id` is ever set → field stays
  empty / nullptr / None everywhere. Backward compatible.
- A fresh detection matched by multiple tracker detections: the best-IoU tracker wins
  the ID (deterministic).
- Same-bbox duplicates folded by `DetectionsDeduplicateCalculator`: the kept
  representative's `track_id` survives.

## Testing

- **Suppression calc unit test:** (a) tracker-overlaps-fresh → fresh survives with the
  transferred id; (b) non-overlapping tracker → gap-fill keeps its own id; (c) tracker
  dets WITHOUT id (BoxTracker-style) → no id written (no-op); (d) id survives the final
  `GreedyDetectionNms`.
- **`BotsortTrackingCalculator` test:** output Detections now carry `track_id` (flip the
  old `EXPECT_FALSE(has_track_id())`); a track's id is stable across frames.
- **Container conversion test:** proto with `track_id` → `Detection.track_id` optional
  set; proto without → `nullopt`.
- **C-API test:** `MpDetection.track_id` populated for a tracked detection / nullptr
  otherwise; CloseResult frees it (no leak / no double-free).
- **Python:** `Detection` dataclass `track_id` parsed from the C result (static-verified
  on this machine — toolchain broken; py_compile + a standalone/conversion check; full
  tests run where the toolchain works).
- **Real YOLO BOTSORT e2e:** extend the existing `TiledVideoTracksBoatsWithBotsort` to
  assert output detections carry non-empty `track_id`s that stay **stable across frames**
  (same object → same id on consecutive frames).

## Scope boundaries (YAGNI)

**In scope:** populate `track_id` on the BOTSORT path (calculator + suppression ID
propagation); thread the existing proto `track_id` (string) through the shared C++
container, C-API, and Python `Detection`; reuse the framework proto field (no proto
change). YOLO is the only producer.

**Out of scope:** `detection_id` (int64); OBB tracking; surfacing IDs for the BoxTracker
path (it has no persistent IDs); any change to the BoxTracker fusion; exposing IDs as
integers (kept as the proto's string form).

## Verifiability

The calculator, suppression, container, and C-API changes build+run here under
`--define MEDIAPIPE_DISABLE_GPU=1`; the YOLO BOTSORT e2e runs against the present
yolov8n fixture. Python is static-verified (py_compile + conversion check), consistent
with the prior Python bindings work.
