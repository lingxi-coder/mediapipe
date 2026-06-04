# Spec — Category names + allowlist/denylist for the YOLO & OBB detectors (Phase 2)

Date: 2026-06-04
Status: Design (pre-plan)
Branch: `dev` (long-lived integration branch; commit only when asked)
Roadmap: `docs/superpowers/specs/2026-06-01-roadmap.md` Phase 2 (Tasks API exposure).
Follows: the YOLO + OBB verify specs (`2026-06-04-verify-{yolo,oriented}-object-detector-task-design.md`), which flagged this gap.

## Problem

Both detector Tasks APIs accept `category_allowlist`/`category_denylist` and a
`display_names_locale`, and both copy the allow/deny lists into their options
protos, but **neither graph applies them**, and **neither populates
`category_name`** — results carry only the integer class `index`. (Verified: the
YOLO and OBB graphs have no label-mapping or category-filter calculator; the OBB
result conversion hardcodes `category_name = std::nullopt`.) This closes that gap
for both detectors, in-graph, reusing upstream MediaPipe mechanisms.

## Goal & success criteria

1. `category_name` (and `display_name` when present) is populated on YOLO and OBB
   detection results, from the model metadata's label file.
2. `category_allowlist` filters results to only the named classes;
   `category_denylist` excludes the named classes — for both detectors, applied
   **before** the `max_results` cap (upstream semantics).
3. Defaults unchanged: no allow/deny + no label file ⇒ current behavior
   (no `category_name`, no filtering); the existing tiled-detection pipeline that
   reuses the YOLO decoder is byte-identical (new options default empty).
4. CPU-verifiable: decoder unit tests (filter-by-index), a new OBB
   label-calculator unit test, and both integration tests assert names + allow/deny.

## Background facts (verified)

- **`DetectionLabelIdToTextCalculator`** (`mediapipe/calculators/util/`,
  `Detection` stream): options `label_map_path` | `label` (repeated string) |
  `label_items` (`Map<int64,LabelMapItem>`); sets the `Detection.label` (and
  display name) string from `label_id`. Reusable as-is for YOLO.
- **`OrientedDetection` proto** (`mediapipe/framework/formats/oriented_detection.proto`)
  has `repeated string label = 6` and `repeated string display_name = 9`
  (parallel to `Detection`) — so an analogous calculator can set them.
- **YOLO result conversion** `ConvertToDetectionResult` already maps
  `detection_proto.label(idx) → category_name` and `display_name(idx) →
  display_name`. So once the label string is set in-graph, YOLO gets
  `category_name` with NO container change.
- **OBB result conversion** `ConvertToOrientedObjectDetectionResult`
  (`oriented_object_detection_result.cc`) currently sets `category_name =
  std::nullopt` unconditionally — must be updated to read `d.label(i)` /
  `d.display_name(i)`.
- **Filtering by class index** is the upstream pattern:
  `TensorsToDetectionsCalculatorOptions` has `repeated int32 allow_classes` /
  `ignore_classes`; the base detector resolves allow/deny NAMES → index set via
  `GetAllowOrDenyCategoryIndicesIfAny(config, label_items)` (a small loop over
  `label_items` matching `.name()`). There is NO stock post-hoc category filter
  (`filter_detections_calculator` filters by score/size only).
- **Label map from metadata:** the export scripts attach the label file as the
  OUTPUT tensor's associated file (type `TENSOR_AXIS_LABELS`). The base detector's
  `GetLabelItemsIfAny` reads it via `metadata_extractor.GetAssociatedFile(name)` +
  `mediapipe::BuildLabelMapFromFiles(labels, display_names)`
  (`mediapipe/util/label_map_util.h`) → `Map<int64,LabelMapItem>`
  (`LabelMapItem{ name, display_name }`).
- **The two Yolo decoders** (`YoloTensorsToDetectionsCalculator`,
  `YoloObbTensorsToOrientedDetectionsCalculator`) compute the argmax class `best`
  per candidate before the conf-threshold check — the natural place to drop
  non-allowed classes. Both graphs already RET_CHECK allowlist+denylist mutual
  exclusion.

## Approach (in-graph; reuse upstream patterns; minimal new code)

### Names
- **YOLO:** insert the stock `DetectionLabelIdToTextCalculator` configured with
  `label_items` from metadata, after `NonMaxSuppression`, before
  `DetectionProjection`. (No result-container change.)
- **OBB:** new calculator `OrientedDetectionLabelIdToTextCalculator`
  (`mediapipe/calculators/util/oriented_detection_label_id_to_text_calculator.{cc,proto}`),
  modeled on the stock `DetectionLabelIdToTextCalculator` but operating on
  `std::vector<OrientedDetection>` — sets `label`/`display_name` from `label_id`
  using `label_map_path` | `label` | `label_items`. Insert after
  `RotatedNonMaxSuppression`, before `OrientedDetectionProjection`. Update
  `ConvertToOrientedObjectDetectionResult` to read `d.label(i)`/`d.display_name(i)`.

### Filtering (before the cap)
- Add `repeated int32 allow_classes` and `repeated int32 ignore_classes`
  (mutually exclusive; default empty) to BOTH
  `YoloTensorsToDetectionsCalculatorOptions` and
  `YoloObbTensorsToOrientedDetectionsCalculatorOptions`. In each decoder's
  per-candidate loop, after computing `best`, skip the candidate when an
  allow-set is non-empty and `best ∉ allow`, or a deny-set contains `best`.
  Default empty ⇒ no filtering (unchanged).
- Each graph resolves `category_allowlist`/`category_denylist` NAMES → index set
  via the metadata `label_items` (same logic as
  `GetAllowOrDenyCategoryIndicesIfAny`) and sets `allow_classes`/`ignore_classes`
  on its decoder options.

### Shared metadata→labels helper
Factor "read the output-tensor `TENSOR_AXIS_LABELS` associated file → ordered
label list + `LabelMapItem` map" into one small helper reused by both graphs
(e.g. `mediapipe/tasks/cc/vision/utils/`), so the two graphs don't duplicate the
metadata-reading + name→index logic.

### Data flow (YOLO)
`tensors → YoloTensorsToDetections(decode + allow/deny by index) →
BatchToSingle → NonMaxSuppression(max_results cap) →
DetectionLabelIdToText(label_items) → DetectionProjection →
DetectionTransformation → Dedup → result (category_name populated)`. OBB is
isomorphic with the new calculator + `RotatedNonMaxSuppression`.

## Error handling

- Using allowlist/denylist with NO label file in metadata → invalid-argument
  error (mirror upstream `kMetadataMissingLabelsError`), since names can't be
  resolved to indices.
- Allowlist + denylist both set → existing `RET_CHECK` (unchanged).
- Unknown/duplicate category names in the lists → ignored (upstream behavior).
- No metadata labels + no allow/deny → names simply absent (`category_name`
  unset); not an error.

## Testing (CPU)

- **Decoder unit tests** (`yolo_tensors_to_detections_calculator_test`,
  `yolo_obb_tensors_to_oriented_detections_calculator_test`): `allow_classes`
  keeps only those classes; `ignore_classes` drops them; empty = unchanged.
- **New OBB label calculator unit test**: `label_id`→`label`/`display_name` from
  a `label`/`label_items` config; `keep_label_id` honored.
- **Integration tests** (both, fixture-gated): `category_name` matches the
  expected class names (YOLO: e.g. "dog"/"cat" on cats_and_dogs; OBB: "ship" on
  boats.jpg); a `category_allowlist` of one class returns only that class; a
  `category_denylist` of that class excludes it.

## Out of scope

- Language bindings (iOS/Java/Web); letterbox/aspect-preserving preprocessing
  (shared follow-ups); score calibration; per-locale display-name selection
  beyond passing `display_names_locale` through to the label map.

## Verification environment

CPU-verifiable here: `bazel test --define MEDIAPIPE_DISABLE_GPU=1` for the two
decoder tests, the new label-calculator test, and the two task integration tests
(with the gitignored model fixtures present).
