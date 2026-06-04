# Spec — Category names + allowlist/denylist for the YOLO & OBB detectors (Phase 2)

Date: 2026-06-04
Status: Design (pre-plan)
Branch: `dev` (long-lived integration branch; commit only when asked)
Roadmap: `docs/superpowers/specs/2026-06-01-roadmap.md` Phase 2 (Tasks API exposure).
Follows: the YOLO + OBB verify specs (`2026-06-04-verify-{yolo,oriented}-object-detector-task-design.md`), which flagged this gap.

## Problem

The YOLO detector Tasks API already exposes `display_names_locale`,
`category_allowlist`, and `category_denylist`, and copies them into its graph
options, but the YOLO graph does not apply those fields and does not populate
`category_name` from model metadata.

The OBB detector has the same graph-level gap plus a public API gap: its graph has
no label-mapping/category-filter stage, and its current C++/C/Python options do
not expose `display_names_locale`, `category_allowlist`, or `category_denylist`.
The OBB result conversion also hardcodes `category_name = std::nullopt`.

This closes both gaps in-graph, reusing upstream MediaPipe mechanisms, while
adding the missing OBB API surface needed to configure the feature.

## Goal & success criteria

1. `category_name` (and `display_name` when present) is populated on YOLO and OBB
   detection results, from the model metadata's label file, without losing the
   original integer `Category.index`.
2. `category_allowlist` filters results to only the resolved known classes;
   `category_denylist` excludes the resolved known classes — for both detectors,
   applied **before** the `max_results` cap (upstream semantics; unknown names
   are ignored as described below).
3. OBB exposes the same C++/C/Python option surface as YOLO for
   `display_names_locale`, `category_allowlist`, and `category_denylist`.
4. Defaults unchanged: no allow/deny + no label file ⇒ current behavior
   (no `category_name`, no filtering); the existing tiled-detection pipeline that
   reuses the YOLO decoder is byte-identical (new options default empty).
5. CPU-verifiable: decoder unit tests (filter-by-index), metadata-resolution/API
   passthrough tests, a new OBB label-calculator unit test, and both integration
   tests assert names + allow/deny.

## Background facts (verified)

- **`DetectionLabelIdToTextCalculator`** (`mediapipe/calculators/util/`,
  `Detection` stream): options `label_map_path` | `label` (repeated string) |
  `label_items` (`Map<int64,LabelMapItem>`); sets the `Detection.label` (and
  display name) string from `label_id`. Reusable as-is for YOLO, but it clears
  `label_id` by default after adding text labels. The YOLO graph MUST configure
  `keep_label_id = true`, otherwise `ConvertToDetectionResult` converts the
  category index to `-1`.
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
- **OBB public options are missing these fields today.** Add them to the OBB
  proto, C++ options, C API options struct, Python dataclass/ctypes struct, and
  options→proto conversion. The C/C++/Python result containers already carry
  `Category.category_name`/`display_name`, so no result ABI expansion is needed.
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
  non-allowed classes. YOLO already RET_CHECKs allowlist+denylist mutual
  exclusion; OBB must add the same check when its public fields are added.

## Approach (in-graph; reuse upstream patterns; minimal new code)

### Names
- **YOLO:** insert the stock `DetectionLabelIdToTextCalculator` configured with
  `label_items` from metadata and `keep_label_id = true`, after
  `NonMaxSuppression`, before `DetectionProjection`. (No result-container change.)
- **OBB:** new calculator `OrientedDetectionLabelIdToTextCalculator`
  (`mediapipe/calculators/util/oriented_detection_label_id_to_text_calculator.{cc,proto}`),
  modeled on the stock `DetectionLabelIdToTextCalculator` but operating on
  `std::vector<OrientedDetection>` — sets `label`/`display_name` from `label_id`
  using `label_map_path` | `label` | `label_items`, and supports
  `keep_label_id` with the same default as the stock calculator. Configure it
  with `keep_label_id = true` in the OBB graph, insert after
  `RotatedNonMaxSuppression`, before `OrientedDetectionProjection`, and update
  `ConvertToOrientedObjectDetectionResult` to read `d.label(i)`/`d.display_name(i)`.
- Do not clear or rewrite `label_id` in either detector graph. The integer class
  index is part of the public result contract and remains the source of
  `Category.index`.

### Public API surface
- **YOLO:** no public API additions. Use the existing
  `display_names_locale`, `category_allowlist`, and `category_denylist` fields in
  the proto, C++ options, C API struct, and Python dataclass/ctypes struct.
- **OBB:** add `display_names_locale` (default `"en"`), `category_allowlist`, and
  `category_denylist` to:
  - `OrientedObjectDetectorOptions` proto (use the next field numbers after
    `num_classes`).
  - C++ `OrientedObjectDetectorOptions` and options→proto conversion.
  - C `MpOrientedObjectDetectorOptions` and C→C++ conversion.
  - Python `OrientedObjectDetectorOptions` dataclass and
    `MpOrientedObjectDetectorOptionsC` ctypes layout.
- Add OBB mutual-exclusion validation for allowlist+denylist at the graph/API
  boundary, matching YOLO's current behavior.
- Keep iOS/Java/Web bindings out of this phase, but do not exclude C or Python:
  they are required for the Python task integration tests.

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
- Match upstream unknown-name behavior exactly: unknown and duplicate names are
  ignored. If an allowlist contains only unknown names, the resolved allow set is
  empty and filtering is a no-op, not "drop all detections." This is surprising
  but is the documented upstream-compatible behavior; lock it with tests so it
  does not become accidental.

### Shared metadata→labels helper
Factor "read the output-tensor `TENSOR_AXIS_LABELS` associated file → ordered
label list + `LabelMapItem` map" into one small helper reused by both graphs
(e.g. `mediapipe/tasks/cc/vision/utils/`), so the two graphs don't duplicate the
metadata-reading + name→index logic.

The helper should expose both:
- `label_items` for the label calculators and result names.
- resolved allow/deny class-index sets for decoder options, including the
  upstream error when allow/deny is requested but no label file exists.

### Python label fallback
The current YOLO/OBB Python tasks have a best-effort metadata scan that fills
`category_name` after C API conversion. Do not rely on that fallback for this
feature. The authoritative names/filtering path is the C++ graph:
- Prefer graph-provided `category_name`/`display_name` whenever present.
- Keep or remove the fallback as a separate compatibility decision, but if kept,
  it must remain display-only and must not be used to implement allow/deny or
  `display_names_locale` semantics.
- Add tests that prove the graph path works through Python without depending on
  `_load_label_map()` / `_enrich_with_label_map()`.

### Data flow (YOLO)
`tensors → YoloTensorsToDetections(decode + allow/deny by index) →
BatchToSingle → NonMaxSuppression(max_results cap) →
DetectionLabelIdToText(label_items, keep_label_id=true) → DetectionProjection →
DetectionTransformation → Dedup → result (category_name populated, index
preserved)`. OBB is isomorphic with the new calculator +
`RotatedNonMaxSuppression`.

## Error handling

- Using allowlist/denylist with NO label file in metadata → invalid-argument
  error (mirror upstream `kMetadataMissingLabelsError`), since names can't be
  resolved to indices.
- Allowlist + denylist both set → invalid argument. YOLO keeps its existing
  validation; OBB adds equivalent validation with the new fields.
- Unknown/duplicate category names in the lists → ignored (upstream behavior).
  All-unknown allowlist resolves to an empty allow set and therefore no-ops.
- No metadata labels + no allow/deny → no label mapping is applied; the
  missing-name representation remains whatever the current binding already uses
  (for example empty string, `std::nullopt`, or Python `None`). Not an error.

## Testing (CPU)

- **Decoder unit tests** (`yolo_tensors_to_detections_calculator_test`,
  `yolo_obb_tensors_to_oriented_detections_calculator_test`): `allow_classes`
  keeps only those classes; `ignore_classes` drops them; empty = unchanged.
- **Metadata/filter-resolution tests**: allow/deny with missing labels returns the
  upstream metadata-missing-labels error; duplicate/unknown names are ignored;
  all-unknown allowlist is a no-op.
- **New OBB label calculator unit test**: `label_id`→`label`/`display_name` from
  a `label`/`label_items` config; `keep_label_id` false clears `label_id` and
  `keep_label_id` true preserves it.
- **OBB API passthrough tests**: C++ options→proto, C API conversion, and Python
  ctypes/dataclass construction copy `display_names_locale`,
  `category_allowlist`, and `category_denylist` correctly, including mutual
  exclusion validation.
- **Integration tests** (both, fixture-gated): `category_name` matches the
  expected class names (YOLO: e.g. "dog"/"cat" on cats_and_dogs; OBB: "ship" on
  boats.jpg); `Category.index` remains the expected class id after name mapping;
  a `category_allowlist` of one known class returns only that class; a
  `category_denylist` of that class excludes it.

## Out of scope

- Language bindings (iOS/Java/Web); letterbox/aspect-preserving preprocessing
  (shared follow-ups); score calibration; per-locale display-name selection
  beyond passing `display_names_locale` through to the label map.

## Verification environment

CPU-verifiable here: `bazel test --define MEDIAPIPE_DISABLE_GPU=1` for the two
decoder tests, the metadata/filter helper tests, the new label-calculator test,
OBB API passthrough tests, and the two task integration tests (with the
gitignored model fixtures present).
