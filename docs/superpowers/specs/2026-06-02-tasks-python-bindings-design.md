# Spec 2.2 — Python Bindings for YOLO & Oriented Object Detectors

Date: 2026-06-02
Status: Design (pre-plan)
Branch: `dev` (long-lived integration branch; commit only when asked)

## Goal

Expose two already-shipped C++ Tasks to Python:

- `YoloObjectDetector` — axis-aligned YOLO detector (Phase 2.1a)
- `OrientedObjectDetector` — oriented bounding box / OBB detector (Phase 2.1b)

Both are bound via MediaPipe's established **ctypes-over-C-API** pattern (NOT pybind11),
matching the existing `object_detector` Python task exactly.

## Resolved decisions

| Decision | Choice |
|---|---|
| Scope | Both detectors (YOLO + OBB) |
| Running modes | All three: image, video, live-stream (async + result_callback) — full parity |
| Label names | Best-effort display-only label lookup in the **Python** layer (parse TFLite metadata label map; fall back to `None` when absent). C layer stays a pure index+score mirror. cc graphs untouched. Python label enrichment does not affect filtering, NMS, or `max_results`. |
| Test gating | Gate + skip on absent model fixtures (`yolov8n.tflite` / `yolov8n-obb.tflite`), matching the cc integration tests |
| Structure | Approach C — two MediaPipe-faithful sibling bindings, sequenced YOLO-first to prove the FFI/async/label harness before building the new oriented container |

## Architecture (three layers)

```
Python ctypes class   tasks/python/vision/{yolo,oriented}_object_detector.py
        ↓
C API layer           tasks/c/vision/{yolo,oriented}_object_detector/*.{h,cc}   (extern "C", into libmediapipe.so)  ← NEW
        ↓
C++ Task              tasks/cc/vision/{yolo,oriented}_object_detector/           (DONE — Phase 2.1)
```

Plus result containers at three sub-layers:
- C struct + C++→C converter — `tasks/c/components/containers/`
- Python ctypes mirror — `*_c.py`
- Python dataclass with `.from_ctypes` — `*.py`

YOLO reuses the existing axis-aligned `MpDetectionResult` chain. OBB needs a brand-new
oriented result container at all three sub-layers.

## Reference templates (verified)

- C API impl: `tasks/c/vision/object_detector/object_detector.cc` — mechanical mirror.
  Async callback lambda pattern at lines 105-127 is reused verbatim (shape).
- C API header: `tasks/c/vision/object_detector/object_detector.h`.
- Converter: `tasks/c/components/containers/detection_result_converter.{h,cc}`.
- C struct: `tasks/c/components/containers/detection_result.h`.
- Python task: `tasks/python/vision/object_detector.py` (ctypes, `_CTYPES_SIGNATURES`,
  `_AsyncResultDispatcher`, `from_ctypes`).
- cc Options field sets:
  - `YoloObjectDetectorOptions`: base_options, running_mode, display_names_locale("en"),
    max_results(-1), score_threshold(0.0f), category_allowlist, category_denylist,
    iou_threshold(0.45f), layout enum {kChannelsFirst=1,kChannelsLast=2},
    num_classes(0), result_callback. There is no YOLO `class_agnostic_nms` option.
    IMPORTANT (verified against `yolo_object_detector_graph.cc`): the 2.1a graph
    pipeline is `YoloTensorsToDetections → BatchToSingle → NonMaxSuppression →
    DetectionProjection → DetectionTransformation → DetectionsDeduplicate`. There is
    NO `DetectionLabelIdToTextCalculator` and NO allowlist/denylist filtering node.
    So `display_names_locale`, `category_allowlist`, and `category_denylist` are
    validated for mutual exclusivity but otherwise INERT today (never applied);
    category names come out empty. Only `score_threshold` / `iou_threshold` /
    `max_results` actually take effect (via NMS). The C/Python bindings faithfully
    mirror these fields so behavior matches the C++ API exactly, but must document
    them as currently no-op pass-throughs.
  - `OrientedObjectDetectorOptions`: base_options, running_mode, max_results(-1),
    score_threshold(0.25f), iou_threshold(0.45f), class_agnostic_nms(false),
    layout enum {kChannelsFirst=1,kChannelsLast=2}, num_classes(0), result_callback.
  - Result alias: `OrientedObjectDetectorResult = components::containers::OrientedObjectDetectionResult`.

## Coordinate-system contract

Follow the existing `master` SSD/ObjectDetector contract exactly: lower-level detector
packets use relative/normalized coordinates, while public Tasks results use original-image
pixel coordinates.

| Stage | SSD / `ObjectDetector` on `master` | YOLO axis-aligned | OBB |
|---|---|---|---|
| Raw tensor decode | `TensorsToDetectionsCalculator` emits `LocationData::RELATIVE_BOUNDING_BOX` | `YoloTensorsToDetectionsCalculator` emits `LocationData::RELATIVE_BOUNDING_BOX` in model-input-normalized space | `YoloObbTensorsToOrientedDetectionsCalculator` emits normalized `OrientedDetection` in model-input-normalized space |
| NMS / postprocess | Runs on relative detections | Runs on relative detections | Runs on normalized oriented detections |
| Projection to original image | `DetectionProjectionCalculator` keeps relative coordinates, projected to the original image | `DetectionProjectionCalculator` keeps relative coordinates, projected to the original image | `OrientedDetectionProjectionCalculator` keeps normalized coordinates, projected to the original image |
| Public graph / Task conversion | `DetectionTransformationCalculator` converts relative bbox to pixel `BOUNDING_BOX` before `ObjectDetectorGraph` output | `DetectionTransformationCalculator` converts relative bbox to pixel `BOUNDING_BOX` before `YoloObjectDetectorGraph` output | `ConvertToOrientedObjectDetectionResult(..., image_size)` converts normalized OBB to pixel `OrientedObjectDetectionResult` at the C++ Task result boundary |
| C++/C/Python Tasks result | Pixel bbox in `[0,image_width) x [0,image_height)` | Pixel bbox in `[0,image_width) x [0,image_height)` | Pixel OBB: `cx`, `cy`, `width`, `height` in original-image pixels; `rotation` in radians CCW |

Design rule: do not expose raw normalized graph packets as Python results. Python bindings
mirror the public C++ Task result containers, not intermediate graph packets. If future cc
work adds an OBB equivalent of `DetectionTransformationCalculator`, the Python/C API contract
stays pixel-unit because the public Task result is already pixel-unit.

## Design sections

### Section 1 — New C API result container (OBB only)

`tasks/c/components/containers/oriented_detection_result.h`:
```c
struct MpOrientedDetection {
  struct MpCategory* categories;
  uint32_t categories_count;
  float cx, cy, width, height, rotation;   // pixels + radians CCW
};
struct MpOrientedDetectionResult {
  struct MpOrientedDetection* detections;
  uint32_t detections_count;
};
```

`oriented_detection_result_converter.{h,cc}`:
- `CppConvertToOrientedDetectionResult(const components::containers::OrientedObjectDetectionResult&, MpOrientedDetectionResult*)`
- `CppCloseOrientedDetectionResult(MpOrientedDetectionResult*)`
- Reuses `category_converter` for the per-detection categories array, mirroring
  `detection_result_converter.cc`.

The C API mirrors the shipped C++ `OrientedObjectDetectionResult` exactly: `cx`, `cy`,
`width`, and `height` are in original-image pixel units, not normalized coordinates.
The C/Python binding must not re-normalize or multiply these values again. This matches the
`master` `ObjectDetector` pattern: normalized detector packets are converted to pixel-unit
Tasks results before crossing the public API boundary. No keypoints field is needed because
OBB detections carry none.

### Section 2 — C API layer (both detectors)

Two sibling dirs mirroring `tasks/c/vision/object_detector/`:

`tasks/c/vision/yolo_object_detector/yolo_object_detector.{h,cc}`:
- `typedef struct MpYoloObjectDetectorInternal* MpYoloObjectDetectorPtr;`
- `typedef MpDetectionResult MpYoloObjectDetectorResult;`
- `struct MpYoloObjectDetectorOptions { MpBaseOptions base_options; MpRunningMode running_mode; const char* display_names_locale; int max_results; float score_threshold; const char** category_allowlist; uint32_t category_allowlist_count; const char** category_denylist; uint32_t category_denylist_count; float iou_threshold; int layout; int num_classes; result_callback_fn result_callback; }`
- Functions: `MpYoloObjectDetectorCreate`, `...DetectImage`, `...DetectForVideo`,
  `...DetectAsync`, `...CloseResult`, `...Close`.

`tasks/c/vision/oriented_object_detector/oriented_object_detector.{h,cc}`:
- `typedef struct MpOrientedObjectDetectorInternal* MpOrientedObjectDetectorPtr;`
- `typedef MpOrientedDetectionResult MpOrientedObjectDetectorResult;`
- `struct MpOrientedObjectDetectorOptions { MpBaseOptions base_options; MpRunningMode running_mode; int max_results; float score_threshold; float iou_threshold; bool class_agnostic_nms; int layout; int num_classes; result_callback_fn result_callback; }`
- Uses the Section 1 converter.

Both: async callback reuses the `object_detector.cc:105-127` lambda pattern (marshal
`absl::StatusOr<CppResult>` → C struct → `result_callback(status, &result, &mp_image, ts)`
→ close). The C result and image pointers are valid only for the lifetime of the callback.
Python callback conversion must synchronously deep-copy all detections and categories before
the C callback closes the result.

Register both `*_c_lib` aggregator targets in `tasks/c/BUILD` next to
`object_detector_c_lib` so they compile into `libmediapipe.so`.

### Section 3 — Python ctypes result containers

- YOLO: reuse existing `tasks/python/components/containers/detections.py` +
  `detections_c.py`.
- OBB: new
  - `oriented_detections_c.py` — `MpOrientedDetectionResultC` / `MpOrientedDetectionC`
    ctypes structures mirroring Section 1.
  - `oriented_detections.py` — `@dataclass OrientedDetection { cx, cy, width, height,
    rotation, categories }` where the box fields are original-image pixels, and
    `@dataclass OrientedObjectDetectionResult` with `from_ctypes(c_result, label_map=None)`.
  - `from_ctypes` must deep-copy nested category arrays immediately; Python objects must
    never retain C pointers because live-stream C results are freed before the callback
    returns to MediaPipe.

### Section 4 — Python task classes

Mirror `tasks/python/vision/object_detector.py`:

`tasks/python/vision/yolo_object_detector.py`:
- `class MpYoloObjectDetectorOptionsC(ctypes.Structure)` with `_fields_` matching Section 2.
- `_CTYPES_SIGNATURES` for the 6 C functions (`CStatusFunction` for the status-returning
  ones, `CFunction` for `CloseResult`).
- `@dataclasses.dataclass YoloObjectDetectorOptions` (base_options, running_mode,
  display_names_locale, max_results, score_threshold, category_allowlist, category_denylist,
  iou_threshold, layout, num_classes, result_callback).
- `class YoloObjectDetector` with `create_from_options` / `detect` /
  `detect_for_video` / `detect_async`, returning `ObjectDetectorResult` (reused).

`tasks/python/vision/oriented_object_detector.py`:
- Same shape; options are base_options, running_mode, max_results, score_threshold,
  iou_threshold, class_agnostic_nms, layout, num_classes, result_callback; returns
  `OrientedObjectDetectionResult`.

Both load the lib via `mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)`;
live-stream wires the C callback through `_AsyncResultDispatcher` + `serial_dispatcher`.

### Section 5 — Label lookup (Python, best-effort)

Applies to BOTH detectors. Verified: neither the YOLO nor the OBB cc graph maps
category IDs to text (no `DetectionLabelIdToTextCalculator`), so both emit index+score
only. Python label lookup is therefore the SOLE source of category names for both. And
because no graph applies `category_allowlist`/`category_denylist`, allowlist/denylist
filtering happens NOWHERE end-to-end today — the binding must not imply otherwise.

- In `create_from_options`, when a model path/buffer is set, parse the TFLite metadata
  label map using MediaPipe's existing Python metadata utilities into an ordered
  index→name list stored on the detector instance.
- `from_ctypes` maps `category.index → category_name` using that list; when no label map
  is present (the common case for stock ultralytics YOLO/OBB exports), `category_name`
  stays `None`.
- Documented as best-effort. The C layer remains a pure index+score mirror; the shipped
  2.1 cc graphs are not modified. Because this happens after C++ detection, it is display-only
  enrichment and cannot implement `category_allowlist`, `category_denylist`, NMS behavior, or
  `max_results` behavior.
- Open implementation detail for the plan: confirm the exact metadata utility entry point
  (`tasks/python/metadata/...`) and the graceful path when the model carries no
  `TENSOR_AXIS_LABELS` / associated label file.

### Section 6 — Tests (gate + skip)

- C API tests: `tasks/c/vision/{yolo,oriented}_object_detector/*_test.cc`, gated on
  `yolov8n.tflite` / `yolov8n-obb.tflite`; `GTEST_SKIP()` with an explanatory message when
  the fixture is absent. Build and options-conversion tests run unconditionally; detector
  creation, inference, and async callback tests are fixture-gated because `Create` requires
  a valid model.
- Python tests: `tasks/python/test/vision/{yolo,oriented}_object_detector_test.py`,
  `unittest.skipUnless(fixture present)` for detect assertions; import + options
  construction run unconditionally. Live-stream tests must assert that the Python callback
  receives deep-copied result objects, not borrowed C memory.
- Coordinate assertions:
  - YOLO Python results must match existing `ObjectDetector` semantics: pixel
    `BoundingBox`, not `relative_bounding_box`.
  - OBB Python results must assert pixel-valued `cx`, `cy`, `width`, `height` for a known
    image size. Calculator-level tests may continue to assert normalized intermediate
    packets, but binding tests assert public Task result units.

### Section 7 — Build sequencing (Approach C)

1. YOLO C API (reuse containers) → YOLO C test → YOLO Python class → YOLO Python test.
2. Verify FFI + async + label-lookup harness end-to-end on YOLO.
3. OBB: new C container + converter → OBB C API → OBB C test → OBB Python container +
   class → OBB Python test.

## Out of scope

- cc-graph label-id-to-text wiring (kept best-effort in Python instead).
- Changing public Python/C Task results to normalized coordinates. Normalized coordinates
  remain an internal graph packet convention, matching `master` SSD/ObjectDetector.
- Web / Java / iOS bindings (2.3 / 2.4 / 2.5).
- Backend inference formats (Phase 6) and BoTSORT (Phase 8).

## Done criteria

- All new C API + Python targets build (`--define MEDIAPIPE_DISABLE_GPU=1`).
- C and Python tests build + run, skipping cleanly when fixtures absent.
- `object_detector` (C + Python) and the 2.1 cc tasks untouched.
- Both detectors importable and constructable from Python; all three running modes wired.
- Public Python results follow the `master` Tasks coordinate contract: raw/postprocess
  packets may be normalized, but C++/C/Python Task results are original-image pixel units.
- Label names populated when metadata present, `None` otherwise, as Python-only display
  enrichment.
