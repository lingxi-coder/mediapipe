# Spec — Verify & harden the OBB OrientedObjectDetector Tasks API (Phase 2)

Date: 2026-06-04
Status: Design (pre-plan)
Branch: `dev` (long-lived integration branch; commit only when asked)
Roadmap: `docs/superpowers/specs/2026-06-01-roadmap.md` Phase 2 (Tasks API exposure).
Follows: `docs/superpowers/specs/2026-06-04-verify-yolo-object-detector-task-design.md` (same recipe; this is the oriented/OBB analogue).

## Problem

The C++ `OrientedObjectDetector` Tasks API exists on `dev`
(`mediapipe/tasks/cc/vision/oriented_object_detector/`: task class + graph +
options proto + integration test), but its integration test is **gated on a
`yolov8n-obb.tflite` fixture that does not exist** — every assertion path calls
`GTEST_SKIP()`, so the task has **never run end-to-end**. Additionally, the test
points at `cats_and_dogs.jpg`, but `yolov8n-obb` is a **DOTA** model (aerial
classes: plane, ship, vehicle, harbor, …) that detects nothing in a cat/dog
photo — so even with a model the current assertions could not pass. This effort
makes the OBB task provable on CPU with an appropriate image.

The local Python env can export + metadata-equip the model (verified in the YOLO
effort: `ultralytics` + lower-level `tflite_support.metadata`).

## Goal & success criteria

1. A real, MediaPipe-metadata-equipped `yolov8n-obb.tflite` and a DOTA-appropriate
   image (`boats.jpg`) exist at `mediapipe/tasks/testdata/vision/` (reproducible
   via a committed export script; the blobs are gitignored, glob-gated).
2. `oriented_object_detector_test` **runs** (not skips) on `boats.jpg` and passes
   with meaningful assertions: at least one oriented detection, ship class
   present, box dims `> 0`, `rotation` finite, scores ≥ threshold, `max_results`
   capped.
3. Detections **match the ultralytics `yolov8n-obb` oracle** on `boats.jpg`
   within tolerance (boxes + classes), validated once at bring-up.
4. A `YoloObbTensorsToOrientedDetectionsCalculator` unit test exercises the
   rotated-box decode directly (synthetic tensor → decoded `cx,cy,w,h,angle` +
   class), independent of the model fixture.
5. CPU build stays green; with the fixture absent the test still `GTEST_SKIP`s
   (CI-safe) and the build still works (glob `allow_empty`).

## Background facts (verified against the code)

- **The OBB decoder is already normalized-by-default — no pixel-space saga.**
  `YoloObbTensorsToOrientedDetectionsCalculator` (`...calculator.cc`) writes
  `d.set_cx(cx); d.set_cy(cy); d.set_width(w); d.set_height(h);` straight from the
  detect-head tensor (and the angle), assuming normalized `[0,1]` boxes — which
  is correct for the ultralytics TFLite export (the YOLO effort confirmed these
  exports emit normalized boxes). The result container
  (`oriented_object_detection_result.cc`) and `OrientedDetectionProjectionCalculator`
  scale to pixels (`od.width = d.width() * image_w`, etc.). So, unlike the YOLO
  verify, NO box-normalization option, heuristic, or graph change is needed.
- **Channel layout:** `channels = 4 + num_classes + 1` (the `+1` is the angle, at
  index `4 + num_classes`); box `cx,cy,w,h` at 0–3, class scores at `4 .. 4+nc-1`.
  For DOTA (`num_classes = 15`) the output is `[1, 20, 8400]`, CHANNELS_FIRST
  (the proto/option default; matches ultralytics export). No layout change.
- **Graph:** `oriented_object_detector_graph.cc` →
  `YoloObbTensorsToOrientedDetectionsCalculator` (sets `num_classes`, layout) →
  `YoloObbBatchDetectionsToSingleCalculator` → `RotatedNonMaxSuppressionCalculator`
  → `OrientedDetectionProjectionCalculator`. Requires metadata to exist + reads
  input `NormalizationOptions`; requires `num_classes` in options (15 for DOTA).
- **Result type:** `OrientedObjectDetectorResult` = vector of `OrientedDetection
  { std::vector<Category> categories; float cx, cy, width, height, rotation; }`
  (pixel units after projection). `Category{ int index; float score;
  optional<string> category_name; }` — the OBB graph does NOT map label
  ids→names (`category_name` empty) and does NOT apply category allow/deny
  (same gap as YOLO); assert on `index`/`score`, not name.
- **Existing test** (`oriented_object_detector_test.cc`): gated on
  `yolov8n-obb.tflite`, `num_classes = 15`, currently `kTestImage =
  "cats_and_dogs.jpg"` (wrong for DOTA — switch to `boats.jpg`), asserts box dims
  positive + `rotation` finite. `ModelPath()`/`ImagePath()` use
  `JoinPath("./", "/mediapipe/tasks/testdata/vision/", name)`.
- **`boats.jpg`** is ultralytics' canonical OBB demo image (a harbor of boats →
  DOTA `ship` class id 1), at `https://ultralytics.com/images/boats.jpg`.

## Approach (chosen)

Mirror the YOLO verify recipe; the OBB decoder needs no normalization work.

### Components

1. **`mediapipe/tasks/testdata/vision/export_yolov8n_obb_tflite.py`** (committed,
   parallel to `export_yolov8n_tflite.py`):
   - `YOLO("yolov8n-obb.pt").export(format="tflite", imgsz=640, nms=False)` →
     float32 `[1,20,8400]`; copy to `mediapipe/tasks/testdata/vision/yolov8n-obb.tflite`.
   - Hand-attach metadata via `tflite_support.metadata.MetadataPopulator` +
     `metadata_schema_py_generated`: input `NormalizationOptions(mean=[0],
     std=[255])` + RGB ImageProperties; DOTA 15-class label file
     (`yolov8n_obb_labels.txt`, from `model.names`).
   - Download `boats.jpg` (from `https://ultralytics.com/images/boats.jpg`) to
     `mediapipe/tasks/testdata/vision/boats.jpg` if absent.
   - Documented "how to regenerate"; all outputs gitignored.

2. **Fixture wiring:** add `filegroup(name="yolo_obb_test_model",
   srcs=glob(["yolov8n-obb.tflite","yolov8n_obb_labels.txt","boats.jpg"],
   allow_empty=True))` to `testdata/vision/BUILD`; add it to the
   `oriented_object_detector_test` `data` deps; `.gitignore` the three blobs.

3. **Decoder unit test:** add a test to
   `yolo_obb_tensors_to_oriented_detections_calculator_test.cc` — a synthetic
   CHANNELS_FIRST `[1, 4+nc+1, 1]` tensor with a known box `cx,cy,w,h`, a known
   angle in the last channel, and class scores → assert the decoded
   `OrientedDetection` has the expected `cx,cy,w,h,rotation`, argmax class index,
   and conf-threshold behavior. (If such a test file/case already covers this,
   extend it; otherwise add the case.)

4. **Harden the integration test** (`oriented_object_detector_test.cc`): set
   `kTestImage = "boats.jpg"`; keep `num_classes = 15`; in `DetectOnImage` assert:
   ≥1 detection; every detection has exactly one category with `index ∈ [0,15)`
   and `score ≥ score_threshold`; `width > 0 && height > 0`;
   `std::isfinite(rotation)`; `max_results` respected; and the `ship` class
   (index 1) present. Ensure `DetectForVideo` sets `num_classes = 15` and runs.

### Data flow

`Image → ImagePreprocessing (resize+normalize via metadata) → TENSORS →
Inference (yolov8n-obb.tflite) → [1,20,8400] → YoloObbTensorsToOrientedDetections
(decode xywh+angle+class, normalized) → flatten → RotatedNMS →
OrientedDetectionProjection (→ pixels) → OrientedObjectDetectorResult`.

## Validation

- **Decoder unit test** (CPU, model-independent) — the rotated-box decode math.
- **Integration gtest** (gated on `boats.jpg` + model) — the hardened assertions.
- **One-time oracle check** at bring-up — run ultralytics `yolov8n-obb` on
  `boats.jpg`, confirm the task's oriented detections match (class `ship`, box
  centers/sizes, rotation) within tolerance. Documented in the plan; not a
  committed target.

## Error handling

Preserve existing task error paths (missing metadata → actionable error;
`num_classes` unset → `RET_CHECK`; ROI unsupported → invalid argument).

## Out of scope (explicit)

- OBB language bindings (Python ctypes OBB task exists but is not re-verified
  here; iOS/Java/Web are build-deferred).
- Category allowlist/denylist + label-name mapping (same unimplemented gap as the
  YOLO graph — a shared follow-up).
- Letterbox/aspect-preserving preprocessing (MediaPipe stretches to 640×640;
  diverges from ultralytics on non-square inputs — shared follow-up).
- Quantized/GPU; metadata-derived `num_classes`.

## Verification environment

Fully CPU-verifiable here: `bazel test --define MEDIAPIPE_DISABLE_GPU=1
//mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test`
(with fixtures present runs; absent skips) and the decoder unit test. Model
export + image download run in the local Python env.
