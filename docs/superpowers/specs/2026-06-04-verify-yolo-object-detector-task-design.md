# Spec — Verify & harden the YOLO object detector Tasks API (Phase 2)

Date: 2026-06-04
Status: Design (pre-plan)
Branch: `dev` (long-lived integration branch; commit only when asked)
Roadmap: `docs/superpowers/specs/2026-06-01-roadmap.md` Phase 2 (Tasks API exposure).

## Problem

The C++ `YoloObjectDetector` Tasks API already exists on `dev`
(`mediapipe/tasks/cc/vision/yolo_object_detector/`: task class + graph + options
proto + integration test, plus a C API and Python ctypes binding). But its
integration test (`yolo_object_detector_test.cc`) is **gated on a
`yolov8n.tflite` fixture that does not exist** — every assertion path calls
`GTEST_SKIP()`. So the task has **never actually run end-to-end**; we cannot
currently prove it detects anything. This effort makes it provable on this
machine (desktop CPU, the only locally-verifiable target).

The full TFLite-export toolchain is present in the local Python env
(`ultralytics`, `tensorflow`, `onnx`, `onnx2tf`, `tf_keras`, `ai_edge_litert`,
`sng4onnx`, `onnxslim`), so a real model fixture can be produced here.

## Goal & success criteria

1. A real, MediaPipe-metadata-equipped `yolov8n.tflite` exists at
   `mediapipe/tasks/testdata/vision/yolov8n.tflite` (reproducible via a committed
   export script; the blob itself is gitignored, matching the repo's
   GCS-fetched-testdata convention).
2. `yolo_object_detector_test` **runs** (not skips) locally and passes with
   meaningful assertions on `bus.jpg`: the expected COCO classes are detected,
   boxes lie within the image, `score_threshold` filters, `max_results` caps,
   and `category_allowlist`/`category_denylist` work.
3. Detections **match the ultralytics oracle** within tolerance (box IoU ≥ 0.7,
   same classes) — validated once at bring-up, reusing the comparison approach
   from `mediapipe/examples/pytorch_yolo`.
4. The CPU build stays green; with the fixture absent the test still
   `GTEST_SKIP()`s so CI is unaffected.

## Background facts (verified against the code)

- **The graph requires TFLite Model Metadata.** `yolo_object_detector_graph.cc`
  errors `kMetadataNotFoundError` if the model has no
  metadata/subgraph_metadata, and runs `ImagePreprocessingGraph`, which reads
  `NormalizationOptions` from metadata. A raw ultralytics export has no MediaPipe
  metadata, so we must attach it.
- **`num_classes` must be set in options** (graph `RET_CHECK_GT(num_classes,0)`;
  metadata-derived `num_classes` is an explicit future enhancement). For COCO,
  80.
- **Output layout** default `kChannelsFirst` = `[1, 4+num_classes, anchors]` =
  `[1, 84, 8400]`, which matches ultralytics' default export. No change needed.
- **Box coordinate space — the primary integration gap.**
  `YoloTensorsToDetectionsCalculator` writes the detect-head `cx,cy,w,h`
  **directly** into a `relative_bounding_box` (`bb->set_xmin(cx - w/2)`,
  `set_width(w)`, …) with **no** input-dimension/scale option — i.e. it assumes
  the model already emits **normalized `[0,1]`** boxes. The `pytorch_yolo` demo
  confirms this: it divides `cx,cy,w,h` by input W/H in Python before calling the
  calculator. Ultralytics' TFLite export emits **pixel-space** boxes
  (`[0,640]`), so fed directly the boxes land far off-image.

## Approach (chosen)

Export-script + gitignored fixture + gated test (vs. committing a blob, or a
Python-only harness). Reproducible, uses the right tools, respects the no-blob
convention, and directly achieves "make the gated C++ test run".

### Components

1. **`mediapipe/tasks/testdata/vision/export_yolov8n_tflite.py`** (committed).
   - `YOLO("yolov8n.pt").export(format="tflite")` → float32 `yolov8n_float32.tflite`,
     input `[1,640,640,3]`, output `[1,84,8400]`.
   - Attach MediaPipe metadata with the **generic** metadata writer
     (`metadata_writers.metadata_writer.MetadataWriter`), NOT the
     `object_detector` writer — the latter assumes the 4-output SSD/TFLite_Detection_PostProcess
     head, whereas YOLO has a single raw `[1,84,8400]` output. The graph only
     needs metadata to *exist* + input `NormalizationOptions` (it derives
     `num_classes` from options and decodes in the calculator), so attach: input
     `NormalizationOptions` mean=`[0,0,0]`, std=`[255,255,255]` (ImageToTensor
     emits pixel/255 = [0,1], matching ultralytics' float32 input), and a COCO
     80-class label file (for category display names). Write to
     `mediapipe/tasks/testdata/vision/yolov8n.tflite`.
   - Documented "how to regenerate" header; the `.tflite` output is gitignored.

2. **Fixture wiring** — uncomment the gated `yolov8n.tflite` `data` dep in
   `yolo_object_detector/BUILD`; add the model path to `.gitignore`. Test keeps
   its `GTEST_SKIP()` guard.

3. **Box-normalization option (the fix the gap forces).** Add a default-off
   option to `YoloTensorsToDetectionsCalculatorOptions`:
   ```proto
   // When > 0, the model emits box cx,cy,w,h in input-PIXEL space; divide cx,w
   // by input_width and cy,h by input_height to produce normalized
   // relative_bounding_box. 0 (default) = boxes are already normalized [0,1]
   // (unchanged: the tiled-detection pipeline feeds normalized boxes).
   optional int32 input_width = 8 [default = 0];
   optional int32 input_height = 9 [default = 0];
   ```
   The calculator divides when both > 0. `YoloObjectDetectorGraph` sets them from
   the model's input tensor dimensions (which it already knows via the
   image-preprocessing/ImageToTensor output size). **Default 0 keeps Phase 1's
   tiled pipeline byte-identical** (it already feeds normalized boxes).

4. **Test hardening** — replace the weak `non-empty + score>0` checks with:
   detected classes include the expected COCO set on `bus.jpg` (persons + bus);
   every box within `[0,w)×[0,h)`; `score_threshold` raises → fewer/no detections;
   `max_results=k` caps the count; `category_allowlist`/`denylist` filter as
   specified. Expected values are derived once from the oracle and pinned with a
   tolerance, not hard-coded to brittle exact coordinates.

5. **Bring-up wiring** — set `num_classes=80` in the test options; confirm layout.

### Data flow (image mode)

`Image → ImagePreprocessingGraph (resize+normalize via metadata) → TENSORS →
InferenceCalculator (yolov8n.tflite) → [1,84,8400] → YoloTensorsToDetections
(decode + per-class conf threshold + NMS; normalize boxes via input_width/height)
→ Detections → projected to input-image pixels → YoloObjectDetectorResult`.

## Validation

- **C++ gtest** (`yolo_object_detector_test`, gated) — the hardened assertions
  above; this is the durable regression test.
- **One-time oracle check** at bring-up — run ultralytics `predict` on `bus.jpg`
  and IoU-match against the task's detections (≥ 0.7, same classes), reusing
  `pytorch_yolo`'s `compare()` logic. Documented in the plan; not a committed
  build target (it needs the Python env + oracle).

## Error handling

Preserve existing task error paths (missing metadata → actionable error;
`num_classes` unset → `RET_CHECK`; ROI unsupported → invalid-argument). The new
calculator option is validated (`RET_CHECK_GE(input_width,0)` etc.).

## Out of scope (explicit)

- OBB / `oriented_object_detector` verification (clean follow-up; same recipe
  with `yolov8n-obb.tflite` + the oriented result type).
- All language bindings (Python / iOS / Java / Web) — build-deferred here.
- Metadata-derived `num_classes` (keep it options-supplied).
- Quantized/int8 models; GPU/Metal delegate for this task.

## Verification environment

Fully CPU-verifiable here: `bazel test --define MEDIAPIPE_DISABLE_GPU=1
//mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test` with
the fixture present runs the assertions; without it, the test skips (CI-safe).
Model export runs in the local Python env.
