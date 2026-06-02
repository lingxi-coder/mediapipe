# Spec 2.5 — iOS bindings for YOLO & Oriented Object Detectors

Date: 2026-06-02
Status: Design (pre-plan)
Branch: `dev` (long-lived integration branch; commit only when asked)

## Goal

Expose two already-shipped C++ detectors to iOS as Objective-C(++) tasks:

- `YoloObjectDetector` — axis-aligned YOLO detector (Phase 2.1a)
- `OrientedObjectDetector` — oriented bounding box / OBB detector (Phase 2.1b)

Bound via MediaPipe's established iOS Tasks pattern: an ObjC(++) class wrapping
`MPPVisionTaskRunner`, which runs the cc **graph** (referenced by registered
name) — NOT the cc Task class. Mirrors `tasks/ios/vision/object_detector`.

## Why iOS (context)

This phase was reached by pivoting off Phase 2.3 (Web). The web vision WASM is a
**prebuilt GCS blob** with no Emscripten/`wasm_cc_binary` toolchain in this fork,
so our custom graphs can't be linked in and web bindings can't run. iOS (and
Android) instead **compile the cc graphs from source**, so our
`YoloObjectDetectorGraph` / `OrientedObjectDetectorGraph` link in and run. iOS was
chosen first: Xcode 26.3 is installed and we're on macOS, giving native build (and
simulator) verification on this machine. (Android/Java remains a viable follow-on.)

## Resolved decisions

| Decision | Choice |
|---|---|
| Scope | Both detectors (YOLO + OBB) |
| Running modes | All three: image, video, liveStream (delegate) — full parity with the object_detector sibling |
| Label names | Mirror the graph: category name/displayName left empty (graphs emit index+score only). NO TFLite-metadata label extractor dependency in the framework. |
| Structure | Approach C — two `MPPObjectDetector`-faithful sibling bindings, sequenced YOLO-first to prove the graph + task-runner + delegate harness before the new oriented result + pixel-conversion helper |
| OBB coordinates | Pixel-unit results produced in the ObjC result `+Helpers` by replicating the cc Task's normalized→pixel scale using the `IMAGE` output packet dims. Graph stays untouched. |

## Architecture (verified against the object_detector sibling)

The iOS task does NOT call the cc Task class. It builds an `MPPTaskInfo` naming the
graph + streams + options, hands it to `MPPVisionTaskRunner`, then converts output
packets to an ObjC result via a `+Helpers` category. Reference (read before
implementing):
- `tasks/ios/vision/object_detector/sources/MPPObjectDetector.{h,mm}` — the task.
  Key constants: graph name `mediapipe.tasks.vision.ObjectDetectorGraph`; streams
  `image_in`/`image_out` (IMAGE), `norm_rect_in` (NORM_RECT), `detections_out`
  (DETECTIONS). LiveStream uses a `packetsCallback` + a serial dispatch queue +
  a weak delegate.
- `.../sources/MPPObjectDetectorOptions.{h,m}` — options + `…LiveStreamDelegate`.
- `.../sources/MPPObjectDetectorResult.{h,m}` — `MPPTaskResult` subclass holding
  `NSArray<MPPDetection*>`.
- `.../utils/sources/MPPObjectDetectorResult+Helpers.mm` — converts the
  `std::vector<Detection>` packet → `MPPObjectDetectorResult` via
  `[MPPDetection detectionWithProto:]`.
- `.../utils/sources/MPPObjectDetectorOptions+Helpers.mm` — writes the options
  proto ext: `optionsProto->MutableExtension(<Options>Proto::ext)` + field setters.
- `.../BUILD` — three `objc_library` targets (`MPPObjectDetectorResult`,
  `MPPObjectDetectorOptions`, `MPPObjectDetector`); the detector target depends on
  `//mediapipe/tasks/cc/vision/object_detector:object_detector_graph` (from-source
  graph linkage) + the utils helpers + the vision/core runner.
- `mediapipe/tasks/ios/BUILD` — the framework umbrella: cc graph deps (~line 106),
  ObjC libs (~line 149), public-header manifests (~lines 222-224, 334).

### Graph output facts (verified)
- YOLO graph `mediapipe.tasks.vision.yolo_object_detector.YoloObjectDetectorGraph`
  outputs `DETECTIONS: std::vector<Detection>` in **pixel** units (its graph has the
  DetectionTransformation → pixel step). → YOLO reuses `MPPObjectDetectorResult`.
- OBB graph `mediapipe.tasks.vision.oriented_object_detector.OrientedObjectDetectorGraph`
  outputs `ORIENTED_DETECTIONS: std::vector<mediapipe::OrientedDetection>` in
  **normalized** [0,1] coords (graph ends at `OrientedDetectionProjectionCalculator`;
  no pixel-transform node) + `IMAGE`. The cc Task does
  `ConvertToOrientedObjectDetectionResult(vec, {image.width, image.height})` to get
  pixels — the iOS helper replicates this.
- `OrientedDetection` proto fields: `cx, cy, width, height` (normalized), `rotation`
  (radians CCW), `label[]`, `label_id[]`, `score[]`, `display_name[]`, …

## Design sections

### Section 1 — Structure (Approach C)

Two per-task directories mirroring `tasks/ios/vision/object_detector/`:
```
tasks/ios/vision/yolo_object_detector/
  sources/        MPPYoloObjectDetector.{h,mm}, MPPYoloObjectDetectorOptions.{h,m}
  utils/sources/  MPPYoloObjectDetectorOptions+Helpers.{h,mm}
  BUILD
  # NOTE: no YOLO result class or result helper. The YOLO graph emits the same
  # pixel std::vector<Detection> as object_detector, so the .mm reuses
  # MPPObjectDetectorResult and its existing
  # MPPObjectDetectorResult+Helpers (objectDetectorResultWithDetectionsPacket:)
  # directly.
tasks/ios/vision/oriented_object_detector/
  sources/        MPPOrientedObjectDetector.{h,mm},
                  MPPOrientedObjectDetectorOptions.{h,m},
                  MPPOrientedObjectDetectorResult.{h,m},
                  MPPOrientedDetection.{h,m}                   # new oriented detection value type
  utils/sources/  MPPOrientedObjectDetectorOptions+Helpers.{h,mm},
                  MPPOrientedObjectDetectorResult+Helpers.{h,mm}
  BUILD
```

### Section 2 — YOLO binding (reuses the axis-aligned result)

- `MPPYoloObjectDetector.{h,mm}`: a faithful clone of `MPPObjectDetector` with:
  - graph name `mediapipe.tasks.vision.yolo_object_detector.YoloObjectDetectorGraph`;
  - same streams (image_in/out, norm_rect_in, detections_out / DETECTIONS);
  - `detect(image:)`, `detect(videoFrame:timestampInMilliseconds:)`,
    `detectAsync(image:timestampInMilliseconds:)` + a
    `MPPYoloObjectDetectorLiveStreamDelegate`.
  - **Result type AND converter reused: `MPPObjectDetectorResult` +
    `MPPObjectDetectorResult+Helpers`** (the YOLO graph emits the same pixel
    `std::vector<Detection>`). The `.mm` imports the existing
    `MPPObjectDetectorResult+Helpers.h` and calls
    `objectDetectorResultWithDetectionsPacket:` directly — no new YOLO result class
    or result helper is created.
- `MPPYoloObjectDetectorOptions.{h,m}` + `…+Helpers`: writes
  `YoloObjectDetectorOptions::ext` — base_options, running_mode (via the proto's
  use_stream_mode), display_names_locale, max_results, score_threshold,
  category_allowlist/denylist (carried for parity; inert in the graph), iou_threshold,
  layout (enum int), num_classes.
- `MPPYoloObjectDetectorLiveStreamDelegate` protocol +
  `yoloObjectDetector:didFinishDetectionWithResult:timestampInMilliseconds:error:`.

### Section 3 — OBB binding (new oriented result + pixel conversion)

- `MPPOrientedDetection.{h,m}` — value type: `NSArray<MPPCategory*> *categories`,
  `float cx, cy, width, height` (PIXELS), `float rotation` (radians CCW). (Reuses
  `MPPCategory` from components/containers.)
- `MPPOrientedObjectDetectorResult.{h,m}` — `MPPTaskResult` subclass with
  `NSArray<MPPOrientedDetection*> *detections`, `initWithDetections:timestampInMilliseconds:`.
- `MPPOrientedObjectDetector.{h,mm}`: clone with graph name
  `…oriented_object_detector.OrientedObjectDetectorGraph`, output stream tag
  `ORIENTED_DETECTIONS` (+ `IMAGE`). All three running modes +
  `MPPOrientedObjectDetectorLiveStreamDelegate`. The output-packet→result step passes
  **both** the `ORIENTED_DETECTIONS` packet and the `IMAGE` packet to the helper
  (the axis-aligned sibling passes only DETECTIONS — this is the one structural
  difference in the `.mm`).
- `MPPOrientedObjectDetectorResult+Helpers.mm`: validates the packet as
  `std::vector<mediapipe::OrientedDetection>`; reads the `IMAGE` packet's width/height;
  for each proto builds an `MPPOrientedDetection` with
  `cx*w, cy*h, width*w, height*h, rotation`, and categories from `label_id[]`/`score[]`
  (name/displayName left nil). Mirrors `ConvertToOrientedObjectDetectionResult`.
- `MPPOrientedObjectDetectorOptions.{h,m}` + `…+Helpers`: writes
  `OrientedObjectDetectorOptions::ext` — base_options, running_mode, max_results,
  score_threshold, iou_threshold, class_agnostic_nms, layout (enum int), num_classes.
  NO display_names_locale/allowlist/denylist (the OBB options proto lacks them).

### Section 4 — Framework registration (`mediapipe/tasks/ios/BUILD`)

Add next to the `object_detector` entries at the three touchpoints:
- cc graph deps (~line 106): `//mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_graph`
  and `//mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_graph`.
- ObjC libs (~line 149): `//mediapipe/tasks/ios/vision/yolo_object_detector:MPPYoloObjectDetector`
  and `//mediapipe/tasks/ios/vision/oriented_object_detector:MPPOrientedObjectDetector`.
- public-header manifests (~lines 222-224 and 334): the new public `.h` files
  (`MPPYoloObjectDetector.h`, `MPPYoloObjectDetectorOptions.h`,
  `MPPOrientedObjectDetector.h`, `MPPOrientedObjectDetectorOptions.h`,
  `MPPOrientedObjectDetectorResult.h`, `MPPOrientedDetection.h`).
  (Confirm the exact list/format of these manifests when implementing; mirror how
  object_detector's headers are listed.)

### Section 5 — Coordinate & label contract

- YOLO: pixel results, reusing `MPPObjectDetectorResult` (graph already pixel).
- OBB: pixel results via the helper's scale-by-image-size (matches the cc Task and the
  2.1b/2.2 pixel contract). The new `MPPOrientedDetection` doc comments state pixels +
  radians CCW.
- Category names/displayNames empty for both (faithful to the graphs, which emit
  index+score). No metadata-extractor dependency in the framework.

### Section 6 — Tests

Gated XCTest mirroring `tasks/ios/test/vision/object_detector`, under
`tasks/ios/test/vision/{yolo_object_detector,oriented_object_detector}/`, gated on the
absent `yolov8n.tflite` / `yolov8n-obb.tflite` fixtures (skip cleanly). Per-detection
assertions: YOLO — ≥1 detection, one category, score>0, pixel bbox; OBB — ≥1 detection,
one category, score>0, pixel width/height>0, finite rotation. Primary verification gate:
the `objc_library` targets **build** under the iOS config (see Section 7).

### Section 7 — Build sequencing (Approach C)

0. **iOS toolchain smoke test FIRST:** `bazel build --config=ios_arm64
   //mediapipe/tasks/ios/vision/object_detector:MPPObjectDetector` — confirm the iOS
   build path works in this env before writing new code (the lesson from the Web/Python
   toolchain gaps). If it fails on a toolchain gap, surface it before proceeding.
1. YOLO: options + `+Helpers` → detector `.{h,mm}` (reuses `MPPObjectDetectorResult`
   + its existing `+Helpers`; no new result files) → BUILD → build-verify
   (`--config=ios_arm64`) → gated test.
2. OBB: `MPPOrientedDetection` + `MPPOrientedObjectDetectorResult` → result `+Helpers`
   (pixel) → options + `+Helpers` → detector `.{h,mm}` → BUILD → build-verify → gated test.
3. Register both in `mediapipe/tasks/ios/BUILD`; build the vision framework target.

## Out of scope

- Android/Java bindings (separate phase; viable since it also links graphs from source).
- Web bindings (blocked on WASM-from-source infra).
- Swift-specific wrappers beyond the `NS_SWIFT_NAME` annotations the ObjC headers carry.
- TFLite-metadata label population (graphs emit index+score; names left empty).
- Backend inference formats (Phase 6) and BoTSORT (Phase 8).

## Done criteria

- iOS toolchain smoke build passes (step 0).
- New `objc_library` targets build under the iOS config; `object_detector` (iOS) and the
  2.1 cc tasks untouched.
- YOLO reuses `MPPObjectDetectorResult`; OBB exposes pixel-unit `MPPOrientedObjectDetectorResult`.
- All three running modes wired (image/video/liveStream-delegate) for both detectors.
- Both registered in the vision framework umbrella (`tasks/ios/BUILD`).
- Gated XCTests build and skip cleanly without the model fixtures.
