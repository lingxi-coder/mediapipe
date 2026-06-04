# Verify & Harden the OBB OrientedObjectDetector Tasks API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing (never-run) `OrientedObjectDetector` C++ Tasks API provably work on CPU: a metadata-equipped `yolov8n-obb.tflite` + a DOTA-appropriate `boats.jpg`, the gated integration test turned into a real oracle-validated regression test, and a multi-class decoder unit test for the rotated-box decode.

**Architecture:** Same recipe as the YOLO verify, minus the box-normalization work — the OBB decoder (`YoloObbTensorsToOrientedDetectionsCalculator`) already writes normalized `cx,cy,w,h,angle` directly (the result container + `OrientedDetectionProjectionCalculator` scale to pixels), which is correct for the normalized TFLite export. A committed Python script exports the model + hand-attaches MediaPipe metadata + downloads `boats.jpg`; blobs are gitignored and glob-gated.

**Tech Stack:** C++ (api2 calculator, MediaPipe Tasks vision graph), Bazel (`--define MEDIAPIPE_DISABLE_GPU=1`), GoogleTest/CalculatorRunner; Python (`ultralytics`, `tflite_support.metadata`, `flatbuffers`, `urllib`).

**Spec:** `docs/superpowers/specs/2026-06-04-verify-oriented-object-detector-task-design.md`.

---

## Reference facts (verified)

- **Decoder** `mediapipe/calculators/tensor/yolo_obb_tensors_to_oriented_detections_calculator.{cc,proto}`: api2 node, input `TENSORS` (one rank-3 `[N, 4+num_classes+1, A]` CHANNELS_FIRST default), output `DETECTIONS` (`std::vector<std::vector<OrientedDetection>>`). `DecodeRow`: box `cx,cy,w,h` at channels 0-3, class scores at `4..4+nc-1`, **angle at channel `4+num_classes`**; argmax over classes, `conf_threshold` filter; writes `OrientedDetection` proto via `set_cx/set_cy/set_width/set_height/set_rotation/add_score/add_label_id`. Boxes are passed through normalized (NO division) — correct for the export.
- **Decoder test** `yolo_obb_tensors_to_oriented_detections_calculator_test.cc` exists: helper `MakeTensor(Tensor::Shape, std::vector<float>)`, `CalculatorRunner`, `ParseTextProtoOrDie<CalculatorGraphConfig::Node>`, output type alias `BatchOrientedDetections = std::vector<std::vector<OrientedDetection>>`, proto accessors `d.cx()/cy()/width()/height()/rotation()/score(0)/label_id(0)`. Existing cases use `num_classes:1` only (so they never exercise the angle-channel shift for `nc>1`).
- **Graph** `oriented_object_detector_graph.cc`: requires metadata + `num_classes` in options; `YoloObbTensorsToOrientedDetectionsCalculator` → `YoloObbBatchDetectionsToSingleCalculator` → `RotatedNonMaxSuppressionCalculator` → `OrientedDetectionProjectionCalculator`. Default layout CHANNELS_FIRST.
- **Result type** `mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h`: `OrientedObjectDetectorResult{ std::vector<OrientedDetection> detections; }`; `OrientedDetection{ std::vector<Category> categories; float cx, cy, width, height, rotation; }` (pixels after projection). `Category{ int index; float score; std::optional<std::string> category_name; }`. The OBB graph does NOT map label ids→names or apply allow/deny → `category_name` empty; assert on `index`.
- **Integration test** `oriented_object_detector_test.cc`: gated (`GTEST_SKIP` if `yolov8n-obb.tflite` absent), `num_classes=15`. TWO BUGS to fix: it sets `options->layout = kChannelsLast` (lines 99,141) but the export is `[1,20,8400]` = CHANNELS_FIRST → remove the override; and `kTestImage="cats_and_dogs.jpg"` (line 59) which has no DOTA objects → switch to `boats.jpg`. `ModelPath()`/`ImagePath()` = `JoinPath("./", "/mediapipe/tasks/testdata/vision/", name)`.
- **Env (verified in the YOLO effort):** `ultralytics`, `tensorflow`, `onnx2tf` present; the MediaPipe pip metadata writer + `tflite_support.metadata_writers` are NOT importable, but `tflite_support.metadata.MetadataPopulator` + `metadata_schema_py_generated` ARE. The YOLO export script `mediapipe/tasks/testdata/vision/export_yolov8n_tflite.py` is the working template.
- **DOTAv1 class order** (ultralytics `yolov8n-obb` `model.names`): 0 plane, **1 ship**, 2 storage tank, 3 baseball diamond, 4 tennis court, 5 basketball court, 6 ground track field, 7 harbor, 8 bridge, 9 large vehicle, 10 small vehicle, 11 helicopter, 12 roundabout, 13 soccer ball field, 14 swimming pool. `boats.jpg` → `ship` (1).
- Build/test: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 <target> --test_output=...`. A piped `exit 0` is NOT success — read the real output.

## File structure

- **Modify** `mediapipe/calculators/tensor/yolo_obb_tensors_to_oriented_detections_calculator_test.cc` — add a multi-class+angle case.
- **Create** `mediapipe/tasks/testdata/vision/export_yolov8n_obb_tflite.py` — export + metadata + boats.jpg download.
- **Modify** `mediapipe/tasks/testdata/vision/BUILD` — `yolo_obb_test_model` glob filegroup.
- **Modify** `mediapipe/tasks/cc/vision/oriented_object_detector/BUILD` — test `data` dep.
- **Modify** `.gitignore` — ignore the OBB blobs.
- **Modify** `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_test.cc` — boats.jpg, drop kChannelsLast, harden.

---

### Task 1: Multi-class rotated-box decoder unit test

**Files:** Modify `mediapipe/calculators/tensor/yolo_obb_tensors_to_oriented_detections_calculator_test.cc`.

Verifies the decoder for `num_classes>1`: argmax picks the right class AND the angle is read from the shifted channel `4+num_classes` (not a fixed index). Existing cases only use `num_classes:1`, so this closes a real gap (a hardcoded `angle_idx` bug would pass nc=1 but fail here).

- [ ] **Step 1: Add the test** (in the anonymous namespace, after the existing tests):

```cpp
TEST(YoloObbCalculatorTest, MultiClassArgmaxAndAngleChannel) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloObbTensorsToOrientedDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:dets"
    options {
      [mediapipe.YoloObbTensorsToOrientedDetectionsCalculatorOptions.ext] {
        num_classes: 3
        conf_threshold: 0.25
      }
    }
  )pb"));
  // CHANNELS_FIRST [1, 4+3+1=8, 1]: cx,cy,w,h, s0,s1,s2, angle (index c*A+a, A=1).
  // argmax over {0.1,0.8,0.3} -> class 1 @ 0.8; angle lives at channel 7 (4+3).
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 8, 1},
                       {0.5f, 0.5f, 0.2f, 0.4f, 0.1f, 0.8f, 0.3f, 0.7f})
                .release())
          .At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<BatchOrientedDetections>();
  ASSERT_EQ(batch.size(), 1u);
  ASSERT_EQ(batch[0].size(), 1u);
  const OrientedDetection& d = batch[0][0];
  EXPECT_NEAR(d.cx(), 0.5f, 1e-5);
  EXPECT_NEAR(d.cy(), 0.5f, 1e-5);
  EXPECT_NEAR(d.width(), 0.2f, 1e-5);
  EXPECT_NEAR(d.height(), 0.4f, 1e-5);
  EXPECT_EQ(d.label_id(0), 1);          // argmax picked class 1
  EXPECT_NEAR(d.score(0), 0.8f, 1e-5);
  EXPECT_NEAR(d.rotation(), 0.7f, 1e-5);  // angle read from channel 4+num_classes
}
```

- [ ] **Step 2: Run it; verify PASS** (the decoder already computes `angle_idx = 4 + num_classes`, so this characterizes + locks in correct multi-class behavior). If it FAILS (e.g. `rotation()` ≈ 0.8, meaning the angle was read from a class channel), that's a real decoder bug — STOP and report.

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_obb_tensors_to_oriented_detections_calculator_test --test_output=errors`
Expected: PASS (all cases).

- [ ] **Step 3: Commit.**

```bash
git add mediapipe/calculators/tensor/yolo_obb_tensors_to_oriented_detections_calculator_test.cc
git commit -m "test(yolo-obb-decode): multi-class argmax + angle-channel-shift case

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Model export + metadata + image download script

**Files:** Create `mediapipe/tasks/testdata/vision/export_yolov8n_obb_tflite.py`.

- [ ] **Step 1: Write the script.** Full content:

```python
# Copyright 2026 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exports yolov8n-obb.pt to a metadata-equipped yolov8n-obb.tflite fixture and
fetches a DOTA-appropriate test image (boats.jpg).

Outputs (next to this script, all gitignored); regenerate with:
    python3 mediapipe/tasks/testdata/vision/export_yolov8n_obb_tflite.py

(1) ultralytics exports the float32 TFLite OBB model (input [1,640,640,3],
output [1, 4+num_classes+1, 8400] = [1,20,8400] for DOTAv1's 15 classes).
(2) MediaPipe-readable metadata is hand-attached (input NormalizationOptions
mean=0/std=255 so ImageToTensor feeds pixel/255=[0,1]; DOTA label file) via
tflite_support's lower-level API (the high-level writers aren't importable here).
(3) boats.jpg (ultralytics' canonical OBB demo image) is downloaded for the test.
"""

import os
import shutil
import urllib.request

import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _fb
from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_MODEL = os.path.join(_HERE, "yolov8n-obb.tflite")
_LABELS = os.path.join(_HERE, "yolov8n_obb_labels.txt")
_IMAGE = os.path.join(_HERE, "boats.jpg")
_IMAGE_URL = "https://ultralytics.com/images/boats.jpg"


def export_float32_tflite():
  """ultralytics export -> (path to the float32 .tflite, names dict)."""
  model = YOLO("yolov8n-obb.pt")  # auto-downloads if absent
  out = model.export(format="tflite", imgsz=640, nms=False)
  path = str(out)
  assert path.endswith(".tflite") and os.path.exists(path), f"bad export: {path}"
  return path, model.names  # names: {0: "plane", 1: "ship", ...}


def write_labels(names):
  with open(_LABELS, "w") as f:
    for i in range(len(names)):
      f.write(f"{names[i]}\n")


def build_metadata_buffer():
  """ModelMetadata with input NormalizationOptions + RGB image properties."""
  input_meta = _fb.TensorMetadataT()
  input_meta.name = "image"
  input_meta.description = "Input image to be detected."
  input_meta.content = _fb.ContentT()
  input_meta.content.contentProperties = _fb.ImagePropertiesT()
  input_meta.content.contentProperties.colorSpace = _fb.ColorSpaceType.RGB
  input_meta.content.contentPropertiesType = (
      _fb.ContentProperties.ImageProperties)
  normalization = _fb.ProcessUnitT()
  normalization.optionsType = _fb.ProcessUnitOptions.NormalizationOptions
  normalization.options = _fb.NormalizationOptionsT()
  normalization.options.mean = [0.0]
  normalization.options.std = [255.0]
  input_meta.processUnits = [normalization]
  input_stats = _fb.StatsT()
  input_stats.max = [1.0]
  input_stats.min = [0.0]
  input_meta.stats = input_stats

  output_meta = _fb.TensorMetadataT()
  output_meta.name = "detections"
  output_meta.description = "Raw YOLO-OBB head [1, 4+num_classes+1, anchors]."
  output_meta.content = _fb.ContentT()
  output_meta.content.contentProperties = _fb.FeaturePropertiesT()
  output_meta.content.contentPropertiesType = (
      _fb.ContentProperties.FeatureProperties)
  label_file = _fb.AssociatedFileT()
  label_file.name = os.path.basename(_LABELS)
  label_file.description = "Label names (DOTAv1)."
  label_file.type = _fb.AssociatedFileType.TENSOR_AXIS_LABELS
  output_meta.associatedFiles = [label_file]

  subgraph = _fb.SubGraphMetadataT()
  subgraph.inputTensorMetadata = [input_meta]
  subgraph.outputTensorMetadata = [output_meta]

  model_meta = _fb.ModelMetadataT()
  model_meta.name = "YOLOv8n-OBB oriented object detector"
  model_meta.description = "Ultralytics YOLOv8n-OBB exported for MediaPipe Tasks."
  model_meta.subgraphMetadata = [subgraph]

  b = flatbuffers.Builder(0)
  b.Finish(model_meta.Pack(b),
           _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
  return b.Output()


def download_image():
  if not os.path.exists(_IMAGE):
    print(f"downloading {_IMAGE_URL} -> {_IMAGE}")
    urllib.request.urlretrieve(_IMAGE_URL, _IMAGE)


def main():
  src_tflite, names = export_float32_tflite()
  write_labels(names)
  shutil.copyfile(src_tflite, _OUT_MODEL)
  populator = _metadata.MetadataPopulator.with_model_file(_OUT_MODEL)
  populator.load_metadata_buffer(build_metadata_buffer())
  populator.load_associated_files([_LABELS])
  populator.populate()
  displayer = _metadata.MetadataDisplayer.with_model_file(_OUT_MODEL)
  assert displayer.get_metadata_json(), "metadata not attached"
  download_image()
  print(f"wrote {_OUT_MODEL} ({os.path.getsize(_OUT_MODEL)} bytes) with metadata; "
        f"image {_IMAGE} ({os.path.getsize(_IMAGE)} bytes)")


if __name__ == "__main__":
  main()
```

- [ ] **Step 2: Run it.** `python3 mediapipe/tasks/testdata/vision/export_yolov8n_obb_tflite.py`
Expected: prints `wrote .../yolov8n-obb.tflite (… bytes) with metadata; image .../boats.jpg (… bytes)`.
ADAPTATION ALLOWED (same as the YOLO script): if `model.export` rejects `nms=False`, drop it; if a `tflite_support` enum name differs, correct it (the YOLO script `export_yolov8n_tflite.py` uses the identical, known-good enum names — copy from there). Note any adaptation.

- [ ] **Step 3: Verify shape + metadata + image.** Run:
```bash
python3 - <<'PY'
import tensorflow as tf
from tflite_support import metadata as md
import os
p="mediapipe/tasks/testdata/vision/yolov8n-obb.tflite"
i=tf.lite.Interpreter(model_path=p); i.allocate_tensors()
print("in", i.get_input_details()[0]["shape"], i.get_input_details()[0]["dtype"])
print("out", i.get_output_details()[0]["shape"])
print("has_metadata", bool(md.MetadataDisplayer.with_model_file(p).get_metadata_json()))
print("boats.jpg", os.path.exists("mediapipe/tasks/testdata/vision/boats.jpg"))
PY
```
Expected: `in [1 640 640 3] float32`, `out [1 20 8400]`, `has_metadata True`, `boats.jpg True`. RECORD the actual output. If output is `[1 8400 20]` (transposed), note it (Task 4 would then set `kChannelsLast`); the default-CHANNELS_FIRST path expects `[1 20 8400]`.

- [ ] **Step 4: Commit ONLY the script** (blobs are gitignored in Task 3; do NOT add them):
```bash
git add mediapipe/tasks/testdata/vision/export_yolov8n_obb_tflite.py
git commit -m "feat(tasks-obb): yolov8n-obb.tflite export + metadata + boats.jpg fetch

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Then `git status --short mediapipe/tasks/testdata/vision/` — confirm `yolov8n-obb.tflite`, `yolov8n_obb_labels.txt`, `boats.jpg` show as UNTRACKED (present on disk, uncommitted).

---

### Task 3: Wire the optional OBB fixtures into the build (glob-safe)

**Files:** Modify `mediapipe/tasks/testdata/vision/BUILD`, `mediapipe/tasks/cc/vision/oriented_object_detector/BUILD`, `.gitignore`.

- [ ] **Step 1: Add a glob filegroup.** In `mediapipe/tasks/testdata/vision/BUILD`, near the existing `yolo_test_model` filegroup (from the YOLO effort):
```python
# Locally-exported OBB fixtures (gitignored; produced by
# export_yolov8n_obb_tflite.py). glob+allow_empty so the build still works when
# absent (CI / fresh clone) — the gated test then GTEST_SKIP()s.
filegroup(
    name = "yolo_obb_test_model",
    srcs = glob(
        [
            "yolov8n-obb.tflite",
            "yolov8n_obb_labels.txt",
            "boats.jpg",
        ],
        allow_empty = True,
    ),
)
```

- [ ] **Step 2: Use it in the OBB test's data deps.** In `mediapipe/tasks/cc/vision/oriented_object_detector/BUILD`, set the `oriented_object_detector_test` `data`:
```python
    data = [
        # Image fixture set (always present — fetched from GCS via mediapipe_files).
        "//mediapipe/tasks/testdata/vision:test_images",
        # OBB model + DOTA image: empty unless exported locally (export_yolov8n_obb_tflite.py).
        "//mediapipe/tasks/testdata/vision:yolo_obb_test_model",
    ],
```

- [ ] **Step 3: Gitignore the blobs.** Append to `.gitignore` (root):
```gitignore
# Locally-exported OBB Tasks test fixtures (regenerate via export_yolov8n_obb_tflite.py)
/mediapipe/tasks/testdata/vision/yolov8n-obb.tflite
/mediapipe/tasks/testdata/vision/yolov8n_obb_labels.txt
/mediapipe/tasks/testdata/vision/boats.jpg
```

- [ ] **Step 4: Verify.**
(a) `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test` → `Build completed successfully`.
(b) `git check-ignore mediapipe/tasks/testdata/vision/yolov8n-obb.tflite mediapipe/tasks/testdata/vision/boats.jpg` prints both; `git status --porcelain mediapipe/tasks/testdata/vision/` shows no `yolov8n-obb.*`/`boats.jpg` lines.

- [ ] **Step 5: Commit.**
```bash
git add mediapipe/tasks/testdata/vision/BUILD \
        mediapipe/tasks/cc/vision/oriented_object_detector/BUILD .gitignore
git commit -m "build(tasks-obb): glob-gated yolov8n-obb fixtures wired into the OBB task test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Harden + run the OBB integration test (oracle-validated)

**Files:** Modify `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_test.cc`.

- [ ] **Step 1: Oracle expectation.** Learn what `yolov8n-obb` emits on `boats.jpg`:
```bash
python3 - <<'PY'
import glob
from collections import Counter
from ultralytics import YOLO
img = glob.glob("**/tasks/testdata/vision/boats.jpg", recursive=True)[0]
r = YOLO("yolov8n-obb.pt").predict(img, conf=0.25, iou=0.45, verbose=False)[0]
cls = r.obb.cls.cpu().numpy().astype(int)
print("image:", img)
print("classes:", Counter(cls.tolist()), "names:", r.names)
print("xywhr (first 5):", r.obb.xywhr.cpu().numpy()[:5].round(3).tolist())
PY
```
Expected: dominant class `ship` (id 1), several oriented boxes with non-trivial rotation. Record the class ids; Step 2 asserts the `ship` id the oracle reports (1).

- [ ] **Step 2: Switch the test image to boats.jpg.** Change the constant:
```cpp
constexpr char kTestImage[] = "boats.jpg";
```

- [ ] **Step 3: Drop the wrong layout override + harden `DetectOnImage`.** In `DetectOnImage`, REMOVE the line `options->layout = OrientedObjectDetectorOptions::kChannelsLast;` (the export is CHANNELS_FIRST `[1,20,8400]`, the default). Replace the assertion block with:
```cpp
  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                           OrientedObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));
  MP_ASSERT_OK_AND_ASSIGN(OrientedObjectDetectorResult result,
                           detector->Detect(image));

  ASSERT_FALSE(result.detections.empty())
      << "Expected oriented detections on " << kTestImage << ".";
  EXPECT_LE(result.detections.size(), 10u);  // max_results respected
  std::set<int> labels;
  for (const auto& det : result.detections) {
    ASSERT_EQ(det.categories.size(), 1u);
    const auto& cat = det.categories[0];
    EXPECT_GE(cat.score, 0.25f);          // score_threshold honored
    EXPECT_GE(cat.index, 0);
    EXPECT_LT(cat.index, 15);             // valid DOTA id
    labels.insert(cat.index);
    EXPECT_GT(det.width, 0.0f);
    EXPECT_GT(det.height, 0.0f);
    EXPECT_TRUE(std::isfinite(det.rotation));
  }
  // boats.jpg -> DOTA "ship" (class 1) per Step 1 oracle.
  EXPECT_TRUE(labels.count(1)) << "expected a ship detection";
  MP_ASSERT_OK(detector->Close());
```
(If Step 1's oracle reports a different dominant id, use that id in the final `labels.count(...)`.) Add `#include <set>` to the test includes if absent.

- [ ] **Step 4: Fix `DetectForVideo`.** REMOVE its `options->layout = OrientedObjectDetectorOptions::kChannelsLast;` line too (keep `num_classes=15`, VIDEO mode, and its existing non-empty assertions on both frames).

- [ ] **Step 5: Run the test; it must RUN (not skip) and PASS.**
`bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test --test_output=all`
Expected: both tests run and pass. If `DetectOnImage` fails (no detections / wrong class / non-finite rotation), debug: confirm the layout is CHANNELS_FIRST (Step removed the override), that `boats.jpg` is the image, and compare to Step 1's oracle (print the detections temporarily). Investigate root cause; do NOT weaken assertions. The OBB decoder needs no normalization (it passes boxes through; the projection scales to pixels), so detections should match the oracle.

- [ ] **Step 6: Oracle cross-check (document).** Compare the gtest's detected classes + box centers/sizes/rotation to Step 1's `xywhr`. Confirm broad agreement (ship class, similar boxes). Note the result in the "Bring-up notes" below. Not a committed target.

- [ ] **Step 7: Commit.**
```bash
git add mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_test.cc
git commit -m "test(tasks-obb): real oriented-detection assertions on boats.jpg (oracle-validated)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Bring-up notes (fill during Task 4)
- Exported OBB output shape / layout: …
- Oracle classes + rotations on boats.jpg: …
- C++ vs oracle agreement: …

## Self-review checklist (before final review)
- Decoder unit test exercises `num_classes>1` (angle read from channel `4+nc`, argmax class).
- The gated integration test builds when the fixture is absent (glob `allow_empty`) and runs+passes when present; uses `boats.jpg`; the wrong `kChannelsLast` override removed in BOTH tests.
- No model/image blobs committed (`.gitignore` covers all three); only the export script + test/BUILD text committed.
- `num_classes=15` set; assertions are real (ship present, box dims > 0, rotation finite, score/max_results); the OBB decoder is unchanged (already normalized-by-default).

## Final verification
- [ ] `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_obb_tensors_to_oriented_detections_calculator_test` — green.
- [ ] `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test --test_output=all` — runs (not skipped) and green.
- [ ] Oracle cross-check recorded.
- [ ] `git status` clean except the (gitignored) exported blobs.

## Out of scope
OBB language bindings; category allowlist/denylist + label-name mapping (shared follow-up with YOLO); letterbox/aspect-preserving preprocessing (shared follow-up); quantized/GPU; metadata-derived `num_classes`.
