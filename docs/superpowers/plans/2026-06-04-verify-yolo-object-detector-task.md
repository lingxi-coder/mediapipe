# Verify & Harden the YOLO Object Detector Tasks API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing (never-run) `YoloObjectDetector` C++ Tasks API provably work on CPU: add the box-normalization option the real model forces, produce a metadata-equipped `yolov8n.tflite`, and turn the skipped integration test into a real, oracle-validated regression test.

**Architecture:** A new default-off `input_width`/`input_height` option on `YoloTensorsToDetectionsCalculator` normalizes pixel-space detect-head boxes; `YoloObjectDetectorGraph` sets it from the model's input tensor dims (`BuildInputImageTensorSpecs`). A committed Python script exports `yolov8n.pt → yolov8n.tflite` and hand-attaches MediaPipe metadata (input `NormalizationOptions`, COCO labels) via `tflite_support`. The model blob stays gitignored; a `glob`-based filegroup keeps the gated test build-safe when the blob is absent.

**Tech Stack:** C++ (api2 calculator, MediaPipe Tasks vision graph), proto2, Bazel (`--define MEDIAPIPE_DISABLE_GPU=1`), GoogleTest/CalculatorRunner; Python (`ultralytics`, `tflite_support.metadata`, `flatbuffers`).

**Spec:** `docs/superpowers/specs/2026-06-04-verify-yolo-object-detector-task-design.md`.

---

## Reference facts (verified)

- **Calculator** `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.{cc,proto}`: api2 node, input `TENSORS` (`std::vector<Tensor>`, one rank-3 `[N, 4+num_classes, A]` CHANNELS_FIRST default), output `DETECTIONS` (`std::vector<std::vector<Detection>>`). `DecodeRow` writes `relative_bounding_box` straight from `cx,cy,w,h` (no scaling). Options proto fields 1–7 used; next free = 8. Test `yolo_tensors_to_detections_calculator_test.cc` exists with a `MakeTensor(Tensor::Shape, std::vector<float>)` helper + `CalculatorRunner` + `ParseTextProtoOrDie<...Node>`.
- **Graph** `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc`: configures `YoloTensorsToDetectionsCalculator` options in a `{ ... }` block (~lines 210–227) inside a Status-returning method that has `model_resources` in scope. `BuildInputImageTensorSpecs(const tasks::core::ModelResources&) -> absl::StatusOr<ImageTensorSpecs>` (`mediapipe/tasks/cc/vision/utils/image_tensor_specs.h`, namespace `mediapipe::tasks::vision`) yields `.image_width`/`.image_height`. BUILD dep label: `//mediapipe/tasks/cc/vision/utils:image_tensor_specs`.
- **Task test** `yolo_object_detector_test.cc`: `kTestImage="cats_and_dogs.jpg"` (always available via `test_images`), `kYoloModel="yolov8n.tflite"`, `ModelPath()`/`ImagePath()` use `JoinPath("./", "/mediapipe/tasks/testdata/vision/", name)`. Tests `GTEST_SKIP()` when `!file::Exists(ModelPath())`. Options need `num_classes` set (graph `RET_CHECK_GT`).
- **Metadata**: the graph requires model metadata to exist + reads input `NormalizationOptions`. The MediaPipe/`tflite_support` *writers* are NOT importable locally, but lower-level `tflite_support.metadata.MetadataPopulator` + `tflite_support.metadata_schema_py_generated` (`ModelMetadataT`, `NormalizationOptionsT`, …) ARE — enough to hand-build + attach metadata.
- **Build flavor:** all C++ here is `bazel {build,test} -c opt --define MEDIAPIPE_DISABLE_GPU=1 <target>`. Read the real output tail — a piped `exit 0` is not success.

## File structure

- **Modify** `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.proto` — add `input_width`/`input_height` (fields 8/9).
- **Modify** `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.cc` — precompute inverses in `Open`, normalize in `DecodeRow`.
- **Modify** `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc` — add a pixel-space-normalization test.
- **Modify** `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc` + `BUILD` — set the option from input specs.
- **Create** `mediapipe/tasks/testdata/vision/export_yolov8n_tflite.py` — export + metadata.
- **Modify** `mediapipe/tasks/testdata/vision/BUILD` — `glob` filegroup for the optional model.
- **Modify** `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD` — test `data` dep.
- **Modify** `.gitignore` — ignore the exported model blob.
- **Modify** `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc` — harden assertions.

---

### Task 1: Box-normalization option on YoloTensorsToDetectionsCalculator

**Files:**
- Modify: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.proto`
- Modify: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.cc`
- Test: `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc`

- [ ] **Step 1: Add the proto fields.** After field 7 (`tile_local_nms_class_agnostic`), before the closing `}` of `message YoloTensorsToDetectionsCalculatorOptions`:

```proto
  // When BOTH > 0, the model emits box cx,cy,w,h in input-PIXEL space; divide
  // cx,w by input_width and cy,h by input_height to produce the normalized
  // [0,1] relative_bounding_box this calculator outputs. 0 (default) = boxes are
  // already normalized [0,1] (unchanged: the tiled-detection pipeline and prior
  // callers feed normalized boxes).
  optional int32 input_width = 8 [default = 0];
  optional int32 input_height = 9 [default = 0];
```

- [ ] **Step 2: Write the failing test.** Add to `yolo_tensors_to_detections_calculator_test.cc` (inside the anonymous namespace, after the existing tests). One box at pixel `(cx,cy,w,h)=(320,240,64,48)` on a `640x480` input, single class score 0.9:

```cpp
TEST(YoloTensorsToDetectionsCalculatorTest, PixelSpaceBoxesNormalizedByInputDims) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:dets"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        num_classes: 1
        conf_threshold: 0.25
        input_width: 640
        input_height: 480
      }
    }
  )pb"));
  // CHANNELS_FIRST [1, 5, 1]: rows = cx,cy,w,h,score0 ; one anchor.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 5, 1},
                       {320.0f, 240.0f, 64.0f, 48.0f, 0.9f})
                .release())
          .At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<std::vector<std::vector<Detection>>>();
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].size(), 1u);
  const auto& bb = out[0][0].location_data().relative_bounding_box();
  // (cx-w/2)/W = (320-32)/640 = 0.45 ; (cy-h/2)/H = (240-24)/480 = 0.45
  EXPECT_NEAR(bb.xmin(), 0.45f, 1e-5);
  EXPECT_NEAR(bb.ymin(), 0.45f, 1e-5);
  EXPECT_NEAR(bb.width(), 64.0f / 640.0f, 1e-5);   // 0.1
  EXPECT_NEAR(bb.height(), 48.0f / 480.0f, 1e-5);  // 0.1
}
```

- [ ] **Step 3: Run it; verify it FAILS.**

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_test --test_output=errors`
Expected: FAIL — without the fix the box is emitted in pixel space (`xmin≈288`), so `EXPECT_NEAR(bb.xmin(), 0.45, …)` fails. (If the proto field isn't compiled yet it fails to build — also acceptable as "fails".)

- [ ] **Step 4: Implement the normalization.** In `yolo_tensors_to_detections_calculator.cc`:

  (a) In `Open`, after the `num_classes` check, validate + precompute inverses:
```cpp
    RET_CHECK_GE(options_.input_width(), 0);
    RET_CHECK_GE(options_.input_height(), 0);
    if (options_.input_width() > 0 && options_.input_height() > 0) {
      inv_w_ = 1.0f / static_cast<float>(options_.input_width());
      inv_h_ = 1.0f / static_cast<float>(options_.input_height());
    }
```

  (b) In `DecodeRow`, change the `cx,cy,w,h` line from `const float` to mutable and normalize:
```cpp
      float cx = at(0), cy = at(1), w = at(2), h = at(3);
      if (inv_w_ > 0.0f) {  // model emits pixel-space boxes -> normalize to [0,1]
        cx *= inv_w_;
        w *= inv_w_;
        cy *= inv_h_;
        h *= inv_h_;
      }
```

  (c) Add the members next to `options_`:
```cpp
  float inv_w_ = 0.0f;  // 1/input_width when normalizing pixel-space boxes
  float inv_h_ = 0.0f;  // 1/input_height; 0 => boxes already normalized
```

- [ ] **Step 5: Run the full calculator test; verify PASS (incl. the existing tests — default behavior unchanged).**

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_test --test_output=errors`
Expected: PASS (all tests). The existing tests set no `input_width`/`input_height`, so `inv_w_==0` and behavior is byte-identical.

- [ ] **Step 6: Commit.**

```bash
git add mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.proto \
        mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.cc \
        mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc
git commit -m "feat(yolo-decode): default-off input_width/height to normalize pixel-space boxes

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Wire input dims in YoloObjectDetectorGraph

**Files:**
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc`
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`

- [ ] **Step 1: Add the include.** With the other `#include "mediapipe/tasks/cc/vision/..."` lines in the graph `.cc`:

```cpp
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
```

- [ ] **Step 2: Set the option from the model's input specs.** In the `auto& opts = yolo_decode.GetOptions<...>();` block, after the `opts.set_layout(...)` call and before the block's closing `}`:

```cpp
      // The model's detect head emits boxes in input-pixel space; give the
      // decoder the input dims so it normalizes to [0,1] before projection.
      MP_ASSIGN_OR_RETURN(
          auto yolo_input_specs,
          ::mediapipe::tasks::vision::BuildInputImageTensorSpecs(
              model_resources));
      opts.set_input_width(yolo_input_specs.image_width);
      opts.set_input_height(yolo_input_specs.image_height);
```

- [ ] **Step 3: Add the BUILD dep.** In the `yolo_object_detector_graph` `cc_library` target's `deps` in `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`, add:

```python
        "//mediapipe/tasks/cc/vision/utils:image_tensor_specs",
```

- [ ] **Step 4: Build the graph; verify it compiles.**

Run: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_graph`
Expected: `Build completed successfully`. (Runtime effect is exercised by the integration test in Task 5.)

- [ ] **Step 5: Commit.**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph.cc \
        mediapipe/tasks/cc/vision/yolo_object_detector/BUILD
git commit -m "feat(tasks-yolo): set decoder input_width/height from model input specs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Model export + metadata script

**Files:**
- Create: `mediapipe/tasks/testdata/vision/export_yolov8n_tflite.py`

- [ ] **Step 1: Write the export script.** Full content:

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
"""Exports yolov8n.pt to a MediaPipe-metadata-equipped yolov8n.tflite fixture.

The output (yolov8n.tflite, next to this script) is gitignored; regenerate with:
    python3 mediapipe/tasks/testdata/vision/export_yolov8n_tflite.py

It (1) exports the float32 TFLite model via ultralytics (input [1,640,640,3],
output [1,84,8400]) and (2) hand-attaches MediaPipe-readable metadata:
input NormalizationOptions(mean=0,std=255) so MediaPipe's ImageToTensor feeds the
model pixel/255 = [0,1], plus a COCO label file. The MediaPipe/tflite_support
metadata *writers* are not importable in this env, so we build the metadata
flatbuffer directly with tflite_support's lower-level API.
"""

import os
import shutil

import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _fb
from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.cfg import ROOT as ULTRA_ROOT

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_MODEL = os.path.join(_HERE, "yolov8n.tflite")
_LABELS = os.path.join(_HERE, "yolov8n_labels.txt")


def export_float32_tflite():
  """ultralytics export -> path to the float32 .tflite."""
  model = YOLO("yolov8n.pt")  # auto-downloads if absent
  # imgsz default 640; nms=False keeps the raw [1,84,8400] detect head that the
  # MediaPipe graph decodes itself.
  out = model.export(format="tflite", imgsz=640, nms=False)
  # ultralytics returns the export path (str or Path).
  path = str(out)
  assert path.endswith(".tflite") and os.path.exists(path), f"bad export: {path}"
  return path, model.names  # names: {id: "person", ...}


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
  output_meta.description = "Raw YOLO detect head [1, 4+num_classes, anchors]."
  output_meta.content = _fb.ContentT()
  output_meta.content.contentProperties = _fb.FeaturePropertiesT()
  output_meta.content.contentPropertiesType = (
      _fb.ContentProperties.FeatureProperties)
  # Associate the label file (informational; the graph decodes label ids).
  label_file = _fb.AssociatedFileT()
  label_file.name = os.path.basename(_LABELS)
  label_file.description = "Label names (COCO)."
  label_file.type = _fb.AssociatedFileType.TENSOR_AXIS_LABELS
  output_meta.associatedFiles = [label_file]

  subgraph = _fb.SubGraphMetadataT()
  subgraph.inputTensorMetadata = [input_meta]
  subgraph.outputTensorMetadata = [output_meta]

  model_meta = _fb.ModelMetadataT()
  model_meta.name = "YOLOv8n object detector"
  model_meta.description = "Ultralytics YOLOv8n exported for MediaPipe Tasks."
  model_meta.subgraphMetadata = [subgraph]

  b = flatbuffers.Builder(0)
  b.Finish(model_meta.Pack(b),
           _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
  return b.Output()


def main():
  src_tflite, names = export_float32_tflite()
  write_labels(names)
  shutil.copyfile(src_tflite, _OUT_MODEL)
  populator = _metadata.MetadataPopulator.with_model_file(_OUT_MODEL)
  populator.load_metadata_buffer(build_metadata_buffer())
  populator.load_associated_files([_LABELS])
  populator.populate()
  # Sanity: metadata round-trips.
  displayer = _metadata.MetadataDisplayer.with_model_file(_OUT_MODEL)
  assert displayer.get_metadata_json(), "metadata not attached"
  print(f"wrote {_OUT_MODEL} ({os.path.getsize(_OUT_MODEL)} bytes) with metadata")


if __name__ == "__main__":
  main()
```

- [ ] **Step 2: Run the export; verify the model + metadata are produced.**

Run: `python3 mediapipe/tasks/testdata/vision/export_yolov8n_tflite.py`
Expected: prints `wrote .../yolov8n.tflite (… bytes) with metadata`. If `model.export` raises about a missing field, drop the `nms=False` kwarg (older ultralytics export raw by default) and re-run.

- [ ] **Step 3: Verify the model shape + metadata from Python (quick check).**

Run:
```bash
python3 - <<'PY'
import numpy as np, tensorflow as tf
from tflite_support import metadata as md
p="mediapipe/tasks/testdata/vision/yolov8n.tflite"
i=tf.lite.Interpreter(model_path=p); i.allocate_tensors()
print("in", i.get_input_details()[0]["shape"], i.get_input_details()[0]["dtype"])
print("out", i.get_output_details()[0]["shape"])
print("has_metadata", bool(md.MetadataDisplayer.with_model_file(p).get_metadata_json()))
PY
```
Expected: `in [1 640 640 3] <float32>`, `out [  1  84 8400]`, `has_metadata True`. (If the output is `[1 8400 84]`, note it — the task options default to CHANNELS_FIRST; a transposed export needs `layout=kChannelsLast`, set in Task 5's options.)

- [ ] **Step 4: Commit the script (NOT the model/labels blobs — those are gitignored in Task 4).**

```bash
git add mediapipe/tasks/testdata/vision/export_yolov8n_tflite.py
git commit -m "feat(tasks-yolo): yolov8n.tflite export + MediaPipe metadata script

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Wire the optional model fixture into the build (glob-safe)

**Files:**
- Modify: `mediapipe/tasks/testdata/vision/BUILD`
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`
- Modify: `.gitignore`

- [ ] **Step 1: Add a glob filegroup for the optional model.** In `mediapipe/tasks/testdata/vision/BUILD`, near the other `filegroup(...)` declarations:

```python
# Locally-exported YOLO fixtures (gitignored; produced by
# export_yolov8n_tflite.py). glob with allow_empty so the build still works when
# the blob is absent (CI / fresh clone) — the gated test then GTEST_SKIP()s.
filegroup(
    name = "yolo_test_model",
    srcs = glob(
        [
            "yolov8n.tflite",
            "yolov8n_labels.txt",
        ],
        allow_empty = True,
    ),
)
```

- [ ] **Step 2: Use it in the task test's data deps.** In `mediapipe/tasks/cc/vision/yolo_object_detector/BUILD`, replace the commented model line in the `yolo_object_detector_test` `data`:

```python
    data = [
        # Image fixture (always present — fetched from GCS via mediapipe_files).
        "//mediapipe/tasks/testdata/vision:test_images",
        # Model fixture: empty unless exported locally (see export_yolov8n_tflite.py).
        "//mediapipe/tasks/testdata/vision:yolo_test_model",
    ],
```

- [ ] **Step 3: Gitignore the exported blobs.** Append to `.gitignore` (repo root):

```gitignore
# Locally-exported YOLO Tasks test fixtures (regenerate via export_yolov8n_tflite.py)
/mediapipe/tasks/testdata/vision/yolov8n.tflite
/mediapipe/tasks/testdata/vision/yolov8n_labels.txt
```

- [ ] **Step 4: Verify the test target builds with the fixture present.**

Run: `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test`
Expected: `Build completed successfully` (the model exists from Task 3, so the filegroup is non-empty).
Also confirm git ignores the blob: `git status --porcelain mediapipe/tasks/testdata/vision/yolov8n.tflite` prints nothing.

- [ ] **Step 5: Commit.**

```bash
git add mediapipe/tasks/testdata/vision/BUILD \
        mediapipe/tasks/cc/vision/yolo_object_detector/BUILD .gitignore
git commit -m "build(tasks-yolo): glob-gated yolov8n.tflite fixture wired into the task test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Harden + run the integration test (oracle-validated)

**Files:**
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc`

- [ ] **Step 1: Establish the oracle expectation.** Run ultralytics on the same image to learn the ground-truth classes the model produces:

```bash
python3 - <<'PY'
from ultralytics import YOLO
# Resolve the runfiles/testdata image (same file the gtest loads).
import glob, os
img = glob.glob("**/tasks/testdata/vision/cats_and_dogs.jpg", recursive=True)[0]
r = YOLO("yolov8n.pt").predict(img, conf=0.25, iou=0.45, verbose=False)[0]
names = r.names
print("detections (class -> count):")
from collections import Counter
c = Counter(int(b.cls) for b in r.boxes)
for k,v in c.items(): print(f"  {k} {names[k]}: {v}")
print("xyxy:", r.boxes.xyxy.cpu().numpy().round().tolist())
PY
```
Expected: a small set of COCO classes (cats_and_dogs.jpg → `cat` id 15 and `dog` id 16, possibly others). Record the dominant class ids; Step 2's assertions use COCO ids `15` (cat) and `16` (dog). If the oracle shows different ids, use those exact ids in Step 2 (the test must reflect what the model actually emits).

- [ ] **Step 2: Replace the gated assertion bodies with meaningful checks.** In `yolo_object_detector_test.cc`, the `DetectOnImage` test, after the `GTEST_SKIP` guard, set options + assertions (keep the existing `Create`/`Detect` calls; add `num_classes`, thresholds, and the checks):

```cpp
  options->base_options.model_asset_path = model_path;
  options->running_mode = core::RunningMode::IMAGE;
  options->num_classes = 80;          // COCO
  options->score_threshold = 0.25f;
  options->iou_threshold = 0.45f;
  options->max_results = 10;
  // layout defaults to kChannelsFirst ([1,84,8400]); if Task 3 Step 3 showed a
  // transposed export, set: options->layout = YoloObjectDetectorOptions::kChannelsLast;

  MP_ASSERT_OK_AND_ASSIGN(auto detector,
                          YoloObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));
  MP_ASSERT_OK_AND_ASSIGN(YoloObjectDetectorResult result,
                          detector->Detect(image));

  ASSERT_FALSE(result.detections.empty()) << "expected detections on " << kTestImage;
  EXPECT_LE(result.detections.size(), 10u);  // max_results respected
  const int w = image.width(), h = image.height();
  std::set<int> labels;
  for (const auto& det : result.detections) {
    ASSERT_EQ(det.categories.size(), 1u);
    const auto& cat = det.categories[0];
    EXPECT_GE(cat.score, 0.25f);                 // score_threshold honored
    EXPECT_GE(cat.index, 0);
    EXPECT_LT(cat.index, 80);                    // valid COCO id
    labels.insert(cat.index);
    // Pixel bbox within image bounds (DetectionTransformation output). The
    // container Rect is {left, top, right, bottom} in pixels.
    const auto& bb = det.bounding_box;
    EXPECT_GE(bb.left, 0);
    EXPECT_GE(bb.top, 0);
    EXPECT_LE(bb.right, w + 1);
    EXPECT_LE(bb.bottom, h + 1);
    EXPECT_GT(bb.right, bb.left);
    EXPECT_GT(bb.bottom, bb.top);
  }
  // cats_and_dogs.jpg: model detects a cat (15) and a dog (16) (per Step 1).
  EXPECT_TRUE(labels.count(15) || labels.count(16))
      << "expected a cat or dog detection";
  MP_ASSERT_OK(detector->Close());
```

- [ ] **Step 3: Add a score-threshold monotonicity test (filtering that IS implemented).** NOTE: `category_allowlist`/`category_denylist` are validated + copied into the options proto but the YOLO graph has **no** label-map/filter calculator, so they are not applied and `category_name` is not populated — do NOT assert those (see Findings). Score-threshold (decoder `conf_threshold`) and `max_results` (NMS `max_num_detections`) ARE applied. Add a new test after `DetectOnImage`:

```cpp
TEST(YoloObjectDetectorTest, ScoreThresholdAndMaxResultsFilter) {
  const std::string model_path = ModelPath();
  if (!mediapipe::file::Exists(model_path).ok()) {
    GTEST_SKIP() << "YOLO model fixture not available at " << model_path;
  }
  auto base = [&]() {
    auto o = std::make_unique<YoloObjectDetectorOptions>();
    o->base_options.model_asset_path = model_path;
    o->running_mode = core::RunningMode::IMAGE;
    o->num_classes = 80;
    o->iou_threshold = 0.45f;
    return o;
  };
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(ImagePath()));

  // High score threshold -> no more detections than a low threshold, and every
  // surviving detection scores >= the high threshold.
  auto lo = base(); lo->score_threshold = 0.25f;
  MP_ASSERT_OK_AND_ASSIGN(auto det_lo, YoloObjectDetector::Create(std::move(lo)));
  MP_ASSERT_OK_AND_ASSIGN(auto r_lo, det_lo->Detect(image));
  MP_ASSERT_OK(det_lo->Close());

  auto hi = base(); hi->score_threshold = 0.9f;
  MP_ASSERT_OK_AND_ASSIGN(auto det_hi, YoloObjectDetector::Create(std::move(hi)));
  MP_ASSERT_OK_AND_ASSIGN(auto r_hi, det_hi->Detect(image));
  MP_ASSERT_OK(det_hi->Close());
  EXPECT_LE(r_hi.detections.size(), r_lo.detections.size());
  for (const auto& d : r_hi.detections) EXPECT_GE(d.categories[0].score, 0.9f);

  // max_results caps the count.
  auto cap = base(); cap->score_threshold = 0.25f; cap->max_results = 1;
  MP_ASSERT_OK_AND_ASSIGN(auto det_cap, YoloObjectDetector::Create(std::move(cap)));
  MP_ASSERT_OK_AND_ASSIGN(auto r_cap, det_cap->Detect(image));
  MP_ASSERT_OK(det_cap->Close());
  EXPECT_LE(r_cap.detections.size(), 1u);
}
```

Add `#include <set>` to the test includes if not present (used by `DetectOnImage`).

- [ ] **Step 4: Run the integration test; verify it PASSES (and no longer skips).**

Run: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_output=all`
Expected: tests RUN (no `GTEST_SKIP` line for the model) and PASS. If boxes fail the bounds check or `labels` is unexpected, the layout/normalization is off → revisit Task 1/2 wiring and Task 3 Step 3 layout note. Read the real output, not the pipe exit.

- [ ] **Step 5: Cross-check against the oracle (one-time, documented).** Confirm the C++ detections match ultralytics within tolerance using the same image; reuse the IoU `compare()` approach from `mediapipe/examples/pytorch_yolo/yolov8n_demo.py` (run the detector via the existing Python binding `mediapipe/tasks/python/vision/yolo_object_detector.py` if its `libmediapipe.so` is buildable, else compare the gtest's printed boxes to the oracle's `xyxy` by hand). Record the result (matched classes + mean IoU) in this plan's "Bring-up notes" section below. This is a validation step, not a committed target.

- [ ] **Step 6: Commit.**

```bash
git add mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_test.cc
git commit -m "test(tasks-yolo): real integration assertions for YoloObjectDetector (oracle-validated)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Bring-up notes (fill during Task 5)

- Exported model output shape / layout used: …
- Oracle classes on cats_and_dogs.jpg: …
- C++ vs oracle: matched classes …, mean IoU …

## Self-review checklist (before final review)
- Default behavior unchanged: `input_width/height` default 0 ⇒ `inv_w_==0` ⇒ boxes treated as normalized (the existing calculator tests + the tiled pipeline are byte-identical).
- The gated test builds AND runs only when the blob is present; absent ⇒ `glob` empty ⇒ build OK ⇒ `GTEST_SKIP`. CI unaffected.
- No model blob committed (`.gitignore` covers it); only the export script + test/BUILD changes are committed.
- `num_classes` set in the test options; layout confirmed against the actual export.
- Boxes asserted within image bounds and scores above threshold; `max_results` capped.

## Final verification
- [ ] `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:yolo_tensors_to_detections_calculator_test` — green.
- [ ] `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test --test_output=all` — runs (not skipped) and green.
- [ ] Oracle cross-check recorded.
- [ ] `git status` clean except the (gitignored) exported model.

## Findings / follow-ups (discovered during planning)
- **`category_allowlist`/`category_denylist` are accepted but NOT applied.** The
  task copies them into the options proto and validates mutual exclusion, but the
  YOLO graph has no label-map / category-filter calculator, so they have no
  effect; `category_name`/`display_name` are likewise unpopulated. This refines
  the spec (which listed allowlist/denylist under hardening): implementing them
  needs a `DetectionLabelIdToTextCalculator` (fed the metadata label map) + a
  category filter in `yolo_object_detector_graph.cc`. Scoped as a **follow-up**,
  not this plan, to keep this effort focused on proving detection correctness.
  Task 5 therefore asserts only the filters that ARE implemented (score
  threshold, `max_results`).

## Out of scope
OBB / `oriented_object_detector` (follow-up, same recipe); category
allowlist/denylist + label-name mapping (follow-up, above); Python/iOS/Java/Web
bindings; metadata-derived `num_classes`; quantized/GPU.
