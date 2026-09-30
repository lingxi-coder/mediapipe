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

Downloaded weights and export intermediates stay in .yolo_export_cache/ next
to this script, regardless of the caller's working directory.

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

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPORT_CACHE = os.path.join(_HERE, ".yolo_export_cache")
_WEIGHTS = os.path.join(_EXPORT_CACHE, "yolov8n.pt")
_OUT_MODEL = os.path.join(_HERE, "yolov8n.tflite")
_LABELS = os.path.join(_HERE, "yolov8n_labels.txt")


def export_float32_tflite():
  """ultralytics export -> (path to the float32 .tflite, names dict)."""
  os.makedirs(_EXPORT_CACHE, exist_ok=True)
  caller_cwd = os.getcwd()
  try:
    os.chdir(_EXPORT_CACHE)
    model = YOLO(_WEIGHTS)  # auto-downloads if absent
    out = model.export(format="tflite", imgsz=640, nms=False)
    path = os.path.abspath(str(out))
  finally:
    os.chdir(caller_cwd)
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
  displayer = _metadata.MetadataDisplayer.with_model_file(_OUT_MODEL)
  assert displayer.get_metadata_json(), "metadata not attached"
  print(f"wrote {_OUT_MODEL} ({os.path.getsize(_OUT_MODEL)} bytes) with metadata")


if __name__ == "__main__":
  main()
