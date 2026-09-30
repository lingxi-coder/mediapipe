#!/usr/bin/env python3
"""Exports yolov8n (COCO) and yolov8n-obb (DOTA) to local ONNX fixtures.

Mirrors export_yolov8n_tflite.py / export_yolov8n_obb_tflite.py. The produced
files are gitignored test-data resources for ONNX backend development.

Usage:
    pip install ultralytics
    python3 mediapipe/tasks/testdata/vision/export_yolov8n_onnx.py

Produces, alongside this script:
    yolov8n.onnx        (COCO, 80 classes, input [1,3,640,640] NCHW)
    yolov8n-obb.onnx    (DOTA, 15 classes + angle, input [1,3,640,640] NCHW)

Downloaded weights and export intermediates stay in .yolo_export_cache/ next
to this script, regardless of the caller's working directory.

Opset 17 is broadly supported by onnxruntime 1.24; bump only if export fails.
"""
import os
import shutil

from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPORT_CACHE = os.path.join(_HERE, ".yolo_export_cache")


def main():
  os.makedirs(_EXPORT_CACHE, exist_ok=True)
  caller_cwd = os.getcwd()
  try:
    os.chdir(_EXPORT_CACHE)
    for weights in ("yolov8n.pt", "yolov8n-obb.pt"):
      model = YOLO(os.path.join(_EXPORT_CACHE, weights))
      out = model.export(format="onnx", opset=17, imgsz=640, simplify=True)
      source = os.path.abspath(str(out))
      assert source.endswith(".onnx") and os.path.exists(source), (
          f"bad export: {source}")
      destination = os.path.join(_HERE, weights.removesuffix(".pt") + ".onnx")
      shutil.copyfile(source, destination)
      print(f"wrote {destination} ({os.path.getsize(destination)} bytes)")
  finally:
    os.chdir(caller_cwd)


if __name__ == "__main__":
  main()
