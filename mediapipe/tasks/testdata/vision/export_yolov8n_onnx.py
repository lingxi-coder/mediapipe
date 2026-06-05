#!/usr/bin/env python3
"""Exports yolov8n (COCO) and yolov8n-obb (DOTA) to ONNX for the ONNX backend tests.

Mirrors export_yolov8n_tflite.py / export_yolov8n_obb_tflite.py. The produced
files are gitignored; they enable the (otherwise skipped) ONNX backend tests.

Usage:
    pip install ultralytics
    python3 export_yolov8n_onnx.py

Produces, alongside this script:
    yolov8n.onnx        (COCO, 80 classes, input [1,3,640,640] NCHW)
    yolov8n-obb.onnx    (DOTA, 15 classes + angle, input [1,3,640,640] NCHW)

Opset 17 is broadly supported by onnxruntime 1.24; bump only if export fails.
"""
from ultralytics import YOLO

for weights in ("yolov8n.pt", "yolov8n-obb.pt"):
    YOLO(weights).export(format="onnx", opset=17, imgsz=640, simplify=True)
