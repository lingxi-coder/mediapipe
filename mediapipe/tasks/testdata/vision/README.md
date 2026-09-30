# YOLO test fixtures

The YOLO export scripts belong to the vision test-data package. Run them from
any directory; their output paths are anchored to this directory.

```bash
python3 mediapipe/tasks/testdata/vision/export_yolov8n_tflite.py
python3 mediapipe/tasks/testdata/vision/export_yolov8n_obb_tflite.py
python3 mediapipe/tasks/testdata/vision/export_yolov8n_onnx.py
```

The final fixtures stay directly in this directory so existing Bazel filegroups
and tests can use them: `yolov8n.tflite`, `yolov8n-obb.tflite`, `yolov8n.onnx`,
`yolov8n-obb.onnx`, label files, and `boats.jpg`. They are generated local files
and remain ignored by Git. The TFLite exports also attach MediaPipe metadata;
the OBB export downloads the demo image when absent.

Downloaded `.pt` weights, ONNX intermediates, SavedModel directories, and
calibration data belong in the ignored `.yolo_export_cache/` directory. Export
scripts run Ultralytics there and copy only the final models into the fixture
directory. The ONNX script regenerates both final ONNX fixtures on each run.

These scripts require Ultralytics and its export dependencies; the TFLite
scripts additionally use FlatBuffers and `tflite_support` metadata utilities.
The models are optional local fixtures: the Bazel filegroups allow them to be
absent, and tests that require them may skip in a fresh checkout.
