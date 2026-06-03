# yolov8n.pt through the MediaPipe C++ YOLO graph (Python-driven)

Runs a PyTorch `yolov8n.pt` model on this machine and decodes its output with
the **real MediaPipe C++ calculators** built in this fork. This is the runnable
realization of the "PyTorch backend" on a Mac (no in-graph LibTorch needed):

```
yolov8n.pt ──(torch, Python)──► raw [1,84,8400] detect-head tensor
                                      │  (box channels normalized to [0,1])
                                      ▼  numpy → pybind bridge (_yolo_pt_graph)
        ┌──────────────────────── MediaPipe CalculatorGraph (C++) ────────────────────────┐
        │ YoloTensorsToDetectionsCalculator → YoloBatchDetectionsToSingle → NonMaxSuppression│
        └────────────────────────────────────────────────────────────────────────────────┘
                                      ▼  std::vector<Detection> → numpy
                       un-letterbox → draw (cv2) → annotated image / video
```

torch runs the model; **MediaPipe's C++ graph does the anchor-free decode + NMS**.

## Files
- `yolo_pt_graph_bridge.cc` — pybind extension `_yolo_pt_graph`: builds a
  `mediapipe::Tensor` from the numpy raw output, runs the real
  `YoloTensorsToDetections → flatten → NonMaxSuppression` graph, returns
  detections. Uses pybind11_bazel's `pybind_extension` (single consistent
  pybind11) and `//third_party:opencv`; deliberately avoids
  `//mediapipe/python:builtin_task_graphs` (which pulls a tasks package missing
  from this fork).
- `yolov8n_demo.py` — loads `yolov8n.pt` (ultralytics), letterboxes input, runs
  the raw torch model, feeds the tensor through the bridge, un-letterboxes,
  draws, and cross-checks against ultralytics' own postprocess as an oracle.

## Run
```bash
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/examples/pytorch_yolo:_yolo_pt_graph
KMP_DUPLICATE_LIB_OK=TRUE python3 mediapipe/examples/pytorch_yolo/yolov8n_demo.py
```
Requires `torch`, `ultralytics`, `opencv-python`, `numpy` (all present in the
project's Python env). Downloads `yolov8n.pt`, an open-source image
(`ultralytics.com/images/bus.jpg`) and a CC sample video on first run; writes
annotated outputs to `out/`.

## Result on this machine (Apple Silicon, CPU graph)
- **Image (bus.jpg):** the C++ graph produces the same high-confidence detections
  as ultralytics (bus + 4 person). The only difference is a single `stop sign`
  at conf `0.2551` — `0.005` above the `0.25` threshold — which a sub-1%
  letterbox-rounding difference drops in our preprocessing. Not a decode error.
- **Video:** ~120 frames at ~20 fps on CPU, annotated to `out/sample_mediapipe.mp4`.

## Notes
- `KMP_DUPLICATE_LIB_OK=TRUE`: torch and the extension each link an OpenMP
  runtime; on macOS that trips OMP error #15. The flag allows the duplicate
  (demo-only workaround); the demo also sets it in-process.
- The box channels of the raw YOLOv8 output are in 640-px model space; the demo
  normalizes them to `[0,1]` before the graph (the decoder's `xywh_normalized`
  convention), then un-letterboxes to original pixels for drawing.
- This is the on-Mac, Python-driven validation path. The in-graph C++ LibTorch
  inference backend (Phase 6 / M5) is separate and not built here (no LibTorch
  wired into Bazel).
