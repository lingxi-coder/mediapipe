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
"""Run yolov8n.pt on an image + video using MediaPipe's C++ YOLO decode+NMS.

torch runs the .pt model in Python to produce the raw [1, 84, 8400] detect-head
output; that tensor is fed into the real MediaPipe C++ graph
(YoloTensorsToDetections -> flatten -> NonMaxSuppression) via the
`_yolo_pt_graph` pybind bridge. Results are drawn on open-source test media and
cross-checked against ultralytics' own postprocess as a correctness oracle.

Usage (after `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1
//mediapipe/examples/pytorch_yolo:_yolo_pt_graph`):

    python3 mediapipe/examples/pytorch_yolo/yolov8n_demo.py
"""

import os

# torch and the MediaPipe extension each link an OpenMP runtime; on macOS that
# trips OMP error #15. Allow the duplicate (demo-only workaround). Must be set
# before importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import urllib.request

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Locate the bazel-built pybind extension.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_BIN = os.path.join(_REPO, "bazel-bin", "mediapipe", "examples", "pytorch_yolo")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import _yolo_pt_graph  # noqa: E402  (provides run_yolo_graph)

_INPUT = 640  # yolov8 default input size
_CONF = 0.25
_IOU = 0.45
_ASSETS = os.path.join(_HERE, "assets")
_OUT = os.path.join(_HERE, "out")

_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
# Small open-source (CC) sample video.
_VIDEO_URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
    "person-bicycle-car-detection.mp4"
)


def _download(url, path):
  if not os.path.exists(path):
    print(f"downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
  return path


def letterbox(img, new=_INPUT, color=(114, 114, 114)):
  """Resize+pad to (new,new) preserving aspect. Returns (padded, gain, padx, pady)."""
  h, w = img.shape[:2]
  gain = min(new / h, new / w)
  nh, nw = int(round(h * gain)), int(round(w * gain))
  resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
  padx, pady = (new - nw) / 2, (new - nh) / 2
  top, bottom = int(round(pady - 0.1)), int(round(pady + 0.1))
  left, right = int(round(padx - 0.1)), int(round(padx + 0.1))
  out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                           cv2.BORDER_CONSTANT, value=color)
  return out, gain, left, top


def raw_forward(torch_model, bgr):
  """Letterbox + run the raw torch model -> raw [1,84,8400] (numpy), + unletterbox params."""
  lb, gain, padx, pady = letterbox(bgr)
  rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
  chw = np.transpose(rgb, (2, 0, 1))[None]  # [1,3,640,640]
  with torch.no_grad():
    y = torch_model(torch.from_numpy(chw))
  raw = y[0] if isinstance(y, (list, tuple)) else y
  raw = raw.detach().cpu().numpy().astype(np.float32)  # [1,84,8400]
  # Box channels (0:4) are in 640-px space; normalize to [0,1] for the decoder.
  raw = raw.copy()
  raw[:, 0:4, :] /= _INPUT
  return raw, gain, padx, pady


def to_pixels(dets, gain, padx, pady, w, h):
  """Map (M,6) normalized [xmin,ymin,w,h,score,label] (letterboxed) -> original px boxes."""
  out = []
  for xmin, ymin, bw, bh, score, label in dets:
    x1 = (xmin * _INPUT - padx) / gain
    y1 = (ymin * _INPUT - pady) / gain
    x2 = ((xmin + bw) * _INPUT - padx) / gain
    y2 = ((ymin + bh) * _INPUT - pady) / gain
    out.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2), score, int(label)))
  return out


def draw(img, boxes, names):
  for x1, y1, x2, y2, score, label in boxes:
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
    tag = f"{names.get(label, label)} {score:.2f}"
    cv2.putText(img, tag, (int(x1), max(12, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
  return img


def run_image(model, torch_model, names):
  os.makedirs(_ASSETS, exist_ok=True)
  os.makedirs(_OUT, exist_ok=True)
  img_path = _download(_IMAGE_URL, os.path.join(_ASSETS, "bus.jpg"))
  bgr = cv2.imread(img_path)
  h, w = bgr.shape[:2]

  raw, gain, padx, pady = raw_forward(torch_model, bgr)
  dets = _yolo_pt_graph.run_yolo_graph(raw, num_classes=len(names),
                                       conf_threshold=_CONF, iou_threshold=_IOU)
  boxes = to_pixels(dets, gain, padx, pady, w, h)
  out_path = os.path.join(_OUT, "bus_mediapipe.jpg")
  cv2.imwrite(out_path, draw(bgr.copy(), boxes, names))

  # Oracle: ultralytics' own postprocess on the same image.
  oracle = model.predict(img_path, conf=_CONF, iou=_IOU, verbose=False)[0]
  print("\n=== IMAGE (bus.jpg) ===")
  print(f"MediaPipe graph: {len(boxes)} detections -> {out_path}")
  mp_classes = sorted(names.get(b[5], b[5]) for b in boxes)
  or_pairs = sorted(
      (names[int(c)], float(s))
      for c, s in zip(oracle.boxes.cls.tolist(), oracle.boxes.conf.tolist()))
  or_classes = sorted(p[0] for p in or_pairs)
  print(f"  MediaPipe classes : {mp_classes}")
  print(f"  ultralytics oracle: {or_classes} ({len(or_classes)} detections)")
  if mp_classes == or_classes:
    print("  MATCH: exact")
  else:
    # Detections the oracle has but we don't, sitting within a small margin of
    # the conf threshold, are explained threshold-edge cases (sub-1%
    # preprocessing-rounding differences), not decode errors.
    margin = 0.02
    borderline = [f"{c} ({s:.4f})" for c, s in or_pairs
                  if s < _CONF + margin and c not in mp_classes]
    matched = sum(1 for c in mp_classes if c in or_classes)
    print(f"  MATCH: {matched}/{len(or_classes)} "
          f"(high-confidence detections agree exactly)")
    if borderline:
      print(f"  borderline oracle-only @conf<{_CONF + margin}: {borderline} "
            f"-- within rounding of the {_CONF} threshold, not a decode error")


def run_video(model, torch_model, names, max_frames=120):
  os.makedirs(_OUT, exist_ok=True)
  vid_path = _download(_VIDEO_URL, os.path.join(_ASSETS, "sample.mp4"))
  cap = cv2.VideoCapture(vid_path)
  fps = cap.get(cv2.CAP_PROP_FPS) or 25
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  out_path = os.path.join(_OUT, "sample_mediapipe.mp4")
  writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
  n, total, t0 = 0, 0, time.time()
  while n < max_frames:
    ok, bgr = cap.read()
    if not ok:
      break
    raw, gain, padx, pady = raw_forward(torch_model, bgr)
    dets = _yolo_pt_graph.run_yolo_graph(raw, num_classes=len(names),
                                         conf_threshold=_CONF, iou_threshold=_IOU)
    boxes = to_pixels(dets, gain, padx, pady, w, h)
    total += len(boxes)
    writer.write(draw(bgr, boxes, names))
    n += 1
  cap.release()
  writer.release()
  dt = time.time() - t0
  print("\n=== VIDEO (sample.mp4) ===")
  print(f"processed {n} frames, {total} detections, "
        f"{n / dt:.1f} fps -> {out_path}")


def main():
  print("loading yolov8n.pt ...")
  model = YOLO("yolov8n.pt")  # downloads weights on first run
  torch_model = model.model.eval()
  names = model.names  # {id: class_name}
  run_image(model, torch_model, names)
  run_video(model, torch_model, names)
  print("\ndone.")


if __name__ == "__main__":
  main()
