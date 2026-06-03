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

torch runs the .pt model in Python (CPU or Apple-GPU/MPS) to produce the raw
[1, 84, A] detect-head output; that tensor is fed into the real MediaPipe C++
graph (YoloTensorsToDetections -> flatten -> NonMaxSuppression) via the
`_yolo_pt_graph` pybind bridge. To make the comparison exact, the input is
ultralytics' OWN preprocessed tensor (so only decode+NMS is being compared), and
results are matched to ultralytics' postprocess by IoU as a correctness oracle.

Usage (after building the bridge):
    python3 mediapipe/examples/pytorch_yolo/yolov8n_demo.py [--device cpu|mps]
"""

import os

# torch and the MediaPipe extension each link an OpenMP runtime; on macOS that
# trips OMP error #15. Allow the duplicate (demo-only workaround). Must be set
# before importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
import time
import urllib.request

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ops

# Locate the bazel-built pybind extension.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_BIN = os.path.join(_REPO, "bazel-bin", "mediapipe", "examples", "pytorch_yolo")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import _yolo_pt_graph  # noqa: E402  (provides run_yolo_graph)

_CONF = 0.25
_IOU = 0.45
_ASSETS = os.path.join(_HERE, "assets")
_OUT = os.path.join(_HERE, "out")

_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
_VIDEO_URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
    "person-bicycle-car-detection.mp4"
)


def _download(url, path):
  if not os.path.exists(path):
    print(f"downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
  return path


def init_predictor(model, sample_bgr, device):
  """Initialize ultralytics' predictor (sets up its exact preprocessing)."""
  model.predict(sample_bgr, conf=_CONF, iou=_IOU, device=device, verbose=False,
                save=False)
  return model.predictor


def raw_forward(model, predictor, bgr):
  """ultralytics-preprocess + raw model forward.

  Returns (raw[1,84,A] with box channels normalized to [0,1], (in_h, in_w)).
  Using the predictor's own preprocessing means the input is identical to the
  oracle's, so any difference is purely decode+NMS.
  """
  im = predictor.preprocess([bgr])  # [1,3,H,W] float, on the model's device
  in_h, in_w = int(im.shape[2]), int(im.shape[3])
  with torch.no_grad():
    y = model.model(im)
  raw = (y[0] if isinstance(y, (list, tuple)) else y)
  raw = raw.detach().to("cpu").numpy().astype(np.float32).copy()  # [1,84,A]
  # Box channels are in input-pixel space; normalize (cx,w by W; cy,h by H).
  raw[:, 0, :] /= in_w
  raw[:, 2, :] /= in_w
  raw[:, 1, :] /= in_h
  raw[:, 3, :] /= in_h
  return raw, (in_h, in_w)


def to_orig(dets, in_hw, orig_hw):
  """Map (M,6) normalized [xmin,ymin,w,h,score,label] (letterboxed) -> original
  xyxy pixels using ultralytics' scale_boxes."""
  if len(dets) == 0:
    return []
  in_h, in_w = in_hw
  xyxy = np.empty((len(dets), 4), dtype=np.float32)
  meta = []
  for i, (xmin, ymin, bw, bh, score, label) in enumerate(dets):
    xyxy[i] = [xmin * in_w, ymin * in_h, (xmin + bw) * in_w, (ymin + bh) * in_h]
    meta.append((float(score), int(label)))
  xyxy = ops.scale_boxes(in_hw, xyxy, orig_hw)
  return [(float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]),
           float(xyxy[i][3]), meta[i][0], meta[i][1]) for i in range(len(dets))]


def iou_xyxy(a, b):
  ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
  ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
  inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
  aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
  ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
  uni = aa + ab - inter
  return inter / uni if uni > 0 else 0.0


def compare(ours, oracle, names):
  """Greedy IoU-match our detections to the oracle; report box+class agreement."""
  o_boxes = oracle.boxes.xyxy.cpu().numpy().tolist()
  o_cls = [int(c) for c in oracle.boxes.cls.tolist()]
  o_conf = [float(s) for s in oracle.boxes.conf.tolist()]
  used = [False] * len(o_boxes)
  matched, ious = 0, []
  for d in ours:
    best, best_iou = -1, 0.0
    for j, ob in enumerate(o_boxes):
      if used[j] or o_cls[j] != d[5]:
        continue
      v = iou_xyxy(d[:4], ob)
      if v > best_iou:
        best, best_iou = j, v
    if best >= 0 and best_iou >= 0.7:
      used[best] = True
      matched += 1
      ious.append(best_iou)
  unmatched_oracle = [
      f"{names[o_cls[j]]}({o_conf[j]:.4f})" for j in range(len(o_boxes))
      if not used[j]
  ]
  print(f"  ours={len(ours)} oracle={len(o_boxes)} "
        f"box-matched={matched}/{len(o_boxes)} "
        f"meanIoU={np.mean(ious):.4f}" if ious else
        f"  ours={len(ours)} oracle={len(o_boxes)} box-matched=0")
  if matched == len(o_boxes) and len(ours) == len(o_boxes):
    print("  RESULT: exact match with the ultralytics oracle (boxes + classes)")
  else:
    print(f"  RESULT: {matched}/{len(o_boxes)} matched; "
          f"unmatched oracle: {unmatched_oracle}")


def draw(img, boxes, names):
  for x1, y1, x2, y2, score, label in boxes:
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
    cv2.putText(img, f"{names.get(label, label)} {score:.2f}",
                (int(x1), max(12, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 200, 0), 1, cv2.LINE_AA)
  return img


def run_image(model, predictor, names, device):
  os.makedirs(_ASSETS, exist_ok=True)
  os.makedirs(_OUT, exist_ok=True)
  img_path = _download(_IMAGE_URL, os.path.join(_ASSETS, "bus.jpg"))
  bgr = cv2.imread(img_path)
  h, w = bgr.shape[:2]
  raw, in_hw = raw_forward(model, predictor, bgr)
  dets = _yolo_pt_graph.run_yolo_graph(raw, num_classes=len(names),
                                       conf_threshold=_CONF, iou_threshold=_IOU)
  boxes = to_orig(dets, in_hw, (h, w))
  out_path = os.path.join(_OUT, f"bus_mediapipe_{device}.jpg")
  cv2.imwrite(out_path, draw(bgr.copy(), boxes, names))
  oracle = model.predict(img_path, conf=_CONF, iou=_IOU, device=device,
                         verbose=False)[0]
  print(f"\n=== IMAGE (bus.jpg) [device={device}] -> {out_path} ===")
  compare(boxes, oracle, names)


def run_video(model, predictor, names, device, max_frames=120):
  os.makedirs(_OUT, exist_ok=True)
  vid_path = _download(_VIDEO_URL, os.path.join(_ASSETS, "sample.mp4"))
  cap = cv2.VideoCapture(vid_path)
  fps = cap.get(cv2.CAP_PROP_FPS) or 25
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  out_path = os.path.join(_OUT, f"sample_mediapipe_{device}.mp4")
  writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
  n, total, t0 = 0, 0, time.time()
  while n < max_frames:
    ok, bgr = cap.read()
    if not ok:
      break
    raw, in_hw = raw_forward(model, predictor, bgr)
    dets = _yolo_pt_graph.run_yolo_graph(raw, num_classes=len(names),
                                         conf_threshold=_CONF, iou_threshold=_IOU)
    writer.write(draw(bgr, to_orig(dets, in_hw, (h, w)), names))
    total += len(dets)
    n += 1
  cap.release()
  writer.release()
  dt = time.time() - t0
  print(f"\n=== VIDEO (sample.mp4) [device={device}] -> {out_path} ===")
  print(f"  {n} frames, {total} detections, {n / dt:.1f} fps "
        f"(model forward on {device}; decode+NMS on the CPU MediaPipe graph)")


def run_device(device):
  print(f"\n############ DEVICE = {device} ############")
  model = YOLO("yolov8n.pt")
  names = model.names
  os.makedirs(_ASSETS, exist_ok=True)
  sample = _download(_IMAGE_URL, os.path.join(_ASSETS, "bus.jpg"))
  predictor = init_predictor(model, sample, device)
  run_image(model, predictor, names, device)
  run_video(model, predictor, names, device)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--device", default="both", choices=["cpu", "mps", "both"])
  args = ap.parse_args()
  if args.device == "both":
    devices = ["cpu"]
    if torch.backends.mps.is_available():
      devices.append("mps")
    else:
      print("MPS not available; running CPU only.")
  else:
    devices = [args.device]
  for d in devices:
    run_device(d)
  print("\ndone.")


if __name__ == "__main__":
  main()
