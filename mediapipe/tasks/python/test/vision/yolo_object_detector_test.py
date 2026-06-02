# Copyright 2026 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for YOLO object detector."""

import os
import unittest

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.tasks.python.components.containers import detections as detections_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import yolo_object_detector
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_DetectionResult = detections_module.DetectionResult
_Image = image_module.Image
_YoloObjectDetector = yolo_object_detector.YoloObjectDetector
_YoloObjectDetectorOptions = yolo_object_detector.YoloObjectDetectorOptions
_Layout = yolo_object_detector.Layout
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode

_MODEL_FILE = 'yolov8n.tflite'
_IMAGE_FILE = 'cats_and_dogs.jpg'
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/vision'


def _model_available() -> bool:
  """Returns True iff the YOLO model fixture is present on disk."""
  try:
    path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _MODEL_FILE)
    )
    return os.path.exists(path)
  except Exception:  # pylint: disable=broad-except
    # get_test_data_path raises RuntimeError when TEST_SRCDIR is missing and
    # ValueError when the file is not found under TEST_SRCDIR.
    return False


_MODEL_PRESENT = _model_available()


class YoloObjectDetectorTest(parameterized.TestCase):

  def test_options_construct_without_model(self):
    """Constructs YoloObjectDetectorOptions and checks defaults; no model needed."""
    base_options = _BaseOptions(model_asset_path='/dummy/model.tflite')
    options = _YoloObjectDetectorOptions(
        base_options=base_options,
        running_mode=_RUNNING_MODE.IMAGE,
        max_results=10,
        layout=_Layout.CHANNELS_LAST,
        num_classes=80,
    )
    # Verify YOLO-specific defaults.
    self.assertAlmostEqual(options.iou_threshold, 0.45, places=5)
    self.assertEqual(int(options.layout), 2)
    # Verify constructor arguments were stored correctly.
    self.assertEqual(options.max_results, 10)
    self.assertEqual(options.num_classes, 80)
    self.assertEqual(options.running_mode, _RUNNING_MODE.IMAGE)

  @unittest.skipUnless(_MODEL_PRESENT, 'yolov8n.tflite fixture not present; skipping inference test')
  def test_detect_image(self):
    """Runs inference and validates the detection result structure."""
    model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _MODEL_FILE)
    )
    image_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _IMAGE_FILE)
    )
    image = _Image.create_from_file(image_path)

    options = _YoloObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RUNNING_MODE.IMAGE,
        score_threshold=0.25,
        layout=_Layout.CHANNELS_LAST,
        num_classes=80,
    )
    with _YoloObjectDetector.create_from_options(options) as detector:
      result = detector.detect(image)

    # Result must be a DetectionResult with at least one detection.
    self.assertIsInstance(result, _DetectionResult)
    self.assertGreater(
        len(result.detections),
        0,
        'Expected at least one detection on cats_and_dogs.jpg',
    )

    for detection in result.detections:
      # Each detection has exactly one category with a positive score.
      self.assertLen(detection.categories, 1)
      score = detection.categories[0].score
      self.assertGreater(score, 0.0, f'Category score must be positive: {detection}')

      # Bounding box must be within the image plane (pixel units, non-negative).
      bb = detection.bounding_box
      self.assertGreaterEqual(bb.origin_x, 0, f'origin_x must be >= 0: {bb}')
      self.assertGreaterEqual(bb.origin_y, 0, f'origin_y must be >= 0: {bb}')
      self.assertGreater(bb.width, 0, f'width must be > 0: {bb}')
      self.assertGreater(bb.height, 0, f'height must be > 0: {bb}')


if __name__ == '__main__':
  absltest.main()
