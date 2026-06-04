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
"""Tests for oriented (OBB) object detector."""

import math
import os
import unittest

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.tasks.python.components.containers import oriented_detections as detections_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import oriented_object_detector
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_OrientedObjectDetectionResult = detections_module.OrientedObjectDetectionResult
_Image = image_module.Image
_OrientedObjectDetector = oriented_object_detector.OrientedObjectDetector
_OrientedObjectDetectorOptions = oriented_object_detector.OrientedObjectDetectorOptions
_Layout = oriented_object_detector.Layout
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode

_MODEL_FILE = 'yolov8n-obb.tflite'
_IMAGE_FILE = 'cats_and_dogs.jpg'
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/vision'


def _model_available() -> bool:
  """Returns True iff the OBB model fixture is present on disk."""
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


class OrientedObjectDetectorTest(parameterized.TestCase):

  def test_options_construct_without_model(self):
    """Constructs OrientedObjectDetectorOptions and checks defaults; no model needed."""
    base_options = _BaseOptions(model_asset_path='/dummy/model.tflite')
    options = _OrientedObjectDetectorOptions(
        base_options=base_options,
        running_mode=_RUNNING_MODE.IMAGE,
        max_results=10,
        layout=_Layout.CHANNELS_LAST,
        num_classes=15,
    )
    # Verify OBB-specific defaults.
    self.assertAlmostEqual(options.score_threshold, 0.25, places=5)
    self.assertAlmostEqual(options.iou_threshold, 0.45, places=5)
    self.assertEqual(int(options.layout), 2)
    self.assertFalse(options.class_agnostic_nms)
    # Verify constructor arguments were stored correctly.
    self.assertEqual(options.num_classes, 15)

  @unittest.skipUnless(
      _MODEL_PRESENT,
      'yolov8n-obb.tflite fixture not present; skipping inference test',
  )
  def test_detect_image(self):
    """Runs inference and validates the oriented detection result structure."""
    model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _MODEL_FILE)
    )
    image_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _IMAGE_FILE)
    )
    image = _Image.create_from_file(image_path)

    options = _OrientedObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RUNNING_MODE.IMAGE,
        score_threshold=0.25,
        layout=_Layout.CHANNELS_FIRST,
        num_classes=15,
    )
    with _OrientedObjectDetector.create_from_options(options) as detector:
      result = detector.detect(image)

    # Result must be an OrientedObjectDetectionResult with at least one detection.
    self.assertIsInstance(result, _OrientedObjectDetectionResult)
    self.assertGreater(
        len(result.detections),
        0,
        'Expected at least one detection on cats_and_dogs.jpg',
    )

    for detection in result.detections:
      # Each detection has exactly one category with a positive score.
      self.assertLen(detection.categories, 1)
      score = detection.categories[0].score
      self.assertGreater(
          score, 0.0, f'Category score must be positive: {detection}'
      )

      # OBB box fields must be in pixel units: positive width/height and finite
      # rotation angle (radians).
      self.assertGreater(
          detection.width, 0, f'width must be > 0: {detection}'
      )
      self.assertGreater(
          detection.height, 0, f'height must be > 0: {detection}'
      )
      self.assertTrue(
          math.isfinite(detection.rotation),
          f'rotation must be finite: {detection}',
      )

  @unittest.skipUnless(
      _MODEL_PRESENT,
      'yolov8n-obb.tflite fixture not present; skipping inference test',
  )
  def test_category_names_and_allow_deny_filter(self):
    """Asserts graph-provided category names and allow/deny filtering on boats.jpg.

    Verifies that:
      1. The OBB graph populates category_name from the model metadata label
         file (not from the Python _load_label_map fallback), so index 1 maps
         to 'ship' on the DOTA boats.jpg fixture.
      2. category_allowlist=['ship'] retains only ship detections.
      3. category_denylist=['ship'] suppresses all ship detections.
    """
    model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _MODEL_FILE)
    )
    # boats.jpg is a DOTA aerial image where YOLO-OBB reliably detects ships
    # (DOTA class index 1).  The image lives in the same testdata directory as
    # the other OBB fixtures.
    boats_image_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, 'boats.jpg')
    )
    image = _Image.create_from_file(boats_image_path)

    # --- default options: all classes, verify names come from the graph ---
    options = _OrientedObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RUNNING_MODE.IMAGE,
        num_classes=15,
        score_threshold=0.25,
        max_results=10,
        layout=_Layout.CHANNELS_FIRST,
    )
    with _OrientedObjectDetector.create_from_options(options) as detector:
      # Isolate the graph path: disable the best-effort Python label fallback so
      # category_name can ONLY come from the graph (read off the C result),
      # proving the in-graph metadata label mapping works end-to-end through
      # Python rather than being filled by _load_label_map / _enrich_with_label_map.
      detector._label_map = None  # pylint: disable=protected-access
      result = detector.detect(image)
    self.assertIsInstance(result, _OrientedObjectDetectionResult)
    self.assertGreater(
        len(result.detections),
        0,
        'Expected at least one OBB detection on boats.jpg',
    )
    saw_ship = False
    for det in result.detections:
      cat = det.categories[0]
      self.assertIsNotNone(
          cat.category_name,
          f'category_name must not be None (graph must populate it): {det}',
      )
      if cat.index == 1:
        self.assertEqual(
            cat.category_name,
            'ship',
            f'DOTA index 1 must map to "ship", got {cat.category_name!r}',
        )
        saw_ship = True
    self.assertTrue(saw_ship, 'Expected at least one "ship" detection on boats.jpg')

    # --- allowlist=['ship']: only ship detections must survive ---
    allow_options = _OrientedObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RUNNING_MODE.IMAGE,
        num_classes=15,
        score_threshold=0.25,
        max_results=10,
        layout=_Layout.CHANNELS_FIRST,
        category_allowlist=['ship'],
    )
    with _OrientedObjectDetector.create_from_options(allow_options) as detector:
      result = detector.detect(image)
    self.assertGreater(
        len(result.detections),
        0,
        'allowlist=["ship"] must still yield detections on boats.jpg',
    )
    for det in result.detections:
      self.assertEqual(
          det.categories[0].index,
          1,
          f'allowlist=["ship"] must keep only index 1, got {det.categories[0]}',
      )
      self.assertEqual(
          det.categories[0].category_name,
          'ship',
          f'allowlist=["ship"] must keep only "ship", got {det.categories[0]}',
      )

    # --- denylist=['ship']: no ship detection must survive ---
    deny_options = _OrientedObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RUNNING_MODE.IMAGE,
        num_classes=15,
        score_threshold=0.25,
        max_results=10,
        layout=_Layout.CHANNELS_FIRST,
        category_denylist=['ship'],
    )
    with _OrientedObjectDetector.create_from_options(deny_options) as detector:
      result = detector.detect(image)
    for det in result.detections:
      self.assertNotEqual(
          det.categories[0].index,
          1,
          f'denylist=["ship"] must suppress index 1, got {det.categories[0]}',
      )


if __name__ == '__main__':
  absltest.main()
