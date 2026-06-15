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

  def test_tiling_defaults_disabled(self):
    """A default YoloObjectDetectorOptions has disabled (1x1) tiling; no model."""
    options = _YoloObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path='/dummy/model.tflite')
    )
    self.assertEqual(options.tiling.tile_rows, 1)
    self.assertEqual(options.tiling.tile_cols, 1)
    self.assertIsNone(options.tiling.explicit_tiles)
    self.assertFalse(options.tiling.enable_motion_scheduling)
    self.assertAlmostEqual(options.tiling.tile_overlap_fraction, 0.0, places=5)
    self.assertAlmostEqual(
        options.tiling.tile_local_nms_iou_threshold, 0.0, places=5
    )
    self.assertEqual(options.tiling.max_detections_after_tile_nms, 0)
    self.assertEqual(options.tiling.max_scheduled_tiles, 0)

  def test_options_with_tiling_construct_without_model(self):
    """Constructs TilingOptions (grid + explicit_tiles + caps); no model needed."""
    tiling = yolo_object_detector.TilingOptions(
        tile_rows=2,
        tile_cols=2,
        tile_overlap_fraction=0.2,
        explicit_tiles=[
            yolo_object_detector.TileRect(0.25, 0.25, 0.5, 0.5),
            yolo_object_detector.TileRect(0.75, 0.6, 0.45, 0.3),
        ],
        tile_local_nms_iou_threshold=0.5,
        max_detections_after_tile_nms=50,
        max_scheduled_tiles=7,
    )
    options = _YoloObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path='/dummy/model.tflite'),
        running_mode=_RUNNING_MODE.IMAGE,
        tiling=tiling,
    )
    self.assertEqual(options.tiling.tile_rows, 2)
    self.assertEqual(options.tiling.tile_cols, 2)
    self.assertAlmostEqual(options.tiling.tile_overlap_fraction, 0.2, places=5)
    self.assertLen(options.tiling.explicit_tiles, 2)
    self.assertAlmostEqual(
        options.tiling.explicit_tiles[1].x_center, 0.75, places=5
    )
    self.assertAlmostEqual(
        options.tiling.explicit_tiles[1].height, 0.3, places=5
    )
    self.assertEqual(options.tiling.max_detections_after_tile_nms, 50)
    self.assertFalse(options.tiling.enable_motion_scheduling)
    self.assertAlmostEqual(
        options.tiling.tile_local_nms_iou_threshold, 0.5, places=5
    )
    self.assertEqual(options.tiling.max_scheduled_tiles, 7)

  def test_build_tiling_options_c_marshalling(self):
    """The dataclass->ctypes marshalling fills the struct; no model needed."""
    import ctypes  # pylint: disable=g-import-not-at-top

    tiling = yolo_object_detector.TilingOptions(
        tile_rows=2,
        tile_cols=3,
        tile_overlap_fraction=0.2,
        explicit_tiles=[
            yolo_object_detector.TileRect(0.1, 0.2, 0.5, 0.4),
            yolo_object_detector.TileRect(0.75, 0.6, 0.45, 0.3),
        ],
        tile_local_nms_iou_threshold=0.5,
        max_detections_after_tile_nms=50,
        max_scheduled_tiles=7,
    )
    tiling_c, keepalive = yolo_object_detector._build_tiling_options_c(tiling)  # keepalive kept alive for the explicit_tiles reads below  # pylint: disable=protected-access

    self.assertEqual(tiling_c.tile_rows, 2)
    self.assertEqual(tiling_c.tile_cols, 3)
    self.assertAlmostEqual(tiling_c.tile_overlap_fraction, 0.2, places=5)
    self.assertEqual(tiling_c.explicit_tiles_count, 2)
    self.assertTrue(bool(tiling_c.explicit_tiles))  # non-null pointer
    self.assertAlmostEqual(tiling_c.explicit_tiles[0].x_center, 0.1, places=5)
    self.assertAlmostEqual(tiling_c.explicit_tiles[0].y_center, 0.2, places=5)
    self.assertAlmostEqual(tiling_c.explicit_tiles[1].width, 0.45, places=5)
    self.assertAlmostEqual(tiling_c.explicit_tiles[1].height, 0.3, places=5)
    self.assertAlmostEqual(tiling_c.tile_local_nms_iou_threshold, 0.5, places=5)
    self.assertEqual(tiling_c.max_detections_after_tile_nms, 50)
    self.assertFalse(tiling_c.enable_motion_scheduling)
    self.assertEqual(tiling_c.max_scheduled_tiles, 7)

  def test_build_tiling_options_c_empty(self):
    """Empty explicit_tiles -> null pointer + zero count (C-safe)."""
    tiling_c, _ = yolo_object_detector._build_tiling_options_c(  # pylint: disable=protected-access
        yolo_object_detector.TilingOptions()
    )
    self.assertEqual(tiling_c.explicit_tiles_count, 0)
    self.assertFalse(bool(tiling_c.explicit_tiles))  # null pointer

  def test_ctypes_tiling_layout_matches_c_abi(self):
    """ctypes tiling structs byte-match the C header (see tiling_options_abi_test.cc)."""
    import ctypes  # pylint: disable=g-import-not-at-top

    self.assertEqual(ctypes.sizeof(yolo_object_detector.MpTileRectC), 16)
    self.assertEqual(ctypes.sizeof(yolo_object_detector.MpTilingOptionsC), 48)

    rect_c = yolo_object_detector.MpTileRectC
    self.assertEqual(rect_c.x_center.offset, 0)
    self.assertEqual(rect_c.y_center.offset, 4)
    self.assertEqual(rect_c.width.offset, 8)
    self.assertEqual(rect_c.height.offset, 12)

    tiling_c = yolo_object_detector.MpTilingOptionsC
    self.assertEqual(tiling_c.tile_rows.offset, 0)
    self.assertEqual(tiling_c.tile_cols.offset, 4)
    self.assertEqual(tiling_c.tile_overlap_fraction.offset, 8)
    self.assertEqual(tiling_c.explicit_tiles.offset, 16)
    self.assertEqual(tiling_c.explicit_tiles_count.offset, 24)
    self.assertEqual(tiling_c.tile_local_nms_iou_threshold.offset, 28)
    self.assertEqual(tiling_c.max_detections_after_tile_nms.offset, 32)
    self.assertEqual(tiling_c.enable_motion_scheduling.offset, 36)
    self.assertEqual(tiling_c.max_scheduled_tiles.offset, 40)
    # c_bool width pin: an offset-only check cannot catch a c_bool->c_int swap.
    self.assertEqual(tiling_c.enable_motion_scheduling.size, 1)

    options_c = yolo_object_detector.MpYoloObjectDetectorOptionsC
    # Absolute anchors mirroring tiling_options_abi_test.cc: these cross-check
    # the whole parent prefix (base_options + scalars), not just tiling's
    # relative placement.
    self.assertEqual(options_c.tiling.offset, 136)
    self.assertEqual(options_c.result_callback.offset, 184)
    self.assertEqual(ctypes.sizeof(options_c), 192)

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

  @unittest.skipUnless(_MODEL_PRESENT, 'yolov8n.tflite fixture not present; skipping inference test')
  def test_detect_image_tiled(self):
    """Smoke-checks the tiled detect path end-to-end (model-gated).

    Note: cats_and_dogs.jpg is detectable without tiling, so this verifies the
    tiling path runs and returns a valid result, not tiling efficacy.
    """
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
        tiling=yolo_object_detector.TilingOptions(
            tile_rows=2, tile_cols=2, tile_overlap_fraction=0.2
        ),
    )
    with _YoloObjectDetector.create_from_options(options) as detector:
      result = detector.detect(image)

    self.assertIsInstance(result, _DetectionResult)
    self.assertGreater(
        len(result.detections),
        0,
        'Expected at least one detection on cats_and_dogs.jpg with tiling',
    )


if __name__ == '__main__':
  absltest.main()
