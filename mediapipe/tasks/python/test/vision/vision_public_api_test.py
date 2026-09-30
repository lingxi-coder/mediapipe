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
"""Checks the public vision package exposes the custom detector APIs."""

import importlib
import unittest

from mediapipe.tasks.python import vision


class VisionPublicApiTest(unittest.TestCase):

  def test_exports_detector_types_and_configuration(self):
    for stem, prefix in (
        ('yolo_object_detector', 'YoloObjectDetector'),
        ('oriented_object_detector', 'OrientedObjectDetector'),
    ):
      module = importlib.import_module(
          'mediapipe.tasks.python.vision.' + stem
      )
      for suffix in ('', 'Options', 'Result'):
        with self.subTest(detector=prefix, symbol=suffix):
          self.assertIs(
              getattr(vision, prefix + suffix),
              getattr(module, prefix + suffix),
          )
      for suffix in (
          'Layout', 'TileRect', 'TilingOptions', 'TrackerType', 'TrackingOptions'
      ):
        with self.subTest(detector=prefix, symbol=suffix):
          self.assertIs(
              getattr(vision, prefix + suffix), getattr(module, suffix)
          )


if __name__ == '__main__':
  unittest.main()
