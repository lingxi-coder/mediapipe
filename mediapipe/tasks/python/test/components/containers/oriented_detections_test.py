# Copyright 2025 The MediaPipe Authors.
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
"""Tests for OrientedDetections conversion between Python and C."""

from absl.testing import absltest

from mediapipe.tasks.python.components.containers import category_c as category_c_lib
from mediapipe.tasks.python.components.containers import oriented_detections as oriented_detections_lib
from mediapipe.tasks.python.components.containers import oriented_detections_c as oriented_detections_c_lib


_CATEGORY = category_c_lib.MpCategoryC(
    index=1,
    score=0.9,
    category_name=b'ship',
    display_name=b'Ship',
)


class OrientedDetectionsTest(absltest.TestCase):

  def test_create_oriented_detection_from_ctypes_with_track_id(self):
    c_categories = (category_c_lib.MpCategoryC * 1)(_CATEGORY)
    c_detection = oriented_detections_c_lib.MpOrientedDetectionC(
        categories=c_categories,
        categories_count=1,
        cx=10.0,
        cy=20.0,
        width=30.0,
        height=40.0,
        rotation=0.3,
        track_id=b'42',
    )

    actual_detection = oriented_detections_lib.OrientedDetection.from_ctypes(
        c_detection
    )

    self.assertEqual(actual_detection.track_id, '42')

  def test_create_oriented_detection_from_ctypes_without_track_id(self):
    c_categories = (category_c_lib.MpCategoryC * 1)(_CATEGORY)
    c_detection = oriented_detections_c_lib.MpOrientedDetectionC(
        categories=c_categories,
        categories_count=1,
        cx=10.0,
        cy=20.0,
        width=30.0,
        height=40.0,
        rotation=0.3,
        track_id=None,
    )

    actual_detection = oriented_detections_lib.OrientedDetection.from_ctypes(
        c_detection
    )

    self.assertIsNone(actual_detection.track_id)


if __name__ == '__main__':
  absltest.main()
