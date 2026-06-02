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
"""Oriented (rotated bounding box) detections data class."""

import dataclasses
from typing import Any, List

from mediapipe.tasks.python.components.containers import category as category_lib
from mediapipe.tasks.python.components.containers import category_c as category_c_lib
from mediapipe.tasks.python.components.containers import oriented_detections_c as oriented_detections_c_lib
from mediapipe.tasks.python.core.optional_dependencies import doc_controls


@dataclasses.dataclass
class OrientedDetection:
  """One oriented (rotated) bounding box detection, in original-image PIXELS.

  Attributes:
    categories: A list of Category objects.
    cx: Box center x, in pixels.
    cy: Box center y, in pixels.
    width: Box width, in pixels.
    height: Box height, in pixels.
    rotation: Rotation angle in radians, counter-clockwise.
  """

  categories: List[category_lib.Category]
  cx: float
  cy: float
  width: float
  height: float
  rotation: float

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, OrientedDetection):
      return False
    return (
        self.categories == other.categories
        and self.cx == other.cx
        and self.cy == other.cy
        and self.width == other.width
        and self.height == other.height
        and self.rotation == other.rotation
    )

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(
      cls, c_obj: oriented_detections_c_lib.MpOrientedDetectionC
  ) -> 'OrientedDetection':
    c_categories = category_c_lib.MpCategoriesC(
        categories=c_obj.categories, categories_count=c_obj.categories_count
    )
    py_categories = category_lib.create_list_of_categories_from_ctypes(
        c_categories
    )
    return OrientedDetection(
        categories=py_categories,
        cx=c_obj.cx,
        cy=c_obj.cy,
        width=c_obj.width,
        height=c_obj.height,
        rotation=c_obj.rotation,
    )


@dataclasses.dataclass
class OrientedObjectDetectionResult:
  """The list of detected oriented objects.

  Attributes:
    detections: A list of `OrientedDetection` objects.
  """

  detections: List[OrientedDetection]

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, OrientedObjectDetectionResult):
      return False
    return self.detections == other.detections

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(
      cls, c_obj: oriented_detections_c_lib.MpOrientedDetectionResultC
  ) -> 'OrientedObjectDetectionResult':
    return OrientedObjectDetectionResult(
        detections=[
            OrientedDetection.from_ctypes(c_obj.detections[i])
            for i in range(c_obj.detections_count)
        ]
    )
