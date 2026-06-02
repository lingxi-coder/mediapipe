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
"""MediaPipe Oriented Detection Result C API types."""

import ctypes

from mediapipe.tasks.python.components.containers import category_c


class MpOrientedDetectionC(ctypes.Structure):
  """CTypes for a single oriented detection (pixel units)."""

  _fields_ = [
      ('categories', ctypes.POINTER(category_c.MpCategoryC)),
      ('categories_count', ctypes.c_uint32),
      ('cx', ctypes.c_float),
      ('cy', ctypes.c_float),
      ('width', ctypes.c_float),
      ('height', ctypes.c_float),
      ('rotation', ctypes.c_float),
  ]


class MpOrientedDetectionResultC(ctypes.Structure):
  """CTypes for the oriented detection result."""

  _fields_ = [
      ('detections', ctypes.POINTER(MpOrientedDetectionC)),
      ('detections_count', ctypes.c_uint32),
  ]
