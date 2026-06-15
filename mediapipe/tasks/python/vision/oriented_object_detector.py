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
"""MediaPipe oriented (OBB) object detector task."""

import ctypes
import dataclasses
import enum
from typing import Callable, List, Optional

from mediapipe.tasks.python.components.containers import oriented_detections as oriented_detections_module
from mediapipe.tasks.python.components.containers import oriented_detections_c as oriented_detections_c_module
from mediapipe.tasks.python.core import async_result_dispatcher
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c as base_options_c_module
from mediapipe.tasks.python.core import mediapipe_c_bindings as mediapipe_c_bindings_c_module
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import image_processing_options_c as image_processing_options_c_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

OrientedObjectDetectorResult = oriented_detections_module.OrientedObjectDetectionResult
_BaseOptions = base_options_module.BaseOptions
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_AsyncResultDispatcher = async_result_dispatcher.AsyncResultDispatcher


class Layout(enum.IntEnum):
  """OBB detect-head output tensor layout (numerically matches the C++ enum)."""

  CHANNELS_FIRST = 1
  CHANNELS_LAST = 2


_C_TYPES_RESULT_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32,  # MpStatus
    ctypes.POINTER(oriented_detections_c_module.MpOrientedDetectionResultC),
    ctypes.c_void_p,  # MpImage
    ctypes.c_int64,  # timestamp_ms
)


def _load_label_map(model_path: Optional[str]) -> Optional[List[str]]:
  """Best-effort: ordered category names from the model's TFLite metadata.

  Returns None when no label file is present (the common case for stock
  ultralytics exports) or on any parsing error. Display-only enrichment.
  """
  if not model_path:
    return None
  try:
    from mediapipe.tasks.python.metadata import metadata as _metadata  # pylint: disable=g-import-not-at-top

    displayer = _metadata.MetadataDisplayer.with_model_file(model_path)
    for name in displayer.get_packed_associated_file_list():
      if name.endswith('.txt') or 'label' in name.lower():
        buf = displayer.get_associated_file_buffer(name)
        text = buf.decode('utf-8') if isinstance(buf, bytes) else buf
        labels = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if labels:
          return labels
  except Exception:  # pylint: disable=broad-except
    return None
  return None


def _enrich_with_label_map(
    result: OrientedObjectDetectorResult, label_map: Optional[List[str]]
) -> OrientedObjectDetectorResult:
  """Sets category_name from label_map by index, in place. No-op if None."""
  if not label_map:
    return result
  for detection in result.detections:
    for category in detection.categories:
      idx = category.index
      if category.category_name is None and idx is not None and (
          0 <= idx < len(label_map)
      ):
        category.category_name = label_map[idx]
  return result


class MpOrientedTileRectC(ctypes.Structure):
  """Byte-matches struct MpOrientedTileRect in the OBB C header."""

  _fields_ = [
      ('x_center', ctypes.c_float),
      ('y_center', ctypes.c_float),
      ('width', ctypes.c_float),
      ('height', ctypes.c_float),
  ]


class MpOrientedTilingOptionsC(ctypes.Structure):
  """Byte-matches struct MpOrientedTilingOptions in the OBB C header (6 fields).

  Field order/types MUST stay in sync with the C header (pinned by
  tiling_options_abi_test.cc). OBB has no motion-scheduling fields.
  """

  _fields_ = [
      ('tile_rows', ctypes.c_int),
      ('tile_cols', ctypes.c_int),
      ('tile_overlap_fraction', ctypes.c_float),
      ('explicit_tiles', ctypes.POINTER(MpOrientedTileRectC)),
      ('explicit_tiles_count', ctypes.c_uint32),
      ('tile_local_nms_iou_threshold', ctypes.c_float),
      ('max_detections_after_tile_nms', ctypes.c_int),
  ]


class MpOrientedObjectDetectorOptionsC(ctypes.Structure):
  """OBB detector options for the C API.

  Field order MUST byte-match struct MpOrientedObjectDetectorOptions in
  mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h.
  """

  _fields_ = [
      ('base_options', base_options_c_module.MpBaseOptionsC),
      ('running_mode', ctypes.c_int),
      ('display_names_locale', ctypes.c_char_p),
      ('max_results', ctypes.c_int),
      ('score_threshold', ctypes.c_float),
      ('category_allowlist', ctypes.POINTER(ctypes.c_char_p)),
      ('category_allowlist_count', ctypes.c_uint32),
      ('category_denylist', ctypes.POINTER(ctypes.c_char_p)),
      ('category_denylist_count', ctypes.c_uint32),
      ('iou_threshold', ctypes.c_float),
      ('class_agnostic_nms', ctypes.c_bool),
      ('layout', ctypes.c_int),
      ('num_classes', ctypes.c_int),
      ('tiling', MpOrientedTilingOptionsC),
      ('result_callback', _C_TYPES_RESULT_CALLBACK),
  ]


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpOrientedObjectDetectorCreate',
        (
            ctypes.POINTER(MpOrientedObjectDetectorOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpOrientedObjectDetectorDetectImage',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.MpImageProcessingOptionsC
            ),
            ctypes.POINTER(
                oriented_detections_c_module.MpOrientedDetectionResultC
            ),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpOrientedObjectDetectorDetectForVideo',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.MpImageProcessingOptionsC
            ),
            ctypes.c_int64,
            ctypes.POINTER(
                oriented_detections_c_module.MpOrientedDetectionResultC
            ),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpOrientedObjectDetectorDetectAsync',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.MpImageProcessingOptionsC
            ),
            ctypes.c_int64,
        ),
    ),
    mediapipe_c_utils.CFunction(
        'MpOrientedObjectDetectorCloseResult',
        [ctypes.POINTER(oriented_detections_c_module.MpOrientedDetectionResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpOrientedObjectDetectorClose',
        (ctypes.c_void_p,),
    ),
)


@dataclasses.dataclass
class OrientedObjectDetectorOptions:
  """Options for the oriented (OBB) object detector task.

  Attributes:
    base_options: Base options for the oriented object detector task.
    running_mode: The running mode of the task. Default to the image mode.
      Oriented object detector task has three running modes: 1) The image mode
      for detecting objects on single image inputs. 2) The video mode for
      detecting objects on the decoded frames of a video. 3) The live stream
      mode for detecting objects on a live stream of input data, such as from
      camera.
    display_names_locale: The locale to use for display names specified through
      the TFLite Model Metadata.
    max_results: The maximum number of top-scored detection results to return.
    score_threshold: Overrides the ones provided in the model metadata. Results
      below this value are rejected. Default 0.25.
    category_allowlist: Allowlist of category names. If non-empty, detection
      results whose category name is not in this set will be filtered out.
      Duplicate or unknown category names are ignored. Mutually exclusive with
      `category_denylist`.
    category_denylist: Denylist of category names. If non-empty, detection
      results whose category name is in this set will be filtered out. Duplicate
      or unknown category names are ignored. Mutually exclusive with
      `category_allowlist`.
    iou_threshold: IoU threshold for rotated non-maximum suppression. Default
      0.45.
    class_agnostic_nms: If True, NMS is applied across all classes jointly.
    layout: The output tensor layout of the OBB detect head. Default
      CHANNELS_FIRST.
    num_classes: Number of classes. If 0, derived from model metadata.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.

  Category names are populated by the OBB graph from the model metadata's label
  file, and category_allowlist / category_denylist filter results by class name.
  The Python `_load_label_map` fallback below is a display-only safety net; it
  never overrides a name the graph already provided and does not implement
  filtering.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  display_names_locale: Optional[str] = None
  max_results: Optional[int] = -1
  score_threshold: Optional[float] = 0.25
  category_allowlist: Optional[List[str]] = None
  category_denylist: Optional[List[str]] = None
  iou_threshold: float = 0.45
  class_agnostic_nms: bool = False
  layout: Layout = Layout.CHANNELS_FIRST
  num_classes: int = 0
  result_callback: Optional[
      Callable[[OrientedObjectDetectorResult, image_module.Image, int], None]
  ] = None


class OrientedObjectDetector:
  """Performs oriented (rotated bounding box) object detection on images.

  The API expects an OBB TFLite model with the appropriate metadata.

  Example usage:
    detector = OrientedObjectDetector.create_from_model_path(
        '/path/to/model.tflite'
    )
    result = detector.detect(image)
    detector.close()

  Or as a context manager:
    with OrientedObjectDetector.create_from_model_path(
        '/path/to/model.tflite'
    ) as d:
      result = d.detect(image)
  """

  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p
  _dispatcher: _AsyncResultDispatcher
  _async_callback: _C_TYPES_RESULT_CALLBACK
  _label_map: Optional[List[str]]

  def __init__(
      self,
      lib: serial_dispatcher.SerialDispatcher,
      handle: ctypes.c_void_p,
      dispatcher: _AsyncResultDispatcher,
      async_callback: _C_TYPES_RESULT_CALLBACK,
      label_map: Optional[List[str]],
  ):
    """Initializes the oriented object detector.

    Args:
      lib: The dispatch library to use for the oriented object detector.
      handle: The C pointer to the oriented object detector.
      dispatcher: The async result handler for the oriented object detector.
      async_callback: The c callback for the oriented object detector.
      label_map: Optional ordered list of category name strings loaded from
        TFLite metadata; used for best-effort label enrichment.
    """
    self._lib = lib
    self._handle = handle
    self._dispatcher = dispatcher
    self._async_callback = async_callback
    self._label_map = label_map

  @classmethod
  def create_from_model_path(
      cls, model_path: str
  ) -> 'OrientedObjectDetector':
    """Creates an `OrientedObjectDetector` object from a TFLite model path.

    Note that the created `OrientedObjectDetector` instance is in image mode,
    for detecting objects on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `OrientedObjectDetector` object created from the model file and default
      `OrientedObjectDetectorOptions`.

    Raises:
      ValueError: If failed to create `OrientedObjectDetector` object from the
        provided file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    options = OrientedObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RunningMode.IMAGE,
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: OrientedObjectDetectorOptions
  ) -> 'OrientedObjectDetector':
    """Creates the `OrientedObjectDetector` from oriented object detector options.

    Args:
      options: Options for the oriented object detector task.

    Returns:
      `OrientedObjectDetector` object created from `options`.

    Raises:
      ValueError: If failed to create `OrientedObjectDetector` object from
        `OrientedObjectDetectorOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    running_mode_module.validate_running_mode(
        options.running_mode, options.result_callback
    )
    lib = mediapipe_c_bindings_c_module.load_shared_library(_CTYPES_SIGNATURES)
    label_map = _load_label_map(
        getattr(options.base_options, 'model_asset_path', None)
    )

    def convert_result(c_result_ptr, image_ptr, timestamp_ms):
      c_result = c_result_ptr[0]
      py_result = _enrich_with_label_map(
          OrientedObjectDetectorResult.from_ctypes(c_result), label_map
      )
      py_image = image_module.Image.create_from_ctypes(image_ptr)
      return (py_result, py_image, timestamp_ms)

    dispatcher = _AsyncResultDispatcher(converter=convert_result)
    c_callback = dispatcher.wrap_callback(
        options.result_callback, _C_TYPES_RESULT_CALLBACK
    )

    allowlist_c = mediapipe_c_bindings_c_module.convert_strings_to_ctypes_array(
        options.category_allowlist
    )
    denylist_c = mediapipe_c_bindings_c_module.convert_strings_to_ctypes_array(
        options.category_denylist
    )
    ctypes_options = MpOrientedObjectDetectorOptionsC(
        base_options=options.base_options.to_ctypes(),
        running_mode=options.running_mode.ctype,
        display_names_locale=(
            options.display_names_locale.encode('utf-8')
            if options.display_names_locale
            else None
        ),
        max_results=options.max_results,
        score_threshold=options.score_threshold,
        category_allowlist=allowlist_c,
        category_allowlist_count=(
            len(options.category_allowlist) if options.category_allowlist else 0
        ),
        category_denylist=denylist_c,
        category_denylist_count=(
            len(options.category_denylist) if options.category_denylist else 0
        ),
        iou_threshold=options.iou_threshold,
        class_agnostic_nms=options.class_agnostic_nms,
        layout=int(options.layout),
        num_classes=options.num_classes,
        result_callback=c_callback,
    )

    detector_handle = ctypes.c_void_p()
    lib.MpOrientedObjectDetectorCreate(
        ctypes.byref(ctypes_options), ctypes.byref(detector_handle)
    )
    return OrientedObjectDetector(
        lib=lib,
        handle=detector_handle,
        dispatcher=dispatcher,
        async_callback=c_callback,
        label_map=label_map,
    )

  def detect(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> OrientedObjectDetectorResult:
    """Performs oriented object detection on the provided MediaPipe Image.

    Only use this method when the OrientedObjectDetector is created with the
    image running mode.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      An oriented detection result object that contains a list of oriented
      detections, each detection has a rotated bounding box expressed in the
      unrotated input frame of reference coordinates system, in pixel units.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If oriented object detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = oriented_detections_c_module.MpOrientedDetectionResultC()
    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpOrientedObjectDetectorDetectImage(
        self._handle,
        c_image,
        c_image_processing_options,
        ctypes.byref(c_result),
    )
    py_result = _enrich_with_label_map(
        OrientedObjectDetectorResult.from_ctypes(c_result), self._label_map
    )
    self._lib.MpOrientedObjectDetectorCloseResult(ctypes.byref(c_result))
    return py_result

  def detect_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> OrientedObjectDetectorResult:
    """Performs oriented object detection on the provided video frames.

    Only use this method when the OrientedObjectDetector is created with the
    video running mode. It's required to provide the video frame's timestamp
    (in milliseconds) along with the video frame. The input timestamps should
    be monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      An oriented detection result object that contains a list of oriented
      detections, each detection has a rotated bounding box expressed in the
      unrotated input frame of reference coordinates system, in pixel units.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If oriented object detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = oriented_detections_c_module.MpOrientedDetectionResultC()
    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpOrientedObjectDetectorDetectForVideo(
        self._handle,
        c_image,
        c_image_processing_options,
        timestamp_ms,
        ctypes.byref(c_result),
    )
    py_result = _enrich_with_label_map(
        OrientedObjectDetectorResult.from_ctypes(c_result), self._label_map
    )
    self._lib.MpOrientedObjectDetectorCloseResult(ctypes.byref(c_result))
    return py_result

  def detect_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data to perform oriented object detection.

    Only use this method when the OrientedObjectDetector is created with the
    live stream running mode. The input timestamps should be monotonically
    increasing for adjacent calls of this method. This method will return
    immediately after the input image is accepted. The results will be
    available via the `result_callback` provided in the
    `OrientedObjectDetectorOptions`.

    The `result_callback` provides:
      - An oriented detection result object that contains a list of oriented
        detections, each detection has a rotated bounding box expressed in the
        unrotated input frame of reference coordinates system.
      - The input image that the oriented object detector runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the
        oriented object detector has already processed.
      RuntimeError: If oriented object detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpOrientedObjectDetectorDetectAsync(
        self._handle,
        c_image,
        c_image_processing_options,
        timestamp_ms,
    )

  def close(self):
    """Shuts down the MediaPipe oriented object detector task instance."""
    if self._handle:
      self._lib.MpOrientedObjectDetectorClose(self._handle)
      self._handle = None
      self._dispatcher.close()
      self._lib.close()

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Shuts down the oriented object detector task instance on context exit.

    Args:
      exc_type: The exception type that caused the exit.
      exc_value: The exception value that caused the exit.
      traceback: The exception traceback that caused the exit.

    Raises:
      RuntimeError: If the MediaPipe OrientedObjectDetector task failed to
        close.
    """
    del exc_type, exc_value, traceback  # Unused.
    self.close()

  def __del__(self):
    self.close()
