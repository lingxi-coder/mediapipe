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
"""MediaPipe YOLO object detector task."""

import ctypes
import dataclasses
import enum
from typing import Callable, List, Optional

from mediapipe.tasks.python.components.containers import detections as detections_module
from mediapipe.tasks.python.components.containers import detections_c as detections_c_module
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

YoloObjectDetectorResult = detections_module.DetectionResult
_BaseOptions = base_options_module.BaseOptions
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_AsyncResultDispatcher = async_result_dispatcher.AsyncResultDispatcher


class Layout(enum.IntEnum):
  """YOLO detect-head output tensor layout (numerically matches the C++ enum)."""

  CHANNELS_FIRST = 1
  CHANNELS_LAST = 2


class TrackerType(enum.IntEnum):
  """Tracker selection for the tiled VIDEO/LIVE_STREAM path.

  Values are numerically equal to the C++/proto TrackerType enum.
  """

  BOX_TRACKER = 1  # optical-flow propagation (default)
  BOTSORT = 2  # tracking-by-detection, motion-only


_C_TYPES_RESULT_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32,  # MpStatus
    ctypes.POINTER(detections_c_module.MpDetectionResultC),
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
    result: YoloObjectDetectorResult, label_map: Optional[List[str]]
) -> YoloObjectDetectorResult:
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


class MpTileRectC(ctypes.Structure):
  """Byte-matches struct MpTileRect in the YOLO C header.

  Field order/types MUST stay in sync with
  mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h (pinned
  by tiling_options_abi_test.cc).
  """

  _fields_ = [
      ('x_center', ctypes.c_float),
      ('y_center', ctypes.c_float),
      ('width', ctypes.c_float),
      ('height', ctypes.c_float),
  ]


class MpTilingOptionsC(ctypes.Structure):
  """Byte-matches struct MpTilingOptions in the YOLO C header.

  Field order/types MUST stay in sync with the C header (pinned by
  tiling_options_abi_test.cc). enable_motion_scheduling is c_bool (1 byte) to
  match the C `bool`. (A c_int here would NOT change any offset -- the field is
  followed by alignment padding that absorbs the extra 3 bytes -- so the
  layout/offset tests cannot catch such a swap; it must be reviewed by hand. The
  c_bool width is pinned by tiling_options_abi_test.cc and
  test_ctypes_tiling_layout_matches_c_abi.)
  """

  _fields_ = [
      ('tile_rows', ctypes.c_int),
      ('tile_cols', ctypes.c_int),
      ('tile_overlap_fraction', ctypes.c_float),
      ('explicit_tiles', ctypes.POINTER(MpTileRectC)),
      ('explicit_tiles_count', ctypes.c_uint32),
      ('tile_local_nms_iou_threshold', ctypes.c_float),
      ('max_detections_after_tile_nms', ctypes.c_int),
      ('enable_motion_scheduling', ctypes.c_bool),
      ('max_scheduled_tiles', ctypes.c_int),
  ]


class MpTrackingOptionsC(ctypes.Structure):
  """Byte-matches struct MpTrackingOptions in the YOLO C header.

  Layout pinned by tracking_options_abi_test.cc (sizeof 32; offsets
  0/4/8/12/16/20/24/28; enable_gmc is c_bool = 1 byte). tracker_type is c_int with
  values 0=unspecified->BOX_TRACKER, 1=BOX_TRACKER, 2=BOTSORT.
  """

  _fields_ = [
      ('tracker_type', ctypes.c_int),
      ('track_high_threshold', ctypes.c_float),
      ('track_low_threshold', ctypes.c_float),
      ('new_track_threshold', ctypes.c_float),
      ('track_buffer', ctypes.c_int),
      ('match_threshold', ctypes.c_float),
      ('enable_gmc', ctypes.c_bool),
      ('nominal_frame_rate', ctypes.c_int),
  ]


class MpYoloObjectDetectorOptionsC(ctypes.Structure):
  """YOLO detector options for the C API.

  Field order MUST byte-match struct MpYoloObjectDetectorOptions in
  mediapipe/tasks/c/vision/yolo_object_detector/yolo_object_detector.h.
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
      ('layout', ctypes.c_int),
      ('num_classes', ctypes.c_int),
      ('tiling', MpTilingOptionsC),
      ('tracking', MpTrackingOptionsC),
      ('result_callback', _C_TYPES_RESULT_CALLBACK),
  ]


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpYoloObjectDetectorCreateV2',
        (
            ctypes.POINTER(MpYoloObjectDetectorOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpYoloObjectDetectorDetectImage',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.MpImageProcessingOptionsC
            ),
            ctypes.POINTER(detections_c_module.MpDetectionResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpYoloObjectDetectorDetectForVideo',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.MpImageProcessingOptionsC
            ),
            ctypes.c_int64,
            ctypes.POINTER(detections_c_module.MpDetectionResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpYoloObjectDetectorDetectAsync',
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
        'MpYoloObjectDetectorCloseResult',
        [ctypes.POINTER(detections_c_module.MpDetectionResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpYoloObjectDetectorClose',
        (ctypes.c_void_p,),
    ),
)


@dataclasses.dataclass
class TileRect:
  """A frame-normalized tile given by its center point and size.

  Attributes:
    x_center: Tile center x, normalized to [0, 1].
    y_center: Tile center y, normalized to [0, 1].
    width: Tile width, normalized to [0, 1].
    height: Tile height, normalized to [0, 1].
  """

  x_center: float = 0.0
  y_center: float = 0.0
  width: float = 0.0
  height: float = 0.0


@dataclasses.dataclass
class TilingOptions:
  """Static tiling configuration for the YOLO object detector.

  Mirrors the C++ YoloObjectDetectorOptions.TilingOptions. Tiling is enabled
  when tile_rows * tile_cols > 1 or explicit_tiles is non-empty. The defaults
  (1x1, no explicit tiles) mean tiling disabled. To tile with a grid set BOTH
  tile_rows and tile_cols (each >= 1); a zero in either disables tiling.

  Attributes:
    tile_rows: Number of grid rows. Mutually exclusive with explicit_tiles.
    tile_cols: Number of grid columns. Mutually exclusive with explicit_tiles.
    tile_overlap_fraction: Fractional overlap added around each grid tile.
    explicit_tiles: Explicit (non-grid) tiles. Mutually exclusive with the grid
      params.
    tile_local_nms_iou_threshold: Per-tile (in-decoder) NMS IoU threshold;
      <= 0 disables.
    max_detections_after_tile_nms: Per-tile cap after tile-local NMS;
      <= 0 disables.
    enable_motion_scheduling: VIDEO/LIVE_STREAM only; gate per-frame tiled
      inference with a motion scheduler. Rejected in IMAGE mode by the
      underlying task.
    max_scheduled_tiles: Per DETECT-frame cap on inferred tiles
      (motion-prioritized). 0 = all.
  """

  tile_rows: int = 1
  tile_cols: int = 1
  tile_overlap_fraction: float = 0.0
  explicit_tiles: Optional[List[TileRect]] = None
  tile_local_nms_iou_threshold: float = 0.0
  max_detections_after_tile_nms: int = 0
  enable_motion_scheduling: bool = False
  max_scheduled_tiles: int = 0


@dataclasses.dataclass
class TrackingOptions:
  """Tracker selection for the YOLO object detector (tiled stream path).

  Mirrors the C++ YoloObjectDetectorOptions.TrackingOptions. Honored only when
  tiling is enabled and running mode is not IMAGE; otherwise ignored. BOTSORT is
  motion-only (no ReID). The knobs are used only for BOTSORT.

  NOTE: BOTSORT only emits a track_id for confirmed tracks. Set
  track_high_threshold / new_track_threshold at or below your detection
  score_threshold, otherwise low-confidence detections never confirm and no
  track_id is ever produced.
  """

  tracker_type: TrackerType = TrackerType.BOX_TRACKER
  track_high_threshold: float = 0.6
  track_low_threshold: float = 0.1
  new_track_threshold: float = 0.7
  track_buffer: int = 30
  match_threshold: float = 0.7
  enable_gmc: bool = False
  nominal_frame_rate: int = 30


@dataclasses.dataclass
class YoloObjectDetectorOptions:
  """Options for the YOLO object detector task.

  Attributes:
    base_options: Base options for the YOLO object detector task.
    running_mode: The running mode of the task. Default to the image mode.
      YOLO object detector task has three running modes: 1) The image mode for
      detecting objects on single image inputs. 2) The video mode for detecting
      objects on the decoded frames of a video. 3) The live stream mode for
      detecting objects on a live stream of input data, such as from camera.
    display_names_locale: The locale to use for display names specified through
      the TFLite Model Metadata.
    max_results: The maximum number of top-scored detection results to return.
    score_threshold: Overrides the ones provided in the model metadata. Results
      below this value are rejected.
    category_allowlist: Allowlist of category names. If non-empty, detection
      results whose category name is not in this set will be filtered out.
      Duplicate or unknown category names are ignored. Mutually exclusive with
      `category_denylist`.
    category_denylist: Denylist of category names. If non-empty, detection
      results whose category name is in this set will be filtered out. Duplicate
      or unknown category names are ignored. Mutually exclusive with
      `category_allowlist`.
    iou_threshold: IoU threshold for non-maximum suppression. Default 0.45.
    layout: The output tensor layout of the YOLO detect head. Default
      CHANNELS_FIRST.
    num_classes: Number of classes. Must be greater than 0. Metadata derivation
      is not yet implemented.
    tiling: Static tiling configuration. Defaults to disabled (1x1).
    tracking: Tracker selection for the tiled VIDEO/LIVE_STREAM path. Honored
      only when tiling is enabled and running mode is not IMAGE; otherwise
      ignored. Defaults to BOX_TRACKER.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.

  Category names are populated by the YOLO graph from the model metadata's
  label file, and category_allowlist / category_denylist filter results by
  class name. The Python `_load_label_map` fallback below is a display-only
  safety net for models whose graph did not populate names; it never overrides
  a name the graph already provided and does not implement filtering.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  display_names_locale: Optional[str] = None
  max_results: Optional[int] = -1
  score_threshold: Optional[float] = 0.0
  category_allowlist: Optional[List[str]] = None
  category_denylist: Optional[List[str]] = None
  iou_threshold: float = 0.45
  layout: Layout = Layout.CHANNELS_FIRST
  num_classes: int = 0
  tiling: TilingOptions = dataclasses.field(default_factory=TilingOptions)
  tracking: TrackingOptions = dataclasses.field(default_factory=TrackingOptions)
  result_callback: Optional[
      Callable[
          [detections_module.DetectionResult, image_module.Image, int], None
      ]
  ] = None


def _build_tiling_options_c(
    tiling: TilingOptions,
) -> tuple['MpTilingOptionsC', object]:
  """Builds the ctypes MpTilingOptionsC from a TilingOptions dataclass.

  Returns the populated ctypes struct AND the backing explicit_tiles array. The
  caller MUST keep the returned array referenced until the C call that consumes
  the parent options struct returns: explicit_tiles is a raw pointer into that
  array. (ctypes also records the array in the parent struct's _objects when the
  nested struct is copied in by value, but returning it makes the lifetime
  explicit instead of relying on that internal behavior.) The C converter copies
  the tiles into a std::vector synchronously during Create, so outliving the
  Create call is sufficient.
  """
  max_grid_size = 2**31 - 1
  for name, value in (
      ('tile_rows', tiling.tile_rows), ('tile_cols', tiling.tile_cols)
  ):
    if isinstance(value, bool) or not isinstance(value, int):
      raise TypeError(f'tiling.{name} must be an integer.')
    if value < 0 or value > max_grid_size:
      raise ValueError(f'tiling.{name} must be in [0, {max_grid_size}].')
  if tiling.tile_rows * tiling.tile_cols > max_grid_size:
    raise ValueError(f'tiling grid tile count must be <= {max_grid_size}.')

  explicit_tiles = tiling.explicit_tiles or []
  tiles_array = (MpTileRectC * len(explicit_tiles))(
      *[
          MpTileRectC(t.x_center, t.y_center, t.width, t.height)
          for t in explicit_tiles
      ]
  )
  tiling_c = MpTilingOptionsC(
      tile_rows=tiling.tile_rows,
      tile_cols=tiling.tile_cols,
      tile_overlap_fraction=tiling.tile_overlap_fraction,
      explicit_tiles=(
          ctypes.cast(tiles_array, ctypes.POINTER(MpTileRectC))
          if explicit_tiles
          else None
      ),
      explicit_tiles_count=len(explicit_tiles),
      tile_local_nms_iou_threshold=tiling.tile_local_nms_iou_threshold,
      max_detections_after_tile_nms=tiling.max_detections_after_tile_nms,
      enable_motion_scheduling=tiling.enable_motion_scheduling,
      max_scheduled_tiles=tiling.max_scheduled_tiles,
  )
  return tiling_c, tiles_array


def _build_tracking_options_c(
    tracking: TrackingOptions,
) -> 'MpTrackingOptionsC':
  """Builds the ctypes MpTrackingOptionsC from a TrackingOptions dataclass.

  No pointer fields, so (unlike tiling) there is no backing array to keep alive.
  """
  return MpTrackingOptionsC(
      tracker_type=int(tracking.tracker_type),
      track_high_threshold=tracking.track_high_threshold,
      track_low_threshold=tracking.track_low_threshold,
      new_track_threshold=tracking.new_track_threshold,
      track_buffer=tracking.track_buffer,
      match_threshold=tracking.match_threshold,
      enable_gmc=tracking.enable_gmc,
      nominal_frame_rate=tracking.nominal_frame_rate,
  )


class YoloObjectDetector:
  """Performs YOLO (axis-aligned) object detection on images.

  The API expects a YOLO TFLite model with the appropriate metadata.

  Example usage:
    detector = YoloObjectDetector.create_from_model_path(
        '/path/to/model.tflite', num_classes=80
    )
    result = detector.detect(image)
    detector.close()

  Or as a context manager:
    with YoloObjectDetector.create_from_model_path(
        '/path/to/model.tflite', num_classes=80
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
    """Initializes the YOLO object detector.

    Args:
      lib: The dispatch library to use for the YOLO object detector.
      handle: The C pointer to the YOLO object detector.
      dispatcher: The async result handler for the YOLO object detector.
      async_callback: The c callback for the YOLO object detector.
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
      cls, model_path: str, *, num_classes: int
  ) -> 'YoloObjectDetector':
    """Creates a `YoloObjectDetector` object from a TFLite model path.

    Note that the created `YoloObjectDetector` instance is in image mode, for
    detecting objects on single image inputs.

    Args:
      model_path: Path to the model.
      num_classes: Number of classes in the model's detection head. Must be a
        positive integer; it is not inferred from model metadata.

    Returns:
      `YoloObjectDetector` object created from the model file and class count,
      using the remaining default `YoloObjectDetectorOptions`.

    Raises:
      TypeError: If `num_classes` is not an integer.
      ValueError: If `num_classes` is not positive, or if failed to create
        `YoloObjectDetector` from the provided file, such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
      raise TypeError('num_classes must be a positive integer.')
    if num_classes <= 0:
      raise ValueError('num_classes must be greater than 0.')
    options = YoloObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RunningMode.IMAGE,
        num_classes=num_classes,
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: YoloObjectDetectorOptions
  ) -> 'YoloObjectDetector':
    """Creates the `YoloObjectDetector` object from YOLO object detector options.

    Args:
      options: Options for the YOLO object detector task.

    Returns:
      `YoloObjectDetector` object created from `options`.

    Raises:
      ValueError: If failed to create `YoloObjectDetector` object from
        `YoloObjectDetectorOptions` such as missing the model.
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
          YoloObjectDetectorResult.from_ctypes(c_result), label_map
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
    # tiles_keepalive holds the explicit_tiles backing array; it must stay
    # referenced through the MpYoloObjectDetectorCreateV2 call below (the C
    # converter copies the tiles into a std::vector synchronously during Create).
    tiling_c, tiles_keepalive = _build_tiling_options_c(options.tiling)
    tracking_c = _build_tracking_options_c(options.tracking)
    ctypes_options = MpYoloObjectDetectorOptionsC(
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
        layout=int(options.layout),
        num_classes=options.num_classes,
        tiling=tiling_c,
        tracking=tracking_c,
        result_callback=c_callback,
    )

    detector_handle = ctypes.c_void_p()
    lib.MpYoloObjectDetectorCreateV2(
        ctypes.byref(ctypes_options), ctypes.byref(detector_handle)
    )
    return YoloObjectDetector(
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
  ) -> YoloObjectDetectorResult:
    """Performs YOLO object detection on the provided MediaPipe Image.

    Only use this method when the YoloObjectDetector is created with the image
    running mode.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      A detection result object that contains a list of detections, each
      detection has a bounding box expressed in the unrotated input frame of
      reference coordinates system, i.e. in `[0,image_width) x [0,
      image_height)`, which are the dimensions of the underlying image data.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If YOLO object detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = detections_c_module.MpDetectionResultC()
    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpYoloObjectDetectorDetectImage(
        self._handle,
        c_image,
        c_image_processing_options,
        ctypes.byref(c_result),
    )
    py_result = _enrich_with_label_map(
        YoloObjectDetectorResult.from_ctypes(c_result), self._label_map
    )
    self._lib.MpYoloObjectDetectorCloseResult(ctypes.byref(c_result))
    return py_result

  def detect_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> YoloObjectDetectorResult:
    """Performs YOLO object detection on the provided video frames.

    Only use this method when the YoloObjectDetector is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      A detection result object that contains a list of detections, each
      detection has a bounding box expressed in the unrotated input frame of
      reference coordinates system, i.e. in `[0,image_width) x [0,
      image_height)`, which are the dimensions of the underlying image data.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If YOLO object detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = detections_c_module.MpDetectionResultC()
    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpYoloObjectDetectorDetectForVideo(
        self._handle,
        c_image,
        c_image_processing_options,
        timestamp_ms,
        ctypes.byref(c_result),
    )
    py_result = _enrich_with_label_map(
        YoloObjectDetectorResult.from_ctypes(c_result), self._label_map
    )
    self._lib.MpYoloObjectDetectorCloseResult(ctypes.byref(c_result))
    return py_result

  def detect_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data to perform YOLO object detection.

    Only use this method when the YoloObjectDetector is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `YoloObjectDetectorOptions`.

    The `result_callback` provides:
      - A detection result object that contains a list of detections, each
        detection has a bounding box expressed in the unrotated input frame of
        reference coordinates system.
      - The input image that the YOLO object detector runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the YOLO
        object detector has already processed.
      RuntimeError: If YOLO object detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpYoloObjectDetectorDetectAsync(
        self._handle,
        c_image,
        c_image_processing_options,
        timestamp_ms,
    )

  def close(self):
    """Frees the task and dispatchers, including when native shutdown fails."""
    if self._handle:
      try:
        self._lib.MpYoloObjectDetectorClose(self._handle)
      finally:
        # The C API consumes the handle even when it reports a graph error.
        self._handle = None
        try:
          self._dispatcher.close()
        finally:
          self._lib.close()

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Shuts down the MediaPipe YOLO task instance on exit of the context manager.

    Args:
      exc_type: The exception type that caused the exit.
      exc_value: The exception value that caused the exit.
      traceback: The exception traceback that caused the exit.

    Raises:
      RuntimeError: If the MediaPipe YoloObjectDetector task failed to close.
    """
    del exc_type, exc_value, traceback  # Unused.
    self.close()

  def __del__(self):
    self.close()
