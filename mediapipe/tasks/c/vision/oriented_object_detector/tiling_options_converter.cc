/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/c/vision/oriented_object_detector/tiling_options_converter.h"

#include <cstdint>

#include "mediapipe/tasks/c/vision/oriented_object_detector/oriented_object_detector.h"
#include "mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h"

namespace mediapipe::tasks::c::vision::oriented_object_detector {

namespace ObbNs = ::mediapipe::tasks::vision::oriented_object_detector;

void CppConvertToTilingOptions(
    const MpOrientedTilingOptions& in,
    ObbNs::OrientedObjectDetectorOptions::TilingOptions* out) {
  out->tile_rows = in.tile_rows;
  out->tile_cols = in.tile_cols;
  out->tile_overlap_fraction = in.tile_overlap_fraction;

  // explicit_tiles requires a 1x1 grid: TileGridCalculator rejects rows/cols != 1
  // when explicit tiles are present, and the graph forwards rows/cols verbatim. A
  // zero-initialized C options struct leaves rows/cols at 0; promote them to 1 when
  // the caller supplies explicit tiles without a grid, matching the C++ struct's
  // (1,1) defaults that the graph assumes. Grid-only callers are unaffected.
  if (in.explicit_tiles_count > 0) {
    if (out->tile_rows == 0) out->tile_rows = 1;
    if (out->tile_cols == 0) out->tile_cols = 1;
  }

  out->explicit_tiles.clear();
  out->explicit_tiles.reserve(in.explicit_tiles_count);
  for (uint32_t i = 0; i < in.explicit_tiles_count; ++i) {
    out->explicit_tiles.push_back({in.explicit_tiles[i].x_center,
                                   in.explicit_tiles[i].y_center,
                                   in.explicit_tiles[i].width,
                                   in.explicit_tiles[i].height});
  }

  out->tile_local_nms_iou_threshold = in.tile_local_nms_iou_threshold;
  out->max_detections_after_tile_nms = in.max_detections_after_tile_nms;
}

}  // namespace mediapipe::tasks::c::vision::oriented_object_detector
