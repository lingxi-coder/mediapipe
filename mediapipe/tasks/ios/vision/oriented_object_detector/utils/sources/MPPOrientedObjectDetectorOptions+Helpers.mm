// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "mediapipe/tasks/ios/vision/oriented_object_detector/utils/sources/MPPOrientedObjectDetectorOptions+Helpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/cc/vision/oriented_object_detector/proto/oriented_object_detector_options.pb.h"

namespace {
using CalculatorOptionsProto = ::mediapipe::CalculatorOptions;
using OrientedObjectDetectorOptionsProto =
    ::mediapipe::tasks::vision::oriented_object_detector::proto::OrientedObjectDetectorOptions;
}  // namespace

@implementation MPPOrientedObjectDetectorOptions (Helpers)

- (void)copyToProto:(CalculatorOptionsProto *)optionsProto {
  OrientedObjectDetectorOptionsProto *graphOptions =
      optionsProto->MutableExtension(OrientedObjectDetectorOptionsProto::ext);

  graphOptions->Clear();

  [self.baseOptions copyToProto:graphOptions->mutable_base_options()];
  graphOptions->mutable_base_options()->set_use_stream_mode(self.runningMode != MPPRunningModeImage);
  graphOptions->set_max_results((int)self.maxResults);
  graphOptions->set_score_threshold(self.scoreThreshold);
  graphOptions->set_iou_threshold(self.iouThreshold);
  graphOptions->set_class_agnostic_nms(self.classAgnosticNMS);
  graphOptions->set_layout(
      static_cast<OrientedObjectDetectorOptionsProto::Layout>(self.layout));
  graphOptions->set_num_classes((int)self.numClasses);
  graphOptions->set_display_names_locale(self.displayNamesLocale.cppString);

  for (NSString *category in self.categoryAllowlist) {
    graphOptions->add_category_allowlist(category.cppString);
  }
  for (NSString *category in self.categoryDenylist) {
    graphOptions->add_category_denylist(category.cppString);
  }

  auto *tiling = graphOptions->mutable_tiling();
  tiling->set_tile_rows((int)self.tilingOptions.tileRows);
  tiling->set_tile_cols((int)self.tilingOptions.tileCols);
  tiling->set_tile_overlap_fraction(self.tilingOptions.tileOverlapFraction);
  for (MPPOrientedObjectDetectorTileRect *tileRect in self.tilingOptions.explicitTiles) {
    auto *explicitTile = tiling->add_explicit_tiles();
    explicitTile->set_x_center(tileRect.xCenter);
    explicitTile->set_y_center(tileRect.yCenter);
    explicitTile->set_width(tileRect.width);
    explicitTile->set_height(tileRect.height);
  }
  tiling->set_tile_local_nms_iou_threshold(self.tilingOptions.tileLocalNMSIOUThreshold);
  tiling->set_max_detections_after_tile_nms(
      (int)self.tilingOptions.maxDetectionsAfterTileNMS);

  auto *tracking = graphOptions->mutable_tracking();
  tracking->set_tracker_type(
      static_cast<OrientedObjectDetectorOptionsProto::TrackingOptions::TrackerType>(
          self.trackingOptions.trackerType));
  tracking->set_track_high_threshold(self.trackingOptions.trackHighThreshold);
  tracking->set_track_low_threshold(self.trackingOptions.trackLowThreshold);
  tracking->set_new_track_threshold(self.trackingOptions.newTrackThreshold);
  tracking->set_track_buffer((int)self.trackingOptions.trackBuffer);
  tracking->set_match_threshold(self.trackingOptions.matchThreshold);
  tracking->set_enable_gmc(self.trackingOptions.enableGMC);
  tracking->set_nominal_frame_rate((int)self.trackingOptions.nominalFrameRate);
}

@end
