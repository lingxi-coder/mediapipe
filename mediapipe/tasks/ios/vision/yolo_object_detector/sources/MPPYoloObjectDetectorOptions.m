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

#import "mediapipe/tasks/ios/vision/yolo_object_detector/sources/MPPYoloObjectDetectorOptions.h"

@implementation MPPYoloObjectDetectorTileRect

- (id)copyWithZone:(NSZone *)zone {
  MPPYoloObjectDetectorTileRect *tileRect = [[[self class] allocWithZone:zone] init];
  tileRect.xCenter = self.xCenter;
  tileRect.yCenter = self.yCenter;
  tileRect.width = self.width;
  tileRect.height = self.height;
  return tileRect;
}

@end

@implementation MPPYoloObjectDetectorTilingOptions

- (instancetype)init {
  self = [super init];
  if (self) {
    _tileRows = 1;
    _tileCols = 1;
    _tileOverlapFraction = 0.0f;
    _explicitTiles = @[];
    _tileLocalNMSIOUThreshold = 0.0f;
    _maxDetectionsAfterTileNMS = 0;
    _enableMotionScheduling = NO;
    _maxScheduledTiles = 0;
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  MPPYoloObjectDetectorTilingOptions *tilingOptions = [[[self class] allocWithZone:zone] init];
  tilingOptions.tileRows = self.tileRows;
  tilingOptions.tileCols = self.tileCols;
  tilingOptions.tileOverlapFraction = self.tileOverlapFraction;
  tilingOptions.explicitTiles = [[NSArray alloc] initWithArray:self.explicitTiles copyItems:YES];
  tilingOptions.tileLocalNMSIOUThreshold = self.tileLocalNMSIOUThreshold;
  tilingOptions.maxDetectionsAfterTileNMS = self.maxDetectionsAfterTileNMS;
  tilingOptions.enableMotionScheduling = self.enableMotionScheduling;
  tilingOptions.maxScheduledTiles = self.maxScheduledTiles;
  return tilingOptions;
}

@end

@implementation MPPYoloObjectDetectorTrackingOptions

- (instancetype)init {
  self = [super init];
  if (self) {
    _trackerType = MPPYoloObjectDetectorTrackerTypeBoxTracker;
    _trackHighThreshold = 0.6f;
    _trackLowThreshold = 0.1f;
    _newTrackThreshold = 0.7f;
    _trackBuffer = 30;
    _matchThreshold = 0.7f;
    _enableGMC = NO;
    _nominalFrameRate = 30;
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  MPPYoloObjectDetectorTrackingOptions *trackingOptions =
      [[[self class] allocWithZone:zone] init];
  trackingOptions.trackerType = self.trackerType;
  trackingOptions.trackHighThreshold = self.trackHighThreshold;
  trackingOptions.trackLowThreshold = self.trackLowThreshold;
  trackingOptions.newTrackThreshold = self.newTrackThreshold;
  trackingOptions.trackBuffer = self.trackBuffer;
  trackingOptions.matchThreshold = self.matchThreshold;
  trackingOptions.enableGMC = self.enableGMC;
  trackingOptions.nominalFrameRate = self.nominalFrameRate;
  return trackingOptions;
}

@end

@implementation MPPYoloObjectDetectorOptions

- (instancetype)init {
  self = [super init];
  if (self) {
    _displayNamesLocale = @"en";
    _maxResults = -1;
    _scoreThreshold = 0.0f;
    _categoryAllowlist = @[];
    _categoryDenylist = @[];
    _iouThreshold = 0.45f;
    _layout = MPPYoloObjectDetectorLayoutChannelsFirst;
    _numClasses = 0;
    _tilingOptions = [[MPPYoloObjectDetectorTilingOptions alloc] init];
    _trackingOptions = [[MPPYoloObjectDetectorTrackingOptions alloc] init];
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  MPPYoloObjectDetectorOptions *options = [super copyWithZone:zone];
  options.runningMode = self.runningMode;
  options.yoloObjectDetectorLiveStreamDelegate = self.yoloObjectDetectorLiveStreamDelegate;
  options.displayNamesLocale = self.displayNamesLocale;
  options.maxResults = self.maxResults;
  options.scoreThreshold = self.scoreThreshold;
  options.categoryAllowlist = self.categoryAllowlist;
  options.categoryDenylist = self.categoryDenylist;
  options.iouThreshold = self.iouThreshold;
  options.layout = self.layout;
  options.numClasses = self.numClasses;
  options.tilingOptions = [self.tilingOptions copy];
  options.trackingOptions = [self.trackingOptions copy];
  return options;
}

@end
