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

#import "mediapipe/tasks/ios/vision/oriented_object_detector/sources/MPPOrientedObjectDetector.h"

#import <cmath>
#import <cstdint>
#import <limits>

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionPacketCreator.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/oriented_object_detector/utils/sources/MPPOrientedObjectDetectorOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/oriented_object_detector/utils/sources/MPPOrientedObjectDetectorResult+Helpers.h"

namespace {
using ::mediapipe::Timestamp;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
}  // namespace

static NSString *const kOrientedDetectionsStreamName = @"oriented_detections_out";
static NSString *const kOrientedDetectionsTag = @"ORIENTED_DETECTIONS";
static NSString *const kImageInStreamName = @"image_in";
static NSString *const kImageOutStreamName = @"image_out";
static NSString *const kImageTag = @"IMAGE";
static NSString *const kNormRectStreamName = @"norm_rect_in";
static NSString *const kNormRectTag = @"NORM_RECT";
static NSString *const kTaskGraphName =
    @"mediapipe.tasks.vision.oriented_object_detector.OrientedObjectDetectorGraph";
static NSString *const kTaskName = @"orientedObjectDetector";

#define OrientedObjectDetectorResultWithOutputPacketMap(outputPacketMap)             \
  ([MPPOrientedObjectDetectorResult orientedObjectDetectorResultWithDetectionsPacket: \
                                      outputPacketMap[kOrientedDetectionsStreamName.cppString] \
                                                                             imagePacket:       \
                                      outputPacketMap[kImageOutStreamName.cppString]])

namespace {

BOOL CreateOrientedInvalidArgumentError(NSString *description, NSError **error) {
  [MPPCommonUtils createCustomError:error
                           withCode:MPPTasksErrorCodeInvalidArgumentError
                        description:[@"INVALID_ARGUMENT: " stringByAppendingString:description]];
  return NO;
}

BOOL ValidateUnitInterval(float value, NSString *fieldName, NSError **error) {
  if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
    return CreateOrientedInvalidArgumentError(
        [NSString stringWithFormat:@"%@ must be finite and in [0, 1].", fieldName], error);
  }
  return YES;
}

BOOL OrientedTilingEnabled(MPPOrientedObjectDetectorOptions *options) {
  return (options.tilingOptions.tileRows > 0 && options.tilingOptions.tileCols > 0 &&
          (options.tilingOptions.tileRows > 1 || options.tilingOptions.tileCols > 1)) ||
         options.tilingOptions.explicitTiles.count > 0;
}

BOOL ValidateOrientedOptions(MPPOrientedObjectDetectorOptions *options, NSError **error) {
  if (options.maxResults == 0) {
    return CreateOrientedInvalidArgumentError(@"Invalid `max_results` option: value must be != 0",
                                              error);
  }

  if (options.categoryAllowlist.count > 0 && options.categoryDenylist.count > 0) {
    return CreateOrientedInvalidArgumentError(
        @"`category_allowlist` and `category_denylist` are mutually exclusive options.", error);
  }

  if (options.numClasses <= 0) {
    return CreateOrientedInvalidArgumentError(
        @"num_classes must be set in OrientedObjectDetectorOptions", error);
  }

  if (options.tilingOptions.explicitTiles.count > 0 &&
      options.tilingOptions.tileOverlapFraction != 0.0f) {
    return CreateOrientedInvalidArgumentError(
        @"tiling.tile_overlap_fraction is ignored with tiling.explicit_tiles; do not set both",
        error);
  }

  if (options.tilingOptions.tileRows < 0 || options.tilingOptions.tileCols < 0) {
    return CreateOrientedInvalidArgumentError(
        @"tiling.tile_rows and tiling.tile_cols must be >= 0.", error);
  }

  // Check each NSInteger before narrowing to the int32 proto fields.
  if (options.tilingOptions.tileRows > std::numeric_limits<int32_t>::max() ||
      options.tilingOptions.tileCols > std::numeric_limits<int32_t>::max()) {
    return CreateOrientedInvalidArgumentError(
        @"tiling.tile_rows and tiling.tile_cols must be <= 2147483647.", error);
  }
  if (static_cast<int64_t>(options.tilingOptions.tileRows) * options.tilingOptions.tileCols >
      std::numeric_limits<int32_t>::max()) {
    return CreateOrientedInvalidArgumentError(
        @"tiling.tile_rows * tiling.tile_cols must be <= 2147483647.", error);
  }

  if (options.tilingOptions.explicitTiles.count > 0 &&
      (options.tilingOptions.tileRows > 1 || options.tilingOptions.tileCols > 1)) {
    return CreateOrientedInvalidArgumentError(
        @"tiling.explicit_tiles is mutually exclusive with a tiling.tile_rows / tiling.tile_cols "
        @"grid (> 1).",
        error);
  }

  if (options.tilingOptions.tileOverlapFraction < 0.0f ||
      options.tilingOptions.tileOverlapFraction >= 1.0f) {
    return CreateOrientedInvalidArgumentError(
        @"tiling.tile_overlap_fraction must be in [0.0, 1.0).", error);
  }

  if (options.trackingOptions.trackerType ==
      MPPOrientedObjectDetectorTrackerTypeBoxTracker) {
    return CreateOrientedInvalidArgumentError(
        @"tracking.tracker_type=BOX_TRACKER: BoxTracker is not supported for oriented detection; "
        @"use BOTSORT.",
        error);
  }

  if (options.trackingOptions.trackerType !=
      MPPOrientedObjectDetectorTrackerTypeBotsort) {
    return YES;
  }

  if (!ValidateUnitInterval(options.trackingOptions.trackHighThreshold,
                            @"tracking.track_high_threshold", error) ||
      !ValidateUnitInterval(options.trackingOptions.trackLowThreshold,
                            @"tracking.track_low_threshold", error) ||
      !ValidateUnitInterval(options.trackingOptions.newTrackThreshold,
                            @"tracking.new_track_threshold", error) ||
      !ValidateUnitInterval(options.trackingOptions.matchThreshold,
                            @"tracking.match_threshold", error)) {
    return NO;
  }

  if (options.trackingOptions.trackLowThreshold > options.trackingOptions.trackHighThreshold) {
    return CreateOrientedInvalidArgumentError(
        @"tracking.track_low_threshold must be <= tracking.track_high_threshold.", error);
  }

  if (options.trackingOptions.trackBuffer < 0 || options.trackingOptions.trackBuffer > 255) {
    return CreateOrientedInvalidArgumentError(@"tracking.track_buffer must be in [0, 255].",
                                              error);
  }

  if (options.trackingOptions.nominalFrameRate < 1 ||
      options.trackingOptions.nominalFrameRate > 255) {
    return CreateOrientedInvalidArgumentError(
        @"tracking.nominal_frame_rate must be in [1, 255].", error);
  }

  if (std::floor(static_cast<double>(options.trackingOptions.nominalFrameRate) / 30.0 *
                 options.trackingOptions.trackBuffer) > 255.0) {
    return CreateOrientedInvalidArgumentError(
        @"effective lost-track window must be <= 255 frames.", error);
  }

  if (options.runningMode == MPPRunningModeImage) {
    return CreateOrientedInvalidArgumentError(
        @"tracking.tracker_type=BOTSORT requires VIDEO or LIVE_STREAM running mode; tracking is "
        @"not available in IMAGE mode.",
        error);
  }

  if (!OrientedTilingEnabled(options)) {
    return CreateOrientedInvalidArgumentError(
        @"tracking.tracker_type=BOTSORT requires tiling to be enabled; the non-tiled path has no "
        @"tracker stage.",
        error);
  }

  if (options.numClasses < 1 || options.numClasses > 256) {
    return CreateOrientedInvalidArgumentError(
        @"tracking.tracker_type=BOTSORT requires num_classes in [1, 256] (BoTSORT stores the "
        @"class id as uint8).",
        error);
  }

  return YES;
}

}  // namespace

@interface MPPOrientedObjectDetector () {
  MPPVisionTaskRunner *_visionTaskRunner;
  dispatch_queue_t _callbackQueue;
  BOOL _tilingEnabled;
}
@property(nonatomic, weak) id<MPPOrientedObjectDetectorLiveStreamDelegate>
    orientedObjectDetectorLiveStreamDelegate;

+ (nullable MPPOrientedObjectDetectorResult *)orientedObjectDetectorResultWithOptionalOutputPacketMap:
    (std::optional<PacketMap> &)outputPacketMap;
@end

@implementation MPPOrientedObjectDetector

- (instancetype)initWithOptions:(MPPOrientedObjectDetectorOptions *)options
                          error:(NSError **)error {
  self = [super init];
  if (self) {
    if (!ValidateOrientedOptions(options, error)) {
      return nil;
    }

    _tilingEnabled = OrientedTilingEnabled(options);
    NSArray<NSString *> *inputStreams = _tilingEnabled
                                            ? @[ [NSString stringWithFormat:@"%@:%@", kImageTag,
                                                                             kImageInStreamName] ]
                                            : @[
                                                [NSString stringWithFormat:@"%@:%@", kImageTag,
                                                                           kImageInStreamName],
                                                [NSString stringWithFormat:@"%@:%@", kNormRectTag,
                                                                           kNormRectStreamName]
                                              ];

    MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc]
          initWithTaskName:kTaskName
             taskGraphName:kTaskGraphName
              inputStreams:inputStreams
             outputStreams:@[
               [NSString stringWithFormat:@"%@:%@", kOrientedDetectionsTag,
                                          kOrientedDetectionsStreamName],
               [NSString stringWithFormat:@"%@:%@", kImageTag, kImageOutStreamName]
             ]
               taskOptions:options
        enableFlowLimiting:options.runningMode == MPPRunningModeLiveStream
               runningMode:MPPCoreRunningModeFromVisionRunningMode(options.runningMode)
                     error:error];

    if (!taskInfo) {
      return nil;
    }

    PacketsCallback packetsCallback = nullptr;
    if (options.orientedObjectDetectorLiveStreamDelegate) {
      _orientedObjectDetectorLiveStreamDelegate = options.orientedObjectDetectorLiveStreamDelegate;
      _callbackQueue = dispatch_queue_create(
          [MPPVisionTaskRunner uniqueDispatchQueueNameWithSuffix:kTaskName], NULL);

      MPPOrientedObjectDetector *__weak weakSelf = self;
      packetsCallback = [=](absl::StatusOr<PacketMap> liveStreamResult) {
        [weakSelf processLiveStreamResult:liveStreamResult];
      };
    }

    _visionTaskRunner = [[MPPVisionTaskRunner alloc] initWithTaskInfo:taskInfo
                                                          runningMode:options.runningMode
                                                           roiAllowed:NO
                                                      packetsCallback:std::move(packetsCallback)
                                                 imageInputStreamName:kImageInStreamName
                                              normRectInputStreamName:_tilingEnabled
                                                                           ? nil
                                                                           : kNormRectStreamName
                                                                error:error];

    if (!_visionTaskRunner) {
      return nil;
    }
  }

  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath
                       numClasses:(NSInteger)numClasses
                             error:(NSError **)error {
  MPPOrientedObjectDetectorOptions *options = [[MPPOrientedObjectDetectorOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  options.numClasses = numClasses;
  return [self initWithOptions:options error:error];
}

- (nullable MPPOrientedObjectDetectorResult *)detectImage:(MPPImage *)image
                                                    error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap = [_visionTaskRunner processImage:image error:error];
  return [MPPOrientedObjectDetector
      orientedObjectDetectorResultWithOptionalOutputPacketMap:outputPacketMap];
}

- (nullable MPPOrientedObjectDetectorResult *)detectVideoFrame:(MPPImage *)image
                                       timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                         error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap =
      [_visionTaskRunner processVideoFrame:image
                   timestampInMilliseconds:timestampInMilliseconds
                                     error:error];
  return [MPPOrientedObjectDetector
      orientedObjectDetectorResultWithOptionalOutputPacketMap:outputPacketMap];
}

- (BOOL)detectAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  return [_visionTaskRunner processLiveStreamImage:image
                           timestampInMilliseconds:timestampInMilliseconds
                                             error:error];
}

- (void)processLiveStreamResult:(absl::StatusOr<PacketMap>)liveStreamResult {
  if (![self.orientedObjectDetectorLiveStreamDelegate
          respondsToSelector:@selector(orientedObjectDetector:
                                 didFinishDetectionWithResult:timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  if (![MPPCommonUtils checkCppError:liveStreamResult.status() toError:&callbackError]) {
    dispatch_async(_callbackQueue, ^{
      [self.orientedObjectDetectorLiveStreamDelegate orientedObjectDetector:self
                                               didFinishDetectionWithResult:nil
                                                    timestampInMilliseconds:Timestamp::Unset().Value()
                                                                      error:callbackError];
    });
    return;
  }

  PacketMap &outputPacketMap = liveStreamResult.value();
  if (outputPacketMap[kImageOutStreamName.cppString].IsEmpty()) {
    return;
  }

  MPPOrientedObjectDetectorResult *result =
      OrientedObjectDetectorResultWithOutputPacketMap(outputPacketMap);
  NSInteger timestampInMilliseconds =
      outputPacketMap[kImageOutStreamName.cppString].Timestamp().Value() /
      kMicrosecondsPerMillisecond;
  dispatch_async(_callbackQueue, ^{
    [self.orientedObjectDetectorLiveStreamDelegate orientedObjectDetector:self
                                             didFinishDetectionWithResult:result
                                                  timestampInMilliseconds:timestampInMilliseconds
                                                                    error:callbackError];
  });
}

+ (nullable MPPOrientedObjectDetectorResult *)orientedObjectDetectorResultWithOptionalOutputPacketMap:
    (std::optional<PacketMap> &)outputPacketMap {
  if (!outputPacketMap.has_value()) {
    return nil;
  }

  return OrientedObjectDetectorResultWithOutputPacketMap(outputPacketMap.value());
}

@end
