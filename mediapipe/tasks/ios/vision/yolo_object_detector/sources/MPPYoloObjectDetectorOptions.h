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

#import <Foundation/Foundation.h>

#import "mediapipe/tasks/ios/core/sources/MPPTaskOptions.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPRunningMode.h"
#import "mediapipe/tasks/ios/vision/object_detector/sources/MPPObjectDetectorResult.h"

NS_ASSUME_NONNULL_BEGIN

@class MPPYoloObjectDetector;

typedef NS_ENUM(NSInteger, MPPYoloObjectDetectorLayout) {
  MPPYoloObjectDetectorLayoutChannelsFirst = 1,
  MPPYoloObjectDetectorLayoutChannelsLast = 2,
} NS_SWIFT_NAME(YoloObjectDetectorLayout);

typedef NS_ENUM(NSInteger, MPPYoloObjectDetectorTrackerType) {
  MPPYoloObjectDetectorTrackerTypeBoxTracker = 1,
  MPPYoloObjectDetectorTrackerTypeBotsort = 2,
} NS_SWIFT_NAME(YoloObjectDetectorTrackerType);

NS_SWIFT_NAME(YoloObjectDetectorTileRect)
@interface MPPYoloObjectDetectorTileRect : NSObject <NSCopying>

@property(nonatomic) float xCenter;
@property(nonatomic) float yCenter;
@property(nonatomic) float width;
@property(nonatomic) float height;

@end

NS_SWIFT_NAME(YoloObjectDetectorTilingOptions)
@interface MPPYoloObjectDetectorTilingOptions : NSObject <NSCopying>

@property(nonatomic) NSInteger tileRows;
@property(nonatomic) NSInteger tileCols;
@property(nonatomic) float tileOverlapFraction;
@property(nonatomic, copy) NSArray<MPPYoloObjectDetectorTileRect *> *explicitTiles;
@property(nonatomic) float tileLocalNMSIOUThreshold;
@property(nonatomic) NSInteger maxDetectionsAfterTileNMS;
@property(nonatomic) BOOL enableMotionScheduling;
@property(nonatomic) NSInteger maxScheduledTiles;

@end

NS_SWIFT_NAME(YoloObjectDetectorTrackingOptions)
@interface MPPYoloObjectDetectorTrackingOptions : NSObject <NSCopying>

@property(nonatomic) MPPYoloObjectDetectorTrackerType trackerType;
@property(nonatomic) float trackHighThreshold;
@property(nonatomic) float trackLowThreshold;
@property(nonatomic) float newTrackThreshold;
@property(nonatomic) NSInteger trackBuffer;
@property(nonatomic) float matchThreshold;
@property(nonatomic) BOOL enableGMC;
@property(nonatomic) NSInteger nominalFrameRate;

@end

NS_SWIFT_NAME(YoloObjectDetectorLiveStreamDelegate)
@protocol MPPYoloObjectDetectorLiveStreamDelegate <NSObject>

@optional

- (void)yoloObjectDetector:(MPPYoloObjectDetector *)yoloObjectDetector
    didFinishDetectionWithResult:(nullable MPPObjectDetectorResult *)result
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(nullable NSError *)error
    NS_SWIFT_NAME(yoloObjectDetector(_:didFinishDetection:timestampInMilliseconds:error:));

@end

/** Options for setting up a `YoloObjectDetector`. */
NS_SWIFT_NAME(YoloObjectDetectorOptions)
@interface MPPYoloObjectDetectorOptions : MPPTaskOptions <NSCopying>

@property(nonatomic) MPPRunningMode runningMode;
@property(nonatomic, weak, nullable) id<MPPYoloObjectDetectorLiveStreamDelegate>
    yoloObjectDetectorLiveStreamDelegate;
@property(nonatomic, copy) NSString *displayNamesLocale;
@property(nonatomic) NSInteger maxResults;
@property(nonatomic) float scoreThreshold;
@property(nonatomic, copy) NSArray<NSString *> *categoryAllowlist;
@property(nonatomic, copy) NSArray<NSString *> *categoryDenylist;
@property(nonatomic) float iouThreshold;
@property(nonatomic) MPPYoloObjectDetectorLayout layout;
/** Number of classes in the model detect head. This value must be greater than zero. */
@property(nonatomic) NSInteger numClasses;
@property(nonatomic, copy) MPPYoloObjectDetectorTilingOptions *tilingOptions;
@property(nonatomic, copy) MPPYoloObjectDetectorTrackingOptions *trackingOptions;

@end

NS_ASSUME_NONNULL_END
