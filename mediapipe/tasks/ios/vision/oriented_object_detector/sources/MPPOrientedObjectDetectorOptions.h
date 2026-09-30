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
#import "mediapipe/tasks/ios/vision/oriented_object_detector/sources/MPPOrientedObjectDetectorResult.h"

NS_ASSUME_NONNULL_BEGIN

@class MPPOrientedObjectDetector;

typedef NS_ENUM(NSInteger, MPPOrientedObjectDetectorLayout) {
  MPPOrientedObjectDetectorLayoutChannelsFirst = 1,
  MPPOrientedObjectDetectorLayoutChannelsLast = 2,
} NS_SWIFT_NAME(OrientedObjectDetectorLayout);

typedef NS_ENUM(NSInteger, MPPOrientedObjectDetectorTrackerType) {
  MPPOrientedObjectDetectorTrackerTypeUnspecified = 0,
  MPPOrientedObjectDetectorTrackerTypeBoxTracker = 1,
  MPPOrientedObjectDetectorTrackerTypeBotsort = 2,
} NS_SWIFT_NAME(OrientedObjectDetectorTrackerType);

NS_SWIFT_NAME(OrientedObjectDetectorTileRect)
@interface MPPOrientedObjectDetectorTileRect : NSObject <NSCopying>

@property(nonatomic) float xCenter;
@property(nonatomic) float yCenter;
@property(nonatomic) float width;
@property(nonatomic) float height;

@end

NS_SWIFT_NAME(OrientedObjectDetectorTilingOptions)
@interface MPPOrientedObjectDetectorTilingOptions : NSObject <NSCopying>

@property(nonatomic) NSInteger tileRows;
@property(nonatomic) NSInteger tileCols;
@property(nonatomic) float tileOverlapFraction;
@property(nonatomic, copy) NSArray<MPPOrientedObjectDetectorTileRect *> *explicitTiles;
@property(nonatomic) float tileLocalNMSIOUThreshold;
@property(nonatomic) NSInteger maxDetectionsAfterTileNMS;

@end

NS_SWIFT_NAME(OrientedObjectDetectorTrackingOptions)
@interface MPPOrientedObjectDetectorTrackingOptions : NSObject <NSCopying>

@property(nonatomic) MPPOrientedObjectDetectorTrackerType trackerType;
@property(nonatomic) float trackHighThreshold;
@property(nonatomic) float trackLowThreshold;
@property(nonatomic) float newTrackThreshold;
@property(nonatomic) NSInteger trackBuffer;
@property(nonatomic) float matchThreshold;
@property(nonatomic) BOOL enableGMC;
@property(nonatomic) NSInteger nominalFrameRate;

@end

NS_SWIFT_NAME(OrientedObjectDetectorLiveStreamDelegate)
@protocol MPPOrientedObjectDetectorLiveStreamDelegate <NSObject>

@optional

- (void)orientedObjectDetector:(MPPOrientedObjectDetector *)orientedObjectDetector
    didFinishDetectionWithResult:(nullable MPPOrientedObjectDetectorResult *)result
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(nullable NSError *)error
    NS_SWIFT_NAME(orientedObjectDetector(_:didFinishDetection:timestampInMilliseconds:error:));

@end

/** Options for setting up an `OrientedObjectDetector`. */
NS_SWIFT_NAME(OrientedObjectDetectorOptions)
@interface MPPOrientedObjectDetectorOptions : MPPTaskOptions <NSCopying>

@property(nonatomic) MPPRunningMode runningMode;
@property(nonatomic, weak, nullable) id<MPPOrientedObjectDetectorLiveStreamDelegate>
    orientedObjectDetectorLiveStreamDelegate;
@property(nonatomic) NSInteger maxResults;
@property(nonatomic) float scoreThreshold;
@property(nonatomic) float iouThreshold;
@property(nonatomic) BOOL classAgnosticNMS;
@property(nonatomic) MPPOrientedObjectDetectorLayout layout;
/** Number of classes in the model detect head. This value must be greater than zero. */
@property(nonatomic) NSInteger numClasses;
@property(nonatomic, copy) NSString *displayNamesLocale;
@property(nonatomic, copy) NSArray<NSString *> *categoryAllowlist;
@property(nonatomic, copy) NSArray<NSString *> *categoryDenylist;
@property(nonatomic, copy) MPPOrientedObjectDetectorTilingOptions *tilingOptions;
@property(nonatomic, copy) MPPOrientedObjectDetectorTrackingOptions *trackingOptions;

@end

NS_ASSUME_NONNULL_END
