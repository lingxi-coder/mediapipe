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

#import "mediapipe/tasks/ios/vision/core/sources/MPPImage.h"
#import "mediapipe/tasks/ios/vision/oriented_object_detector/sources/MPPOrientedObjectDetectorOptions.h"

NS_ASSUME_NONNULL_BEGIN

/** Class that performs oriented object detection on images. */
NS_SWIFT_NAME(OrientedObjectDetector)
@interface MPPOrientedObjectDetector : NSObject

- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                numClasses:(NSInteger)numClasses
                                      error:(NSError **)error
    NS_SWIFT_NAME(init(modelPath:numClasses:));

- (nullable instancetype)initWithOptions:(MPPOrientedObjectDetectorOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (nullable MPPOrientedObjectDetectorResult *)detectImage:(MPPImage *)image
                                                    error:(NSError **)error NS_SWIFT_NAME(detect(image:));

- (nullable MPPOrientedObjectDetectorResult *)detectVideoFrame:(MPPImage *)image
                                       timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                         error:(NSError **)error
    NS_SWIFT_NAME(detect(videoFrame:timestampInMilliseconds:));

- (BOOL)detectAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error
    NS_SWIFT_NAME(detectAsync(image:timestampInMilliseconds:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
