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
#import <UIKit/UIKit.h>

#import "mediapipe/tasks/ios/components/containers/sources/MPPCategory.h"

NS_ASSUME_NONNULL_BEGIN

/** Represents one oriented (rotated) object detection in pixel units. */
NS_SWIFT_NAME(OrientedDetection)
@interface MPPOrientedDetection : NSObject

/** An array of `Category` objects containing the predicted categories. */
@property(nonatomic, readonly) NSArray<MPPCategory *> *categories;

/** The center point of the oriented bounding box in pixels. */
@property(nonatomic, readonly) CGPoint center;

/** The size of the oriented bounding box in pixels. */
@property(nonatomic, readonly) CGSize size;

/** The rotation of the oriented bounding box in radians, counter-clockwise. */
@property(nonatomic, readonly) float rotation;

/** The optional persistent track identifier associated with the detection. */
@property(nonatomic, readonly, nullable) NSString *trackID;

/**
 * Initializes a new `OrientedDetection` object.
 *
 * @param categories A list of `Category` objects that contain category name, display name,
 * score, and the label index.
 * @param center The center point of the oriented bounding box in pixels.
 * @param size The size of the oriented bounding box in pixels.
 * @param rotation The rotation of the oriented bounding box in radians, counter-clockwise.
 * @param trackID An optional persistent track identifier associated with the detection.
 *
 * @return An instance of `OrientedDetection` initialized with the given values.
 */
- (instancetype)initWithCategories:(NSArray<MPPCategory *> *)categories
                            center:(CGPoint)center
                              size:(CGSize)size
                          rotation:(float)rotation
                           trackID:(nullable NSString *)trackID NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
