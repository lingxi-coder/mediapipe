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

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPOrientedDetection+Helpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"

#include "mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h"

namespace {
using OrientedDetectionProto = ::mediapipe::OrientedDetection;
using ::mediapipe::tasks::components::containers::ConvertToOrientedObjectDetectionResult;
}  // namespace

@implementation MPPOrientedDetection (Helpers)

+ (MPPOrientedDetection *)orientedDetectionWithProto:(const OrientedDetectionProto &)detectionProto
                                          imageWidth:(NSInteger)imageWidth
                                         imageHeight:(NSInteger)imageHeight {
  const auto convertedResult = ConvertToOrientedObjectDetectionResult(
      {detectionProto}, {static_cast<int>(imageWidth), static_cast<int>(imageHeight)});
  const auto &convertedDetection = convertedResult.detections.front();
  NSMutableArray<MPPCategory *> *categories =
      [NSMutableArray arrayWithCapacity:(NSUInteger)convertedDetection.categories.size()];

  for (const auto &category : convertedDetection.categories) {
    NSString *categoryName = category.category_name.has_value()
                                 ? [NSString stringWithCppString:*category.category_name]
                                 : nil;
    NSString *displayName = category.display_name.has_value()
                                ? [NSString stringWithCppString:*category.display_name]
                                : nil;
    [categories addObject:[[MPPCategory alloc] initWithIndex:category.index
                                                       score:category.score
                                                categoryName:categoryName
                                                 displayName:displayName]];
  }

  CGPoint center = CGPointMake(convertedDetection.cx, convertedDetection.cy);
  CGSize size = CGSizeMake(convertedDetection.width, convertedDetection.height);
  NSString *trackID = convertedDetection.track_id.has_value()
                          ? [NSString stringWithCppString:*convertedDetection.track_id]
                          : nil;

  return [[MPPOrientedDetection alloc] initWithCategories:categories
                                                   center:center
                                                     size:size
                                                 rotation:convertedDetection.rotation
                                                  trackID:trackID];
}

@end
