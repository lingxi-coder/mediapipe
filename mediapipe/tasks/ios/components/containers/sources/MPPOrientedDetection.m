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

#import "mediapipe/tasks/ios/components/containers/sources/MPPOrientedDetection.h"

@implementation MPPOrientedDetection

- (instancetype)initWithCategories:(NSArray<MPPCategory *> *)categories
                            center:(CGPoint)center
                              size:(CGSize)size
                          rotation:(float)rotation
                           trackID:(nullable NSString *)trackID {
  self = [super init];
  if (self) {
    _categories = categories;
    _center = center;
    _size = size;
    _rotation = rotation;
    _trackID = trackID;
  }
  return self;
}

@end
