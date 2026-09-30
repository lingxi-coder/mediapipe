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

#import "mediapipe/tasks/ios/vision/oriented_object_detector/utils/sources/MPPOrientedObjectDetectorResult+Helpers.h"

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPOrientedDetection+Helpers.h"

#include "mediapipe/framework/formats/image.h"

namespace {
using OrientedDetectionProto = ::mediapipe::OrientedDetection;
using ::mediapipe::Image;
using ::mediapipe::Packet;
}  // namespace

@implementation MPPOrientedObjectDetectorResult (Helpers)

+ (nullable MPPOrientedObjectDetectorResult *)orientedObjectDetectorResultWithDetectionsPacket:
                                              (const Packet &)detectionsPacket
                                                                             imagePacket:
                                              (const Packet &)imagePacket {
  NSInteger timestampInMilliseconds =
      (NSInteger)(detectionsPacket.Timestamp().Value() / kMicrosecondsPerMillisecond);
  if (!detectionsPacket.ValidateAsType<std::vector<OrientedDetectionProto>>().ok() ||
      !imagePacket.ValidateAsType<Image>().ok()) {
    return [[MPPOrientedObjectDetectorResult alloc] initWithDetections:@[]
                                               timestampInMilliseconds:timestampInMilliseconds];
  }

  const Image &image = imagePacket.Get<Image>();
  const std::vector<OrientedDetectionProto> &detectionProtos =
      detectionsPacket.Get<std::vector<OrientedDetectionProto>>();

  NSMutableArray<MPPOrientedDetection *> *detections =
      [NSMutableArray arrayWithCapacity:(NSUInteger)detectionProtos.size()];
  for (const auto &detectionProto : detectionProtos) {
    [detections addObject:[MPPOrientedDetection orientedDetectionWithProto:detectionProto
                                                                imageWidth:image.width()
                                                               imageHeight:image.height()]];
  }

  return [[MPPOrientedObjectDetectorResult alloc] initWithDetections:detections
                                             timestampInMilliseconds:timestampInMilliseconds];
}

@end
