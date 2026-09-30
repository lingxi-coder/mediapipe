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

#import <XCTest/XCTest.h>

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/vision/yolo_object_detector/sources/MPPYoloObjectDetector.h"

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

@interface MPPYoloObjectDetectorTests : XCTestCase
@end

@implementation MPPYoloObjectDetectorTests

- (void)testOversizedTileGridFailsBeforeLoadingModel {
  for (NSNumber *dimension in @[ @65536, @46341 ]) {
    MPPYoloObjectDetectorOptions *options = [[MPPYoloObjectDetectorOptions alloc] init];
    options.numClasses = 80;
    options.tilingOptions.tileRows = dimension.integerValue;
    options.tilingOptions.tileCols = dimension.integerValue;
    NSError *error = nil;
    MPPYoloObjectDetector *detector =
        [[MPPYoloObjectDetector alloc] initWithOptions:options error:&error];
    XCTAssertNil(detector);
    XCTAssertEqual(error.code, MPPTasksErrorCodeInvalidArgumentError);
    XCTAssertEqualObjects(error.localizedDescription,
                          @"INVALID_ARGUMENT: tiling.tile_rows * tiling.tile_cols must be <= "
                          @"2147483647.");
  }
}

#if __LP64__
- (void)testTileDimensionCannotTruncateToInt32 {
  MPPYoloObjectDetectorOptions *options = [[MPPYoloObjectDetectorOptions alloc] init];
  options.numClasses = 80;
  options.tilingOptions.tileRows = NSIntegerMax;
  options.tilingOptions.tileCols = 0;
  NSError *error = nil;
  MPPYoloObjectDetector *detector =
      [[MPPYoloObjectDetector alloc] initWithOptions:options error:&error];
  XCTAssertNil(detector);
  XCTAssertEqual(error.code, MPPTasksErrorCodeInvalidArgumentError);
  XCTAssertEqualObjects(error.localizedDescription,
                        @"INVALID_ARGUMENT: tiling.tile_rows and tiling.tile_cols must be <= "
                        @"2147483647.");
}
#endif

- (void)testCreateYoloObjectDetectorWithoutNumClassesFails {
  MPPYoloObjectDetectorOptions *options = [[MPPYoloObjectDetectorOptions alloc] init];

  NSError *error = nil;
  MPPYoloObjectDetector *detector = [[MPPYoloObjectDetector alloc] initWithOptions:options
                                                                              error:&error];
  XCTAssertNil(detector);
  XCTAssertEqual(error.code, MPPTasksErrorCodeInvalidArgumentError);
  XCTAssertEqualObjects(
      error.localizedDescription,
      @"INVALID_ARGUMENT: num_classes must be set in YoloObjectDetectorOptions "
       "(metadata-derived num_classes is a future enhancement)");
}

- (void)testCreateYoloObjectDetectorAllowlistAndDenylistFails {
  MPPYoloObjectDetectorOptions *options = [[MPPYoloObjectDetectorOptions alloc] init];
  options.categoryAllowlist = @[ @"cat" ];
  options.categoryDenylist = @[ @"dog" ];

  NSError *error = nil;
  MPPYoloObjectDetector *detector = [[MPPYoloObjectDetector alloc] initWithOptions:options
                                                                              error:&error];
  XCTAssertNil(detector);

  NSError *expectedError = [NSError
      errorWithDomain:kExpectedErrorDomain
                 code:MPPTasksErrorCodeInvalidArgumentError
             userInfo:@{
               NSLocalizedDescriptionKey :
                   @"INVALID_ARGUMENT: `category_allowlist` and `category_denylist` are mutually "
                   @"exclusive options."
             }];
  AssertEqualErrors(error, expectedError);
}

- (void)testCreateYoloObjectDetectorBotsortInImageModeFails {
  MPPYoloObjectDetectorOptions *options = [[MPPYoloObjectDetectorOptions alloc] init];
  options.trackingOptions.trackerType = MPPYoloObjectDetectorTrackerTypeBotsort;
  options.numClasses = 80;

  NSError *error = nil;
  MPPYoloObjectDetector *detector = [[MPPYoloObjectDetector alloc] initWithOptions:options
                                                                              error:&error];
  XCTAssertNil(detector);

  NSError *expectedError = [NSError
      errorWithDomain:kExpectedErrorDomain
                 code:MPPTasksErrorCodeInvalidArgumentError
             userInfo:@{
               NSLocalizedDescriptionKey :
                   @"INVALID_ARGUMENT: tracking.tracker_type=BOTSORT requires VIDEO or "
                   @"LIVE_STREAM running mode; tracking is not available in IMAGE mode."
             }];
  AssertEqualErrors(error, expectedError);
}

@end
