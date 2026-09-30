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
#import "mediapipe/tasks/ios/vision/oriented_object_detector/sources/MPPOrientedObjectDetector.h"

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

@interface MPPOrientedObjectDetectorTests : XCTestCase
@end

@implementation MPPOrientedObjectDetectorTests

- (void)testOversizedTileGridFailsBeforeLoadingModel {
  for (NSNumber *dimension in @[ @65536, @46341 ]) {
    MPPOrientedObjectDetectorOptions *options = [[MPPOrientedObjectDetectorOptions alloc] init];
    options.numClasses = 15;
    options.tilingOptions.tileRows = dimension.integerValue;
    options.tilingOptions.tileCols = dimension.integerValue;
    NSError *error = nil;
    MPPOrientedObjectDetector *detector =
        [[MPPOrientedObjectDetector alloc] initWithOptions:options error:&error];
    XCTAssertNil(detector);
    XCTAssertEqual(error.code, MPPTasksErrorCodeInvalidArgumentError);
    XCTAssertEqualObjects(error.localizedDescription,
                          @"INVALID_ARGUMENT: tiling.tile_rows * tiling.tile_cols must be <= "
                          @"2147483647.");
  }
}

#if __LP64__
- (void)testTileDimensionCannotTruncateToInt32 {
  MPPOrientedObjectDetectorOptions *options = [[MPPOrientedObjectDetectorOptions alloc] init];
  options.numClasses = 15;
  options.tilingOptions.tileRows = NSIntegerMax;
  options.tilingOptions.tileCols = 0;
  NSError *error = nil;
  MPPOrientedObjectDetector *detector =
      [[MPPOrientedObjectDetector alloc] initWithOptions:options error:&error];
  XCTAssertNil(detector);
  XCTAssertEqual(error.code, MPPTasksErrorCodeInvalidArgumentError);
  XCTAssertEqualObjects(error.localizedDescription,
                        @"INVALID_ARGUMENT: tiling.tile_rows and tiling.tile_cols must be <= "
                        @"2147483647.");
}
#endif

- (void)testCreateOrientedObjectDetectorWithoutNumClassesFails {
  MPPOrientedObjectDetectorOptions *options = [[MPPOrientedObjectDetectorOptions alloc] init];

  NSError *error = nil;
  MPPOrientedObjectDetector *detector =
      [[MPPOrientedObjectDetector alloc] initWithOptions:options error:&error];
  XCTAssertNil(detector);
  XCTAssertEqual(error.code, MPPTasksErrorCodeInvalidArgumentError);
  XCTAssertEqualObjects(error.localizedDescription,
                        @"INVALID_ARGUMENT: num_classes must be set in "
                         "OrientedObjectDetectorOptions");
}

- (void)testCreateOrientedObjectDetectorAllowlistAndDenylistFails {
  MPPOrientedObjectDetectorOptions *options = [[MPPOrientedObjectDetectorOptions alloc] init];
  options.categoryAllowlist = @[ @"cat" ];
  options.categoryDenylist = @[ @"dog" ];

  NSError *error = nil;
  MPPOrientedObjectDetector *detector =
      [[MPPOrientedObjectDetector alloc] initWithOptions:options error:&error];
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

- (void)testCreateOrientedObjectDetectorBoxTrackerFails {
  MPPOrientedObjectDetectorOptions *options = [[MPPOrientedObjectDetectorOptions alloc] init];
  options.numClasses = 15;
  options.trackingOptions.trackerType = MPPOrientedObjectDetectorTrackerTypeBoxTracker;

  NSError *error = nil;
  MPPOrientedObjectDetector *detector =
      [[MPPOrientedObjectDetector alloc] initWithOptions:options error:&error];
  XCTAssertNil(detector);

  NSError *expectedError = [NSError
      errorWithDomain:kExpectedErrorDomain
                 code:MPPTasksErrorCodeInvalidArgumentError
             userInfo:@{
               NSLocalizedDescriptionKey :
                   @"INVALID_ARGUMENT: tracking.tracker_type=BOX_TRACKER: BoxTracker is not "
                   @"supported for oriented detection; use BOTSORT."
             }];
  AssertEqualErrors(error, expectedError);
}

@end
