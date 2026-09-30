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

package com.google.mediapipe.tasks.vision.yoloobjectdetector;

import com.google.mediapipe.tasks.core.BaseOptions;

/** Runnable validation coverage for {@link YoloObjectDetector} options. */
public final class YoloObjectDetectorOptionsValidationTest {
  public static void main(String[] args) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath("model.tflite").build();

    expectIllegalArgument(
        () ->
            YoloObjectDetector.YoloObjectDetectorOptions.builder()
                .setBaseOptions(baseOptions)
                .setNumClasses(0)
                .build(),
        "numClasses must be set");

    // Reject both zero-wrapping and negative-wrapping int products before any JNI/model work.
    for (int[] grid : new int[][] {{65536, 65536}, {46341, 46341}, {Integer.MAX_VALUE, 2}}) {
      YoloObjectDetector.YoloObjectDetectorOptions.TilingOptions tiling =
          YoloObjectDetector.YoloObjectDetectorOptions.TilingOptions.builder()
              .setTileRows(grid[0])
              .setTileCols(grid[1])
              .build();
      expectIllegalArgument(
          () ->
              YoloObjectDetector.YoloObjectDetectorOptions.builder()
                  .setBaseOptions(baseOptions)
                  .setNumClasses(80)
                  .setTilingOptions(tiling)
                  .build(),
          "tiling.tile_rows * tiling.tile_cols must be <= 2147483647");
      if (!YoloObjectDetectorOptionsValidator.isTilingEnabled(tiling)) {
        throw new AssertionError("Large grid must not wrap and silently disable tiling.");
      }
    }
    for (int[] grid : new int[][] {{0, Integer.MAX_VALUE}, {Integer.MAX_VALUE, 0}, {1, 1}}) {
      YoloObjectDetector.YoloObjectDetectorOptions.TilingOptions tiling =
          YoloObjectDetector.YoloObjectDetectorOptions.TilingOptions.builder()
              .setTileRows(grid[0])
              .setTileCols(grid[1])
              .build();
      YoloObjectDetectorOptionsValidator.validateTilingOptions(tiling);
      if (YoloObjectDetectorOptionsValidator.isTilingEnabled(tiling)) {
        throw new AssertionError("Zero-sized or single-tile grids must preserve non-tiled mode.");
      }
    }

    YoloObjectDetector.YoloObjectDetectorOptions options =
        YoloObjectDetector.YoloObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setNumClasses(80)
            .build();
    if (options.numClasses() != 80) {
      throw new AssertionError("Expected numClasses to be preserved for valid YOLO options.");
    }
  }

  private static void expectIllegalArgument(ThrowingRunnable runnable, String expectedMessagePart) {
    try {
      runnable.run();
    } catch (IllegalArgumentException e) {
      if (e.getMessage() == null || !e.getMessage().contains(expectedMessagePart)) {
        throw new AssertionError(
            "Expected IllegalArgumentException containing \""
                + expectedMessagePart
                + "\" but got: "
                + e.getMessage(),
            e);
      }
      return;
    } catch (Exception e) {
      throw new AssertionError("Expected IllegalArgumentException but got: " + e, e);
    }
    throw new AssertionError("Expected IllegalArgumentException to be thrown.");
  }

  private interface ThrowingRunnable {
    void run() throws Exception;
  }

  private YoloObjectDetectorOptionsValidationTest() {}
}
