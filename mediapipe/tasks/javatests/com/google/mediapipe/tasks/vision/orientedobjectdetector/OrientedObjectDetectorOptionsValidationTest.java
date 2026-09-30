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

package com.google.mediapipe.tasks.vision.orientedobjectdetector;

import com.google.mediapipe.formats.proto.OrientedDetectionProto;
import com.google.mediapipe.tasks.components.containers.OrientedDetection;
import com.google.mediapipe.tasks.core.BaseOptions;

/** Runnable validation coverage for {@link OrientedObjectDetector} options. */
public final class OrientedObjectDetectorOptionsValidationTest {
  public static void main(String[] args) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath("model.tflite").build();

    expectIllegalArgument(
        () ->
            OrientedObjectDetector.OrientedObjectDetectorOptions.builder()
                .setBaseOptions(baseOptions)
                .setNumClasses(0)
                .build(),
        "numClasses must be set");

    // Reject both zero-wrapping and negative-wrapping int products before any JNI/model work.
    for (int[] grid : new int[][] {{65536, 65536}, {46341, 46341}, {Integer.MAX_VALUE, 2}}) {
      OrientedObjectDetector.OrientedObjectDetectorOptions.TilingOptions tiling =
          OrientedObjectDetector.OrientedObjectDetectorOptions.TilingOptions.builder()
              .setTileRows(grid[0])
              .setTileCols(grid[1])
              .build();
      expectIllegalArgument(
          () ->
              OrientedObjectDetector.OrientedObjectDetectorOptions.builder()
                  .setBaseOptions(baseOptions)
                  .setNumClasses(15)
                  .setTilingOptions(tiling)
                  .build(),
          "tiling.tile_rows * tiling.tile_cols must be <= 2147483647");
      if (!OrientedObjectDetectorOptionsValidator.isTilingEnabled(tiling)) {
        throw new AssertionError("Large grid must not wrap and silently disable tiling.");
      }
    }
    for (int[] grid : new int[][] {{0, Integer.MAX_VALUE}, {Integer.MAX_VALUE, 0}, {1, 1}}) {
      OrientedObjectDetector.OrientedObjectDetectorOptions.TilingOptions tiling =
          OrientedObjectDetector.OrientedObjectDetectorOptions.TilingOptions.builder()
              .setTileRows(grid[0])
              .setTileCols(grid[1])
              .build();
      OrientedObjectDetectorOptionsValidator.validateTilingOptions(tiling);
      if (OrientedObjectDetectorOptionsValidator.isTilingEnabled(tiling)) {
        throw new AssertionError("Zero-sized or single-tile grids must preserve non-tiled mode.");
      }
    }

    OrientedObjectDetector.OrientedObjectDetectorOptions options =
        OrientedObjectDetector.OrientedObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setNumClasses(15)
            .build();
    if (options.numClasses() != 15) {
      throw new AssertionError(
          "Expected numClasses to be preserved for valid oriented detector options.");
    }

    OrientedDetectionProto.OrientedDetection detectionProto =
        OrientedDetectionProto.OrientedDetection.newBuilder()
            .setCx(0.5f)
            .setCy(0.25f)
            .setWidth(0.4f)
            .setHeight(0.2f)
            .setRotation(0.3f)
            .build();
    OrientedDetection detection =
        OrientedDetection.createFromProto(
            detectionProto, /* imageWidth= */ 200, /* imageHeight= */ 100);
    assertNear(detection.centerX(), 100.0f, 1e-4f, "centerX");
    assertNear(detection.centerY(), 25.0f, 1e-4f, "centerY");
    assertNear(detection.width(), 86.0971f, 1e-3f, "width");
    assertNear(detection.height(), 20.6890f, 1e-3f, "height");
    assertNear(detection.rotation(), 0.153452f, 1e-5f, "rotation");
  }

  private static void assertNear(float actual, float expected, float tolerance, String field) {
    if (Math.abs(actual - expected) > tolerance) {
      throw new AssertionError(
          "Expected " + field + " to be " + expected + " but got " + actual + ".");
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

  private OrientedObjectDetectorOptionsValidationTest() {}
}
