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

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.framework.AndroidPacketGetter;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.ErrorListener;
import com.google.mediapipe.tasks.core.OutputHandler;
import com.google.mediapipe.tasks.core.OutputHandler.ResultListener;
import com.google.mediapipe.tasks.core.TaskInfo;
import com.google.mediapipe.tasks.core.TaskOptions;
import com.google.mediapipe.tasks.core.TaskRunner;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.vision.core.BaseVisionTaskApi;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.orientedobjectdetector.proto.OrientedObjectDetectorOptionsProto;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/** Performs oriented object detection on images. */
public final class OrientedObjectDetector extends BaseVisionTaskApi {
  private static final String IMAGE_IN_STREAM_NAME = "image_in";
  private static final String NORM_RECT_IN_STREAM_NAME = "norm_rect_in";
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.oriented_object_detector.OrientedObjectDetectorGraph";

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList("IMAGE:" + IMAGE_IN_STREAM_NAME, "NORM_RECT:" + NORM_RECT_IN_STREAM_NAME));

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> TILED_INPUT_STREAMS =
      Collections.unmodifiableList(Arrays.asList("IMAGE:" + IMAGE_IN_STREAM_NAME));

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> OUTPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList("ORIENTED_DETECTIONS:oriented_detections_out", "IMAGE:image_out"));

  private static final int DETECTIONS_OUT_STREAM_INDEX = 0;
  private static final int IMAGE_OUT_STREAM_INDEX = 1;

  private final boolean tilingEnabled;

  static {
    System.loadLibrary("mediapipe_tasks_jni");
  }

  public static OrientedObjectDetector createFromFile(
      Context context, String modelPath, int numClasses) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context,
        OrientedObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setNumClasses(numClasses)
            .build());
  }

  public static OrientedObjectDetector createFromFile(
      Context context, File modelFile, int numClasses) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context,
          OrientedObjectDetectorOptions.builder()
              .setBaseOptions(baseOptions)
              .setNumClasses(numClasses)
              .build());
    }
  }

  public static OrientedObjectDetector createFromBuffer(
      Context context, ByteBuffer modelBuffer, int numClasses) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetBuffer(modelBuffer).build();
    return createFromOptions(
        context,
        OrientedObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setNumClasses(numClasses)
            .build());
  }

  public static OrientedObjectDetector createFromOptions(
      Context context, OrientedObjectDetectorOptions detectorOptions) {
    final boolean tilingEnabled =
        OrientedObjectDetectorOptionsValidator.isTilingEnabled(detectorOptions.tilingOptions());
    OutputHandler<OrientedObjectDetectorResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<OrientedObjectDetectorResult, MPImage>() {
          @Override
          public OrientedObjectDetectorResult convertToTaskResult(List<Packet> packets) {
            Packet detectionsPacket = packets.get(DETECTIONS_OUT_STREAM_INDEX);
            Packet imagePacket = packets.get(IMAGE_OUT_STREAM_INDEX);
            int imageWidth = PacketGetter.getImageWidth(imagePacket);
            int imageHeight = PacketGetter.getImageHeight(imagePacket);
            if (detectionsPacket.isEmpty()) {
              return OrientedObjectDetectorResult.create(
                  new ArrayList<>(),
                  imageWidth,
                  imageHeight,
                  BaseVisionTaskApi.generateResultTimestampMs(
                      detectorOptions.runningMode(), detectionsPacket));
            }
            return OrientedObjectDetectorResult.create(
                PacketGetter.getProtoVector(
                    detectionsPacket,
                    com.google.mediapipe.formats.proto.OrientedDetectionProto.OrientedDetection
                        .parser()),
                imageWidth,
                imageHeight,
                BaseVisionTaskApi.generateResultTimestampMs(
                    detectorOptions.runningMode(), detectionsPacket));
          }

          @Override
          public MPImage convertToTaskInput(List<Packet> packets) {
            return new BitmapImageBuilder(
                    AndroidPacketGetter.getBitmap(packets.get(IMAGE_OUT_STREAM_INDEX)))
                .build();
          }
        });
    detectorOptions.resultListener().ifPresent(handler::setResultListener);
    detectorOptions.errorListener().ifPresent(handler::setErrorListener);
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<OrientedObjectDetectorOptions>builder()
                .setTaskName(OrientedObjectDetector.class.getSimpleName())
                .setTaskRunningModeName(detectorOptions.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(tilingEnabled ? TILED_INPUT_STREAMS : INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(detectorOptions)
                .setEnableFlowLimiting(detectorOptions.runningMode() == RunningMode.LIVE_STREAM)
                .build(),
            handler);
    return new OrientedObjectDetector(runner, detectorOptions.runningMode(), tilingEnabled);
  }

  private OrientedObjectDetector(
      TaskRunner taskRunner, RunningMode runningMode, boolean tilingEnabled) {
    super(
        taskRunner,
        runningMode,
        IMAGE_IN_STREAM_NAME,
        tilingEnabled ? "" : NORM_RECT_IN_STREAM_NAME);
    this.tilingEnabled = tilingEnabled;
  }

  public OrientedObjectDetectorResult detect(MPImage image) {
    return detect(image, ImageProcessingOptions.builder().build());
  }

  public OrientedObjectDetectorResult detect(
      MPImage image, ImageProcessingOptions imageProcessingOptions) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (OrientedObjectDetectorResult) processImageData(image, imageProcessingOptions);
  }

  public OrientedObjectDetectorResult detectForVideo(MPImage image, long timestampMs) {
    return detectForVideo(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  public OrientedObjectDetectorResult detectForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (OrientedObjectDetectorResult)
        processVideoData(image, imageProcessingOptions, timestampMs);
  }

  public void detectAsync(MPImage image, long timestampMs) {
    detectAsync(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  public void detectAsync(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    sendLiveStreamData(image, imageProcessingOptions, timestampMs);
  }

  @AutoValue
  public abstract static class OrientedObjectDetectorOptions extends TaskOptions {
    /** Output tensor layout of the OBB detect head. */
    public enum Layout {
      CHANNELS_FIRST,
      CHANNELS_LAST
    }

    @AutoValue
    public abstract static class TilingOptions {
      @AutoValue
      public abstract static class TileRect {
        @AutoValue.Builder
        public abstract static class Builder {
          public abstract Builder setXCenter(float value);

          public abstract Builder setYCenter(float value);

          public abstract Builder setWidth(float value);

          public abstract Builder setHeight(float value);

          abstract TileRect autoBuild();

          public final TileRect build() {
            return autoBuild();
          }
        }

        public abstract float xCenter();

        public abstract float yCenter();

        public abstract float width();

        public abstract float height();

        public static Builder builder() {
          return new AutoValue_OrientedObjectDetector_OrientedObjectDetectorOptions_TilingOptions_TileRect
                  .Builder()
              .setXCenter(0.0f)
              .setYCenter(0.0f)
              .setWidth(0.0f)
              .setHeight(0.0f);
        }
      }

      @AutoValue.Builder
      public abstract static class Builder {
        public abstract Builder setTileRows(int value);

        public abstract Builder setTileCols(int value);

        public abstract Builder setTileOverlapFraction(float value);

        public abstract Builder setExplicitTiles(List<TileRect> value);

        public abstract Builder setTileLocalNmsIouThreshold(float value);

        public abstract Builder setMaxDetectionsAfterTileNms(int value);

        abstract TilingOptions autoBuild();

        public final TilingOptions build() {
          return autoBuild();
        }
      }

      public abstract int tileRows();

      public abstract int tileCols();

      public abstract float tileOverlapFraction();

      @SuppressWarnings("AutoValueImmutableFields")
      public abstract List<TileRect> explicitTiles();

      public abstract float tileLocalNmsIouThreshold();

      public abstract int maxDetectionsAfterTileNms();

      public static Builder builder() {
        return new AutoValue_OrientedObjectDetector_OrientedObjectDetectorOptions_TilingOptions
                .Builder()
            .setTileRows(1)
            .setTileCols(1)
            .setTileOverlapFraction(0.0f)
            .setExplicitTiles(Collections.emptyList())
            .setTileLocalNmsIouThreshold(0.0f)
            .setMaxDetectionsAfterTileNms(0);
      }
    }

    @AutoValue
    public abstract static class TrackingOptions {
      /** Tracker selection for the tiled VIDEO/LIVE_STREAM path. */
      public enum TrackerType {
        TRACKER_UNSPECIFIED,
        BOX_TRACKER,
        BOTSORT
      }

      @AutoValue.Builder
      public abstract static class Builder {
        public abstract Builder setTrackerType(TrackerType value);

        public abstract Builder setTrackHighThreshold(float value);

        public abstract Builder setTrackLowThreshold(float value);

        public abstract Builder setNewTrackThreshold(float value);

        public abstract Builder setTrackBuffer(int value);

        public abstract Builder setMatchThreshold(float value);

        public abstract Builder setEnableGmc(boolean value);

        public abstract Builder setNominalFrameRate(int value);

        abstract TrackingOptions autoBuild();

        public final TrackingOptions build() {
          return autoBuild();
        }
      }

      public abstract TrackerType trackerType();

      public abstract float trackHighThreshold();

      public abstract float trackLowThreshold();

      public abstract float newTrackThreshold();

      public abstract int trackBuffer();

      public abstract float matchThreshold();

      public abstract boolean enableGmc();

      public abstract int nominalFrameRate();

      public static Builder builder() {
        return new AutoValue_OrientedObjectDetector_OrientedObjectDetectorOptions_TrackingOptions
                .Builder()
            .setTrackerType(TrackerType.TRACKER_UNSPECIFIED)
            .setTrackHighThreshold(0.6f)
            .setTrackLowThreshold(0.1f)
            .setNewTrackThreshold(0.7f)
            .setTrackBuffer(30)
            .setMatchThreshold(0.7f)
            .setEnableGmc(false)
            .setNominalFrameRate(30);
      }
    }

    @AutoValue.Builder
    public abstract static class Builder {
      public abstract Builder setBaseOptions(BaseOptions value);

      public abstract Builder setRunningMode(RunningMode value);

      public abstract Builder setMaxResults(int value);

      public abstract Builder setScoreThreshold(float value);

      public abstract Builder setIouThreshold(float value);

      public abstract Builder setClassAgnosticNms(boolean value);

      public abstract Builder setLayout(Layout value);

      public abstract Builder setNumClasses(int value);

      public abstract Builder setDisplayNamesLocale(String value);

      public abstract Builder setCategoryAllowlist(List<String> value);

      public abstract Builder setCategoryDenylist(List<String> value);

      public abstract Builder setTilingOptions(TilingOptions value);

      public abstract Builder setTrackingOptions(TrackingOptions value);

      public abstract Builder setResultListener(
          ResultListener<OrientedObjectDetectorResult, MPImage> value);

      public abstract Builder setErrorListener(ErrorListener value);

      abstract OrientedObjectDetectorOptions autoBuild();

      public final OrientedObjectDetectorOptions build() {
        OrientedObjectDetectorOptions options = autoBuild();
        if (options.runningMode() == RunningMode.LIVE_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The oriented object detector is in the live stream mode, a user-defined result"
                    + " listener must be provided in OrientedObjectDetectorOptions.");
          }
        } else if (options.resultListener().isPresent()) {
          throw new IllegalArgumentException(
              "The oriented object detector is in the image or video mode, a user-defined result"
                  + " listener shouldn't be provided in OrientedObjectDetectorOptions.");
        }
        if (options.maxResults() == 0) {
          throw new IllegalArgumentException("Invalid maxResults option: value must be != 0.");
        }
        if (options.numClasses() <= 0) {
          throw new IllegalArgumentException(
              "numClasses must be set to a positive value for OrientedObjectDetectorOptions.");
        }
        if (!options.categoryAllowlist().isEmpty() && !options.categoryDenylist().isEmpty()) {
          throw new IllegalArgumentException(
              "Category allowlist and denylist are mutually exclusive.");
        }
        OrientedObjectDetectorOptionsValidator.validateTilingOptions(options.tilingOptions());
        boolean tilingEnabled =
            OrientedObjectDetectorOptionsValidator.isTilingEnabled(options.tilingOptions());
        if (options.trackingOptions().trackerType() == TrackingOptions.TrackerType.BOX_TRACKER) {
          throw new IllegalArgumentException(
              "tracking.tracker_type=BOX_TRACKER: BoxTracker is not supported for oriented"
                  + " detection; use BOTSORT.");
        }
        if (options.trackingOptions().trackerType() == TrackingOptions.TrackerType.BOTSORT) {
          OrientedObjectDetectorOptionsValidator.validateBotsortTrackingOptions(
              options.trackingOptions());
          if (options.runningMode() == RunningMode.IMAGE) {
            throw new IllegalArgumentException(
                "tracking.tracker_type=BOTSORT requires VIDEO or LIVE_STREAM running mode;"
                    + " tracking is not available in IMAGE mode.");
          }
          if (!tilingEnabled) {
            throw new IllegalArgumentException(
                "tracking.tracker_type=BOTSORT requires tiling to be enabled; the non-tiled path"
                    + " has no tracker stage.");
          }
          if (options.numClasses() > 256) {
            throw new IllegalArgumentException(
                "tracking.tracker_type=BOTSORT requires numClasses in [1, 256].");
          }
        }
        return options;
      }
    }

    abstract BaseOptions baseOptions();

    abstract RunningMode runningMode();

    abstract int maxResults();

    abstract float scoreThreshold();

    abstract float iouThreshold();

    abstract boolean classAgnosticNms();

    abstract Layout layout();

    abstract int numClasses();

    abstract String displayNamesLocale();

    @SuppressWarnings("AutoValueImmutableFields")
    abstract List<String> categoryAllowlist();

    @SuppressWarnings("AutoValueImmutableFields")
    abstract List<String> categoryDenylist();

    abstract TilingOptions tilingOptions();

    abstract TrackingOptions trackingOptions();

    abstract Optional<ResultListener<OrientedObjectDetectorResult, MPImage>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_OrientedObjectDetector_OrientedObjectDetectorOptions.Builder()
          .setRunningMode(RunningMode.IMAGE)
          .setMaxResults(-1)
          .setScoreThreshold(0.25f)
          .setIouThreshold(0.45f)
          .setClassAgnosticNms(false)
          .setLayout(Layout.CHANNELS_FIRST)
          .setNumClasses(0)
          .setDisplayNamesLocale("en")
          .setCategoryAllowlist(Collections.emptyList())
          .setCategoryDenylist(Collections.emptyList())
          .setTilingOptions(TilingOptions.builder().build())
          .setTrackingOptions(TrackingOptions.builder().build());
    }

    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      BaseOptionsProto.BaseOptions.Builder baseOptionsBuilder =
          BaseOptionsProto.BaseOptions.newBuilder();
      baseOptionsBuilder.setUseStreamMode(runningMode() != RunningMode.IMAGE);
      baseOptionsBuilder.mergeFrom(convertBaseOptionsToProto(baseOptions()));
      OrientedObjectDetectorOptionsProto.OrientedObjectDetectorOptions.Builder taskOptionsBuilder =
          OrientedObjectDetectorOptionsProto.OrientedObjectDetectorOptions.newBuilder()
              .setBaseOptions(baseOptionsBuilder)
              .setMaxResults(maxResults())
              .setScoreThreshold(scoreThreshold())
              .setIouThreshold(iouThreshold())
              .setClassAgnosticNms(classAgnosticNms())
              .setLayout(
                  layout() == Layout.CHANNELS_LAST
                      ? OrientedObjectDetectorOptionsProto.OrientedObjectDetectorOptions.Layout
                          .CHANNELS_LAST
                      : OrientedObjectDetectorOptionsProto.OrientedObjectDetectorOptions.Layout
                          .CHANNELS_FIRST)
              .setNumClasses(numClasses())
              .setDisplayNamesLocale(displayNamesLocale());
      if (!categoryAllowlist().isEmpty()) {
        taskOptionsBuilder.addAllCategoryAllowlist(categoryAllowlist());
      }
      if (!categoryDenylist().isEmpty()) {
        taskOptionsBuilder.addAllCategoryDenylist(categoryDenylist());
      }
      OrientedObjectDetectorOptionsProto.OrientedObjectDetectorOptions.TilingOptions.Builder
          tilingBuilder =
              OrientedObjectDetectorOptionsProto.OrientedObjectDetectorOptions.TilingOptions
                  .newBuilder()
                  .setTileRows(tilingOptions().tileRows())
                  .setTileCols(tilingOptions().tileCols())
                  .setTileOverlapFraction(tilingOptions().tileOverlapFraction())
                  .setTileLocalNmsIouThreshold(tilingOptions().tileLocalNmsIouThreshold())
                  .setMaxDetectionsAfterTileNms(tilingOptions().maxDetectionsAfterTileNms());
      for (TilingOptions.TileRect tileRect : tilingOptions().explicitTiles()) {
        tilingBuilder.addExplicitTiles(
            OrientedObjectDetectorOptionsProto.OrientedObjectDetectorOptions.TilingOptions.TileRect
                .newBuilder()
                .setXCenter(tileRect.xCenter())
                .setYCenter(tileRect.yCenter())
                .setWidth(tileRect.width())
                .setHeight(tileRect.height())
                .build());
      }
      taskOptionsBuilder.setTiling(tilingBuilder);

      OrientedObjectDetectorOptionsProto.OrientedObjectDetectorOptions.TrackingOptions.Builder
          trackingBuilder =
              OrientedObjectDetectorOptionsProto.OrientedObjectDetectorOptions.TrackingOptions
                  .newBuilder()
                  .setTrackerType(
                      trackingOptions().trackerType() == TrackingOptions.TrackerType.BOTSORT
                          ? OrientedObjectDetectorOptionsProto.OrientedObjectDetectorOptions
                              .TrackingOptions.TrackerType.BOTSORT
                          : trackingOptions().trackerType()
                                  == TrackingOptions.TrackerType.BOX_TRACKER
                              ? OrientedObjectDetectorOptionsProto.OrientedObjectDetectorOptions
                                  .TrackingOptions.TrackerType.BOX_TRACKER
                              : OrientedObjectDetectorOptionsProto.OrientedObjectDetectorOptions
                                  .TrackingOptions.TrackerType.TRACKER_UNSPECIFIED)
                  .setTrackHighThreshold(trackingOptions().trackHighThreshold())
                  .setTrackLowThreshold(trackingOptions().trackLowThreshold())
                  .setNewTrackThreshold(trackingOptions().newTrackThreshold())
                  .setTrackBuffer(trackingOptions().trackBuffer())
                  .setMatchThreshold(trackingOptions().matchThreshold())
                  .setEnableGmc(trackingOptions().enableGmc())
                  .setNominalFrameRate(trackingOptions().nominalFrameRate());
      taskOptionsBuilder.setTracking(trackingBuilder);
      return CalculatorOptions.newBuilder()
          .setExtension(
              OrientedObjectDetectorOptionsProto.OrientedObjectDetectorOptions.ext,
              taskOptionsBuilder.build())
          .build();
    }
  }

  private void validateImageProcessingOptions(ImageProcessingOptions imageProcessingOptions) {
    if (imageProcessingOptions.regionOfInterest().isPresent()) {
      if (tilingEnabled) {
        throw new IllegalArgumentException("tiling and ROI are mutually exclusive");
      }
      throw new IllegalArgumentException(
          "OrientedObjectDetector doesn't support region-of-interest.");
    }
    if (tilingEnabled && imageProcessingOptions.rotationDegrees() != 0) {
      throw new IllegalArgumentException("tiling does not support rotation_degrees");
    }
  }

}

final class OrientedObjectDetectorOptionsValidator {
  static boolean isTilingEnabled(
      OrientedObjectDetector.OrientedObjectDetectorOptions.TilingOptions tilingOptions) {
    return (long) tilingOptions.tileRows() * tilingOptions.tileCols() > 1
        || !tilingOptions.explicitTiles().isEmpty();
  }

  static void validateTilingOptions(
      OrientedObjectDetector.OrientedObjectDetectorOptions.TilingOptions tilingOptions) {
    if (!tilingOptions.explicitTiles().isEmpty() && tilingOptions.tileOverlapFraction() != 0.0f) {
      throw new IllegalArgumentException(
          "tiling.tile_overlap_fraction is ignored with tiling.explicit_tiles; do not set both");
    }
    if (tilingOptions.tileRows() < 0 || tilingOptions.tileCols() < 0) {
      throw new IllegalArgumentException("tiling.tile_rows and tiling.tile_cols must be >= 0.");
    }
    if ((long) tilingOptions.tileRows() * tilingOptions.tileCols() > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "tiling.tile_rows * tiling.tile_cols must be <= 2147483647.");
    }
    if (!tilingOptions.explicitTiles().isEmpty()
        && (tilingOptions.tileRows() > 1 || tilingOptions.tileCols() > 1)) {
      throw new IllegalArgumentException(
          "tiling.explicit_tiles is mutually exclusive with a tiling.tile_rows / tiling.tile_cols"
              + " grid (> 1).");
    }
    if (tilingOptions.tileOverlapFraction() < 0.0f || tilingOptions.tileOverlapFraction() >= 1.0f) {
      throw new IllegalArgumentException("tiling.tile_overlap_fraction must be in [0.0, 1.0).");
    }
  }

  static void validateBotsortTrackingOptions(
      OrientedObjectDetector.OrientedObjectDetectorOptions.TrackingOptions trackingOptions) {
    if (!Float.isFinite(trackingOptions.trackHighThreshold())
        || trackingOptions.trackHighThreshold() < 0.0f
        || trackingOptions.trackHighThreshold() > 1.0f
        || !Float.isFinite(trackingOptions.trackLowThreshold())
        || trackingOptions.trackLowThreshold() < 0.0f
        || trackingOptions.trackLowThreshold() > 1.0f
        || !Float.isFinite(trackingOptions.newTrackThreshold())
        || trackingOptions.newTrackThreshold() < 0.0f
        || trackingOptions.newTrackThreshold() > 1.0f
        || !Float.isFinite(trackingOptions.matchThreshold())
        || trackingOptions.matchThreshold() < 0.0f
        || trackingOptions.matchThreshold() > 1.0f) {
      throw new IllegalArgumentException("tracking thresholds must be finite and in [0, 1].");
    }
    if (trackingOptions.trackLowThreshold() > trackingOptions.trackHighThreshold()) {
      throw new IllegalArgumentException(
          "tracking.track_low_threshold must be <= tracking.track_high_threshold.");
    }
    if (trackingOptions.trackBuffer() < 0 || trackingOptions.trackBuffer() > 255) {
      throw new IllegalArgumentException("tracking.track_buffer must be in [0, 255].");
    }
    if (trackingOptions.nominalFrameRate() < 1 || trackingOptions.nominalFrameRate() > 255) {
      throw new IllegalArgumentException("tracking.nominal_frame_rate must be in [1, 255].");
    }
    if (Math.floor(
            (double) trackingOptions.nominalFrameRate() / 30.0 * trackingOptions.trackBuffer())
        > 255.0) {
      throw new IllegalArgumentException("effective lost-track window must be <= 255 frames.");
    }
  }

  private OrientedObjectDetectorOptionsValidator() {}
}
