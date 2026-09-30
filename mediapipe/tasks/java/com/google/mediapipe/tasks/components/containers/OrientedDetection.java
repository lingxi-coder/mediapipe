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

package com.google.mediapipe.tasks.components.containers;

import com.google.auto.value.AutoValue;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/** Represents one oriented object detection with pixel-unit geometry. */
@AutoValue
public abstract class OrientedDetection {
  private static final int DEFAULT_CATEGORY_INDEX = -1;

  private static final class PixelOrientedBox {
    private final float width;
    private final float height;
    private final float rotation;
    private final double area;

    private PixelOrientedBox(float width, float height, float rotation, double area) {
      this.width = width;
      this.height = height;
      this.rotation = rotation;
      this.area = area;
    }
  }

  /** Creates an {@link OrientedDetection} instance. */
  public static OrientedDetection create(
      List<Category> categories,
      float centerX,
      float centerY,
      float width,
      float height,
      float rotation,
      Optional<String> trackId) {
    return new AutoValue_OrientedDetection(
        Collections.unmodifiableList(categories),
        centerX,
        centerY,
        width,
        height,
        rotation,
        trackId);
  }

  /** Creates an {@link OrientedDetection} instance from an oriented detection protobuf message. */
  public static OrientedDetection createFromProto(
      com.google.mediapipe.formats.proto.OrientedDetectionProto.OrientedDetection detectionProto,
      int imageWidth,
      int imageHeight) {
    List<Category> categories = new ArrayList<>();
    for (int idx = 0; idx < detectionProto.getScoreCount(); ++idx) {
      categories.add(
          Category.create(
              detectionProto.getScore(idx),
              detectionProto.getLabelIdCount() > idx
                  ? detectionProto.getLabelId(idx)
                  : DEFAULT_CATEGORY_INDEX,
              detectionProto.getLabelCount() > idx ? detectionProto.getLabel(idx) : "",
              detectionProto.getDisplayNameCount() > idx
                  ? detectionProto.getDisplayName(idx)
                  : ""));
    }
    Optional<String> trackId =
        detectionProto.hasTrackId() ? Optional.of(detectionProto.getTrackId()) : Optional.empty();
    PixelOrientedBox pixelBox = fitPixelOrientedBox(detectionProto, imageWidth, imageHeight);
    return create(
        categories,
        detectionProto.getCx() * imageWidth,
        detectionProto.getCy() * imageHeight,
        pixelBox.width,
        pixelBox.height,
        pixelBox.rotation,
        trackId);
  }

  private static PixelOrientedBox fitPixelOrientedBox(
      com.google.mediapipe.formats.proto.OrientedDetectionProto.OrientedDetection detectionProto,
      int imageWidth,
      int imageHeight) {
    double cosRotation = Math.cos(detectionProto.getRotation());
    double sinRotation = Math.sin(detectionProto.getRotation());
    double widthEdgeX = detectionProto.getWidth() * cosRotation * imageWidth;
    double widthEdgeY = detectionProto.getWidth() * sinRotation * imageHeight;
    double heightEdgeX = -detectionProto.getHeight() * sinRotation * imageWidth;
    double heightEdgeY = detectionProto.getHeight() * cosRotation * imageHeight;
    double centerX = detectionProto.getCx() * imageWidth;
    double centerY = detectionProto.getCy() * imageHeight;
    double[] cornerX = {
      centerX + (widthEdgeX + heightEdgeX) * 0.5,
      centerX + (widthEdgeX - heightEdgeX) * 0.5,
      centerX - (widthEdgeX + heightEdgeX) * 0.5,
      centerX - (widthEdgeX - heightEdgeX) * 0.5
    };
    double[] cornerY = {
      centerY + (widthEdgeY + heightEdgeY) * 0.5,
      centerY + (widthEdgeY - heightEdgeY) * 0.5,
      centerY - (widthEdgeY + heightEdgeY) * 0.5,
      centerY - (widthEdgeY - heightEdgeY) * 0.5
    };

    PixelOrientedBox best = fitAlongEdge(cornerX, cornerY, widthEdgeX, widthEdgeY);
    PixelOrientedBox heightAligned = fitAlongEdge(cornerX, cornerY, heightEdgeX, heightEdgeY);
    double areaTolerance = best == null ? 0.0 : Math.max(1.0, best.area) * 1e-6;
    if (heightAligned != null && (best == null || heightAligned.area < best.area - areaTolerance)) {
      best = heightAligned;
    }
    if (best != null) {
      return best;
    }

    return new PixelOrientedBox(
        0.0f, 0.0f, (float) Math.atan2(imageHeight * sinRotation, imageWidth * cosRotation), 0.0);
  }

  private static PixelOrientedBox fitAlongEdge(
      double[] cornerX, double[] cornerY, double edgeX, double edgeY) {
    double edgeLength = Math.hypot(edgeX, edgeY);
    if (edgeLength <= Math.ulp(1.0f)) {
      return null;
    }

    double axisX = edgeX / edgeLength;
    double axisY = edgeY / edgeLength;
    double normalX = -axisY;
    double normalY = axisX;
    double minAxis = Double.POSITIVE_INFINITY;
    double maxAxis = Double.NEGATIVE_INFINITY;
    double minNormal = Double.POSITIVE_INFINITY;
    double maxNormal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < cornerX.length; ++i) {
      double axisProjection = cornerX[i] * axisX + cornerY[i] * axisY;
      double normalProjection = cornerX[i] * normalX + cornerY[i] * normalY;
      minAxis = Math.min(minAxis, axisProjection);
      maxAxis = Math.max(maxAxis, axisProjection);
      minNormal = Math.min(minNormal, normalProjection);
      maxNormal = Math.max(maxNormal, normalProjection);
    }

    double width = maxAxis - minAxis;
    double height = maxNormal - minNormal;
    return new PixelOrientedBox(
        (float) width, (float) height, (float) Math.atan2(axisY, axisX), width * height);
  }

  /** A list of {@link Category} objects. */
  public abstract List<Category> categories();

  /** The x coordinate of the box center in pixels. */
  public abstract float centerX();

  /** The y coordinate of the box center in pixels. */
  public abstract float centerY();

  /** The box width in pixels. */
  public abstract float width();

  /** The box height in pixels. */
  public abstract float height();

  /** The counter-clockwise box rotation in radians. */
  public abstract float rotation();

  /** An optional persistent tracking ID associated with the detection. */
  public abstract Optional<String> trackId();
}
