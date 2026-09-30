/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {OrientedDetection as OrientedDetectionProto} from '../../../../framework/formats/oriented_detection_pb';
import {OrientedDetection} from '../../../../tasks/web/components/containers/oriented_detection_result';

const DEFAULT_CATEGORY_INDEX = -1;
const FLOAT32_EPSILON = 1.1920928955078125e-7;

interface PixelOrientedBox {
  width: number;
  height: number;
  rotation: number;
  area: number;
}

function fitAlongEdge(
  corners: Array<[number, number]>,
  edgeX: number,
  edgeY: number,
): PixelOrientedBox|undefined {
  const edgeLength = Math.hypot(edgeX, edgeY);
  if (edgeLength <= FLOAT32_EPSILON) {
    return undefined;
  }

  const axisX = edgeX / edgeLength;
  const axisY = edgeY / edgeLength;
  const normalX = -axisY;
  const normalY = axisX;
  let minAxis = Number.POSITIVE_INFINITY;
  let maxAxis = Number.NEGATIVE_INFINITY;
  let minNormal = Number.POSITIVE_INFINITY;
  let maxNormal = Number.NEGATIVE_INFINITY;
  for (const [x, y] of corners) {
    const axisProjection = x * axisX + y * axisY;
    const normalProjection = x * normalX + y * normalY;
    minAxis = Math.min(minAxis, axisProjection);
    maxAxis = Math.max(maxAxis, axisProjection);
    minNormal = Math.min(minNormal, normalProjection);
    maxNormal = Math.max(maxNormal, normalProjection);
  }

  const width = maxAxis - minAxis;
  const height = maxNormal - minNormal;
  return {
    width,
    height,
    rotation: Math.atan2(axisY, axisX),
    area: width * height,
  };
}

function fitPixelOrientedBox(
  source: OrientedDetectionProto,
  imageWidth: number,
  imageHeight: number,
): PixelOrientedBox {
  const rotation = source.getRotation() ?? 0;
  const cosRotation = Math.cos(rotation);
  const sinRotation = Math.sin(rotation);
  const widthEdgeX = (source.getWidth() ?? 0) * cosRotation * imageWidth;
  const widthEdgeY = (source.getWidth() ?? 0) * sinRotation * imageHeight;
  const heightEdgeX =
    -(source.getHeight() ?? 0) * sinRotation * imageWidth;
  const heightEdgeY =
    (source.getHeight() ?? 0) * cosRotation * imageHeight;
  const centerX = (source.getCx() ?? 0) * imageWidth;
  const centerY = (source.getCy() ?? 0) * imageHeight;
  const corners: Array<[number, number]> = [
    [centerX + (widthEdgeX + heightEdgeX) * 0.5,
     centerY + (widthEdgeY + heightEdgeY) * 0.5],
    [centerX + (widthEdgeX - heightEdgeX) * 0.5,
     centerY + (widthEdgeY - heightEdgeY) * 0.5],
    [centerX - (widthEdgeX + heightEdgeX) * 0.5,
     centerY - (widthEdgeY + heightEdgeY) * 0.5],
    [centerX - (widthEdgeX - heightEdgeX) * 0.5,
     centerY - (widthEdgeY - heightEdgeY) * 0.5],
  ];

  const widthAligned = fitAlongEdge(corners, widthEdgeX, widthEdgeY);
  const heightAligned = fitAlongEdge(corners, heightEdgeX, heightEdgeY);
  const areaTolerance = widthAligned ?
    Math.max(1, widthAligned.area) * 1e-6 :
    0;
  if (widthAligned &&
      (!heightAligned ||
       widthAligned.area <= heightAligned.area + areaTolerance)) {
    return widthAligned;
  }
  if (heightAligned) {
    return heightAligned;
  }

  return {
    width: 0,
    height: 0,
    rotation: Math.atan2(
      imageHeight * sinRotation,
      imageWidth * cosRotation,
    ),
    area: 0,
  };
}

/** Converts an OrientedDetection proto into a pixel-space object. */
export function convertFromOrientedDetectionProto(
  source: OrientedDetectionProto,
  imageWidth: number,
  imageHeight: number,
): OrientedDetection {
  const scores = source.getScoreList();
  const indexes = source.getLabelIdList();
  const labels = source.getLabelList();
  const displayNames = source.getDisplayNameList();
  const pixelBox = fitPixelOrientedBox(source, imageWidth, imageHeight);

  const detection: OrientedDetection = {
    categories: [],
    cx: (source.getCx() ?? 0) * imageWidth,
    cy: (source.getCy() ?? 0) * imageHeight,
    width: pixelBox.width,
    height: pixelBox.height,
    rotation: pixelBox.rotation,
  };

  for (let i = 0; i < scores.length; ++i) {
    detection.categories.push({
      score: scores[i],
      index: indexes[i] ?? DEFAULT_CATEGORY_INDEX,
      categoryName: labels[i] ?? '',
      displayName: displayNames[i] ?? '',
    });
  }

  if (source.hasTrackId()) {
    detection.trackId = source.getTrackId();
  }

  return detection;
}
