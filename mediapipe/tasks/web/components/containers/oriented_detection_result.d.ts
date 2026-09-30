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

import {Category} from '../../../../tasks/web/components/containers/category';

/** Represents one oriented detection by a detection task. */
export declare interface OrientedDetection {
  /** A list of `Category` objects. */
  categories: Category[];

  /** The center x coordinate of the detected box, in pixels. */
  cx: number;

  /** The center y coordinate of the detected box, in pixels. */
  cy: number;

  /** The width of the detected box, in pixels. */
  width: number;

  /** The height of the detected box, in pixels. */
  height: number;

  /** The counter-clockwise box rotation in radians. */
  rotation: number;

  /** Optional persistent track ID emitted by tracking. */
  trackId?: string;
}

/** Oriented detection results of a model. */
export declare interface OrientedDetectionResult {
  /** A list of oriented detections. */
  detections: OrientedDetection[];
}
