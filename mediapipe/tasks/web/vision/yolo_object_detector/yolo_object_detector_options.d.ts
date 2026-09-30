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

import {BaseOptions} from '../../../../tasks/web/core/task_runner_options';
import {VisionTaskOptions} from '../../../../tasks/web/vision/core/vision_task_options';

/** Output tensor layout of the YOLO detect head. */
export declare type YoloObjectDetectorLayout =
  'CHANNELS_FIRST' | 'CHANNELS_LAST';

/** Tracker selection for the tiled YOLO video path. */
export declare type YoloTrackingTrackerType = 'BOX_TRACKER' | 'BOTSORT';

/** A frame-normalized tile given by its center point and size. */
export declare interface YoloTileRect {
  xCenter?: number | undefined;
  yCenter?: number | undefined;
  width?: number | undefined;
  height?: number | undefined;
}

/** Static tiling configuration for the YOLO object detector. */
export declare interface YoloTilingOptions {
  tileRows?: number | undefined;
  tileCols?: number | undefined;
  tileOverlapFraction?: number | undefined;
  explicitTiles?: YoloTileRect[] | undefined;
  tileLocalNmsIouThreshold?: number | undefined;
  maxDetectionsAfterTileNms?: number | undefined;
  enableMotionScheduling?: boolean | undefined;
  maxScheduledTiles?: number | undefined;
}

/** Tracker configuration for the tiled YOLO video path. */
export declare interface YoloTrackingOptions {
  trackerType?: YoloTrackingTrackerType | undefined;
  trackHighThreshold?: number | undefined;
  trackLowThreshold?: number | undefined;
  newTrackThreshold?: number | undefined;
  trackBuffer?: number | undefined;
  matchThreshold?: number | undefined;
  enableGmc?: boolean | undefined;
  nominalFrameRate?: number | undefined;
}

/** Options to configure the MediaPipe YOLO object detector task. */
export declare interface YoloObjectDetectorOptions extends VisionTaskOptions {
  /** Options to configure model asset loading. */
  baseOptions: BaseOptions;

  /** The locale to use for display names specified through model metadata. */
  displayNamesLocale?: string | undefined;

  /** The maximum number of top-scored detection results to return. */
  maxResults?: number | undefined;

  /** Results below this value are rejected. */
  scoreThreshold?: number | undefined;

  /** Allowlist of category names, mutually exclusive with `categoryDenylist`. */
  categoryAllowlist?: string[] | undefined;

  /** Denylist of category names, mutually exclusive with `categoryAllowlist`. */
  categoryDenylist?: string[] | undefined;

  /** IoU threshold for non-maximum suppression. */
  iouThreshold?: number | undefined;

  /** Output tensor layout of the YOLO detect head. */
  layout?: YoloObjectDetectorLayout | undefined;

  /**
   * Number of classes for the detect head.
   *
   * This is required for Web because the bundled model-only constructors are
   * intentionally not exposed for this custom task.
   */
  numClasses: number;

  /** Static tiling configuration. Defaults to disabled (1x1). */
  tiling?: YoloTilingOptions | undefined;

  /** Tracker configuration for the tiled video path. */
  tracking?: YoloTrackingOptions | undefined;
}
