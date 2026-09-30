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

import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {CalculatorOptions} from '../../../../framework/calculator_options_pb';
import {OrientedDetection as OrientedDetectionProto} from '../../../../framework/formats/oriented_detection_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {OrientedObjectDetectorOptions as OrientedObjectDetectorOptionsProto} from '../../../../tasks/cc/vision/oriented_object_detector/proto/oriented_object_detector_options_pb';
import {convertFromOrientedDetectionProto} from '../../../../tasks/web/components/processors/oriented_detection_result';
import {assertCustomVisionWasmGraph} from '../../../../tasks/web/vision/core/custom_vision_wasm_fileset';
import type {CustomVisionWasmFileset} from '../../../../tasks/web/vision/core/custom_vision_wasm_fileset';
import {ImageProcessingOptions} from '../../../../tasks/web/vision/core/image_processing_options';
import type {RunningMode} from '../../../../tasks/web/vision/core/vision_task_options';
import {
  VisionGraphRunner,
  VisionTaskRunner,
} from '../../../../tasks/web/vision/core/vision_task_runner';
import {
  ImageSource,
  WasmModule,
  getImageSourceSize,
} from '../../../../web/graph_runner/graph_runner';

import type {
  OrientedObjectDetectorLayout,
  OrientedObjectDetectorOptions,
  OrientedTileRect,
  OrientedTilingOptions,
  OrientedTrackingOptions,
  OrientedTrackingTrackerType,
} from './oriented_object_detector_options';
import type {OrientedObjectDetectorResult} from './oriented_object_detector_result';

const IMAGE_STREAM = 'input_frame_gpu';
const NORM_RECT_STREAM = 'norm_rect';
const ORIENTED_DETECTIONS_STREAM = 'oriented_detections';
export const ORIENTED_OBJECT_DETECTOR_GRAPH =
  'mediapipe.tasks.vision.oriented_object_detector.OrientedObjectDetectorGraph';

const DEFAULT_MAX_RESULTS = -1;
const DEFAULT_SCORE_THRESHOLD = 0.25;
const DEFAULT_IOU_THRESHOLD = 0.45;
const DEFAULT_CLASS_AGNOSTIC_NMS = false;
const DEFAULT_LAYOUT = 'CHANNELS_FIRST';
const DEFAULT_NUM_CLASSES = 0;
const DEFAULT_TILE_ROWS = 1;
const DEFAULT_TILE_COLS = 1;
const DEFAULT_TILE_OVERLAP_FRACTION = 0;
const DEFAULT_TILE_LOCAL_NMS_IOU_THRESHOLD = 0;
const DEFAULT_MAX_DETECTIONS_AFTER_TILE_NMS = 0;
const DEFAULT_TRACKER_TYPE = 'TRACKER_UNSPECIFIED';
const DEFAULT_TRACK_HIGH_THRESHOLD = 0.6;
const DEFAULT_TRACK_LOW_THRESHOLD = 0.1;
const DEFAULT_NEW_TRACK_THRESHOLD = 0.7;
const DEFAULT_TRACK_BUFFER = 30;
const DEFAULT_MATCH_THRESHOLD = 0.7;
const DEFAULT_ENABLE_GMC = false;
const DEFAULT_NOMINAL_FRAME_RATE = 30;

type OrientedObjectDetectorOptionsUpdate =
  Partial<OrientedObjectDetectorOptions>;

export type {OrientedObjectDetectorOptions}
  from './oriented_object_detector_options';
export type {
  Category,
  OrientedDetection,
  OrientedObjectDetectorResult,
} from './oriented_object_detector_result';
export {type ImageSource};

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

function createOrientedTilingDefaults():
  OrientedObjectDetectorOptionsProto.TilingOptions {
  return new OrientedObjectDetectorOptionsProto.TilingOptions();
}

function createOrientedTrackingDefaults():
  OrientedObjectDetectorOptionsProto.TrackingOptions {
  return new OrientedObjectDetectorOptionsProto.TrackingOptions();
}

function getOrientedTrackerTypeProto(
  trackerType: OrientedTrackingTrackerType,
): OrientedObjectDetectorOptionsProto.TrackingOptions.TrackerTypeMap[
  keyof OrientedObjectDetectorOptionsProto.TrackingOptions.TrackerTypeMap
] {
  switch (trackerType) {
    case 'BOX_TRACKER':
      return OrientedObjectDetectorOptionsProto.TrackingOptions.TrackerType
        .BOX_TRACKER;
    case 'BOTSORT':
      return OrientedObjectDetectorOptionsProto.TrackingOptions.TrackerType
        .BOTSORT;
    case 'TRACKER_UNSPECIFIED':
    default:
      return OrientedObjectDetectorOptionsProto.TrackingOptions.TrackerType
        .TRACKER_UNSPECIFIED;
  }
}

function getOrientedLayoutProto(
  layout: OrientedObjectDetectorLayout,
): OrientedObjectDetectorOptionsProto.LayoutMap[
  keyof OrientedObjectDetectorOptionsProto.LayoutMap
] {
  switch (layout) {
    case 'CHANNELS_LAST':
      return OrientedObjectDetectorOptionsProto.Layout.CHANNELS_LAST;
    case 'CHANNELS_FIRST':
    default:
      return OrientedObjectDetectorOptionsProto.Layout.CHANNELS_FIRST;
  }
}

function isOrientedTilingEnabled(
  tiling: OrientedObjectDetectorOptionsProto.TilingOptions | undefined,
): boolean {
  if (!tiling) {
    return false;
  }
  const tileRows = tiling.getTileRows() ?? DEFAULT_TILE_ROWS;
  const tileCols = tiling.getTileCols() ?? DEFAULT_TILE_COLS;
  return (
    tileRows * tileCols > 1 ||
    tiling.getExplicitTilesList().length > 0
  );
}

function validateTiledImageProcessingOptions(
  imageProcessingOptions?: ImageProcessingOptions,
): void {
  if (!imageProcessingOptions) {
    return;
  }
  if (imageProcessingOptions.regionOfInterest) {
    throw new Error('tiling and ROI are mutually exclusive');
  }
  if ((imageProcessingOptions.rotationDegrees ?? 0) !== 0) {
    throw new Error('tiling does not support rotation_degrees');
  }
}

function validateOrientedBotsortOptions(
  tracking: OrientedObjectDetectorOptionsProto.TrackingOptions,
): void {
  const trackHighThreshold =
    tracking.getTrackHighThreshold() ?? DEFAULT_TRACK_HIGH_THRESHOLD;
  const trackLowThreshold =
    tracking.getTrackLowThreshold() ?? DEFAULT_TRACK_LOW_THRESHOLD;
  const newTrackThreshold =
    tracking.getNewTrackThreshold() ?? DEFAULT_NEW_TRACK_THRESHOLD;
  const matchThreshold =
    tracking.getMatchThreshold() ?? DEFAULT_MATCH_THRESHOLD;
  const trackBuffer = tracking.getTrackBuffer() ?? DEFAULT_TRACK_BUFFER;
  const nominalFrameRate =
    tracking.getNominalFrameRate() ?? DEFAULT_NOMINAL_FRAME_RATE;
  const values = [
    trackHighThreshold,
    trackLowThreshold,
    newTrackThreshold,
    matchThreshold,
  ];
  if (values.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new Error('tracking thresholds must be finite and in [0, 1].');
  }
  if (trackLowThreshold > trackHighThreshold) {
    throw new Error(
      'tracking.track_low_threshold must be <= tracking.track_high_threshold.',
    );
  }
  if (trackBuffer < 0 || trackBuffer > 255) {
    throw new Error('tracking.track_buffer must be in [0, 255].');
  }
  if (
    nominalFrameRate < 1 || nominalFrameRate > 255
  ) {
    throw new Error('tracking.nominal_frame_rate must be in [1, 255].');
  }
  if (
    Math.floor(
      (nominalFrameRate / 30) * trackBuffer,
    ) > 255
  ) {
    throw new Error('effective lost-track window must be <= 255 frames.');
  }
}

function validateOrientedOptions(
  optionsProto: OrientedObjectDetectorOptionsProto,
  runningMode: RunningMode,
): void {
  if (optionsProto.getMaxResults() === 0) {
    throw new Error('Invalid `max_results` option: value must be != 0');
  }
  if (
    optionsProto.getCategoryAllowlistList().length > 0 &&
    optionsProto.getCategoryDenylistList().length > 0
  ) {
    throw new Error(
      '`category_allowlist` and `category_denylist` are mutually exclusive options.',
    );
  }

  const tiling =
    optionsProto.getTiling() ?? createOrientedTilingDefaults();
  const tileRows = tiling.getTileRows() ?? DEFAULT_TILE_ROWS;
  const tileCols = tiling.getTileCols() ?? DEFAULT_TILE_COLS;
  const tileOverlapFraction =
    tiling.getTileOverlapFraction() ?? DEFAULT_TILE_OVERLAP_FRACTION;
  if (
    tiling.getExplicitTilesList().length > 0 &&
    tileOverlapFraction !== 0
  ) {
    throw new Error(
      'tiling.tile_overlap_fraction is ignored with tiling.explicit_tiles; do not set both',
    );
  }
  if (tileRows < 0 || tileCols < 0) {
    throw new Error('tiling.tile_rows and tiling.tile_cols must be >= 0.');
  }
  if (
    tiling.getExplicitTilesList().length > 0 &&
    (tileRows > 1 || tileCols > 1)
  ) {
    throw new Error(
      'tiling.explicit_tiles is mutually exclusive with a tiling.tile_rows / tiling.tile_cols grid (> 1).',
    );
  }
  if (
    tileOverlapFraction < 0 || tileOverlapFraction >= 1
  ) {
    throw new Error(
      'tiling.tile_overlap_fraction must be in [0.0, 1.0).',
    );
  }
  const numClasses = optionsProto.getNumClasses() ?? DEFAULT_NUM_CLASSES;
  if (numClasses <= 0) {
    throw new Error('num_classes must be set in OrientedObjectDetectorOptions');
  }

  const tracker =
    optionsProto.getTracking() ?? createOrientedTrackingDefaults();
  if (
    (tracker.getTrackerType() ??
      OrientedObjectDetectorOptionsProto.TrackingOptions.TrackerType
        .TRACKER_UNSPECIFIED) ===
    OrientedObjectDetectorOptionsProto.TrackingOptions.TrackerType.BOX_TRACKER
  ) {
    throw new Error(
      'tracking.tracker_type=BOX_TRACKER: BoxTracker is not supported for oriented detection; use BOTSORT.',
    );
  }
  if (
    (tracker.getTrackerType() ??
      OrientedObjectDetectorOptionsProto.TrackingOptions.TrackerType
        .TRACKER_UNSPECIFIED) ===
    OrientedObjectDetectorOptionsProto.TrackingOptions.TrackerType.BOTSORT
  ) {
    validateOrientedBotsortOptions(tracker);
    if (runningMode === 'IMAGE') {
      throw new Error(
        'tracking.tracker_type=BOTSORT requires VIDEO or LIVE_STREAM running mode; tracking is not available in IMAGE mode.',
      );
    }
    if (!isOrientedTilingEnabled(tiling)) {
      throw new Error(
        'tracking.tracker_type=BOTSORT requires tiling to be enabled; the non-tiled path has no tracker stage.',
      );
    }
    if (numClasses < 1 || numClasses > 256) {
      throw new Error(
        'tracking.tracker_type=BOTSORT requires num_classes in [1, 256] (BoTSORT stores the class id as uint8).',
      );
    }
  }
}

function applyOrientedTileRect(
  tileRect: OrientedTileRect,
): OrientedObjectDetectorOptionsProto.TilingOptions.TileRect {
  const tileRectProto =
    new OrientedObjectDetectorOptionsProto.TilingOptions.TileRect();
  tileRectProto.setXCenter(tileRect.xCenter ?? 0);
  tileRectProto.setYCenter(tileRect.yCenter ?? 0);
  tileRectProto.setWidth(tileRect.width ?? 0);
  tileRectProto.setHeight(tileRect.height ?? 0);
  return tileRectProto;
}

function applyOrientedTilingOptions(
  tilingProto: OrientedObjectDetectorOptionsProto.TilingOptions,
  tilingOptions: OrientedTilingOptions | undefined,
): void {
  if (tilingOptions === undefined) {
    return;
  }
  if ('tileRows' in tilingOptions) {
    tilingProto.setTileRows(tilingOptions.tileRows ?? DEFAULT_TILE_ROWS);
  }
  if ('tileCols' in tilingOptions) {
    tilingProto.setTileCols(tilingOptions.tileCols ?? DEFAULT_TILE_COLS);
  }
  if ('tileOverlapFraction' in tilingOptions) {
    tilingProto.setTileOverlapFraction(
      tilingOptions.tileOverlapFraction ?? DEFAULT_TILE_OVERLAP_FRACTION,
    );
  }
  if ('explicitTiles' in tilingOptions) {
    tilingProto.setExplicitTilesList(
      (tilingOptions.explicitTiles ?? []).map(applyOrientedTileRect),
    );
  }
  if ('tileLocalNmsIouThreshold' in tilingOptions) {
    tilingProto.setTileLocalNmsIouThreshold(
      tilingOptions.tileLocalNmsIouThreshold ??
        DEFAULT_TILE_LOCAL_NMS_IOU_THRESHOLD,
    );
  }
  if ('maxDetectionsAfterTileNms' in tilingOptions) {
    tilingProto.setMaxDetectionsAfterTileNms(
      tilingOptions.maxDetectionsAfterTileNms ??
        DEFAULT_MAX_DETECTIONS_AFTER_TILE_NMS,
    );
  }
}

function applyOrientedTrackingOptions(
  trackingProto: OrientedObjectDetectorOptionsProto.TrackingOptions,
  trackingOptions: OrientedTrackingOptions | undefined,
): void {
  if (trackingOptions === undefined) {
    return;
  }
  if ('trackerType' in trackingOptions) {
    trackingProto.setTrackerType(
      getOrientedTrackerTypeProto(
        trackingOptions.trackerType ?? DEFAULT_TRACKER_TYPE,
      ),
    );
  }
  if ('trackHighThreshold' in trackingOptions) {
    trackingProto.setTrackHighThreshold(
      trackingOptions.trackHighThreshold ?? DEFAULT_TRACK_HIGH_THRESHOLD,
    );
  }
  if ('trackLowThreshold' in trackingOptions) {
    trackingProto.setTrackLowThreshold(
      trackingOptions.trackLowThreshold ?? DEFAULT_TRACK_LOW_THRESHOLD,
    );
  }
  if ('newTrackThreshold' in trackingOptions) {
    trackingProto.setNewTrackThreshold(
      trackingOptions.newTrackThreshold ?? DEFAULT_NEW_TRACK_THRESHOLD,
    );
  }
  if ('trackBuffer' in trackingOptions) {
    trackingProto.setTrackBuffer(
      trackingOptions.trackBuffer ?? DEFAULT_TRACK_BUFFER,
    );
  }
  if ('matchThreshold' in trackingOptions) {
    trackingProto.setMatchThreshold(
      trackingOptions.matchThreshold ?? DEFAULT_MATCH_THRESHOLD,
    );
  }
  if ('enableGmc' in trackingOptions) {
    trackingProto.setEnableGmc(
      trackingOptions.enableGmc ?? DEFAULT_ENABLE_GMC,
    );
  }
  if ('nominalFrameRate' in trackingOptions) {
    trackingProto.setNominalFrameRate(
      trackingOptions.nominalFrameRate ?? DEFAULT_NOMINAL_FRAME_RATE,
    );
  }
}

function applyOrientedOptions(
  optionsProto: OrientedObjectDetectorOptionsProto,
  options: OrientedObjectDetectorOptionsUpdate,
): void {
  if ('maxResults' in options) {
    optionsProto.setMaxResults(options.maxResults ?? DEFAULT_MAX_RESULTS);
  }
  if ('scoreThreshold' in options) {
    optionsProto.setScoreThreshold(
      options.scoreThreshold ?? DEFAULT_SCORE_THRESHOLD,
    );
  }
  if ('iouThreshold' in options) {
    optionsProto.setIouThreshold(
      options.iouThreshold ?? DEFAULT_IOU_THRESHOLD,
    );
  }
  if ('classAgnosticNms' in options) {
    optionsProto.setClassAgnosticNms(
      options.classAgnosticNms ?? DEFAULT_CLASS_AGNOSTIC_NMS,
    );
  }
  if ('layout' in options) {
    optionsProto.setLayout(
      getOrientedLayoutProto(options.layout ?? DEFAULT_LAYOUT),
    );
  }
  if ('numClasses' in options) {
    optionsProto.setNumClasses(options.numClasses ?? DEFAULT_NUM_CLASSES);
  }
  if (options.displayNamesLocale !== undefined) {
    optionsProto.setDisplayNamesLocale(options.displayNamesLocale);
  } else if ('displayNamesLocale' in options) {
    optionsProto.clearDisplayNamesLocale();
  }
  if ('categoryAllowlist' in options) {
    optionsProto.setCategoryAllowlistList(options.categoryAllowlist ?? []);
  }
  if ('categoryDenylist' in options) {
    optionsProto.setCategoryDenylistList(options.categoryDenylist ?? []);
  }
  if ('tiling' in options) {
    if (options.tiling === undefined) {
      optionsProto.setTiling(createOrientedTilingDefaults());
    } else {
      const tilingProto =
        optionsProto.getTiling() ?? createOrientedTilingDefaults();
      applyOrientedTilingOptions(tilingProto, options.tiling);
      optionsProto.setTiling(tilingProto);
    }
  }
  if ('tracking' in options) {
    if (options.tracking === undefined) {
      optionsProto.setTracking(createOrientedTrackingDefaults());
    } else {
      const trackingProto =
        optionsProto.getTracking() ?? createOrientedTrackingDefaults();
      applyOrientedTrackingOptions(trackingProto, options.tracking);
      optionsProto.setTracking(trackingProto);
    }
  }
}

/**
 * Performs oriented object detection on images.
 *
 * Callers must wrap a custom Wasm binary with
 * `createCustomVisionWasmFileset()` and declare
 * `mediapipe.tasks.vision.oriented_object_detector.OrientedObjectDetectorGraph`.
 * The bundled MediaPipe vision Wasm files do not include this custom graph.
 */
export class OrientedObjectDetector extends VisionTaskRunner {
  private result: OrientedObjectDetectorResult = {detections: []};
  private options = new OrientedObjectDetectorOptionsProto();
  private imageSize: [number, number] = [0, 0];

  /**
   * Initializes the Wasm runtime and creates a new oriented object detector
   * from the provided options.
   *
   * The custom fileset must declare the oriented-object graph registration;
   * the bundled vision Wasm files returned by `FilesetResolver` are not
   * accepted.
   */
  static createFromOptions(
    wasmFileset: CustomVisionWasmFileset,
    orientedObjectDetectorOptions: OrientedObjectDetectorOptions,
  ): Promise<OrientedObjectDetector> {
    assertCustomVisionWasmGraph(
      wasmFileset,
      ORIENTED_OBJECT_DETECTOR_GRAPH,
    );
    return VisionTaskRunner.createVisionInstance(
      OrientedObjectDetector,
      wasmFileset,
      orientedObjectDetectorOptions,
    );
  }

  /** @hideconstructor */
  constructor(
    wasmModule: WasmModule,
    glCanvas?: HTMLCanvasElement | OffscreenCanvas | null,
  ) {
    super(
      new VisionGraphRunner(wasmModule, glCanvas),
      IMAGE_STREAM,
      NORM_RECT_STREAM,
      /* roiAllowed= */ false,
    );
    this.options.setBaseOptions(new BaseOptionsProto());
    this.options.setTiling(createOrientedTilingDefaults());
    this.options.setTracking(createOrientedTrackingDefaults());
  }

  protected override getTaskName(): string {
    return 'OrientedObjectDetector';
  }

  protected override get baseOptions(): BaseOptionsProto {
    return this.options.getBaseOptions()!;
  }

  protected override set baseOptions(proto: BaseOptionsProto) {
    this.options.setBaseOptions(proto);
  }

  override setOptions(
    options: OrientedObjectDetectorOptionsUpdate,
  ): Promise<void> {
    const candidate = OrientedObjectDetectorOptionsProto.deserializeBinary(
      this.options.serializeBinary(),
    );
    if (!candidate.getBaseOptions()) {
      candidate.setBaseOptions(new BaseOptionsProto());
    }
    if (!candidate.getTiling()) {
      candidate.setTiling(createOrientedTilingDefaults());
    }
    if (!candidate.getTracking()) {
      candidate.setTracking(createOrientedTrackingDefaults());
    }

    applyOrientedOptions(candidate, options);
    validateOrientedOptions(candidate, this.getRunningMode(options));

    const previous = this.options;
    this.options = candidate;
    try {
      return this.applyOptions(options).catch((error) => {
        this.options = previous;
        throw error;
      });
    } catch (error) {
      this.options = previous;
      throw error;
    }
  }

  detect(
    image: ImageSource,
    imageProcessingOptions?: ImageProcessingOptions,
  ): OrientedObjectDetectorResult {
    this.result = {detections: []};
    this.imageSize = getImageSourceSize(image);
    if (this.isTilingEnabled()) {
      validateTiledImageProcessingOptions(imageProcessingOptions);
      this.processTiledImageData(image);
    } else {
      this.processImageData(image, imageProcessingOptions);
    }
    return this.result;
  }

  detectForVideo(
    videoFrame: ImageSource,
    timestamp: number,
    imageProcessingOptions?: ImageProcessingOptions,
  ): OrientedObjectDetectorResult {
    this.result = {detections: []};
    this.imageSize = getImageSourceSize(videoFrame);
    if (this.isTilingEnabled()) {
      validateTiledImageProcessingOptions(imageProcessingOptions);
      this.processTiledVideoData(videoFrame, timestamp);
    } else {
      this.processVideoData(videoFrame, imageProcessingOptions, timestamp);
    }
    return this.result;
  }

  private getRunningMode(
    options: OrientedObjectDetectorOptionsUpdate,
  ): RunningMode {
    if ('runningMode' in options) {
      return options.runningMode ?? 'IMAGE';
    }
    return this.baseOptions.getUseStreamMode() ? 'VIDEO' : 'IMAGE';
  }

  private isTilingEnabled(): boolean {
    return isOrientedTilingEnabled(this.options.getTiling());
  }

  private processTiledImageData(image: ImageSource): void {
    if (this.baseOptions.getUseStreamMode()) {
      throw new Error(
        "Task is not initialized with image mode. 'runningMode' must be set to 'IMAGE'.",
      );
    }
    this.processTiledData(image, this.getSyntheticTimestamp());
  }

  private processTiledVideoData(
    videoFrame: ImageSource,
    timestamp: number,
  ): void {
    if (!this.baseOptions.getUseStreamMode()) {
      throw new Error(
        "Task is not initialized with video mode. 'runningMode' must be set to 'VIDEO'.",
      );
    }
    this.processTiledData(videoFrame, timestamp);
  }

  private processTiledData(imageSource: ImageSource, timestamp: number): void {
    this.startProcessing(timestamp);
    this.graphRunner.addGpuBufferAsImageToStream(
      imageSource,
      IMAGE_STREAM,
      timestamp,
    );
    this.finishProcessing(timestamp);
  }

  private addJsOrientedDetections(data: Uint8Array[]): void {
    for (const binaryProto of data) {
      const detectionProto = OrientedDetectionProto.deserializeBinary(binaryProto);
      this.result.detections.push(
        convertFromOrientedDetectionProto(
          detectionProto,
          this.imageSize[0],
          this.imageSize[1],
        ),
      );
    }
  }

  protected override refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(IMAGE_STREAM);
    if (!this.isTilingEnabled()) {
      graphConfig.addInputStream(NORM_RECT_STREAM);
    }
    graphConfig.addOutputStream(ORIENTED_DETECTIONS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
      OrientedObjectDetectorOptionsProto.ext,
      this.options,
    );

    const detectorNode = new CalculatorGraphConfig.Node();
    detectorNode.setCalculator(ORIENTED_OBJECT_DETECTOR_GRAPH);
    detectorNode.addInputStream('IMAGE:' + IMAGE_STREAM);
    if (!this.isTilingEnabled()) {
      detectorNode.addInputStream('NORM_RECT:' + NORM_RECT_STREAM);
    }
    detectorNode.addOutputStream(
      'ORIENTED_DETECTIONS:' + ORIENTED_DETECTIONS_STREAM,
    );
    detectorNode.setOptions(calculatorOptions);
    graphConfig.addNode(detectorNode);

    this.graphRunner.attachProtoVectorListener(
      ORIENTED_DETECTIONS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsOrientedDetections(binaryProto);
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      ORIENTED_DETECTIONS_STREAM,
      (timestamp) => {
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    this.setGraph(
      new Uint8Array(graphConfig.serializeBinary()),
      /* isBinary= */ true,
    );
  }
}
