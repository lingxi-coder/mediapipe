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
import {Detection as DetectionProto} from '../../../../framework/formats/detection_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {YoloObjectDetectorOptions as YoloObjectDetectorOptionsProto} from '../../../../tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options_pb';
import {convertFromDetectionProto} from '../../../../tasks/web/components/processors/detection_result';
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
} from '../../../../web/graph_runner/graph_runner';

import type {
  YoloObjectDetectorLayout,
  YoloObjectDetectorOptions,
  YoloTileRect,
  YoloTilingOptions,
  YoloTrackingOptions,
  YoloTrackingTrackerType,
} from './yolo_object_detector_options';
import type {YoloObjectDetectorResult} from './yolo_object_detector_result';

const IMAGE_STREAM = 'input_frame_gpu';
const NORM_RECT_STREAM = 'norm_rect';
const DETECTIONS_STREAM = 'detections';
export const YOLO_OBJECT_DETECTOR_GRAPH =
  'mediapipe.tasks.vision.yolo_object_detector.YoloObjectDetectorGraph';

const DEFAULT_MAX_RESULTS = -1;
const DEFAULT_SCORE_THRESHOLD = 0.25;
const DEFAULT_IOU_THRESHOLD = 0.45;
const DEFAULT_NUM_CLASSES = 0;
const DEFAULT_LAYOUT = 'CHANNELS_FIRST';
const DEFAULT_TILE_ROWS = 1;
const DEFAULT_TILE_COLS = 1;
const DEFAULT_TILE_OVERLAP_FRACTION = 0;
const DEFAULT_TILE_LOCAL_NMS_IOU_THRESHOLD = 0;
const DEFAULT_MAX_DETECTIONS_AFTER_TILE_NMS = 0;
const DEFAULT_ENABLE_MOTION_SCHEDULING = false;
const DEFAULT_MAX_SCHEDULED_TILES = 0;
const DEFAULT_TRACKER_TYPE = 'BOX_TRACKER';
const DEFAULT_TRACK_HIGH_THRESHOLD = 0.6;
const DEFAULT_TRACK_LOW_THRESHOLD = 0.1;
const DEFAULT_NEW_TRACK_THRESHOLD = 0.7;
const DEFAULT_TRACK_BUFFER = 30;
const DEFAULT_MATCH_THRESHOLD = 0.7;
const DEFAULT_ENABLE_GMC = false;
const DEFAULT_NOMINAL_FRAME_RATE = 30;

type YoloObjectDetectorOptionsUpdate = Partial<YoloObjectDetectorOptions>;

export type {YoloObjectDetectorOptions} from './yolo_object_detector_options';
export type {
  BoundingBox,
  Category,
  Detection,
  YoloObjectDetectorResult,
} from './yolo_object_detector_result';
export {type ImageSource};

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

function createYoloTilingDefaults():
  YoloObjectDetectorOptionsProto.TilingOptions {
  return new YoloObjectDetectorOptionsProto.TilingOptions();
}

function createYoloTrackingDefaults():
  YoloObjectDetectorOptionsProto.TrackingOptions {
  return new YoloObjectDetectorOptionsProto.TrackingOptions();
}

function getYoloTrackerTypeProto(
  trackerType: YoloTrackingTrackerType,
): YoloObjectDetectorOptionsProto.TrackingOptions.TrackerTypeMap[
  keyof YoloObjectDetectorOptionsProto.TrackingOptions.TrackerTypeMap
] {
  switch (trackerType) {
    case 'BOTSORT':
      return YoloObjectDetectorOptionsProto.TrackingOptions.TrackerType
        .BOTSORT;
    case 'BOX_TRACKER':
    default:
      return YoloObjectDetectorOptionsProto.TrackingOptions.TrackerType
        .BOX_TRACKER;
  }
}

function getYoloLayoutProto(
  layout: YoloObjectDetectorLayout,
): YoloObjectDetectorOptionsProto.LayoutMap[
  keyof YoloObjectDetectorOptionsProto.LayoutMap
] {
  switch (layout) {
    case 'CHANNELS_LAST':
      return YoloObjectDetectorOptionsProto.Layout.CHANNELS_LAST;
    case 'CHANNELS_FIRST':
    default:
      return YoloObjectDetectorOptionsProto.Layout.CHANNELS_FIRST;
  }
}

function isYoloTilingEnabled(
  tiling: YoloObjectDetectorOptionsProto.TilingOptions | undefined,
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

function validateBotsortThreshold(value: number, field: string): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${field} must be finite and in [0, 1].`);
  }
}

function validateYoloBotsortOptions(
  tracking: YoloObjectDetectorOptionsProto.TrackingOptions,
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
  validateBotsortThreshold(
    trackHighThreshold,
    'tracking.track_high_threshold',
  );
  validateBotsortThreshold(
    trackLowThreshold,
    'tracking.track_low_threshold',
  );
  validateBotsortThreshold(
    newTrackThreshold,
    'tracking.new_track_threshold',
  );
  validateBotsortThreshold(
    matchThreshold,
    'tracking.match_threshold',
  );
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
    throw new Error(
      'floor(nominal_frame_rate / 30 * track_buffer) must be <= 255.',
    );
  }
}

function validateYoloOptions(
  optionsProto: YoloObjectDetectorOptionsProto,
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

  const tiling = optionsProto.getTiling() ?? createYoloTilingDefaults();
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
  if (
    !Number.isInteger(tileRows) || !Number.isInteger(tileCols) ||
    tileRows < 0 || tileCols < 0 ||
    tileRows > 2147483647 || tileCols > 2147483647
  ) {
    throw new Error(
      'tiling.tile_rows and tiling.tile_cols must be non-negative int32 values.',
    );
  }
  if (tileRows * tileCols > 2147483647) {
    throw new Error('tiling grid tile count must be <= 2147483647.');
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
    throw new Error(
      'num_classes must be set in YoloObjectDetectorOptions (metadata-derived num_classes is a future enhancement)',
    );
  }
  if (
    (tiling.getEnableMotionScheduling() ?? DEFAULT_ENABLE_MOTION_SCHEDULING) &&
    runningMode === 'IMAGE'
  ) {
    throw new Error(
      'tiling.enable_motion_scheduling requires VIDEO or LIVE_STREAM running mode; motion scheduling is meaningless in IMAGE mode.',
    );
  }

  const tracking = optionsProto.getTracking() ?? createYoloTrackingDefaults();
  if (
    (tracking.getTrackerType() ??
      YoloObjectDetectorOptionsProto.TrackingOptions.TrackerType.BOX_TRACKER) ===
    YoloObjectDetectorOptionsProto.TrackingOptions.TrackerType.BOTSORT
  ) {
    validateYoloBotsortOptions(tracking);
    if (runningMode === 'IMAGE') {
      throw new Error(
        'tracking.tracker_type=BOTSORT requires VIDEO or LIVE_STREAM running mode; tracking is not available in IMAGE mode.',
      );
    }
    if (!isYoloTilingEnabled(tiling)) {
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

function applyYoloTileRect(
  tileRect: YoloTileRect,
): YoloObjectDetectorOptionsProto.TilingOptions.TileRect {
  const tileRectProto =
    new YoloObjectDetectorOptionsProto.TilingOptions.TileRect();
  tileRectProto.setXCenter(tileRect.xCenter ?? 0);
  tileRectProto.setYCenter(tileRect.yCenter ?? 0);
  tileRectProto.setWidth(tileRect.width ?? 0);
  tileRectProto.setHeight(tileRect.height ?? 0);
  return tileRectProto;
}

function applyYoloTilingOptions(
  tilingProto: YoloObjectDetectorOptionsProto.TilingOptions,
  tilingOptions: YoloTilingOptions | undefined,
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
      (tilingOptions.explicitTiles ?? []).map(applyYoloTileRect),
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
  if ('enableMotionScheduling' in tilingOptions) {
    tilingProto.setEnableMotionScheduling(
      tilingOptions.enableMotionScheduling ?? DEFAULT_ENABLE_MOTION_SCHEDULING,
    );
  }
  if ('maxScheduledTiles' in tilingOptions) {
    tilingProto.setMaxScheduledTiles(
      tilingOptions.maxScheduledTiles ?? DEFAULT_MAX_SCHEDULED_TILES,
    );
  }
}

function applyYoloTrackingOptions(
  trackingProto: YoloObjectDetectorOptionsProto.TrackingOptions,
  trackingOptions: YoloTrackingOptions | undefined,
): void {
  if (trackingOptions === undefined) {
    return;
  }
  if ('trackerType' in trackingOptions) {
    trackingProto.setTrackerType(
      getYoloTrackerTypeProto(
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

function applyYoloOptions(
  optionsProto: YoloObjectDetectorOptionsProto,
  options: YoloObjectDetectorOptionsUpdate,
): void {
  if (options.displayNamesLocale !== undefined) {
    optionsProto.setDisplayNamesLocale(options.displayNamesLocale);
  } else if ('displayNamesLocale' in options) {
    optionsProto.clearDisplayNamesLocale();
  }
  if ('maxResults' in options) {
    optionsProto.setMaxResults(options.maxResults ?? DEFAULT_MAX_RESULTS);
  }
  if ('scoreThreshold' in options) {
    optionsProto.setScoreThreshold(
      options.scoreThreshold ?? DEFAULT_SCORE_THRESHOLD,
    );
  }
  if ('categoryAllowlist' in options) {
    optionsProto.setCategoryAllowlistList(options.categoryAllowlist ?? []);
  }
  if ('categoryDenylist' in options) {
    optionsProto.setCategoryDenylistList(options.categoryDenylist ?? []);
  }
  if ('iouThreshold' in options) {
    optionsProto.setIouThreshold(options.iouThreshold ?? DEFAULT_IOU_THRESHOLD);
  }
  if ('layout' in options) {
    optionsProto.setLayout(
      getYoloLayoutProto(options.layout ?? DEFAULT_LAYOUT),
    );
  }
  if ('numClasses' in options) {
    optionsProto.setNumClasses(options.numClasses ?? DEFAULT_NUM_CLASSES);
  }
  if ('tiling' in options) {
    if (options.tiling === undefined) {
      optionsProto.setTiling(createYoloTilingDefaults());
    } else {
      const tilingProto =
        optionsProto.getTiling() ?? createYoloTilingDefaults();
      applyYoloTilingOptions(tilingProto, options.tiling);
      optionsProto.setTiling(tilingProto);
    }
  }
  if ('tracking' in options) {
    if (options.tracking === undefined) {
      optionsProto.setTracking(createYoloTrackingDefaults());
    } else {
      const trackingProto =
        optionsProto.getTracking() ?? createYoloTrackingDefaults();
      applyYoloTrackingOptions(trackingProto, options.tracking);
      optionsProto.setTracking(trackingProto);
    }
  }
}

/**
 * Performs YOLO object detection on images.
 *
 * Callers must wrap a custom Wasm binary with
 * `createCustomVisionWasmFileset()` and declare
 * `mediapipe.tasks.vision.yolo_object_detector.YoloObjectDetectorGraph`.
 * The bundled MediaPipe vision Wasm files do not include this custom graph.
 */
export class YoloObjectDetector extends VisionTaskRunner {
  private result: YoloObjectDetectorResult = {detections: []};
  private options = new YoloObjectDetectorOptionsProto();

  /**
   * Initializes the Wasm runtime and creates a new YOLO object detector from
   * the provided options.
   *
   * The custom fileset must declare the YOLO graph registration; the bundled
   * vision Wasm files returned by `FilesetResolver` are not accepted.
   */
  static createFromOptions(
    wasmFileset: CustomVisionWasmFileset,
    yoloObjectDetectorOptions: YoloObjectDetectorOptions,
  ): Promise<YoloObjectDetector> {
    assertCustomVisionWasmGraph(wasmFileset, YOLO_OBJECT_DETECTOR_GRAPH);
    return VisionTaskRunner.createVisionInstance(
      YoloObjectDetector,
      wasmFileset,
      yoloObjectDetectorOptions,
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
    this.options.setTiling(createYoloTilingDefaults());
    this.options.setTracking(createYoloTrackingDefaults());
  }

  protected override getTaskName(): string {
    return 'YoloObjectDetector';
  }

  protected override get baseOptions(): BaseOptionsProto {
    return this.options.getBaseOptions()!;
  }

  protected override set baseOptions(proto: BaseOptionsProto) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Applies updates in call order. Await asynchronous updates before inference.
   * Download and validation failures preserve the existing task. If installing
   * model data or rebuilding the graph fails, the task closes and must be
   * recreated.
   */
  override setOptions(options: YoloObjectDetectorOptionsUpdate): Promise<void> {
    return this.runWithOptionsUpdate(() => this.updateOptions(options));
  }

  private updateOptions(options: YoloObjectDetectorOptionsUpdate): Promise<void> {
    const candidate = YoloObjectDetectorOptionsProto.deserializeBinary(
      this.options.serializeBinary(),
    );
    if (!candidate.getBaseOptions()) {
      candidate.setBaseOptions(new BaseOptionsProto());
    }
    if (!candidate.getTiling()) {
      candidate.setTiling(createYoloTilingDefaults());
    }
    if (!candidate.getTracking()) {
      candidate.setTracking(createYoloTrackingDefaults());
    }

    applyYoloOptions(candidate, options);
    validateYoloOptions(candidate, this.getRunningMode(options));

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
  ): YoloObjectDetectorResult {
    this.assertReadyForProcessing();
    this.result = {detections: []};
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
  ): YoloObjectDetectorResult {
    this.assertReadyForProcessing();
    this.result = {detections: []};
    if (this.isTilingEnabled()) {
      validateTiledImageProcessingOptions(imageProcessingOptions);
      this.processTiledVideoData(videoFrame, timestamp);
    } else {
      this.processVideoData(videoFrame, imageProcessingOptions, timestamp);
    }
    return this.result;
  }

  private getRunningMode(
    options: YoloObjectDetectorOptionsUpdate,
  ): RunningMode {
    if ('runningMode' in options) {
      return options.runningMode ?? 'IMAGE';
    }
    return this.baseOptions.getUseStreamMode() ? 'VIDEO' : 'IMAGE';
  }

  private isTilingEnabled(): boolean {
    return isYoloTilingEnabled(this.options.getTiling());
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

  private addJsObjectDetections(data: Uint8Array[]): void {
    for (const binaryProto of data) {
      const detectionProto = DetectionProto.deserializeBinary(binaryProto);
      this.result.detections.push(convertFromDetectionProto(detectionProto));
    }
  }

  protected override refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(IMAGE_STREAM);
    if (!this.isTilingEnabled()) {
      graphConfig.addInputStream(NORM_RECT_STREAM);
    }
    graphConfig.addOutputStream(DETECTIONS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
      YoloObjectDetectorOptionsProto.ext,
      this.options,
    );

    const detectorNode = new CalculatorGraphConfig.Node();
    detectorNode.setCalculator(YOLO_OBJECT_DETECTOR_GRAPH);
    detectorNode.addInputStream('IMAGE:' + IMAGE_STREAM);
    if (!this.isTilingEnabled()) {
      detectorNode.addInputStream('NORM_RECT:' + NORM_RECT_STREAM);
    }
    detectorNode.addOutputStream('DETECTIONS:' + DETECTIONS_STREAM);
    detectorNode.setOptions(calculatorOptions);
    graphConfig.addNode(detectorNode);

    this.graphRunner.attachProtoVectorListener(
      DETECTIONS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsObjectDetections(binaryProto);
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(DETECTIONS_STREAM, (timestamp) => {
      this.setLatestOutputTimestamp(timestamp);
    });

    this.setGraph(
      new Uint8Array(graphConfig.serializeBinary()),
      /* isBinary= */ true,
    );
  }
}
