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

import 'jasmine';

import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {Detection as DetectionProto} from '../../../../framework/formats/detection_pb';
import {LocationData} from '../../../../framework/formats/location_data_pb';
import {
  addJasmineCustomFloatEqualityTester,
  createSpyWasmModule,
  MediapipeTasksFake,
  SpyWasmModule,
  verifyGraph,
  verifyListenersRegistered,
} from '../../../../tasks/web/core/task_runner_test_utils';
import {createCustomVisionWasmFileset} from '../../../../tasks/web/vision/core/custom_vision_wasm_fileset';

import {
  YOLO_OBJECT_DETECTOR_GRAPH,
  YoloObjectDetector,
} from './yolo_object_detector';
import type {YoloObjectDetectorOptions} from './yolo_object_detector_options';

class YoloObjectDetectorFake extends YoloObjectDetector
implements MediapipeTasksFake {
  calculatorName =
    'mediapipe.tasks.vision.yolo_object_detector.YoloObjectDetectorGraph';
  attachListenerSpies: jasmine.Spy[] = [];
  graph: CalculatorGraphConfig | undefined;

  fakeWasmModule: SpyWasmModule;
  addProtoToStreamSpy: jasmine.Spy;
  protoListener:
    | ((binaryProtos: Uint8Array[], timestamp: number) => void)
    | undefined;

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);
    this.fakeWasmModule = this.graphRunner.wasmModule as unknown as SpyWasmModule;

    this.attachListenerSpies[0] = spyOn(
      this.graphRunner,
      'attachProtoVectorListener',
    ).and.callFake((stream, listener) => {
      expect(stream).toEqual('detections');
      this.protoListener = listener;
    });
    spyOn(this.graphRunner, 'setGraph').and.callFake((binaryGraph) => {
      this.graph = CalculatorGraphConfig.deserializeBinary(binaryGraph);
    });
    spyOn(this.graphRunner, 'addGpuBufferAsImageToStream');
    this.addProtoToStreamSpy = spyOn(this.graphRunner, 'addProtoToStream');
  }
}

describe('YoloObjectDetector', () => {
  let yoloObjectDetector: YoloObjectDetectorFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    yoloObjectDetector = new YoloObjectDetectorFake();
    await yoloObjectDetector.setOptions({
      baseOptions: {modelAssetBuffer: new Uint8Array([])},
      numClasses: 80,
    });
  });

  afterEach(() => {
    yoloObjectDetector.close();
  });

  it('initializes graph', async () => {
    verifyGraph(yoloObjectDetector, ['numClasses', 80]);
    verifyListenersRegistered(yoloObjectDetector);
  });

  it('rejects a custom Wasm fileset without its graph', () => {
    const customFileset = createCustomVisionWasmFileset(
      {wasmLoaderPath: 'custom.js', wasmBinaryPath: 'custom.wasm'},
      ['example.OtherGraph'],
    );

    expect(() => YoloObjectDetector.createFromOptions(customFileset, {
      baseOptions: {modelAssetBuffer: new Uint8Array([])},
      numClasses: 80,
    })).toThrowError(new RegExp(YOLO_OBJECT_DETECTOR_GRAPH));
  });

  it('merges options', async () => {
    await yoloObjectDetector.setOptions({scoreThreshold: 0.4});
    await yoloObjectDetector.setOptions({iouThreshold: 0.2});
    verifyGraph(yoloObjectDetector, ['scoreThreshold', 0.4]);
    verifyGraph(yoloObjectDetector, ['iouThreshold', 0.2]);
  });

  describe('setOptions()', () => {
    interface TestCase {
      optionName: keyof YoloObjectDetectorOptions;
      protoName: string;
      customValue: unknown;
      defaultValue: unknown;
    }

    const testCases: TestCase[] = [
      {
        optionName: 'displayNamesLocale',
        protoName: 'displayNamesLocale',
        customValue: 'fr',
        defaultValue: 'en',
      },
      {
        optionName: 'maxResults',
        protoName: 'maxResults',
        customValue: 5,
        defaultValue: -1,
      },
      {
        optionName: 'scoreThreshold',
        protoName: 'scoreThreshold',
        customValue: 0.5,
        defaultValue: 0.25,
      },
      {
        optionName: 'categoryAllowlist',
        protoName: 'categoryAllowlistList',
        customValue: ['ship'],
        defaultValue: [],
      },
      {
        optionName: 'categoryDenylist',
        protoName: 'categoryDenylistList',
        customValue: ['boat'],
        defaultValue: [],
      },
      {
        optionName: 'iouThreshold',
        protoName: 'iouThreshold',
        customValue: 0.3,
        defaultValue: 0.45,
      },
      {
        optionName: 'layout',
        protoName: 'layout',
        customValue: 2,
        defaultValue: 1,
      },
      {
        optionName: 'numClasses',
        protoName: 'numClasses',
        customValue: 32,
        defaultValue: 80,
      },
    ];

    for (const testCase of testCases) {
      it(`can set ${testCase.optionName}`, async () => {
        await yoloObjectDetector.setOptions({
          [testCase.optionName]: testCase.optionName === 'layout'
            ? 'CHANNELS_LAST'
            : testCase.customValue,
        });
        verifyGraph(yoloObjectDetector, [testCase.protoName, testCase.customValue]);
      });
    }
  });

  it('resets tiling to defaults', async () => {
    await yoloObjectDetector.setOptions({
      tiling: {tileRows: 2, tileCols: 2},
    });
    verifyGraph(yoloObjectDetector, [['tiling', 'tileRows'], 2]);
    await yoloObjectDetector.setOptions({tiling: undefined});
    verifyGraph(yoloObjectDetector, [['tiling', 'tileRows'], 1]);
    verifyGraph(yoloObjectDetector, [['tiling', 'tileCols'], 1]);
  });

  for (const runningMode of ['IMAGE', 'VIDEO'] as const) {
    const rejectedMode = runningMode === 'IMAGE' ? 'VIDEO' : 'IMAGE';
    const frame = {width: 200, height: 100} as ImageData;

    function expectOriginalRunningMode(): void {
      if (runningMode === 'IMAGE') {
        expect(() => yoloObjectDetector.detect(frame)).not.toThrow();
        expect(() => yoloObjectDetector.detectForVideo(frame, 42)).toThrowError(
          /Task is not initialized with video mode/,
        );
      } else {
        expect(() => yoloObjectDetector.detectForVideo(frame, 42)).not.toThrow();
        expect(() => yoloObjectDetector.detect(frame)).toThrowError(
          /Task is not initialized with image mode/,
        );
      }
    }

    it(`retains ${runningMode} mode after a rejected canvas update`, async () => {
      await yoloObjectDetector.setOptions({runningMode});
      const graph = yoloObjectDetector.graph;

      expect(() => yoloObjectDetector.setOptions({
        runningMode: rejectedMode,
        canvas: {} as HTMLCanvasElement,
      })).toThrowError(/You must create a new task to reset the canvas/);

      expect(yoloObjectDetector.graph).toBe(graph);
      expectOriginalRunningMode();
      await yoloObjectDetector.setOptions({maxResults: 5});
      verifyGraph(yoloObjectDetector, undefined, [
        'useStreamMode', runningMode === 'VIDEO',
      ]);
    });

    it(`retains ${runningMode} mode after a failed model fetch`, async () => {
      await yoloObjectDetector.setOptions({runningMode});
      const graph = yoloObjectDetector.graph;
      spyOn(globalThis, 'fetch').and.returnValue(
        Promise.reject(new Error('Model download failed')),
      );

      await expectAsync(yoloObjectDetector.setOptions({
        runningMode: rejectedMode,
        baseOptions: {modelAssetPath: 'missing-model.tflite'},
        scoreThreshold: 0.9,
      })).toBeRejectedWithError('Model download failed');

      expect(yoloObjectDetector.graph).toBe(graph);
      expectOriginalRunningMode();
      await yoloObjectDetector.setOptions({maxResults: 5});
      verifyGraph(yoloObjectDetector, ['scoreThreshold', 0.25], [
        'useStreamMode', runningMode === 'VIDEO',
      ]);
    });
  }

  it('validates an undefined running mode as an IMAGE reset', async () => {
    await yoloObjectDetector.setOptions({
      runningMode: 'VIDEO',
      tiling: {tileRows: 2, tileCols: 2},
      tracking: {trackerType: 'BOTSORT'},
    });
    // Omitting the property preserves VIDEO mode and its tracker.
    await yoloObjectDetector.setOptions({scoreThreshold: 0.4});
    const graph = yoloObjectDetector.graph;

    expect(() => yoloObjectDetector.setOptions({
      runningMode: undefined,
    })).toThrowError(/BOTSORT requires VIDEO or LIVE_STREAM running mode/);

    expect(yoloObjectDetector.graph).toBe(graph);
    expect(() => yoloObjectDetector.detectForVideo(
      {width: 200, height: 100} as ImageData, 42,
    )).not.toThrow();

    // Resetting the tracker together with the mode is a valid IMAGE update.
    await yoloObjectDetector.setOptions({
      runningMode: undefined,
      tracking: undefined,
    });
    verifyGraph(yoloObjectDetector, undefined, ['useStreamMode', false]);
    expect(() => yoloObjectDetector.detect(
      {width: 200, height: 100} as ImageData,
    )).not.toThrow();
  });

  it('rejects resetting motion scheduling to IMAGE mode', async () => {
    await yoloObjectDetector.setOptions({
      runningMode: 'VIDEO',
      tiling: {tileRows: 2, tileCols: 2, enableMotionScheduling: true},
    });
    const graph = yoloObjectDetector.graph;

    expect(() => yoloObjectDetector.setOptions({
      runningMode: undefined,
    })).toThrowError(/enable_motion_scheduling requires VIDEO/);
    expect(yoloObjectDetector.graph).toBe(graph);

    await yoloObjectDetector.setOptions({
      runningMode: undefined,
      tiling: {enableMotionScheduling: false},
    });
    verifyGraph(yoloObjectDetector, undefined, ['useStreamMode', false]);
  });

  it('drops NORM_RECT when tiling is enabled', async () => {
    await yoloObjectDetector.setOptions({
      tiling: {tileRows: 2, tileCols: 2},
    });

    expect(yoloObjectDetector.graph!.getInputStreamList()).toEqual([
      'input_frame_gpu',
    ]);
    expect(yoloObjectDetector.graph!.getNodeList()[0].getInputStreamList()).toEqual([
      'IMAGE:input_frame_gpu',
    ]);
  });

  it('rejects maxResults = 0', () => {
    expect(() => {
      yoloObjectDetector.setOptions({maxResults: 0});
    }).toThrowError(
      'Invalid `max_results` option: value must be != 0',
    );
  });

  it('rejects allowlist and denylist together', () => {
    expect(() => {
      yoloObjectDetector.setOptions({
        categoryAllowlist: ['ship'],
        categoryDenylist: ['boat'],
      });
    }).toThrowError(
      '`category_allowlist` and `category_denylist` are mutually exclusive options.',
    );
  });

  it('rejects missing numClasses', () => {
    expect(() => {
      yoloObjectDetector.setOptions({numClasses: undefined});
    }).toThrowError(
      'num_classes must be set in YoloObjectDetectorOptions (metadata-derived num_classes is a future enhancement)',
    );
  });

  it('rejects tiled ROI and rotation', async () => {
    await yoloObjectDetector.setOptions({
      tiling: {tileRows: 2, tileCols: 2},
    });

    expect(() => {
      yoloObjectDetector.detect({} as HTMLImageElement, {
        regionOfInterest: {left: 0, top: 0, right: 1, bottom: 1},
      });
    }).toThrowError('tiling and ROI are mutually exclusive');

    expect(() => {
      yoloObjectDetector.detect({} as HTMLImageElement, {
        rotationDegrees: 90,
      });
    }).toThrowError('tiling does not support rotation_degrees');
  });

  it('retains non-tiled rotation', () => {
    yoloObjectDetector.detect({} as HTMLImageElement, {
      rotationDegrees: 90,
    });
    expect(yoloObjectDetector.addProtoToStreamSpy).toHaveBeenCalled();
  });

  it('rejects BOTSORT without tiled video mode', () => {
    expect(() => {
      yoloObjectDetector.setOptions({
        tracking: {trackerType: 'BOTSORT'},
      });
    }).toThrowError(
      'tracking.tracker_type=BOTSORT requires VIDEO or LIVE_STREAM running mode; tracking is not available in IMAGE mode.',
    );
  });

  it('transforms results with track ids', async () => {
    const detection = new DetectionProto();
    detection.addScore(0.1);
    detection.setTrackId('track-1');
    const locationData = new LocationData();
    const boundingBox = new LocationData.BoundingBox();
    locationData.setBoundingBox(boundingBox);
    detection.setLocationData(locationData);
    const binaryProto = detection.serializeBinary();

    yoloObjectDetector.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(yoloObjectDetector);
      yoloObjectDetector.protoListener!([binaryProto], 1337);
    });

    const {detections} = yoloObjectDetector.detect({} as HTMLImageElement);

    expect(detections).toEqual([{
      categories: [{
        score: 0.1,
        index: -1,
        categoryName: '',
        displayName: '',
      }],
      boundingBox: {originX: 0, originY: 0, width: 0, height: 0, angle: 0},
      keypoints: [],
      trackId: 'track-1',
    }]);
  });
});
