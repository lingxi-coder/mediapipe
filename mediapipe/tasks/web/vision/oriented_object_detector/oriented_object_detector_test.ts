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
import {OrientedDetection as OrientedDetectionProto} from '../../../../framework/formats/oriented_detection_pb';
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
  ORIENTED_OBJECT_DETECTOR_GRAPH,
  OrientedObjectDetector,
} from './oriented_object_detector';
import type {
  OrientedObjectDetectorOptions,
} from './oriented_object_detector_options';

class OrientedObjectDetectorFake extends OrientedObjectDetector
implements MediapipeTasksFake {
  calculatorName =
    'mediapipe.tasks.vision.oriented_object_detector.OrientedObjectDetectorGraph';
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
      expect(stream).toEqual('oriented_detections');
      this.protoListener = listener;
    });
    spyOn(this.graphRunner, 'setGraph').and.callFake((binaryGraph) => {
      this.graph = CalculatorGraphConfig.deserializeBinary(binaryGraph);
    });
    spyOn(this.graphRunner, 'addGpuBufferAsImageToStream');
    this.addProtoToStreamSpy = spyOn(this.graphRunner, 'addProtoToStream');
  }
}

describe('OrientedObjectDetector', () => {
  let orientedObjectDetector: OrientedObjectDetectorFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    orientedObjectDetector = new OrientedObjectDetectorFake();
    await orientedObjectDetector.setOptions({
      baseOptions: {modelAssetBuffer: new Uint8Array([])},
      numClasses: 15,
    });
  });

  afterEach(() => {
    orientedObjectDetector.close();
  });

  it('initializes graph', async () => {
    verifyGraph(orientedObjectDetector, ['numClasses', 15]);
    verifyListenersRegistered(orientedObjectDetector);
  });

  it('rejects a custom Wasm fileset without its graph', () => {
    const customFileset = createCustomVisionWasmFileset(
      {wasmLoaderPath: 'custom.js', wasmBinaryPath: 'custom.wasm'},
      ['example.OtherGraph'],
    );

    expect(() => OrientedObjectDetector.createFromOptions(customFileset, {
      baseOptions: {modelAssetBuffer: new Uint8Array([])},
      numClasses: 15,
    })).toThrowError(new RegExp(ORIENTED_OBJECT_DETECTOR_GRAPH));
  });

  it('merges options', async () => {
    await orientedObjectDetector.setOptions({scoreThreshold: 0.4});
    await orientedObjectDetector.setOptions({classAgnosticNms: true});
    verifyGraph(orientedObjectDetector, ['scoreThreshold', 0.4]);
    verifyGraph(orientedObjectDetector, ['classAgnosticNms', true]);
  });

  describe('setOptions()', () => {
    interface TestCase {
      optionName: keyof OrientedObjectDetectorOptions;
      protoName: string;
      customValue: unknown;
      defaultValue: unknown;
      setValue?: unknown;
    }

    const testCases: TestCase[] = [
      {
        optionName: 'maxResults',
        protoName: 'maxResults',
        customValue: 5,
        defaultValue: -1,
      },
      {
        optionName: 'scoreThreshold',
        protoName: 'scoreThreshold',
        customValue: 0.4,
        defaultValue: 0.25,
      },
      {
        optionName: 'iouThreshold',
        protoName: 'iouThreshold',
        customValue: 0.3,
        defaultValue: 0.45,
      },
      {
        optionName: 'classAgnosticNms',
        protoName: 'classAgnosticNms',
        customValue: true,
        defaultValue: false,
      },
      {
        optionName: 'layout',
        protoName: 'layout',
        customValue: 2,
        defaultValue: 1,
        setValue: 'CHANNELS_LAST',
      },
      {
        optionName: 'displayNamesLocale',
        protoName: 'displayNamesLocale',
        customValue: 'fr',
        defaultValue: 'en',
      },
      {
        optionName: 'numClasses',
        protoName: 'numClasses',
        customValue: 32,
        defaultValue: 15,
      },
    ];

    for (const testCase of testCases) {
      it(`can set ${testCase.optionName}`, async () => {
        await orientedObjectDetector.setOptions({
          [testCase.optionName]: testCase.setValue ?? testCase.customValue,
        });
        verifyGraph(orientedObjectDetector, [testCase.protoName, testCase.customValue]);
      });
    }
  });

  for (const runningMode of ['IMAGE', 'VIDEO'] as const) {
    const rejectedMode = runningMode === 'IMAGE' ? 'VIDEO' : 'IMAGE';
    const frame = {width: 200, height: 100} as ImageData;

    function expectOriginalRunningMode(): void {
      if (runningMode === 'IMAGE') {
        expect(() => orientedObjectDetector.detect(frame)).not.toThrow();
        expect(() => orientedObjectDetector.detectForVideo(frame, 42)).toThrowError(
          /Task is not initialized with video mode/,
        );
      } else {
        expect(() => orientedObjectDetector.detectForVideo(frame, 42)).not.toThrow();
        expect(() => orientedObjectDetector.detect(frame)).toThrowError(
          /Task is not initialized with image mode/,
        );
      }
    }

    it(`retains ${runningMode} mode after a rejected canvas update`, async () => {
      await orientedObjectDetector.setOptions({runningMode});
      const graph = orientedObjectDetector.graph;

      expect(() => orientedObjectDetector.setOptions({
        runningMode: rejectedMode,
        canvas: {} as HTMLCanvasElement,
      })).toThrowError(/You must create a new task to reset the canvas/);

      expect(orientedObjectDetector.graph).toBe(graph);
      expectOriginalRunningMode();
      await orientedObjectDetector.setOptions({maxResults: 5});
      verifyGraph(orientedObjectDetector, undefined, [
        'useStreamMode', runningMode === 'VIDEO',
      ]);
    });

    it(`retains ${runningMode} mode after a failed model fetch`, async () => {
      await orientedObjectDetector.setOptions({runningMode});
      const graph = orientedObjectDetector.graph;
      spyOn(globalThis, 'fetch').and.returnValue(
        Promise.reject(new Error('Model download failed')),
      );

      await expectAsync(orientedObjectDetector.setOptions({
        runningMode: rejectedMode,
        baseOptions: {modelAssetPath: 'missing-model.tflite'},
        scoreThreshold: 0.9,
      })).toBeRejectedWithError('Model download failed');

      expect(orientedObjectDetector.graph).toBe(graph);
      expectOriginalRunningMode();
      await orientedObjectDetector.setOptions({maxResults: 5});
      verifyGraph(orientedObjectDetector, ['scoreThreshold', 0.25], [
        'useStreamMode', runningMode === 'VIDEO',
      ]);
    });
  }

  it('validates an undefined running mode as an IMAGE reset', async () => {
    await orientedObjectDetector.setOptions({
      runningMode: 'VIDEO',
      tiling: {tileRows: 2, tileCols: 2},
      tracking: {trackerType: 'BOTSORT'},
    });
    // Omitting the property preserves VIDEO mode and its tracker.
    await orientedObjectDetector.setOptions({scoreThreshold: 0.4});
    const graph = orientedObjectDetector.graph;

    expect(() => orientedObjectDetector.setOptions({
      runningMode: undefined,
    })).toThrowError(/BOTSORT requires VIDEO or LIVE_STREAM running mode/);

    expect(orientedObjectDetector.graph).toBe(graph);
    expect(() => orientedObjectDetector.detectForVideo(
      {width: 200, height: 100} as ImageData, 42,
    )).not.toThrow();

    // Resetting the tracker together with the mode is a valid IMAGE update.
    await orientedObjectDetector.setOptions({
      runningMode: undefined,
      tracking: undefined,
    });
    verifyGraph(orientedObjectDetector, undefined, ['useStreamMode', false]);
    expect(() => orientedObjectDetector.detect(
      {width: 200, height: 100} as ImageData,
    )).not.toThrow();
  });

  it('drops NORM_RECT when tiling is enabled', async () => {
    await orientedObjectDetector.setOptions({
      tiling: {tileRows: 2, tileCols: 2},
    });

    expect(orientedObjectDetector.graph!.getInputStreamList()).toEqual([
      'input_frame_gpu',
    ]);
    expect(orientedObjectDetector.graph!.getNodeList()[0].getInputStreamList()).toEqual([
      'IMAGE:input_frame_gpu',
    ]);
  });

  it('rejects BOX_TRACKER', () => {
    expect(() => {
      orientedObjectDetector.setOptions({
        tracking: {trackerType: 'BOX_TRACKER'},
      });
    }).toThrowError(
      'tracking.tracker_type=BOX_TRACKER: BoxTracker is not supported for oriented detection; use BOTSORT.',
    );
  });

  it('rejects missing numClasses', () => {
    expect(() => {
      orientedObjectDetector.setOptions({numClasses: undefined});
    }).toThrowError(
      'num_classes must be set in OrientedObjectDetectorOptions',
    );
  });

  it('rejects tiled ROI and rotation', async () => {
    await orientedObjectDetector.setOptions({
      tiling: {tileRows: 2, tileCols: 2},
    });

    expect(() => {
      orientedObjectDetector.detect({width: 200, height: 100} as ImageData, {
        regionOfInterest: {left: 0, top: 0, right: 1, bottom: 1},
      });
    }).toThrowError('tiling and ROI are mutually exclusive');

    expect(() => {
      orientedObjectDetector.detect({width: 200, height: 100} as ImageData, {
        rotationDegrees: 90,
      });
    }).toThrowError('tiling does not support rotation_degrees');
  });

  it('retains non-tiled rotation', () => {
    orientedObjectDetector.detect({width: 200, height: 100} as ImageData, {
      rotationDegrees: 90,
    });
    expect(orientedObjectDetector.addProtoToStreamSpy).toHaveBeenCalled();
  });

  it('transforms oriented results to pixel units', () => {
    const detection = new OrientedDetectionProto();
    detection.addScore(0.9);
    detection.addLabelId(1);
    detection.addLabel('ship');
    detection.addDisplayName('Ship');
    detection.setCx(0.5);
    detection.setCy(0.25);
    detection.setWidth(0.4);
    detection.setHeight(0.2);
    detection.setRotation(0.3);
    detection.setTrackId('track-1');
    const binaryProto = detection.serializeBinary();

    orientedObjectDetector.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(orientedObjectDetector);
      orientedObjectDetector.protoListener!([binaryProto], 1337);
    });

    const {detections} = orientedObjectDetector.detect(
      {width: 200, height: 100} as ImageData,
    );

    expect(detections).toHaveSize(1);
    expect(detections[0].categories).toEqual([{
      score: 0.9,
      index: 1,
      categoryName: 'ship',
      displayName: 'Ship',
    }]);
    expect(detections[0].cx).toBeCloseTo(100);
    expect(detections[0].cy).toBeCloseTo(25);
    expect(detections[0].width).toBeCloseTo(86.0971, 3);
    expect(detections[0].height).toBeCloseTo(20.6890, 3);
    expect(detections[0].rotation).toBeCloseTo(0.153452, 5);
    expect(detections[0].trackId).toBe('track-1');
  });
});
