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
  addImageToStreamSpy: jasmine.Spy;
  setGraphSpy: jasmine.Spy;
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
    this.setGraphSpy = spyOn(this.graphRunner, 'setGraph').and.callFake((binaryGraph) => {
      this.graph = CalculatorGraphConfig.deserializeBinary(binaryGraph);
    });
    this.addImageToStreamSpy = spyOn(this.graphRunner, 'addGpuBufferAsImageToStream');
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

  for (const modelLoads of [true, false]) {
    it(`serializes later options when an earlier model load ${modelLoads ? 'succeeds' : 'fails'}`, async () => {
      let resolveFetch!: (response: Response) => void;
      let rejectFetch!: (error: Error) => void;
      spyOn(globalThis, 'fetch').and.returnValue(new Promise<Response>((resolve, reject) => {
        resolveFetch = resolve;
        rejectFetch = reject;
      }));
      const graph = orientedObjectDetector.graph;
      const first = orientedObjectDetector.setOptions({
        baseOptions: {modelAssetPath: 'pending-model.tflite'},
        scoreThreshold: 0.2,
      });
      const firstResult = modelLoads
        ? expectAsync(first).toBeResolved()
        : expectAsync(first).toBeRejectedWithError('Model download failed');
      const second = orientedObjectDetector.setOptions({
        tiling: {tileRows: 2, tileCols: 2},
      });

      expect(orientedObjectDetector.graph).toBe(graph);
      if (modelLoads) {
        resolveFetch({
          ok: true,
          arrayBuffer: async () => new Uint8Array([1]).buffer,
        } as Response);
      } else {
        rejectFetch(new Error('Model download failed'));
      }
      await firstResult;
      await second;

      verifyGraph(orientedObjectDetector, [
        'scoreThreshold', modelLoads ? 0.2 : 0.25,
      ]);
      verifyGraph(orientedObjectDetector, [['tiling', 'tileRows'], 2]);
      expect(orientedObjectDetector.graph!.getInputStreamList()).toEqual(['input_frame_gpu']);
      orientedObjectDetector.addProtoToStreamSpy.calls.reset();
      orientedObjectDetector.detect({width: 200, height: 100} as ImageData);
      expect(orientedObjectDetector.addProtoToStreamSpy).not.toHaveBeenCalled();
    });
  }

  it('keeps consecutive synchronous updates immediately usable', async () => {
    const first = orientedObjectDetector.setOptions({scoreThreshold: 0.4});
    const second = orientedObjectDetector.setOptions({tiling: {tileRows: 2}});
    verifyGraph(orientedObjectDetector, ['scoreThreshold', 0.4]);
    verifyGraph(orientedObjectDetector, [['tiling', 'tileRows'], 2]);
    expect(() => orientedObjectDetector.detect({width: 200, height: 100} as ImageData))
      .not.toThrow();
    expect(orientedObjectDetector.addImageToStreamSpy).toHaveBeenCalledTimes(1);
    expect(orientedObjectDetector.addProtoToStreamSpy).not.toHaveBeenCalled();
    await Promise.all([first, second]);
  });

  it('blocks all frames until an asynchronous update has settled', async () => {
    await orientedObjectDetector.setOptions({tiling: {tileRows: 2}});
    let rejectFetch!: (error: Error) => void;
    spyOn(globalThis, 'fetch').and.returnValue(new Promise<Response>((_, reject) => {
      rejectFetch = reject;
    }));
    const pending = orientedObjectDetector.setOptions({
      baseOptions: {modelAssetPath: 'pending-model.tflite'},
      tiling: undefined,
      runningMode: 'VIDEO',
    });
    const result = expectAsync(pending).toBeRejectedWithError('Download failed');
    const frame = {width: 200, height: 100} as ImageData;

    expect(() => orientedObjectDetector.detect(frame)).toThrowError(/Await setOptions/);
    expect(() => orientedObjectDetector.detectForVideo(frame, 42)).toThrowError(/Await setOptions/);
    expect(orientedObjectDetector.addImageToStreamSpy).not.toHaveBeenCalled();
    expect(orientedObjectDetector.addProtoToStreamSpy).not.toHaveBeenCalled();

    rejectFetch(new Error('Download failed'));
    await result;
    expect(() => orientedObjectDetector.detect(frame)).not.toThrow();
    expect(orientedObjectDetector.addImageToStreamSpy).toHaveBeenCalledTimes(1);
    expect(orientedObjectDetector.addProtoToStreamSpy).not.toHaveBeenCalled();
  });

  it('closes after downloaded model installation fails', async () => {
    spyOn(globalThis, 'fetch').and.resolveTo({
      ok: true,
      arrayBuffer: async () => new Uint8Array([1]).buffer,
    } as Response);
    orientedObjectDetector.setGraphSpy.and.throwError('Graph initialization failed');

    await expectAsync(orientedObjectDetector.setOptions({
      baseOptions: {modelAssetPath: 'invalid-model.tflite'},
    })).toBeRejectedWithError('Graph initialization failed');

    expect(orientedObjectDetector.fakeWasmModule.FS_createDataFile).toHaveBeenCalled();
    expect(orientedObjectDetector.fakeWasmModule._closeGraph).toHaveBeenCalledTimes(1);
    expect(() => orientedObjectDetector.detect({width: 200, height: 100} as ImageData))
      .toThrowError('Task is closed.');
    expect(() => orientedObjectDetector.setOptions({scoreThreshold: 0.4}))
      .toThrowError('Task is closed.');
  });

  it('closes after buffered model installation fails', () => {
    orientedObjectDetector.setGraphSpy.and.throwError('Graph initialization failed');

    expect(() => orientedObjectDetector.setOptions({
      baseOptions: {modelAssetBuffer: new Uint8Array([1])},
    })).toThrowError('Graph initialization failed');

    expect(orientedObjectDetector.fakeWasmModule._closeGraph).toHaveBeenCalledTimes(1);
    expect(() => orientedObjectDetector.detect({width: 200, height: 100} as ImageData))
      .toThrowError('Task is closed.');
    expect(() => orientedObjectDetector.setOptions({scoreThreshold: 0.4}))
      .toThrowError('Task is closed.');
  });

  it('rejects pending and queued updates when closed', async () => {
    let resolveFetch!: (response: Response) => void;
    spyOn(globalThis, 'fetch').and.returnValue(new Promise<Response>((resolve) => {
      resolveFetch = resolve;
    }));
    const graph = orientedObjectDetector.graph;
    const pending = orientedObjectDetector.setOptions({
      baseOptions: {modelAssetPath: 'pending-model.tflite'},
    });
    const queued = orientedObjectDetector.setOptions({tiling: {tileRows: 2}});
    const pendingResult = expectAsync(pending).toBeRejectedWithError('Task is closed.');
    const queuedResult = expectAsync(queued).toBeRejectedWithError('Task is closed.');

    orientedObjectDetector.close();
    await Promise.all([pendingResult, queuedResult]);
    resolveFetch({
      ok: true,
      arrayBuffer: async () => new Uint8Array([1]).buffer,
    } as Response);
    // Drain the fetch and model-loading continuations even if fetch ignores abort.
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    expect(orientedObjectDetector.graph).toBe(graph);
    expect(orientedObjectDetector.fakeWasmModule.FS_createDataFile).not.toHaveBeenCalled();
    expect(orientedObjectDetector.fakeWasmModule._closeGraph).toHaveBeenCalledTimes(1);
    expect(() => orientedObjectDetector.detect({width: 200, height: 100} as ImageData))
      .toThrowError('Task is closed.');
    expect(() => orientedObjectDetector.setOptions({scoreThreshold: 0.4}))
      .toThrowError('Task is closed.');
  });

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

  it('rejects grid dimensions that cannot be represented safely', () => {
    const invalidGrids = [
      {tileRows: -1, tileCols: 1},
      {tileRows: 1.5, tileCols: 1},
      {tileRows: Number.NaN, tileCols: 1},
      {tileRows: 1, tileCols: Number.POSITIVE_INFINITY},
      {tileRows: 2147483648, tileCols: 0},
      {tileRows: 65536, tileCols: 65536},
      {tileRows: 46341, tileCols: 46341},
    ];
    for (const tiling of invalidGrids) {
      expect(() => orientedObjectDetector.setOptions({tiling})).toThrowError(/tiling/);
    }
  });

  it('keeps zero-sized grid dimensions disabled', async () => {
    await orientedObjectDetector.setOptions({
      tiling: {tileRows: 0, tileCols: 2147483647},
    });
    expect(orientedObjectDetector.graph!.getInputStreamList()).toContain('norm_rect');
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
