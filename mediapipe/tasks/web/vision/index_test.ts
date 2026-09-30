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

import * as runtime from './index';
import * as declarations from './types';

describe('custom vision public exports', () => {
  const exports = [
    'YoloObjectDetector',
    'OrientedObjectDetector',
    'YOLO_OBJECT_DETECTOR_GRAPH',
    'ORIENTED_OBJECT_DETECTOR_GRAPH',
    'createCustomVisionWasmFileset',
    'assertCustomVisionWasmGraph',
  ] as const;

  for (const name of exports) {
    it(`provides the declared ${name} at the runtime entry point`, () => {
      expect(runtime[name]).toBeDefined();
      expect(runtime[name]).toBe(declarations[name]);
    });
  }

  it('creates and checks filesets using the public graph names', () => {
    const graphNames = [
      runtime.YOLO_OBJECT_DETECTOR_GRAPH,
      runtime.ORIENTED_OBJECT_DETECTOR_GRAPH,
    ];
    const fileset = runtime.createCustomVisionWasmFileset({
      wasmLoaderPath: 'custom.js',
      wasmBinaryPath: 'custom.wasm',
    }, graphNames);

    for (const graphName of graphNames) {
      expect(() => runtime.assertCustomVisionWasmGraph(fileset, graphName))
        .not.toThrow();
    }
  });
});
