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

import {
  assertCustomVisionWasmGraph,
  createCustomVisionWasmFileset,
} from './custom_vision_wasm_fileset';
import type {CustomVisionWasmFileset} from './custom_vision_wasm_fileset';

describe('CustomVisionWasmFileset', () => {
  const wasmFileset = {
    wasmLoaderPath: 'custom_vision.js',
    wasmBinaryPath: 'custom_vision.wasm',
  };

  it('records graph registrations', () => {
    const customFileset = createCustomVisionWasmFileset(
      wasmFileset,
      ['example.CustomGraph', 'example.CustomGraph'],
    );

    expect(customFileset.wasmLoaderPath).toBe('custom_vision.js');
    expect(customFileset.registeredGraphNames).toEqual([
      'example.CustomGraph',
    ]);
    expect(() => {
      assertCustomVisionWasmGraph(customFileset, 'example.CustomGraph');
    }).not.toThrow();
  });

  it('rejects an empty graph registration list', () => {
    expect(() => createCustomVisionWasmFileset(wasmFileset, [])).toThrowError(
      /registeredGraphNames/,
    );
  });

  it('rejects a task graph not declared by the fileset', () => {
    const customFileset = createCustomVisionWasmFileset(
      wasmFileset,
      ['example.OtherGraph'],
    );

    expect(() => {
      assertCustomVisionWasmGraph(customFileset, 'example.RequiredGraph');
    }).toThrowError(/example.RequiredGraph/);
  });

  it('rejects the standard fileset at runtime', () => {
    expect(() => {
      assertCustomVisionWasmGraph(
        wasmFileset as CustomVisionWasmFileset,
        'example.RequiredGraph',
      );
    }).toThrowError(/createCustomVisionWasmFileset/);
  });
});
