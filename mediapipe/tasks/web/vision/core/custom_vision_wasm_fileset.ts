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

import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';

const customVisionWasmFilesetBrand: unique symbol =
  Symbol('CustomVisionWasmFileset');

/**
 * A Wasm fileset whose binary is declared to contain custom vision graph
 * registrations.
 */
export interface CustomVisionWasmFileset extends WasmFileset {
  /** Fully qualified graph type names registered by the custom binary. */
  readonly registeredGraphNames: ReadonlyArray<string>;
  readonly [customVisionWasmFilesetBrand]: true;
}

/**
 * Declares the graph registrations provided by a custom vision Wasm binary.
 *
 * This explicit boundary prevents custom-graph tasks from accidentally being
 * initialized with the standard fileset returned by `FilesetResolver`.
 */
export function createCustomVisionWasmFileset(
  wasmFileset: WasmFileset,
  registeredGraphNames: ReadonlyArray<string>,
): CustomVisionWasmFileset {
  if (registeredGraphNames.length === 0 ||
      registeredGraphNames.some(graphName => graphName.trim().length === 0)) {
    throw new Error(
      'registeredGraphNames must contain at least one non-empty graph name.',
    );
  }
  const uniqueGraphNames = Array.from(new Set(registeredGraphNames));
  return Object.freeze({
    ...wasmFileset,
    registeredGraphNames: Object.freeze(uniqueGraphNames),
    [customVisionWasmFilesetBrand]: true as const,
  });
}

/** Verifies that a custom fileset declares the graph required by a task. */
export function assertCustomVisionWasmGraph(
  wasmFileset: CustomVisionWasmFileset,
  graphName: string,
): void {
  if (!Array.isArray(wasmFileset.registeredGraphNames)) {
    throw new Error(
      'Custom graph tasks require a fileset created with ' +
      'createCustomVisionWasmFileset().',
    );
  }
  if (!wasmFileset.registeredGraphNames.includes(graphName)) {
    throw new Error(
      `The custom vision Wasm fileset does not declare graph "${graphName}".`,
    );
  }
}
