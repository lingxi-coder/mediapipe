/**
 * Copyright 2022 The MediaPipe Authors.
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

import {InferenceCalculatorOptions} from '../../../calculators/tensor/inference_calculator_pb';
import {CalculatorGraphConfig} from '../../../framework/calculator_pb';
import {Acceleration} from '../../../tasks/cc/core/proto/acceleration_pb';
import {BaseOptions as BaseOptionsProto} from '../../../tasks/cc/core/proto/base_options_pb';
import {ExternalFile} from '../../../tasks/cc/core/proto/external_file_pb';
import {
  BaseOptions,
  TaskRunnerOptions,
} from '../../../tasks/web/core/task_runner_options';
import {streamToUint8Array} from '../../../tasks/web/genai/llm_inference/model_loading_utils';
import {
  FileLocator,
  GraphRunner,
  WasmMediaPipeConstructor,
  createMediaPipeLib,
} from '../../../web/graph_runner/graph_runner';
import {SupportLogging} from '../../../web/graph_runner/graph_runner_logging_lib';
import {SupportModelResourcesGraphService} from '../../../web/graph_runner/register_model_resources_graph_service';
import {TaskLogger} from './task_logger';
import {createTasksLogger} from './task_logger_factory';

import {WasmFileset} from './wasm_fileset';

// Internal stream names for temporarily keeping memory alive, then freeing it.
const FREE_MEMORY_STREAM = 'free_memory';
const UNUSED_STREAM_SUFFIX = '_unused_out';

// tslint:disable-next-line:enforce-name-casing
const CachedGraphRunnerType = SupportLogging(
  SupportModelResourcesGraphService(GraphRunner),
);

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/**
 * An implementation of the GraphRunner that exposes the resource graph
 * service.
 */
export class CachedGraphRunner extends CachedGraphRunnerType {}

/**
 * Creates a new instance of a Mediapipe Task. Determines if SIMD is
 * supported and loads the relevant WASM binary.
 * @return A fully instantiated instance of `T`.
 */
export async function createTaskRunner<T extends TaskRunner>(
  type: WasmMediaPipeConstructor<T>,
  canvas: HTMLCanvasElement | OffscreenCanvas | null | undefined,
  fileset: WasmFileset,
  options: TaskRunnerOptions,
): Promise<T> {
  const fileLocator: FileLocator = {
    locateFile(file): string {
      // We currently only use a single .wasm file and a single .data file (for
      // the tasks that have to load assets). We need to revisit how we
      // initialize the file locator if we ever need to differentiate between
      // diffferent files.
      if (file.endsWith('.wasm')) {
        return fileset.wasmBinaryPath.toString();
      } else if (fileset.assetBinaryPath && file.endsWith('.data')) {
        return fileset.assetBinaryPath.toString();
      }
      return file;
    },
  };

  const instance = await createMediaPipeLib(
    type,
    fileset.wasmLoaderPath,
    fileset.assetLoaderPath,
    canvas,
    fileLocator,
  );
  try {
    instance.enableLogging(options);
    await instance.setOptions(options);
    return instance;
  } catch (error) {
    try {
      instance.close();
    } catch {
      // Preserve the initialization error if releasing resources also fails.
    }
    throw error;
  }
}

/** Base class for all MediaPipe Tasks. */
export abstract class TaskRunner {
  protected abstract baseOptions: BaseOptionsProto;
  protected logger?: TaskLogger;
  private processingErrors: Error[] = [];
  private latestOutputTimestamp = 0;
  private keepaliveNode?: CalculatorGraphConfig.Node;
  private isClosed = false;
  private pendingOptionsUpdate?: Promise<void>;
  private optionsUpdateInProgress = false;
  private readonly modelLoadControllers = new Set<AbortController>();

  /**
   * Creates a new instance of a Mediapipe Task. Determines if SIMD is
   * supported and loads the relevant WASM binary.
   * @return A fully instantiated instance of `T`.
   */
  protected static async createInstance<T extends TaskRunner>(
    type: WasmMediaPipeConstructor<T>,
    canvas: HTMLCanvasElement | OffscreenCanvas | null | undefined,
    fileset: WasmFileset,
    options: TaskRunnerOptions,
  ): Promise<T> {
    return createTaskRunner(type, canvas, fileset, options);
  }

  /** @hideconstructor protected */
  constructor(protected readonly graphRunner: CachedGraphRunner) {
    // Disables the automatic render-to-screen code, which allows for pure
    // CPU processing.
    this.graphRunner.setAutoRenderToScreen(false);
  }

  /** Configures the task with custom options. */
  abstract setOptions(options: TaskRunnerOptions): Promise<void>;

  /**
   * Serializes complete option updates, including validation and rollback.
   * The first update starts synchronously; later updates wait for it to settle.
   */
  protected runWithOptionsUpdate(update: () => Promise<void>): Promise<void> {
    this.assertNotClosed();
    const runUpdate = () => {
      this.assertNotClosed();
      this.optionsUpdateInProgress = true;
      try {
        const result = update();
        if (this.modelLoadControllers.size === 0) {
          // Updates that install synchronously remain immediately usable.
          this.optionsUpdateInProgress = false;
          return result;
        }
        return result.finally(() => {
          this.optionsUpdateInProgress = false;
        });
      } catch (error) {
        this.optionsUpdateInProgress = false;
        throw error;
      }
    };
    const queued = this.pendingOptionsUpdate !== undefined;
    const pending = this.pendingOptionsUpdate
      ? this.pendingOptionsUpdate.then(runUpdate, runUpdate)
      : runUpdate();
    if (!queued && !this.optionsUpdateInProgress) {
      // A completed synchronous update must not delay the next synchronous one.
      return pending;
    }
    this.pendingOptionsUpdate = pending;
    const clearPending = () => {
      if (this.pendingOptionsUpdate === pending) {
        this.pendingOptionsUpdate = undefined;
      }
    };
    void pending.then(clearPending, clearPending);
    return pending;
  }

  /** Rejects work after the task has released its graph resources. */
  protected assertNotClosed(): void {
    if (this.isClosed) {
      throw new Error('Task is closed.');
    }
  }

  /** Prevents frames from reaching a graph while new options are loading. */
  protected assertReadyForProcessing(): void {
    this.assertNotClosed();
    if (this.optionsUpdateInProgress) {
      throw new Error(
        'Task options are being updated. Await setOptions() before processing.',
      );
    }
  }

  /**
   * Installs model data and its graph. Installation can replace native resources,
   * so failures close the task instead of exposing a partially updated graph.
   */
  private installGraph(updateModel?: () => void): void {
    this.assertNotClosed();
    try {
      updateModel?.();
      this.refreshGraph();
      this.onGraphRefreshed();
    } catch (error) {
      try {
        this.close();
      } catch {
        // Preserve the installation error if releasing resources also fails.
      }
      throw error;
    }
  }

  /** Loads a model while allowing close() to cancel it immediately. */
  private loadModelData(
    load: (signal: AbortSignal) => Promise<Uint8Array>,
  ): Promise<Uint8Array> {
    this.assertNotClosed();
    const controller = new AbortController();
    this.modelLoadControllers.add(controller);
    const closed = new Promise<never>((_, reject) => {
      controller.signal.addEventListener('abort', () => {
        reject(new Error('Task is closed.'));
      }, {once: true});
    });
    let loading: Promise<Uint8Array>;
    try {
      loading = load(controller.signal);
    } catch (error) {
      this.modelLoadControllers.delete(controller);
      throw error;
    }
    return Promise.race([loading, closed]).then(
      (buffer) => {
        this.assertNotClosed();
        return buffer;
      },
      (error) => {
        this.assertNotClosed();
        throw error;
      },
    ).finally(() => {
      this.modelLoadControllers.delete(controller);
    });
  }

  /** Returns the public name of the task (e.g. FaceLandmarker). */
  protected abstract getTaskName(): string;

  enableLogging(options: TaskRunnerOptions): void {
    const runningMode = (options as {runningMode: string}).runningMode ?? '';
    const apiKey = this.graphRunner.getMediapipeApiKey();
    this.logger = createTasksLogger(this.getTaskName(), runningMode, apiKey);
  }

  /**
   * Applies the current set of options, including optionally any base options
   * that have not been processed by the task implementation. The options are
   * applied synchronously unless a `modelAssetPath` is provided. This ensures
   * that for most use cases options are applied directly and immediately affect
   * the next inference.
   *
   * @param options The options for the task.
   * @param loadTfliteModel Whether to load the model specified in
   *     `options.baseOptions`.
   */
  protected applyOptions(
    options: TaskRunnerOptions,
    loadTfliteModel = true,
  ): Promise<void> {
    this.assertNotClosed();
    if (loadTfliteModel) {
      const baseOptions: BaseOptions = options.baseOptions || {};

      // Validate that exactly one model is configured
      if (
        options.baseOptions?.modelAssetBuffer &&
        options.baseOptions?.modelAssetPath
      ) {
        throw new Error(
          'Cannot set both baseOptions.modelAssetPath and baseOptions.modelAssetBuffer',
        );
      } else if (
        !(
          this.baseOptions.getModelAsset()?.hasFileContent() ||
          this.baseOptions.getModelAsset()?.hasFileName() ||
          options.baseOptions?.modelAssetBuffer ||
          options.baseOptions?.modelAssetPath
        )
      ) {
        throw new Error(
          'Either baseOptions.modelAssetPath or baseOptions.modelAssetBuffer must be set',
        );
      }

      this.setAcceleration(baseOptions);
      if (baseOptions.modelAssetPath) {
        // We don't use `await` here since we want to apply most settings
        // synchronously.
        const modelAssetPath = baseOptions.modelAssetPath;
        return this.loadModelData((signal) =>
          fetch(modelAssetPath.toString(), {signal})
            .then((response) => {
              if (!response.ok) {
                throw new Error(
                  `Failed to fetch model: ${modelAssetPath} (${response.status})`,
                );
              }
              return response.arrayBuffer();
            })
            .then((buffer) => new Uint8Array(buffer)),
        ).then((buffer) => {
          this.installGraph(() => {
            try {
              // Try to delete file as we cannot overwrite an existing file
              // using our current API.
              this.graphRunner.wasmModule.FS_unlink('/model.dat');
            } catch {}
            // TODO: Consider passing the model to the graph as an
            // input side packet as this might reduce copies.
            this.graphRunner.wasmModule.FS_createDataFile(
              '/',
              'model.dat',
              buffer,
              /* canRead= */ true,
              /* canWrite= */ false,
              /* canOwn= */ false,
            );
            this.setExternalFile('/model.dat');
          });
        });
      } else if (baseOptions.modelAssetBuffer instanceof Uint8Array) {
        this.setExternalFile(baseOptions.modelAssetBuffer);
      } else if (baseOptions.modelAssetBuffer) {
        const reader = baseOptions.modelAssetBuffer;
        return this.loadModelData((signal) => {
          const cancelReader = () => {
            void reader.cancel().catch(() => {});
          };
          signal.addEventListener('abort', cancelReader, {once: true});
          return streamToUint8Array(reader).finally(() => {
            signal.removeEventListener('abort', cancelReader);
          });
        }).then((buffer) => {
          this.installGraph(() => this.setExternalFile(buffer));
        });
      }
    }

    // If there is no model to download, we can apply the setting synchronously.
    this.installGraph();
    return Promise.resolve();
  }

  /** Appliest the current options to the MediaPipe graph. */
  protected abstract refreshGraph(): void;

  /**
   * Callback that gets invoked once a new graph configuration has been
   * applied.
   */
  protected onGraphRefreshed(): void {}

  /** Returns the current CalculatorGraphConfig. */
  protected getCalculatorGraphConfig(): CalculatorGraphConfig {
    let config: CalculatorGraphConfig | undefined;
    this.graphRunner.getCalculatorGraphConfig((binaryData) => {
      config = CalculatorGraphConfig.deserializeBinary(binaryData);
    });
    if (!config) {
      throw new Error('Failed to retrieve CalculatorGraphConfig');
    }
    return config;
  }

  /**
   * Takes the raw data from a MediaPipe graph, and passes it to C++ to be run
   * over the video stream. Will replace the previously running MediaPipe graph,
   * if there is one.
   * @param graphData The raw MediaPipe graph data, either in binary
   *     protobuffer format (.binarypb), or else in raw text format (.pbtxt or
   *     .textproto).
   * @param isBinary This should be set to true if the graph is in
   *     binary format, and false if it is in human-readable text format.
   */
  protected setGraph(graphData: Uint8Array, isBinary: boolean): void {
    this.assertNotClosed();
    this.graphRunner.attachErrorListener((code, message) => {
      this.processingErrors.push(new Error(message));
    });

    // Enables use of our model resource caching graph service; we apply this to
    // every MediaPipe graph we run.
    this.graphRunner.registerModelResourcesGraphService();

    this.graphRunner.setGraph(graphData, isBinary);
    this.logger?.logSessionStart();
    this.keepaliveNode = undefined;
    this.handleErrors();
  }

  /**
   * Signals beginning of graph processing.
   * @param timestamp The timestamp of the input packets.
   */
  protected startProcessing(timestamp?: number): void {
    this.assertReadyForProcessing();
    if (this.logger && timestamp !== undefined) {
      if (this.baseOptions.getAcceleration()?.hasGpu()) {
        this.logger.recordGpuInputArrival(timestamp);
      } else {
        this.logger.recordCpuInputArrival(timestamp);
      }
    }
  }

  /**
   * Forces all queued-up packets to be pushed through the MediaPipe graph as
   * far as possible, performing all processing until no more processing can be
   * done.
   */
  protected finishProcessing(timestamp?: number): void {
    this.assertNotClosed();
    this.graphRunner.finishProcessing();
    this.handleErrors();
    if (this.logger && timestamp !== undefined) {
      this.logger.recordInvocationEnd(timestamp);
    }
  }

  /*
   * Sets the latest output timestamp received from the graph (in ms).
   * Timestamps that are smaller than the currently latest output timestamp are
   * ignored.
   */
  protected setLatestOutputTimestamp(timestamp: number): void {
    this.latestOutputTimestamp = Math.max(
      this.latestOutputTimestamp,
      timestamp,
    );
  }

  /**
   * Gets a synthetic timestamp in ms that can be used to send data to the
   * next packet. The timestamp is one millisecond past the last timestamp
   * received from the graph.
   */
  protected getSyntheticTimestamp(): number {
    return this.latestOutputTimestamp + 1;
  }

  /** Throws the error from the error listener if an error was raised. */
  private handleErrors() {
    try {
      const errorCount = this.processingErrors.length;
      if (errorCount === 1) {
        // Re-throw error to get a more meaningful stacktrace
        throw new Error(this.processingErrors[0].message);
      } else if (errorCount > 1) {
        throw new Error(
          'Encountered multiple errors: ' +
            this.processingErrors.map((e) => e.message).join(', '),
        );
      }
    } finally {
      this.processingErrors = [];
    }
  }

  /** Configures the `externalFile` option */
  protected setExternalFile(modelAssetPath?: string): void;
  protected setExternalFile(modelAssetBuffer?: Uint8Array): void;
  protected setExternalFile(
    modelAssetPathOrBuffer?: Uint8Array | string,
  ): void {
    const externalFile = this.baseOptions.getModelAsset() || new ExternalFile();
    if (typeof modelAssetPathOrBuffer === 'string') {
      externalFile.setFileName(modelAssetPathOrBuffer);
      externalFile.clearFileContent();
    } else if (modelAssetPathOrBuffer instanceof Uint8Array) {
      externalFile.setFileContent(modelAssetPathOrBuffer);
      externalFile.clearFileName();
    }
    this.baseOptions.setModelAsset(externalFile);
  }

  /** Configures the `acceleration` option. */
  private setAcceleration(options: BaseOptions) {
    let acceleration = this.baseOptions.getAcceleration();

    if (!acceleration) {
      // Create default instance for the initial configuration.
      acceleration = new Acceleration();
      acceleration.setTflite(new InferenceCalculatorOptions.Delegate.TfLite());
    }

    if ('delegate' in options) {
      if (options.delegate === 'GPU') {
        acceleration.setGpu(new InferenceCalculatorOptions.Delegate.Gpu());
      } else {
        acceleration.setTflite(
          new InferenceCalculatorOptions.Delegate.TfLite(),
        );
      }
    }

    this.baseOptions.setAcceleration(acceleration);
  }

  /**
   * Adds a node to the graph to temporarily keep certain streams alive.
   * NOTE: To use this call, PassThroughCalculator must be included in your wasm
   *     dependencies.
   */
  protected addKeepaliveNode(graphConfig: CalculatorGraphConfig) {
    this.keepaliveNode = new CalculatorGraphConfig.Node();
    this.keepaliveNode.setCalculator('PassThroughCalculator');
    this.keepaliveNode.addInputStream(FREE_MEMORY_STREAM);
    this.keepaliveNode.addOutputStream(
      FREE_MEMORY_STREAM + UNUSED_STREAM_SUFFIX,
    );
    graphConfig.addInputStream(FREE_MEMORY_STREAM);
    graphConfig.addNode(this.keepaliveNode);
  }

  /** Adds streams to the keepalive node to be kept alive until callback. */
  protected keepStreamAlive(streamName: string) {
    this.keepaliveNode!.addInputStream(streamName);
    this.keepaliveNode!.addOutputStream(streamName + UNUSED_STREAM_SUFFIX);
  }

  /** Frees any streams being kept alive by the keepStreamAlive callback. */
  protected freeKeepaliveStreams() {
    this.graphRunner.addBoolToStream(
      true,
      FREE_MEMORY_STREAM,
      this.latestOutputTimestamp,
    );
  }

  /**
   * Closes and cleans up the resources held by this task.
   * @export
   */
  close(): void {
    if (this.isClosed) {
      return;
    }
    this.isClosed = true;
    for (const controller of this.modelLoadControllers) {
      controller.abort();
    }
    this.keepaliveNode = undefined;
    this.logger?.logSessionEnd();
    this.logger?.close();
    this.graphRunner.closeGraph();
  }
}


