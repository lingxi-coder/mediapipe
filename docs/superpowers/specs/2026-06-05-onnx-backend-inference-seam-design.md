# Pluggable Inference Backend Seam — ONNX Runtime CPU reference backend (design)

- **Date:** 2026-06-05
- **Status:** Approved design. First sub-project of **Phase 6 (Backend inference)** in `2026-06-01-roadmap.md`.
- **Build model:** Fork / custom build of MediaPipe (not an upstream PR). Work lands on `dev`.
- **Branch:** `dev` (base `master`).

## 1. Purpose & relationship to the roadmap

Phase 6 of the roadmap is "pluggable inference backend + ONNX / PyTorch / TensorRT / CoreML, each
platform-gated." That is **several independent subsystems**, so it is decomposed: this spec covers the
**first sub-project** — establish that a *non-TFLite* backend can run the real tiled YOLO/OBB detector —
proven end-to-end with **one reference backend: ONNX Runtime on CPU**. Each remaining backend
(PyTorch, TensorRT, CoreML) becomes its own later spec following the same template established here.

Phase 7 (LiteRT-Next backend-wide zero-copy) depends on this seam but is out of scope here.

### Decisions locked during brainstorming

1. **Reference backend:** ONNX Runtime, CPU, float32.
2. **Architecture:** a **standalone `OnnxInferenceCalculator`** (a drop-in node), *not* a new branch in the
   shared `InferenceCalculator` selector and *not* a new `Delegate` oneof entry. The shared
   `inference_calculator.{h,cc,proto}` are **not modified**.
3. **Proof depth:** **full detector e2e** — an additive `backend` option (default TFLite) on the YOLO/OBB
   graph builders selects the ONNX calculator; the proof runs the real tiled detector on `boats.jpg`
   against the same oracle the TFLite tests use.
4. **Scope (YAGNI):** CPU / float32 / macOS-wired onnxruntime only. Out: GPU execution providers
   (CoreML/CUDA EP), quantized & multi-signature models, Linux/Windows wiring, the other backends.

## 2. Current architecture (the seam this builds on)

Established by reading the tree on `dev`:

- **`InferenceCalculator` is a selector subgraph.** `InferenceCalculatorSelectorImpl::GetConfig`
  (`mediapipe/calculators/tensor/inference_calculator.cc:46`) reads `options.delegate`, builds an ordered
  list of suffixes (`Metal`,`Gl`,`GlAdvanced`,`Cpu`,`Xnnpack`), and rewrites the node to the **first
  registered** `InferenceCalculator<Suffix>`. This is TFLite-centric: every "delegate" drives the same
  TFLite interpreter. We deliberately do **not** route ONNX through here (Approach C).
- **`InferenceRunner` (`mediapipe/calculators/tensor/inference_runner.h`) is the clean per-backend
  interface:** `Run(cc, tensor_span) -> std::vector<Tensor>`, `GetInputOutputTensorNames()`, and
  `GetModelMetadata() -> InferenceMetadata` (non-pure; backends opt in by overriding). TFLite's impl is
  `InferenceInterpreterDelegateRunner`. **We reuse this interface** for the ONNX backend.
- **The Phase-1 metadata seam was built for exactly this.** `InferenceMetadata`
  (`mediapipe/framework/formats/inference_metadata.proto`) carries `model_id`, `input[]`/`output[]`
  (`TensorSpec`: name, shape, dtype, quant), `backend`, `batch_capacity`, `is_dynamic_batch`,
  `input_height/width/channels`, `tensor_layout`, `class_count`. The TFLite runner populates it
  (`inference_interpreter_delegate_runner.cc:294`: `set_backend("cpu")`, `set_tensor_layout("BHWC")`,
  H/W/C = `dims[1/2/3]`, `batch_capacity = dims[0]`). The detection-core spec §9 explicitly notes the
  metadata is read from the *loaded instance* "because the M5 backends (TensorRT/CoreML/ONNX) can only be
  introspected once loaded," and `backend` (field 4) = "backend actually selected (post-init)." The
  tiler/batcher (`StreamingTilesToTensorBatchCalculator`, `TileSpecToTilePlanCalculator`) consume
  `InferenceMetadata` as the source of truth for input geometry/dtype/layout/`batch_capacity`. **So the
  detector decode path is already backend-agnostic** — a new backend only needs to (a) produce `Tensor`s
  and (b) report `InferenceMetadata`.
- **No ONNX/PyTorch/TensorRT/CoreML/LiteRT presence in-tree.** Backends start from zero.

## 3. Architecture of this sub-project

`OnnxInferenceCalculator` is a drop-in for `InferenceCalculator`: identical I/O contract, backed by an
`OnnxInferenceRunner : InferenceRunner`. Nothing shared is modified.

```
YOLO/OBB graph builder
  ├─ backend = TFLITE (default, unset) → AddInference(...) → InferenceCalculator   [today, unchanged]
  └─ backend = ONNX                    → OnnxInferenceCalculator
                                           └─ OnnxInferenceRunner : InferenceRunner
                                                ├─ Run():  NHWC Tensor → (transpose) → ORT CPU → Tensor
                                                ├─ GetModelMetadata(): InferenceMetadata{backend="onnx", layout="BHWC", H/W/C, ...}
                                                └─ GetInputOutputTensorNames()
  downstream tiler / decode / NMS / projection — IDENTICAL for both backends
```

The detector graph stays backend-agnostic because the tiler/decoder consume `InferenceMetadata`, not
TFLite specifics. `backend=ONNX` is additive and default-off, so the TFLite path is byte-identical when
unset.

## 4. Components (files, each with one responsibility)

### 4.1 Bazel dependency wrap (mirror the OpenCV precedent)

- **Create `third_party/onnxruntime_macos.BUILD`** — a `cc_library` named `onnxruntime`:
  - `srcs = glob(["opt/onnxruntime/lib/libonnxruntime.dylib"])`
  - `hdrs = glob(["opt/onnxruntime/include/onnxruntime/*.h"])`
  - `includes = ["opt/onnxruntime/include/onnxruntime"]`
  - `linkstatic = 1`, `visibility = ["//visibility:public"]`
- **Add to `WORKSPACE`** a `new_local_repository(name = "macos_onnxruntime", build_file =
  "@//third_party:onnxruntime_macos.BUILD", path = "/opt/homebrew")`, modeled exactly on `macos_opencv`.
  The version-independent Homebrew symlink `/opt/homebrew/opt/onnxruntime` resolves to the current Cellar
  version, so this needs no edit after `brew upgrade onnxruntime`.
- The C++ API header is `/opt/homebrew/opt/onnxruntime/include/onnxruntime/onnxruntime_cxx_api.h`; the
  dylib is `/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib` (both verified present, onnxruntime
  1.24.3).

### 4.2 `mediapipe/calculators/tensor/onnx_inference_runner.{h,cc}`

`class OnnxInferenceRunner : public InferenceRunner`. Responsibilities:

- **Construction** (`Create(model_path, intra_op_num_threads)` factory returning
  `absl::StatusOr<std::unique_ptr<OnnxInferenceRunner>>`): build `Ort::Env`, `Ort::SessionOptions`
  (intra-op threads; default CPU execution provider), `Ort::Session(env, model_path.c_str(), opts)`.
  Cache input/output names and the input `TensorSpec`s. Any ORT exception is converted to `absl::Status`.
- **`Run(cc, tensor_span)`**: for each input `Tensor`, take a CPU read view; adapt layout (§5); wrap as an
  `Ort::Value` over a CPU `Ort::MemoryInfo`; call `session_.Run(...)`; copy each output `Ort::Value` into
  a mediapipe `Tensor(Tensor::ElementType::kFloat32, shape)` via its CPU write view. Returns
  `std::vector<Tensor>` in model output order.
- **`GetInputOutputTensorNames()`**: return the cached names as `InputOutputTensorNames` (the
  single-signature map type from `inference_io_mapper.h`), so the base IO-mapping contract is satisfiable.
- **`GetModelMetadata()`**: build `InferenceMetadata` from the session — `set_backend("onnx")`, one
  `TensorSpec` per input/output (name, shape, dtype `"float32"`), and the NHWC-facing convenience fields
  `input_height/width/channels`, `tensor_layout("BHWC")`, `batch_capacity`, `is_dynamic_batch` (see §5).

### 4.3 `mediapipe/calculators/tensor/onnx_inference_calculator.{cc,proto}` (+ BUILD)

- **`onnx_inference_calculator.proto`**: `message OnnxInferenceCalculatorOptions { optional string
  model_path = 1; optional int32 intra_op_num_threads = 2 [default = -1]; }` extending
  `CalculatorOptions`.
- **`onnx_inference_calculator.cc`**: an api2 calculator registered as `OnnxInferenceCalculator` with a
  contract that mirrors `InferenceCalculator`: `Input<std::vector<Tensor>>::Optional{"TENSORS"}`,
  `Output<std::vector<Tensor>>::Optional{"TENSORS"}`, and `SideOutput<InferenceMetadata>::Optional
  {"METADATA"}`. `Open()` creates the `OnnxInferenceRunner` from options and, **only on success**, emits
  `METADATA` exactly once (matching `InferenceCalculator`'s gating). `Process()` forwards `TENSORS`
  through `runner_->Run(...)`. `Close()` releases the runner.
- **BUILD target** is macOS-gated via `select()` (see §8) and depends on `@macos_onnxruntime//:onnxruntime`.

### 4.4 Detector graph backend option

- **Add to `YoloObjectDetectorOptions` and `OrientedObjectDetectorOptions`** (their `proto/*.proto`) an
  additive backend selector:
  ```proto
  message Backend {
    message Tflite {}
    message Onnx { optional string model_path = 1; }   // path to the .onnx model
    oneof backend { Tflite tflite = 1; Onnx onnx = 2; }
  }
  optional Backend backend = 10;   // YoloObjectDetectorOptions: next free field is 10
  // optional Backend backend = 11; // OrientedObjectDetectorOptions: next free field is 11
  ```
  (`YoloObjectDetectorOptions` uses fields 1–9, so `backend` = 10; `OrientedObjectDetectorOptions` uses
  fields 1–10, so `backend` = 11. Unset ⇒ TFLite via `base_options`, unchanged.)
  Keeping the ONNX model path in this `backend.onnx` message leaves `base_options.model_asset_path`
  pristine for the TFLite path (resolved judgment call — see §11).
- **In `yolo_object_detector_graph.cc` / `oriented_object_detector_graph.cc`**: where the graph currently
  calls `AddInference(...)` (e.g. `yolo_object_detector_graph.cc:213`), branch on `options.backend`:
  TFLite (default/unset) keeps the existing `AddInference(...)` call verbatim; ONNX adds an
  `OnnxInferenceCalculator` node, sets its `model_path` from `backend.onnx.model_path`, and wires
  `TENSORS` (from preprocessing) and the optional `METADATA` side output **identically** to the TFLite
  branch. Everything downstream is untouched.

### 4.5 Fixture export

- **`mediapipe/tasks/testdata/vision/export_yolov8n_onnx.py`** — ultralytics
  `YOLO(...).export(format="onnx", opset=17)` producing `yolov8n.onnx` and `yolov8n-obb.onnx` (opset 17 is
  broadly supported by onnxruntime 1.24; adjust only if export fails).
  Gitignored, mirroring the existing `export_yolov8n_tflite.py` / `export_yolov8n_obb_tflite.py` scripts.
- **`mediapipe/tasks/testdata/vision/BUILD`**: extend the existing `yolo_test_model` / `yolo_obb_test_model`
  `glob(..., allow_empty=True)` filegroups to include `yolov8n.onnx` / `yolov8n-obb.onnx`, so absence on a
  fresh clone / CI keeps the build green and the gated test `GTEST_SKIP()`s.

## 5. Layout handling (the one nuanced point)

ultralytics ONNX export is **NCHW** `[1,3,640,640]`; the tiler produces **NHWC** `[1,640,640,3]` (what
TFLite expects). The adaptation is **confined to `OnnxInferenceRunner`**:

- `GetModelMetadata()` reports the NHWC-facing geometry (`input_height/width/channels` from H,W,C and
  `tensor_layout("BHWC")`), so the tiler packs exactly as it does for TFLite. `batch_capacity` and
  `is_dynamic_batch` come from the ONNX input's batch dim (the detector runs N=1).
- `Run()` transposes the incoming NHWC `Tensor` to NCHW before constructing the input `Ort::Value`, and
  copies outputs through unchanged.

Result: the tiler and detector graph are byte-identical across backends; only the runner knows ONNX wants
NCHW. yolov8 output is channels-first `[1, 4+nc(+1 for OBB angle), 8400]`, the same convention the
existing OBB decode path already handles, so the decoder is unchanged. Output-tensor *ordering* parity
with the TFLite decode path is asserted by the tests (§7).

## 6. Error handling

- **Missing/unbuilt dylib** → bazel build error (or a clear ORT load failure), surfaced at build/init.
- **Missing or invalid `.onnx`** → `Open()` returns a non-OK `absl::Status`; `METADATA` is **not** emitted.
- **Input shape/dtype mismatch** vs the session's expected input → `RET_CHECK` reporting expected-vs-actual.
- **Dynamic batch** (input dim `-1`) → reported via `is_dynamic_batch`; v1 supports the detector's fixed
  N=1 and fails loudly on other batch sizes.
- **ORT exceptions** are caught at every boundary and converted to `absl::Status` (no exceptions escape
  into the framework).

## 7. Testing strategy

1. **Runner unit test** (`onnx_inference_runner_test.cc`): load `yolov8n.onnx`; assert `GetModelMetadata()`
   fields (H/W/C, layout `"BHWC"`, `backend=="onnx"`, `class_count`); run a deterministic input; assert
   output shape `[1, C, 8400]`.
2. **Calculator drop-in test** (`onnx_inference_calculator_test.cc`): feed one input tensor through both
   `OnnxInferenceCalculator` and the TFLite `InferenceCalculator`; assert tensor-level closeness within a
   cross-runtime tolerance. Confirms the drop-in contract + layout handling.
3. **Full detector e2e** (the milestone proof): build YOLO and OBB graphs with `backend=ONNX` and run on
   `boats.jpg`, asserting the **same oracle** the TFLite detector tests use (ships at expected locations,
   within the existing positional/score tolerances). Proves a real non-TFLite backend runs the real tiled
   detector and that the metadata seam keeps the tiler backend-agnostic.
4. **CPU regression**: the existing TFLite YOLO/OBB detector tests stay green and byte-identical with
   `backend` unset.

All gated tests `GTEST_SKIP()` when the `.onnx` fixture is absent (fresh clone / CI), consistent with the
existing YOLO/OBB fixture handling.

## 8. Platform gating

- v1 wires `macos_onnxruntime` only. The `onnx_inference_calculator` / `onnx_inference_runner` BUILD
  targets are macOS-gated with `select()` (other OSes compile them out / do not depend on onnxruntime).
- On a non-macOS build, `OnnxInferenceCalculator` is simply **not registered**; selecting `backend=ONNX`
  then fails with a clear "OnnxInferenceCalculator is not registered on this platform" error rather than a
  silent fallback.
- All new options are default-off, so no existing build or target is affected. Linux/Windows onnxruntime
  repos are a later increment (the `select()` leaves explicit room for them).

## 9. Risks & mitigations

1. **Bazel link of the Homebrew dylib** — mitigated by the exact OpenCV `new_local_repository` + `*.BUILD`
   precedent already used in this repo; onnxruntime is verified installed at `/opt/homebrew/opt/onnxruntime`.
2. **NHWC↔NCHW input transpose + output ordering parity** — confined to the runner and verified by tests 2
   and 3.
3. **Cross-runtime numeric drift in the oracle** — use the same tolerance style as the existing oracle
   tests; the semantic oracle (correct ships detected) is the primary assertion, not bit-equality with
   TFLite.
4. **ONNX export fidelity** — pin the ultralytics export opset in `export_yolov8n_onnx.py`; the script
   documents the exact export invocation.
5. **Build verifiability** — this machine builds+runs desktop C++ (`DISABLE_GPU=1`), Metal, and C-API
   `cc_test`s. The ONNX backend is desktop CPU C++ + a Homebrew dylib, so it is fully **buildable and
   runnable here** (unlike GLES). This is why ONNX-CPU-on-macOS was chosen as the first reference backend.

## 10. Deferred work & forward hooks

- Each remaining backend (PyTorch/libtorch, TensorRT on NVIDIA, CoreML on Apple) = its own spec following
  this `InferenceRunner`-impl + standalone-calculator template.
- GPU execution providers (CoreML/CUDA EP for ONNX), quantized & multi-signature models, and Linux/Windows
  onnxruntime repos are later increments on this same seam.
- Phase 7 (LiteRT-Next `TensorBuffer` interop) consumes this backend layer; nothing here needs revisiting
  for it.

## 11. Resolved judgment calls

- **§5 — confine the NCHW transpose inside the runner** (vs exporting an NHWC-input ONNX model). Chosen so
  the tiler/detector graph stay byte-identical to the TFLite path; the only backend-specific knowledge
  lives in `OnnxInferenceRunner`.
- **§4.4 — carry the ONNX model path in a `backend.onnx` message on the detector options** (vs reusing
  `base_options.model_asset_path`). Chosen so `base_options` stays pristine for the unchanged TFLite path
  and the backend choice is explicit and self-contained.
