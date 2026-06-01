# Inference Metadata Seam Implementation Plan (Group 1, Plan 3a of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement M4 (Approach B): source static, model-level `InferenceMetadata` from the **loaded** inference instance via a new read-only `InferenceRunner::GetModelMetadata()`, and surface it from `InferenceCalculator` as an **opt-in `METADATA` output side packet** emitted once after successful init. This metadata becomes the source of truth that drives tile packaging in Plan 3b.

**Architecture:** A contained, additive, guarded fork of the inference core (the spec's one sanctioned exception to "don't touch shared code"). `InferenceRunner` gains a **non-pure** virtual `GetModelMetadata()` defaulting to `Unimplemented` (so the GL/Metal/other runners compile unchanged); only the CPU/TFLite `InferenceInterpreterDelegateRunner` overrides it, reading the live interpreter's input/output `TfLiteTensor`s. `InferenceCalculator` declares an optional `METADATA` output side packet and, when wired, sets it in `Open()` after the runner is built. When `METADATA` is not wired, behavior is byte-for-byte upstream.

**Tech Stack:** C++17, MediaPipe api2, TFLite C++ `Interpreter`, protobuf2, Bazel, GoogleTest.

**Spec:** `docs/superpowers/specs/2026-06-01-detection-core-yolo-obb-tiling-design.md` (§5.5 InferenceMetadata; §9 M4; §3 runner finding).

**Depends on:** nothing in Plans 1–2 (orthogonal); but it is a prerequisite for Plan 3b.

**Key code facts (verified in `inference_interpreter_delegate_runner.cc`):**
- The runner holds `std::unique_ptr<Interpreter> interpreter_`.
- `interpreter_->inputs()` / `interpreter_->outputs()` return `std::vector<int>` tensor indices.
- `interpreter_->tensor(idx)` → `TfLiteTensor*`; `t->dims->size`, `t->dims->data[i]`, `t->type` (`TfLiteType`), `t->params.scale`, `t->params.zero_point`, and `t->dims_signature` (may be null; `-1` marks a dynamic dim).
- The interface `InferenceRunner` lives in `inference_runner.h` with currently two pure-virtual methods.

---

## File structure

- Create: `mediapipe/framework/formats/inference_metadata.proto` — `TensorSpec`, `InferenceMetadata`.
- Modify: `mediapipe/framework/formats/BUILD` — proto target.
- Modify: `mediapipe/calculators/tensor/inference_runner.h` — add non-pure `GetModelMetadata()`.
- Modify: `mediapipe/calculators/tensor/inference_interpreter_delegate_runner.{cc,BUILD-deps}` — override it.
- Modify: `mediapipe/calculators/tensor/inference_calculator.h` — declare optional `METADATA` side output.
- Modify: `mediapipe/calculators/tensor/inference_calculator.cc` — set it in `Open()` when wired.
- Create: `mediapipe/calculators/tensor/inference_calculator_metadata_test.cc` — graph test of the side packet.
- Modify: `mediapipe/calculators/tensor/BUILD` — proto dep + test target.

All commands from repo root with `--define MEDIAPIPE_DISABLE_GPU=1`.

---

### Task 1: `InferenceMetadata` proto

**Files:**
- Create: `mediapipe/framework/formats/inference_metadata.proto`
- Modify: `mediapipe/framework/formats/BUILD`

- [ ] **Step 1: Write the proto** (from spec §5.5):

```proto
// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

syntax = "proto2";

package mediapipe;

message TensorSpec {
  optional string name = 1;
  repeated int32 shape = 2 [packed = true];   // includes batch dim; -1 = dynamic
  optional string dtype = 3;            // "float32", "uint8", "int8", ...
  optional float quant_scale = 4;
  optional int32 quant_zero_point = 5;
}

// Static, model-level metadata. Read from the LOADED inference instance and
// emitted exactly once at InferenceCalculator.Open() as an opt-in side packet.
message InferenceMetadata {
  optional string model_id = 1;
  repeated TensorSpec input = 2;
  repeated TensorSpec output = 3;
  optional string backend = 4;          // backend actually selected (post-init)
  optional int32 batch_capacity = 5;    // model input batch dim (>0 fixed)
  optional bool is_dynamic_batch = 6;
  optional int32 input_height = 7;
  optional int32 input_width = 8;
  optional int32 input_channels = 9;
  optional string tensor_layout = 10;   // e.g. "BHWC"
  optional int32 class_count = 11;
}
```

- [ ] **Step 2: Add the proto target** in `mediapipe/framework/formats/BUILD` (near `detection_proto`):

```python
mediapipe_proto_library(
    name = "inference_metadata_proto",
    srcs = ["inference_metadata.proto"],
)
```

- [ ] **Step 3: Build**

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/framework/formats:inference_metadata_cc_proto
```
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add mediapipe/framework/formats/inference_metadata.proto mediapipe/framework/formats/BUILD
git commit -m "feat(metadata): add InferenceMetadata proto"
```

---

### Task 2: `InferenceRunner::GetModelMetadata()` interface (default Unimplemented)

**Files:**
- Modify: `mediapipe/calculators/tensor/inference_runner.h`

Adding a **non-pure** virtual keeps every existing runner (GL, Metal, advanced) compiling untouched — only the delegate runner overrides it in Task 3.

- [ ] **Step 1: Read the current interface**

Run: `sed -n '1,40p' mediapipe/calculators/tensor/inference_runner.h`
Confirm the class has `Run(...)` and `GetInputOutputTensorNames()` as pure virtuals, and note the includes.

- [ ] **Step 2: Add the include and the default method**

In `mediapipe/calculators/tensor/inference_runner.h`:
(a) add near the other includes:
```cpp
#include "mediapipe/framework/formats/inference_metadata.pb.h"
```
(b) inside `class InferenceRunner`, after `GetInputOutputTensorNames()`'s declaration, add:
```cpp
  // Returns static, model-level metadata read from the loaded instance.
  // Default: not implemented (backends opt in by overriding). Callers that
  // need metadata must use a backend that overrides this (CPU/TFLite does).
  virtual absl::StatusOr<InferenceMetadata> GetModelMetadata() const {
    return absl::UnimplementedError(
        "GetModelMetadata is not implemented for this inference backend");
  }
```
Confirm `absl::StatusOr` and `absl::UnimplementedError` are available (the header already includes `absl/status/statusor.h`; if not, add `#include "absl/status/status.h"` and `#include "absl/status/statusor.h"`).

- [ ] **Step 3: Add the proto dep** to the `inference_runner` cc_library in `mediapipe/calculators/tensor/BUILD`. First locate it: `grep -n "name = \"inference_runner\"" mediapipe/calculators/tensor/BUILD`. Add to its `deps`:
```python
        "//mediapipe/framework/formats:inference_metadata_cc_proto",
```

- [ ] **Step 4: Build the runner header's library to verify it compiles**

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:inference_runner
```
Expected: success (header-only/with its lib). If `inference_runner` is not a standalone target, build a small dependent like `//mediapipe/calculators/tensor:inference_interpreter_delegate_runner` instead and confirm it still compiles.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/inference_runner.h mediapipe/calculators/tensor/BUILD
git commit -m "feat(metadata): add InferenceRunner::GetModelMetadata (default Unimplemented)"
```

---

### Task 3: Implement `GetModelMetadata()` in the CPU/TFLite delegate runner

**Files:**
- Modify: `mediapipe/calculators/tensor/inference_interpreter_delegate_runner.cc`
- Modify: `mediapipe/calculators/tensor/BUILD` (deps)

- [ ] **Step 1: Add a TfLiteType→string helper and the override declaration**

In `inference_interpreter_delegate_runner.cc`, near the top anonymous namespace, add:
```cpp
namespace {
std::string TfLiteTypeName(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32: return "float32";
    case kTfLiteUInt8:   return "uint8";
    case kTfLiteInt8:    return "int8";
    case kTfLiteInt32:   return "int32";
    case kTfLiteInt64:   return "int64";
    case kTfLiteFloat16: return "float16";
    default:             return "unknown";
  }
}
}  // namespace
```
(If an anonymous namespace already exists at the top, add the function inside it instead of opening a new one.)

In the class declaration `class InferenceInterpreterDelegateRunner : public InferenceRunner`, add the override next to `GetInputOutputTensorNames`:
```cpp
  absl::StatusOr<InferenceMetadata> GetModelMetadata() const override;
```

- [ ] **Step 2: Implement the override** (append near the other out-of-line method definitions in the .cc; the interpreter access pattern matches existing code at lines ~149–190):

```cpp
absl::StatusOr<InferenceMetadata>
InferenceInterpreterDelegateRunner::GetModelMetadata() const {
  RET_CHECK(interpreter_ != nullptr);
  InferenceMetadata md;
  md.set_backend("cpu");  // delegate runner = CPU/XNNPACK/NNAPI family

  auto fill_spec = [](const TfLiteTensor* t, TensorSpec* spec) {
    if (t->name != nullptr) spec->set_name(t->name);
    if (t->dims != nullptr) {
      for (int i = 0; i < t->dims->size; ++i) spec->add_shape(t->dims->data[i]);
    }
    spec->set_dtype(TfLiteTypeName(t->type));
    spec->set_quant_scale(t->params.scale);
    spec->set_quant_zero_point(t->params.zero_point);
  };

  for (int idx : interpreter_->inputs()) {
    fill_spec(interpreter_->tensor(idx), md.add_input());
  }
  for (int idx : interpreter_->outputs()) {
    fill_spec(interpreter_->tensor(idx), md.add_output());
  }

  // Derive image-input geometry + batch from the FIRST input tensor (BHWC).
  if (!interpreter_->inputs().empty()) {
    const TfLiteTensor* in = interpreter_->tensor(interpreter_->inputs()[0]);
    if (in->dims != nullptr && in->dims->size == 4) {
      md.set_batch_capacity(in->dims->data[0]);
      md.set_input_height(in->dims->data[1]);
      md.set_input_width(in->dims->data[2]);
      md.set_input_channels(in->dims->data[3]);
      md.set_tensor_layout("BHWC");
    }
    // Dynamic batch: dims_signature carries -1 for unresolved dims.
    bool dynamic_batch = false;
    if (in->dims_signature != nullptr && in->dims_signature->size >= 1) {
      dynamic_batch = (in->dims_signature->data[0] == -1);
    }
    md.set_is_dynamic_batch(dynamic_batch);
  }
  return md;
}
```

Notes for the implementer:
- `TfLiteTensor`, `TfLiteType`, `kTfLite*` come from TFLite headers already transitively included by this file (it already uses `interpreter_->tensor(...)->dims`). If `TfLiteType` enum names differ in this TFLite version, adjust the switch to the names that compile (confirm with the headers under the TFLite repo dep). `class_count` is intentionally NOT set here (it is a decoder option, not a model-introspectable fact for arbitrary models).

- [ ] **Step 3: Ensure the proto dep is on the delegate-runner library**

`grep -n "name = \"inference_interpreter_delegate_runner\"" mediapipe/calculators/tensor/BUILD`, then add to its `deps` (if not already pulled transitively):
```python
        "//mediapipe/framework/formats:inference_metadata_cc_proto",
```

- [ ] **Step 4: Build**

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:inference_interpreter_delegate_runner
```
Expected: success. Fix any TfLiteType name mismatch revealed by the compiler.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/inference_interpreter_delegate_runner.cc mediapipe/calculators/tensor/BUILD
git commit -m "feat(metadata): implement GetModelMetadata in TFLite delegate runner"
```

---

### Task 4: Opt-in `METADATA` side output on `InferenceCalculator`

**Files:**
- Modify: `mediapipe/calculators/tensor/inference_calculator.h`
- Modify: `mediapipe/calculators/tensor/inference_calculator.cc`

- [ ] **Step 1: Read where the contract ports and the runner are declared**

Run:
```bash
grep -n "kSideInModel\|OutputSidePacket\|SideOutput\|class InferenceCalculator\|inference_runner_\|::Open\|MEDIAPIPE_NODE_CONTRACT\|kOutTensors\|GetContract\|UpdateContract" mediapipe/calculators/tensor/inference_calculator.h
grep -n "inference_runner_\|::Open\|runner" mediapipe/calculators/tensor/inference_calculator.cc | head
```
Note the exact api2 port style used (this base uses api2 with `Input/Output/SideInput`), and how/where the runner is created in `Open`.

- [ ] **Step 2: Declare the optional side output** in `inference_calculator.h`

Add the include:
```cpp
#include "mediapipe/framework/formats/inference_metadata.pb.h"
```
In the `InferenceCalculator` interface port block (alongside `kSideInModel`), add:
```cpp
  // Optional: when wired, emits static model metadata once at Open().
  static constexpr SideOutput<InferenceMetadata>::Optional kSideOutMetadata{
      "METADATA"};
```
Add `kSideOutMetadata` to the `MEDIAPIPE_NODE_CONTRACT(...)` argument list.

If this base class uses the older procedural contract (`GetContract`/`UpdateContract`) rather than the declarative `MEDIAPIPE_NODE_CONTRACT`, instead add inside the contract function:
```cpp
  if (cc->OutputSidePackets().HasTag("METADATA")) {
    cc->OutputSidePackets().Tag("METADATA").Set<InferenceMetadata>();
  }
```
Pick whichever matches what Step 1 showed.

- [ ] **Step 3: Populate it in `Open()`** in `inference_calculator.cc`, immediately after the runner (`inference_runner_`) is successfully created:

```cpp
  // Emit static model metadata once, if the graph wired the METADATA side
  // packet. Guarded so non-wired graphs are unchanged.
  if (cc->OutputSidePackets().HasTag("METADATA")) {
    MP_ASSIGN_OR_RETURN(auto metadata, inference_runner_->GetModelMetadata());
    cc->OutputSidePackets().Tag("METADATA").Set(
        api2::MakePacket<InferenceMetadata>(std::move(metadata)));
  }
```
Use the runner member's actual name from Step 1 (e.g. `inference_runner_`), and the packet-making idiom already used in this file (plain `MakePacket<>` vs `api2::MakePacket<>`); match the file. Place this AFTER the point where init can still fail, so a failed load returns before emitting.

- [ ] **Step 4: Add the proto dep** to the `inference_calculator` (interface) library in `mediapipe/calculators/tensor/BUILD`: `grep -n "name = \"inference_calculator_interface\"\|name = \"inference_calculator\"" mediapipe/calculators/tensor/BUILD`, then add to the matching library's `deps`:
```python
        "//mediapipe/framework/formats:inference_metadata_cc_proto",
```

- [ ] **Step 5: Build the CPU inference calculator**

```bash
bazel build --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:inference_calculator_cpu
```
Expected: success.

- [ ] **Step 6: Commit**

```bash
git add mediapipe/calculators/tensor/inference_calculator.h mediapipe/calculators/tensor/inference_calculator.cc mediapipe/calculators/tensor/BUILD
git commit -m "feat(metadata): opt-in METADATA side output on InferenceCalculator"
```

---

### Task 5: Graph test — METADATA side packet is emitted once from a real model

**Files:**
- Create: `mediapipe/calculators/tensor/inference_calculator_metadata_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`

- [ ] **Step 1: Find an existing small .tflite testdata target** to avoid adding new model assets:

```bash
grep -rn "1x3_square_int32.tflite\|\.tflite\"" mediapipe/calculators/tensor/BUILD mediapipe/calculators/tensor/testdata 2>/dev/null | head
grep -rln "InferenceCalculator" mediapipe/calculators/tensor/*test*.cc | head
```
Use a model already referenced by an existing inference test (e.g. the one in `inference_calculator_test.cc`). Note its exact `data` label and on-disk path. Below, `MODEL_PATH` denotes that path and `MODEL_DATA_DEP` its Bazel `data` label — substitute the real values.

- [ ] **Step 2: Write the test** `mediapipe/calculators/tensor/inference_calculator_metadata_test.cc`:

```cpp
// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

// Wires InferenceCalculator with the optional METADATA side output and a model
// side packet path, runs the graph open phase, and inspects the side packet.
TEST(InferenceCalculatorMetadataTest, EmitsModelMetadataOnce) {
  // NOTE: replace MODEL_PATH with the real testdata path found in Step 1.
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_side_packet: "model_path"
    output_side_packet: "metadata"
    node {
      calculator: "InferenceCalculator"
      input_side_packet: "MODEL_PATH:model_path"
      output_side_packet: "METADATA:metadata"
      options {
        [mediapipe.InferenceCalculatorOptions.ext] {
          delegate { xnnpack {} }
        }
      }
    }
  )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun(
      {{"model_path",
        MakePacket<std::string>("MODEL_PATH")}}));   // <-- real path
  MP_ASSERT_OK(graph.WaitUntilIdle());

  MP_ASSERT_OK_AND_ASSIGN(Packet p, graph.GetOutputSidePacket("metadata"));
  const auto& md = p.Get<InferenceMetadata>();
  EXPECT_FALSE(md.input().empty());
  EXPECT_FALSE(md.output().empty());
  EXPECT_EQ(md.backend(), "cpu");
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace
}  // namespace mediapipe
```

Implementer notes:
- The exact way to pass the model (a `MODEL_PATH` string side packet vs a `MODEL` `TfLiteModelPtr` side packet) must match what `InferenceCalculator` accepts — verify against `inference_calculator_test.cc` from Step 1 and copy its model-wiring idiom exactly (tag name, side packet type, options). The assertions on `md` are the point; the wiring should mirror the existing working test.
- If the chosen model is not 4-D BHWC, drop the height/width assertions; always assert `input()`/`output()` non-empty and `backend() == "cpu"`.

- [ ] **Step 3: Add the test target** in `mediapipe/calculators/tensor/BUILD`:

```python
cc_test(
    name = "inference_calculator_metadata_test",
    srcs = ["inference_calculator_metadata_test.cc"],
    data = [MODEL_DATA_DEP],  # <-- real data label from Step 1
    deps = [
        ":inference_calculator_cpu",
        ":inference_calculator_interface",  # or the label that exports InferenceCalculator
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:inference_metadata_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
    ],
)
```
Confirm the InferenceCalculator library label(s) by grepping the existing `inference_calculator_test` target's deps and reuse them.

- [ ] **Step 4: Run the test**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:inference_calculator_metadata_test --test_output=all
```
Expected: PASS — the `metadata` side packet exists, has non-empty input/output specs, backend "cpu". If it fails because the model-wiring idiom is off, align it with the existing inference test from Step 1.

- [ ] **Step 5: Commit**

```bash
git add mediapipe/calculators/tensor/inference_calculator_metadata_test.cc mediapipe/calculators/tensor/BUILD
git commit -m "test(metadata): METADATA side packet emitted from loaded model"
```

---

### Task 6: Regression — non-wired graphs are unchanged

**Files:**
- (verification only; no new production code)

- [ ] **Step 1: Build & run the existing inference tests to confirm no regression**

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:inference_calculator_test --test_output=errors
```
Expected: PASS — proves graphs that do NOT wire `METADATA` are unaffected by the new optional side output (the guard `HasTag("METADATA")` is false there).

- [ ] **Step 2: Commit a note if any incidental fix was needed** (otherwise nothing to commit). If the existing test needed no change, there is no commit for this task.

---

## Done criteria (Plan 3a)

- `InferenceMetadata` proto builds.
- `InferenceRunner::GetModelMetadata()` exists (non-pure, default Unimplemented); CPU/TFLite delegate runner overrides it from the live interpreter.
- `InferenceCalculator` emits the `METADATA` side packet exactly once at `Open()` **only when wired**; `//mediapipe/calculators/tensor:inference_calculator_test` still green (no regression).
- `//mediapipe/calculators/tensor:inference_calculator_metadata_test` green.

## Risks / notes

- **TfLiteType enum spelling** may vary by TFLite version — the switch may need adjusting to compile (Task 3 Step 4 catches this).
- **Dynamic batch detection** relies on `dims_signature`; if null, we report `is_dynamic_batch=false` and `batch_capacity` from resolved dims — acceptable for Plan 3b (which also guards `N <= batch_capacity`).
- This is the spec's sanctioned core fork; keep the diff additive and guarded so rebases stay clean.

## Handoff to Plan 3b

Plan 3b consumes the `METADATA` side packet at the tiler's `Open()` to size/pack tiles to `input_height/width/channels`, `batch_capacity`, and `is_dynamic_batch`, then feeds batched tensors to `InferenceCalculator` and the Plan-1/Plan-2 decoders.
