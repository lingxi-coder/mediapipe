# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

MediaPipe uses **Bazel** as its primary build system (with Bzlmod enabled). The `.bazelrc` sets C++20 and `--jobs 128` by default.

### Common Build Commands

```bash
# Build a target (desktop, GPU disabled)
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 <target>

# Build a specific example
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hello_world:hello_world

# Run a desktop binary
bazel run --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hello_world:hello_world

# Build the C shared library (used by Python Tasks bindings)
bazel build -c opt --linkopt -s --strip always --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/c:libmediapipe.so

# Android build
bazel build -c opt --config=android_arm64 <target>

# iOS build
bazel build -c opt --config=ios_arm64 <target>
```

### Running Tests

```bash
# Run a single C++ test
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/framework:calculator_graph_test

# Run a single Python test
bazel test //mediapipe/tasks/python/test/vision:object_detector_test

# Run all tests in a package
bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/framework/...

# Run Python framework tests
bazel test //mediapipe/python:packet_test
bazel test //mediapipe/python:calculator_graph_test
```

### Python Package (pip)

```bash
# Install from source (uses Bazel internally)
python setup.py install --link-opencv

# Or with GPU disabled
MEDIAPIPE_DISABLE_GPU=1 python setup.py install --link-opencv
```

## Architecture

### Layer Overview

```
Python/Web/Java/iOS APIs
        ↓
C API (tasks/c/libmediapipe.so) — stable ABI boundary for language bindings
        ↓
C++ Task implementations (tasks/cc/)
        ↓
Framework: Graph execution engine (framework/)
        ↓
Calculators (calculators/ + tasks/cc/*/calculators/)
        ↓
TFLite inference / OpenCV / other backends
```

### Core Framework (`mediapipe/framework/`)

The foundation: a streaming data-flow engine. Key abstractions:
- **Packet** — a typed, timestamped unit of data passed between calculators
- **Calculator** — a processing node; implements `GetContract`, `Open`, `Process`, `Close`
- **Graph** — a `CalculatorGraphConfig` proto defining the DAG of calculators connected by named streams
- **Timestamp** — monotonically increasing value coordinating packet synchronization

`api2/` provides a newer, type-safe Calculator API (preferred for new calculators). `api3/` is the newest iteration.

### Calculators (`mediapipe/calculators/`)

Built-in calculators organized by domain: `core/`, `tensor/`, `tflite/`, `image/`, `video/`, `audio/`, `util/`. Graphs reference calculators by their registered name string.

### Tasks API (`mediapipe/tasks/`)

High-level ML task APIs organized by language binding and modality:

| Directory | Contents |
|---|---|
| `tasks/cc/{vision,text,audio}/` | C++ task implementations (e.g., `face_detector/`, `object_detector/`) |
| `tasks/c/` | C API layer — compiled into `libmediapipe.so` |
| `tasks/python/` | Python bindings using the C API via ctypes |
| `tasks/java/` | Android Java bindings |
| `tasks/ios/` | iOS Objective-C bindings |
| `tasks/web/` | TypeScript/JavaScript bindings using WASM |
| `tasks/metadata/` | TFLite model metadata schemas |

Vision tasks include: face detector, face landmarker, hand landmarker, gesture recognizer, pose landmarker, image classifier, image segmenter, object detector, image embedder.

### Python Bindings (`mediapipe/python/`)

Framework-level bindings built via pybind11 (`framework_bindings.cc`). Exposes `CalculatorGraph`, `Packet`, image types. Legacy solutions (face mesh, hands, pose, etc.) are in `python/solutions/`.

The Tasks Python API in `tasks/python/` is separate and uses the C API (`libmediapipe.so`) via ctypes, not pybind11.

### Model Maker (`mediapipe/model_maker/`)

Fine-tuning/customization library. Separate `setup.py` and `requirements.txt`. Supports customizing models for vision, text, and LLM tasks.

### Web (`mediapipe/tasks/web/`)

TypeScript APIs built with Bazel + rollup. Each task has a `index.ts` exporting the public API. Uses WASM binaries compiled from the C++ Tasks.

## Key Conventions

- Bazel target paths mirror the filesystem: `mediapipe/tasks/python/vision:face_detector` → `mediapipe/tasks/python/vision/face_detector.py`
- Proto files define graph configs and task options; generated `*_py_pb2` targets are the Python equivalents
- `--define MEDIAPIPE_DISABLE_GPU=1` is needed for desktop builds without GPU support
- Test data (models, images) is fetched from GCS and declared as `data` deps in BUILD files
