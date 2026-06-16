# Vendored BoT-SORT: re-vendoring manifest

This directory contains a **motion-only** vendoring of BoT-SORT-cpp: the
appearance/Re-ID and TensorRT/ONNX-Runtime paths are intentionally stripped so
the tracker builds and links with only OpenCV + Eigen (no CUDA/TensorRT/ONNX,
no `INIReader`). This file records exactly where the sources came from and which
local edits were applied, so a future re-vendor can be reproduced.

## Upstream

- Repository: https://github.com/viplix3/BoTSORT-cpp
- Branch: `main`
- Commit SHA (the `main` branch tip at vendoring time):
  `06b739af279fb343fb50658fb19fc2bf532b6daf`

To re-check the current upstream tip:

```bash
git ls-remote https://github.com/viplix3/BoTSORT-cpp main
```

## Files intentionally OMITTED

These upstream paths are **not** vendored here:

- `botsort/include/ReID.h` — replaced with a local no-op stub (see patch 1).
- `botsort/include/INIReader.h` — `.ini` config parsing is unused in this build.
- `botsort/src/ReID.cpp` — Re-ID implementation (TensorRT/ONNX dependent).
- `botsort/src/TRT_InferenceEngine/` — TensorRT inference engine directory.

## Local patches applied

Every local edit to a vendored source carries a `Motion-only build:` comment, so
all of them are greppable:

```bash
grep -rn "Motion-only build:" third_party/botsort/
```

1. **`include/ReID.h`** — replaced the upstream (TensorRT-dependent) header with a
   no-op `ReIDModel` stub. This lets `BoTSORT.{h,cpp}` compile and link without
   CUDA/TensorRT/ONNX-Runtime. Re-ID is never enabled at runtime in this build.

2. **`src/TrackerParams.cpp`, `src/ReIDParams.cpp`, `src/GmcParams.cpp`** — the
   `load_config` bodies were replaced with default-returning stubs, and the
   `#include "INIReader.h"` was removed. The `load_config` symbols are kept
   defined (rather than deleted) because `BoTSORT.cpp` takes their address; they
   are never executed (configuration is supplied by the calculator, not `.ini`
   files).

3. **`src/BoTSORT.cpp`** — removed the unused `#include "INIReader.h"`.

4. **`include/DataType.h`, `src/KalmanFilter.cpp`, `src/KalmanFilterAccBased.cpp`**
   — Eigen includes rewritten from `<eigen3/Eigen/...>` to `"Eigen/..."`, to match
   the header root of the vendored `@eigen//:eigen3` (headers resolve as
   `Eigen/...`, not `eigen3/Eigen/...`).

5. **`src/GlobalMotionCompensation.cpp`** — removed a `#ifdef DEBUG` `cv::imshow`
   call (which requires OpenCV `highgui` and drags in macOS Cocoa). The match
   visualization image is still built; the display call was replaced with
   `(void) matches_img;`.
