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

#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_TILING_CACHE_STATS_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_TILING_CACHE_STATS_H_

#include <cstdint>

#include "mediapipe/framework/formats/cache_stats.h"
#include "mediapipe/framework/formats/cpu_buffer_pool.h"

namespace mediapipe {

// Diagnostic-only snapshot of the tiling caches. Emitted on an optional
// CACHE_STATS output; never used to align inference or merge streams.
struct TilingCacheStats {
  CacheStats tile_plan;
  CacheStats tile_matrix;
  CpuBufferPoolStats cpu_tensor_pool;
  // GPU zero-copy path (Plan 4). These stay zero under MEDIAPIPE_DISABLE_GPU=1
  // and whenever enable_gpu_zero_copy is false.
  CacheStats gpu_tensor_buffer;  // Cache 5: GPU/AHWB tensor-buffer pool
  CacheStats tile_surface;       // Cache 4: GL program/tile-surface cache
  // Batches emitted from the most recent GPU-served frame. This is a gauge of
  // the last frame's fan-out, NOT a live in-flight count (nothing decrements
  // on downstream completion).
  int64_t last_frame_gpu_batches = 0;
  // Frames that requested zero-copy (enable_gpu_zero_copy with IMAGE_GPU
  // wired) but arrived without a usable GPU input and were served on the CPU
  // path instead. Cumulative.
  int64_t gpu_to_cpu_fallbacks = 0;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_TILING_CACHE_STATS_H_
