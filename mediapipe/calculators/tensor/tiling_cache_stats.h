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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TILING_CACHE_STATS_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TILING_CACHE_STATS_H_

#include "mediapipe/calculators/tensor/tiling_cache_utils.h"
#include "mediapipe/framework/formats/cpu_buffer_pool.h"

namespace mediapipe {

// Diagnostic-only snapshot of the tiling caches. Emitted on an optional
// CACHE_STATS output; never used to align inference or merge streams.
struct TilingCacheStats {
  CacheStats tile_plan;
  CacheStats tile_matrix;
  CpuBufferPoolStats cpu_tensor_pool;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TILING_CACHE_STATS_H_
