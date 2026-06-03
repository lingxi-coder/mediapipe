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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TILING_GPU_RESOURCE_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TILING_GPU_RESOURCE_H_

#include <cstdint>

namespace mediapipe {

// Context-scope identity for GPU resource caches. A resource created under one
// key must never be reused under a different key. gl_context_identity is the
// GlContext* (or a stable share-group id if available); api_version is the
// GLES major*10+minor. See spec "OpenGL resource scope key".
//
// This header is pure types (no GL includes), so it compiles under
// MEDIAPIPE_DISABLE_GPU=1; the GL converter/calculator populate these from a
// live GlContext only inside the GLES-guarded code paths.
struct GpuResourceScopeKey {
  const void* gl_context_identity = nullptr;
  const void* share_group_identity = nullptr;
  int api_version = 0;
  bool operator==(const GpuResourceScopeKey& o) const {
    return gl_context_identity == o.gl_context_identity &&
           share_group_identity == o.share_group_identity &&
           api_version == o.api_version;
  }
  bool operator!=(const GpuResourceScopeKey& o) const {
    return !operator==(o);
  }
};

// Ownership lifecycle of a pooled GPU resource (spec "In-flight ... ownership").
// A resource must complete the full cycle
//   kFree -> kAcquiredForWrite -> kSubmitted -> kInFlightDownstream
//         -> kReclaimable -> kFree
// before it is reused; it is never reused while still kInFlightDownstream (a
// packet still references it) or before its write fence has signaled.
enum class GpuResourceState {
  kFree,                // safe to acquire
  kAcquiredForWrite,    // owned by the current Process()
  kSubmitted,           // GL writes enqueued; waiting on write fence/sync
  kInFlightDownstream,  // emitted in a packet
  kReclaimable,         // packet released AND GPU work complete
};

}  // namespace mediapipe
#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TILING_GPU_RESOURCE_H_
