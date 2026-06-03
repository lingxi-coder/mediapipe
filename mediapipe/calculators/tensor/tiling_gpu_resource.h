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
#include <vector>

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

// Pure in-flight ownership accounting for a bounded pool of GPU/AHWB batch
// buffers (spec "In-flight GPU/OpenGL ownership model"). It holds NO GL state:
// the calculator feeds it lifecycle events (acquired, write submitted, packet
// emitted, packet released, fence signaled) and the ledger decides when a slot
// becomes reusable and whether an acquisition must fail (ResourceExhausted)
// because the pool is at capacity / the in-flight cap is reached. A slot is
// reclaimed for reuse only after BOTH its downstream packet has been released
// AND its GPU write fence has signaled — never before. This is the one part of
// the GPU pool that is verifiable on a CPU build, so it is unit-tested there;
// binding slots to real GL SSBO / AHardwareBuffer storage happens in the
// GLES-guarded calculator path.
class GpuResourceLedger {
 public:
  GpuResourceLedger() = default;
  // capacity: max total pooled slots (e.g. max_gpu_tensor_buffers). <= 0 means
  // unbounded growth (no pooling cap). max_in_flight: cap on slots that may be
  // kInFlightDownstream at once; <= 0 means bounded only by capacity.
  GpuResourceLedger(int capacity, int max_in_flight)
      : capacity_(capacity), max_in_flight_(max_in_flight) {}

  struct Slot {
    GpuResourceState state = GpuResourceState::kFree;
    bool packet_released = false;
    bool fence_signaled = false;
  };

  // Reclaims any completed slots, then returns the index of a slot moved to
  // kAcquiredForWrite, or -1 if the pool is exhausted (at capacity with no free
  // slot, or the in-flight cap is reached). On reuse `hits_` increments; on
  // growth `misses_` increments.
  int Acquire() {
    ReclaimCompleted();
    if (InFlightCapReached()) return -1;
    for (int i = 0; i < static_cast<int>(slots_.size()); ++i) {
      if (slots_[i].state == GpuResourceState::kFree) {
        slots_[i] = Slot{GpuResourceState::kAcquiredForWrite, false, false};
        ++hits_;
        return i;
      }
    }
    if (capacity_ > 0 && static_cast<int>(slots_.size()) >= capacity_) {
      return -1;  // exhausted: at capacity, nothing free
    }
    slots_.push_back(Slot{GpuResourceState::kAcquiredForWrite, false, false});
    ++misses_;
    return static_cast<int>(slots_.size()) - 1;
  }

  void MarkSubmitted(int slot) {
    if (Valid(slot)) slots_[slot].state = GpuResourceState::kSubmitted;
  }
  // Emitted in a downstream packet: now counts against the in-flight cap.
  void MarkInFlight(int slot) {
    if (Valid(slot)) slots_[slot].state = GpuResourceState::kInFlightDownstream;
  }
  void NotePacketReleased(int slot) {
    if (Valid(slot)) slots_[slot].packet_released = true;
  }
  void NoteFenceSignaled(int slot) {
    if (Valid(slot)) slots_[slot].fence_signaled = true;
  }

  // Frees every kInFlightDownstream slot whose packet is released AND whose GPU
  // fence has signaled.
  void ReclaimCompleted() {
    for (Slot& s : slots_) {
      if (s.state == GpuResourceState::kInFlightDownstream && s.packet_released &&
          s.fence_signaled) {
        s = Slot{GpuResourceState::kFree, false, false};
      }
    }
  }

  int in_flight() const {
    int n = 0;
    for (const Slot& s : slots_) {
      if (s.state == GpuResourceState::kInFlightDownstream) ++n;
    }
    return n;
  }
  int free_count() const {
    int n = 0;
    for (const Slot& s : slots_) {
      if (s.state == GpuResourceState::kFree) ++n;
    }
    return n;
  }
  int total() const { return static_cast<int>(slots_.size()); }
  int64_t hits() const { return hits_; }
  int64_t misses() const { return misses_; }
  int capacity() const { return capacity_; }

 private:
  bool Valid(int slot) const {
    return slot >= 0 && slot < static_cast<int>(slots_.size());
  }
  bool InFlightCapReached() const {
    return max_in_flight_ > 0 && in_flight() >= max_in_flight_;
  }

  int capacity_ = 0;
  int max_in_flight_ = 0;
  std::vector<Slot> slots_;
  int64_t hits_ = 0;
  int64_t misses_ = 0;
};

}  // namespace mediapipe
#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TILING_GPU_RESOURCE_H_
