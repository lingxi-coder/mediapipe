// Copyright 2020 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_CPU_BUFFER_POOL_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_CPU_BUFFER_POOL_H_

#include <cstddef>
#include <cstdint>
#include <list>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"

namespace mediapipe {

// Diagnostic counters for one CPU buffer pool.
struct CpuBufferPoolStats {
  int64_t hits = 0;       // Acquire served from the free-list
  int64_t misses = 0;     // Acquire had to allocate
  int64_t inserts = 0;    // Release retained a buffer
  int64_t evictions = 0;  // Release freed a buffer because the pool was full
};

// A bounded, thread-safe pool of raw CPU buffers keyed by (byte size,
// alignment). `capacity == 0` => disabled: Acquire always allocates, Release
// always frees (returns false), behaving exactly as if there were no pool.
// Allocation uses the same aligned_malloc/aligned_free routines as Tensor, so
// pooled buffers are interchangeable with directly-allocated ones.
//
// Lifetime contract: the pool is intended to be co-owned (shared_ptr) by every
// Tensor that draws from it, so it always outlives in-flight buffers. The pool
// frees everything it still holds on destruction.
class CpuBufferPool {
 public:
  explicit CpuBufferPool(size_t capacity) : capacity_(capacity) {}
  ~CpuBufferPool();

  CpuBufferPool(const CpuBufferPool&) = delete;
  CpuBufferPool& operator=(const CpuBufferPool&) = delete;

  bool enabled() const { return capacity_ > 0; }

  // Returns a buffer of exactly `bytes` with the given `alignment`
  // (alignment 0 => unaligned malloc, matching Tensor). Never returns nullptr
  // unless allocation itself fails. Served from the free-list on a key match,
  // otherwise freshly allocated.
  void* Acquire(size_t bytes, int alignment);

  // Returns `buffer` (which must have been produced by Acquire with the same
  // bytes+alignment) to the pool. Retains it if the pool has spare capacity and
  // returns true; otherwise frees it and returns false. A disabled pool always
  // frees and returns false.
  bool Release(void* buffer, size_t bytes, int alignment);

  CpuBufferPoolStats stats() const;

 private:
  struct Key {
    size_t bytes;
    int alignment;
    bool operator==(const Key& o) const {
      return bytes == o.bytes && alignment == o.alignment;
    }
    template <typename H>
    friend H AbslHashValue(H h, const Key& k) {
      return H::combine(std::move(h), k.bytes, k.alignment);
    }
  };
  static void* Allocate(size_t bytes, int alignment);
  static void Free(void* buffer, int alignment);

  mutable absl::Mutex mu_;
  const size_t capacity_;
  size_t retained_ ABSL_GUARDED_BY(mu_) = 0;  // total buffers held across keys
  absl::flat_hash_map<Key, std::list<void*>> free_ ABSL_GUARDED_BY(mu_);
  CpuBufferPoolStats stats_ ABSL_GUARDED_BY(mu_);
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_CPU_BUFFER_POOL_H_
