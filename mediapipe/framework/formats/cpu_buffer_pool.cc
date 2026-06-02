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

#include "mediapipe/framework/formats/cpu_buffer_pool.h"

#include <algorithm>
#include <cstdlib>

#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port/aligned_malloc_and_free.h"

namespace mediapipe {

void* CpuBufferPool::Allocate(size_t bytes, int alignment) {
  if (alignment > 0) {
    return aligned_malloc(std::max<size_t>(alignment, bytes), alignment);
  }
  return malloc(bytes);
}

void CpuBufferPool::Free(void* buffer, int alignment) {
  if (buffer == nullptr) return;
  if (alignment > 0) {
    aligned_free(buffer);
  } else {
    free(buffer);
  }
}

CpuBufferPool::~CpuBufferPool() {
  for (auto& [key, buffers] : free_) {
    for (void* b : buffers) Free(b, key.alignment);
  }
}

void* CpuBufferPool::Acquire(size_t bytes, int alignment) {
  if (capacity_ > 0) {
    absl::MutexLock lock(&mu_);
    auto it = free_.find(Key{bytes, alignment});
    if (it != free_.end() && !it->second.empty()) {
      void* b = it->second.back();
      it->second.pop_back();
      --retained_;
      ++stats_.hits;
      return b;
    }
    ++stats_.misses;
  }
  return Allocate(bytes, alignment);
}

bool CpuBufferPool::Release(void* buffer, size_t bytes, int alignment) {
  if (buffer == nullptr) return true;
  if (capacity_ > 0) {
    absl::MutexLock lock(&mu_);
    if (retained_ < capacity_) {
      free_[Key{bytes, alignment}].push_back(buffer);
      ++retained_;
      ++stats_.inserts;
      return true;
    }
  }
  Free(buffer, alignment);
  return false;
}

CpuBufferPoolStats CpuBufferPool::stats() const {
  absl::MutexLock lock(&mu_);
  return stats_;
}

}  // namespace mediapipe
