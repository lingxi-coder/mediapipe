// Copyright 2024 The MediaPipe Authors.
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
#ifndef MEDIAPIPE_FRAMEWORK_MEMORY_MANAGER_H_
#define MEDIAPIPE_FRAMEWORK_MEMORY_MANAGER_H_

#include <cstddef>
#include <memory>
#include <utility>

// Defines MEDIAPIPE_TENSOR_USE_AHWB
#include "mediapipe/framework/port.h"
#include "mediapipe/gpu/multi_pool.h"

#include "mediapipe/framework/formats/cpu_buffer_pool.h"

#ifdef MEDIAPIPE_TENSOR_USE_AHWB
#include "mediapipe/framework/formats/hardware_buffer_pool.h"
#endif

namespace mediapipe {

// Owns buffer pools to provide access to pooled buffer objects. Access is
// managed via shared_ptrs to allow clients of buffer objects to control their
// lifetime.
//
// Example usage:
// 1) Instantiate the MemoryManager and pass it to the kMemoryManagerService
//    before graph initialization:
//    CalculatorGraph graph;
//    graph.SetServiceObject(kMemoryManagerService,
//                           std::make_shared<MemoryManager>());
//    graph.Initialize(...);
// 2) Lookup MemoryManager instance in Calculator::Open():
//    absl::Status Calculator::Open(CalculatorContext* cc) {
//      auto memory_manager_service = cc->Service(kMemoryManagerService);
//      if (memory_manager_service.IsAvailable()) {
//        memory_manager_ = &memory_manager_service.GetObject();
//      } else {
//        ABSL_LOG(WARNING) << "MemoryManager not available";
//      }
//      ...
//    }
//    private:
//       MemoryManager* memory_manager_ = nullptr;
// 3) Pass Calculator::memory_manager_ to the Tensor class constructor:
//       Tensor tensor(Tensor::ElementType::kFloat32,
//                     Tensor::Shape{kTensorSize}, &memory_manager_);
class MemoryManager {
 public:
  // If prefer_ahwb is true, Tensors written on CPU and read as OpenGL buffers
  // or vice versa use Android Hardware Buffers as storage when supported.
  explicit MemoryManager(MultiPoolOptions options = kDefaultMultiPoolOptions,
                         bool prefer_ahwb = false,
                         size_t cpu_buffer_pool_capacity = 0) {
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
    hardware_buffer_pool_ =
        std::make_shared<HardwareBufferPool>(std::move(options));
    prefer_ahwb_ = prefer_ahwb;
#else
    static_cast<void>(options);
    static_cast<void>(prefer_ahwb);
#endif
    if (cpu_buffer_pool_capacity > 0) {
      cpu_buffer_pool_ =
          std::make_shared<CpuBufferPool>(cpu_buffer_pool_capacity);
    }
  }

  // Convenience overload for the CPU tensor pool. A zero capacity leaves CPU
  // allocation unchanged and GetCpuBufferPool() returns nullptr.
  explicit MemoryManager(size_t cpu_buffer_pool_capacity)
      : MemoryManager(kDefaultMultiPoolOptions, /*prefer_ahwb=*/false,
                      cpu_buffer_pool_capacity) {}

#ifdef MEDIAPIPE_TENSOR_USE_AHWB
  std::shared_ptr<HardwareBufferPool> GetAndroidHardwareBufferPool() const {
    return hardware_buffer_pool_;
  }
#endif

  // Null when CPU pooling is disabled.
  std::shared_ptr<CpuBufferPool> GetCpuBufferPool() const {
    return cpu_buffer_pool_;
  }

  bool PreferAhwb() const {
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
    return prefer_ahwb_;
#else
    return false;
#endif
  }

 private:
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
  std::shared_ptr<HardwareBufferPool> hardware_buffer_pool_;
  bool prefer_ahwb_ = false;
#endif
  std::shared_ptr<CpuBufferPool> cpu_buffer_pool_;
};

}  //  namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_MEMORY_MANAGER_H_
