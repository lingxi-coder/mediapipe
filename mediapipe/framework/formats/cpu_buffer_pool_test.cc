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

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

TEST(CpuBufferPoolTest, DisabledNeverPools) {
  CpuBufferPool pool(/*capacity=*/0);
  EXPECT_FALSE(pool.enabled());
  void* p = pool.Acquire(/*bytes=*/64, /*alignment=*/0);
  EXPECT_NE(p, nullptr);                 // still allocates
  EXPECT_FALSE(pool.Release(p, 64, 0));  // refuses to retain; caller-freed=false
  EXPECT_EQ(pool.stats().inserts, 0);
}

TEST(CpuBufferPoolTest, RecyclesSameSizeBuffer) {
  CpuBufferPool pool(/*capacity=*/2);
  void* a = pool.Acquire(256, 0);
  ASSERT_NE(a, nullptr);
  EXPECT_TRUE(pool.Release(a, 256, 0));   // retained
  void* b = pool.Acquire(256, 0);         // same key -> same pointer
  EXPECT_EQ(b, a);
  EXPECT_EQ(pool.stats().hits, 1);
  EXPECT_EQ(pool.stats().misses, 1);      // the first Acquire missed
  pool.Release(b, 256, 0);
}

TEST(CpuBufferPoolTest, DifferentKeyDoesNotAlias) {
  CpuBufferPool pool(2);
  void* a = pool.Acquire(256, 0);
  pool.Release(a, 256, 0);
  void* b = pool.Acquire(512, 0);   // different size -> miss, distinct buffer
  EXPECT_NE(b, a);
  void* c = pool.Acquire(256, 16);  // same size, different alignment -> miss
  EXPECT_NE(c, a);
  pool.Release(b, 512, 0);
  pool.Release(c, 256, 16);
}

TEST(CpuBufferPoolTest, CapacityBoundsRetainedBuffers) {
  CpuBufferPool pool(/*capacity=*/1);
  void* a = pool.Acquire(256, 0);
  void* b = pool.Acquire(256, 0);          // miss, both live
  EXPECT_TRUE(pool.Release(a, 256, 0));    // retained (pool now full: 1)
  EXPECT_FALSE(pool.Release(b, 256, 0));   // over capacity -> not retained
  EXPECT_EQ(pool.stats().evictions, 0);    // we never inserted-then-evicted
}

TEST(CpuBufferPoolTest, AlignedBuffersAreAligned) {
  CpuBufferPool pool(2);
  void* p = pool.Acquire(100, 64);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);
  pool.Release(p, 100, 64);
}

}  // namespace
}  // namespace mediapipe
