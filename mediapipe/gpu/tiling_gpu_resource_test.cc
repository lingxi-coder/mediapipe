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

#include "mediapipe/gpu/tiling_gpu_resource.h"

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/multi_pool.h"
#include "mediapipe/gpu/reusable_pool.h"

namespace mediapipe {
namespace {

// Exercise the same MultiPool/ReusablePool retention as AHWB without requiring
// Android allocation. The live count detects cached buffers across shape pools.
struct TestBuffer {
  TestBuffer() { ++live; }
  ~TestBuffer() { --live; }
  void Reuse() { ++reuse_count; }
  int reuse_count = 0;
  inline static int live = 0;
};

class TestBufferPool : public ReusablePool<TestBuffer> {
 public:
  static std::shared_ptr<TestBufferPool> Create(
      int spec, const MultiPoolOptions& options) {
    return std::shared_ptr<TestBufferPool>(new TestBufferPool(options));
  }
  static absl::StatusOr<std::unique_ptr<TestBuffer>> CreateBufferWithoutPool(
      int spec) {
    return std::make_unique<TestBuffer>();
  }

 private:
  explicit TestBufferPool(const MultiPoolOptions& options)
      : ReusablePool<TestBuffer>(
            []() -> absl::StatusOr<std::unique_ptr<TestBuffer>> {
              return std::make_unique<TestBuffer>();
            }, options) {}
};

using TestMultiPool =
    MultiPool<TestBufferPool, int, std::shared_ptr<TestBuffer>>;

TEST(TilingGpuPoolTest, RetainsConfiguredBudgetAndReusesReleasedBuffers) {
  for (int capacity : {1, 4}) {
    SCOPED_TRACE(capacity);
    EXPECT_EQ(TestBuffer::live, 0);
    {
      TestMultiPool pool(TilingGpuPoolOptions(capacity));
      std::vector<std::shared_ptr<TestBuffer>> in_use;
      // In-use buffers must remain valid even beyond the retention budget.
      for (int i = 0; i < capacity + 2; ++i) {
        MP_ASSERT_OK_AND_ASSIGN(auto buffer, pool.Get(1));
        in_use.push_back(std::move(buffer));
      }
      EXPECT_EQ(TestBuffer::live, capacity + 2);
      in_use.clear();
      EXPECT_EQ(TestBuffer::live, capacity);
      for (int i = 0; i < capacity; ++i) {
        MP_ASSERT_OK_AND_ASSIGN(auto buffer, pool.Get(1));
        EXPECT_EQ(buffer->reuse_count, 1);
        in_use.push_back(std::move(buffer));
      }
      EXPECT_EQ(TestBuffer::live, capacity);
    }
    EXPECT_EQ(TestBuffer::live, 0);
  }
}

TEST(TilingGpuPoolTest, ShapeChangesEvictIdleBuffersButPreserveInUseBuffers) {
  EXPECT_EQ(TestBuffer::live, 0);
  {
    TestMultiPool pool(TilingGpuPoolOptions(4));
    MP_ASSERT_OK_AND_ASSIGN(auto pending, pool.Get(1));
    // Run through a frequency-scrub cycle, allowing the cold shape's entry to
    // expire even if the eviction tie initially favors it over the new shape.
    for (int i = 0; i < 64; ++i) {
      MP_ASSERT_OK_AND_ASSIGN(auto next_shape, pool.Get(2));
    }
    EXPECT_EQ(TestBuffer::live, 2);
    pending->Reuse();  // The evicted shape's outstanding buffer remains valid.
    EXPECT_EQ(pending->reuse_count, 1);
    pending.reset();
    EXPECT_EQ(TestBuffer::live, 1);  // Old shape cannot re-enter an evicted pool.
    // Changing dynamic batch shapes cannot accumulate a capacity-sized cache
    // for each shape, even when there are more in-use buffers than the budget.
    for (int shape = 3; shape < 20; ++shape) {
      std::vector<std::shared_ptr<TestBuffer>> buffers;
      for (int i = 0; i < 6; ++i) {
        MP_ASSERT_OK_AND_ASSIGN(auto buffer, pool.Get(shape));
        buffers.push_back(std::move(buffer));
      }
      buffers.clear();
      EXPECT_LE(TestBuffer::live, 4) << "shape " << shape;
    }
  }
  EXPECT_EQ(TestBuffer::live, 0);
}

TEST(GpuResourceScopeKeyTest, EqualityByAllFields) {
  int a = 0, b = 0;
  GpuResourceScopeKey k1{&a, &b, 31};
  GpuResourceScopeKey k2{&a, &b, 31};
  GpuResourceScopeKey k3{&a, &b, 30};  // different api_version
  GpuResourceScopeKey k4{&b, &b, 31};  // different context identity
  EXPECT_EQ(k1, k2);
  EXPECT_NE(k1, k3);
  EXPECT_NE(k1, k4);
}

TEST(GpuResourceLedgerTest, GrowsUpToCapacityThenExhausts) {
  GpuResourceLedger ledger(/*capacity=*/2, /*max_in_flight=*/0);
  const int s0 = ledger.Acquire();
  const int s1 = ledger.Acquire();
  EXPECT_GE(s0, 0);
  EXPECT_GE(s1, 0);
  EXPECT_EQ(ledger.total(), 2);
  EXPECT_EQ(ledger.misses(), 2);  // both grew the pool
  // Both slots still acquired (not free) -> at capacity -> exhausted.
  EXPECT_EQ(ledger.Acquire(), -1);
}

TEST(GpuResourceLedgerTest, ReclaimsOnlyAfterReleaseAndFence) {
  GpuResourceLedger ledger(/*capacity=*/1, /*max_in_flight=*/0);
  const int s = ledger.Acquire();
  ledger.MarkSubmitted(s);
  ledger.MarkInFlight(s);
  EXPECT_EQ(ledger.in_flight(), 1);

  // Packet released but fence NOT signaled: not reclaimable, still exhausted.
  ledger.NotePacketReleased(s);
  ledger.ReclaimCompleted();
  EXPECT_EQ(ledger.in_flight(), 1);
  EXPECT_EQ(ledger.Acquire(), -1);

  // Fence signaled too: now reclaimable -> reused (a hit, not a new slot).
  ledger.NoteFenceSignaled(s);
  const int reused = ledger.Acquire();
  EXPECT_EQ(reused, s);
  EXPECT_EQ(ledger.total(), 1);   // no growth
  EXPECT_EQ(ledger.misses(), 1);  // only the first acquire grew
  EXPECT_EQ(ledger.hits(), 1);    // the reuse
}

TEST(GpuResourceLedgerTest, FenceBeforeReleaseStillWaitsForRelease) {
  GpuResourceLedger ledger(/*capacity=*/1, /*max_in_flight=*/0);
  const int s = ledger.Acquire();
  ledger.MarkInFlight(s);
  ledger.NoteFenceSignaled(s);  // fence first
  ledger.ReclaimCompleted();
  EXPECT_EQ(ledger.in_flight(), 1);  // still in flight: packet not released
  ledger.NotePacketReleased(s);
  ledger.ReclaimCompleted();
  EXPECT_EQ(ledger.in_flight(), 0);
  EXPECT_EQ(ledger.free_count(), 1);
}

TEST(GpuResourceLedgerTest, InFlightCapBlocksEvenWithCapacityLeft) {
  // Capacity 4 but only 1 may be in flight at a time.
  GpuResourceLedger ledger(/*capacity=*/4, /*max_in_flight=*/1);
  const int s0 = ledger.Acquire();
  ledger.MarkInFlight(s0);
  EXPECT_EQ(ledger.in_flight(), 1);
  // In-flight cap reached -> acquire fails despite free capacity.
  EXPECT_EQ(ledger.Acquire(), -1);
  // Complete s0; then a new acquire succeeds.
  ledger.NotePacketReleased(s0);
  ledger.NoteFenceSignaled(s0);
  const int s1 = ledger.Acquire();
  EXPECT_GE(s1, 0);
}

}  // namespace
}  // namespace mediapipe
