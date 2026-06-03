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

#include "mediapipe/calculators/tensor/tiling_gpu_resource.h"

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

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
