// Copyright 2026 The MediaPipe Authors. Licensed under the Apache License 2.0.
#include "mediapipe/calculators/tensor/tiling_cache_utils.h"

#include <string>

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

StableCacheKey IntKey(int64_t v) {
  return StableKeyBuilder().AddInt(v).Build();
}

TEST(StableKeyBuilderTest, IdenticalInputsEqualKeys) {
  EXPECT_EQ(StableKeyBuilder().AddInt(7).AddFloat(0.1f).AddString("a").Build(),
            StableKeyBuilder().AddInt(7).AddFloat(0.1f).AddString("a").Build());
}

TEST(StableKeyBuilderTest, AnyFieldChangeChangesKey) {
  const StableCacheKey base =
      StableKeyBuilder().AddInt(7).AddFloat(0.1f).Build();
  EXPECT_FALSE(base == StableKeyBuilder().AddInt(8).AddFloat(0.1f).Build());
  EXPECT_FALSE(base == StableKeyBuilder().AddInt(7).AddFloat(0.2f).Build());
}

TEST(StableKeyBuilderTest, TypeTaggingPreventsCrossTypeCollision) {
  // AddInt(1) must not encode the same bytes as AddString("\x01...").
  EXPECT_FALSE(StableKeyBuilder().AddInt(1).Build() ==
               StableKeyBuilder().AddString("\x01").Build());
}

TEST(BoundedLruCacheTest, DisabledNeverHits) {
  BoundedLruCache<int> cache(0);
  EXPECT_FALSE(cache.enabled());
  cache.Put(IntKey(1), 100);
  EXPECT_EQ(cache.Get(IntKey(1)), nullptr);
  EXPECT_EQ(cache.stats().hits, 0);
  EXPECT_EQ(cache.stats().inserts, 0);
}

TEST(BoundedLruCacheTest, HitAndMiss) {
  BoundedLruCache<int> cache(2);
  EXPECT_EQ(cache.Get(IntKey(1)), nullptr);
  cache.Put(IntKey(1), 100);
  const int* v = cache.Get(IntKey(1));
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(*v, 100);
  EXPECT_EQ(cache.stats().hits, 1);
  EXPECT_EQ(cache.stats().misses, 1);
}

TEST(BoundedLruCacheTest, EvictsLeastRecentlyUsed) {
  BoundedLruCache<int> cache(2);
  cache.Put(IntKey(1), 1);
  cache.Put(IntKey(2), 2);
  ASSERT_NE(cache.Get(IntKey(1)), nullptr);  // 1 now MRU, 2 is LRU
  cache.Put(IntKey(3), 3);                   // evicts 2
  EXPECT_EQ(cache.Get(IntKey(2)), nullptr);
  ASSERT_NE(cache.Get(IntKey(1)), nullptr);
  ASSERT_NE(cache.Get(IntKey(3)), nullptr);
  EXPECT_EQ(cache.stats().evictions, 1);
}

}  // namespace
}  // namespace mediapipe
