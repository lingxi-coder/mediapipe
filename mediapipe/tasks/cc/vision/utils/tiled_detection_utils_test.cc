/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/vision/utils/tiled_detection_utils.h"

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace {

// A non-positive (dynamic/unspecified) model batch dim is normalized to a
// dynamic batch with capacity 1, so it never trips the positive-capacity
// RET_CHECK downstream and the front emits a variable (valid-count) batch.
TEST(NormalizeTiledBatchDimTest, NonPositiveBecomesDynamicCapacityOne) {
  for (int raw : {0, -1, -8}) {
    const NormalizedBatchDim nb = NormalizeTiledBatchDim(raw);
    EXPECT_EQ(nb.batch_capacity, 1) << "raw=" << raw;
    EXPECT_TRUE(nb.is_dynamic) << "raw=" << raw;
  }
}

// A positive batch dim is a fixed batch: passed through unchanged, not dynamic.
TEST(NormalizeTiledBatchDimTest, PositiveIsFixedPassthrough) {
  for (int raw : {1, 2, 4, 16}) {
    const NormalizedBatchDim nb = NormalizeTiledBatchDim(raw);
    EXPECT_EQ(nb.batch_capacity, raw) << "raw=" << raw;
    EXPECT_FALSE(nb.is_dynamic) << "raw=" << raw;
  }
}

}  // namespace
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
