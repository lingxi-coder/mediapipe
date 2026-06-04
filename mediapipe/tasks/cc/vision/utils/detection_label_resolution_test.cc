/* Copyright 2024 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/vision/utils/detection_label_resolution.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::tasks::vision {
namespace {

using LabelItems = mediapipe::proto_ns::Map<int64_t, mediapipe::LabelMapItem>;

LabelItems MakeItems(std::vector<std::string> names) {
  LabelItems items;
  for (int i = 0; i < (int)names.size(); ++i) items[i].set_name(names[i]);
  return items;
}

google::protobuf::RepeatedPtrField<std::string> Rep(std::vector<std::string> v) {
  google::protobuf::RepeatedPtrField<std::string> r;
  for (auto& s : v) *r.Add() = s;
  return r;
}

TEST(DetectionLabelResolutionTest, ResolvesNamesToIndices) {
  auto items = MakeItems({"cat", "dog", "bird"});
  MP_ASSERT_OK_AND_ASSIGN(auto idx, ResolveCategoryIndices(
      items, Rep({"dog", "bird", "unknown"}), Rep({})));  // unknown ignored
  EXPECT_THAT(idx, ::testing::UnorderedElementsAre(1, 2));
}

TEST(DetectionLabelResolutionTest, DenylistResolves) {
  auto items = MakeItems({"cat", "dog"});
  MP_ASSERT_OK_AND_ASSIGN(auto idx, ResolveCategoryIndices(
      items, Rep({}), Rep({"cat"})));
  EXPECT_THAT(idx, ::testing::UnorderedElementsAre(0));
}

TEST(DetectionLabelResolutionTest, AllUnknownAllowlistIsEmptyNoOp) {
  auto items = MakeItems({"cat"});
  MP_ASSERT_OK_AND_ASSIGN(auto idx, ResolveCategoryIndices(
      items, Rep({"zzz"}), Rep({})));
  EXPECT_TRUE(idx.empty());  // no-op, NOT "drop all"
}

TEST(DetectionLabelResolutionTest, NoFilterIsEmptySet) {
  auto items = MakeItems({"cat"});
  MP_ASSERT_OK_AND_ASSIGN(auto idx, ResolveCategoryIndices(items, Rep({}), Rep({})));
  EXPECT_TRUE(idx.empty());
}

TEST(DetectionLabelResolutionTest, MissingLabelsWithFilterIsError) {
  LabelItems empty;
  EXPECT_FALSE(ResolveCategoryIndices(empty, Rep({"dog"}), Rep({})).ok());
}

}  // namespace
}  // namespace mediapipe::tasks::vision
