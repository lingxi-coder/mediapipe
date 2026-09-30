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

#include "mediapipe/util/detection_nms_util.h"

#include <vector>

#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

// Helper: create a Detection with one score, one label_id, and a
// relative bounding box.
Detection Det(float score, int label_id,
              float xmin, float ymin, float w, float h) {
  Detection d;
  d.add_score(score);
  d.add_label_id(label_id);
  auto* ld = d.mutable_location_data();
  ld->set_format(mediapipe::LocationData::RELATIVE_BOUNDING_BOX);
  auto* rbb = ld->mutable_relative_bounding_box();
  rbb->set_xmin(xmin);
  rbb->set_ymin(ymin);
  rbb->set_width(w);
  rbb->set_height(h);
  return d;
}

// -------------------------------------------------------------------------
// DetectionRelativeIoU tests
// -------------------------------------------------------------------------

TEST(DetectionRelativeIoUTest, IdenticalBoxesReturnOne) {
  Detection a = Det(0.9f, 0, 0.1f, 0.1f, 0.4f, 0.4f);
  Detection b = Det(0.8f, 0, 0.1f, 0.1f, 0.4f, 0.4f);
  EXPECT_NEAR(DetectionRelativeIoU(a, b), 1.0f, 1e-6f);
}

TEST(DetectionRelativeIoUTest, DisjointBoxesReturnZero) {
  Detection a = Det(0.9f, 0, 0.0f, 0.0f, 0.3f, 0.3f);
  Detection b = Det(0.8f, 0, 0.7f, 0.7f, 0.3f, 0.3f);
  EXPECT_NEAR(DetectionRelativeIoU(a, b), 0.0f, 1e-6f);
}

TEST(DetectionRelativeIoUTest, PartialOverlapIsCorrect) {
  // a: [0, 0.5] x [0, 0.5]  area=0.25
  // b: [0.25, 0.75] x [0.25, 0.75]  area=0.25
  // inter: [0.25, 0.5] x [0.25, 0.5]  area=0.0625
  // union: 0.25+0.25-0.0625=0.4375
  Detection a = Det(0.9f, 0, 0.0f, 0.0f, 0.5f, 0.5f);
  Detection b = Det(0.8f, 0, 0.25f, 0.25f, 0.5f, 0.5f);
  EXPECT_NEAR(DetectionRelativeIoU(a, b), 0.0625f / 0.4375f, 1e-5f);
}

TEST(DetectionRelativeIoUTest, ZeroAreaBoxReturnsZero) {
  Detection a = Det(0.9f, 0, 0.1f, 0.1f, 0.0f, 0.4f);  // zero width
  Detection b = Det(0.8f, 0, 0.1f, 0.1f, 0.4f, 0.4f);
  EXPECT_NEAR(DetectionRelativeIoU(a, b), 0.0f, 1e-6f);
}

// -------------------------------------------------------------------------
// GreedyDetectionNms tests
// -------------------------------------------------------------------------

TEST(GreedyDetectionNmsTest, BasicDedupSameClass) {
  // Two heavily-overlapping boxes, same class → keep only higher-scoring one.
  std::vector<Detection> dets = {
      Det(0.7f, 0, 0.0f, 0.0f, 0.5f, 0.5f),
      Det(0.9f, 0, 0.05f, 0.05f, 0.5f, 0.5f),
  };
  auto kept = GreedyDetectionNms(dets, /*iou_threshold=*/0.5f,
                                 /*class_agnostic=*/false);
  ASSERT_EQ(kept.size(), 1u);
  EXPECT_NEAR(kept[0].score(0), 0.9f, 1e-6f);
}

TEST(GreedyDetectionNmsTest, DifferentClassBothKeptPerClass) {
  // Two heavily-overlapping boxes, different classes → both kept (per-class NMS).
  std::vector<Detection> dets = {
      Det(0.9f, 0, 0.0f, 0.0f, 0.5f, 0.5f),
      Det(0.8f, 1, 0.0f, 0.0f, 0.5f, 0.5f),
  };
  auto kept = GreedyDetectionNms(dets, /*iou_threshold=*/0.5f,
                                 /*class_agnostic=*/false);
  EXPECT_EQ(kept.size(), 2u);
}

TEST(GreedyDetectionNmsTest, ClassAgnosticSuppressesDifferentClass) {
  // Two heavily-overlapping boxes, different classes, class-agnostic → 1 kept.
  std::vector<Detection> dets = {
      Det(0.9f, 0, 0.0f, 0.0f, 0.5f, 0.5f),
      Det(0.8f, 1, 0.0f, 0.0f, 0.5f, 0.5f),
  };
  auto kept = GreedyDetectionNms(dets, /*iou_threshold=*/0.5f,
                                 /*class_agnostic=*/true);
  ASSERT_EQ(kept.size(), 1u);
  EXPECT_NEAR(kept[0].score(0), 0.9f, 1e-6f);
}

TEST(GreedyDetectionNmsTest, DisjointBoxesAllKept) {
  std::vector<Detection> dets = {
      Det(0.9f, 0, 0.0f, 0.0f, 0.3f, 0.3f),
      Det(0.8f, 0, 0.7f, 0.7f, 0.3f, 0.3f),
  };
  auto kept = GreedyDetectionNms(dets, /*iou_threshold=*/0.5f,
                                 /*class_agnostic=*/false);
  EXPECT_EQ(kept.size(), 2u);
}

TEST(GreedyDetectionNmsTest, OutputIsDescendingScore) {
  std::vector<Detection> dets = {
      Det(0.4f, 0, 0.0f, 0.0f, 0.2f, 0.2f),
      Det(0.9f, 0, 0.5f, 0.5f, 0.2f, 0.2f),
      Det(0.7f, 0, 0.2f, 0.2f, 0.2f, 0.2f),
  };
  auto kept = GreedyDetectionNms(dets, /*iou_threshold=*/0.5f,
                                 /*class_agnostic=*/false);
  ASSERT_EQ(kept.size(), 3u);
  EXPECT_GE(kept[0].score(0), kept[1].score(0));
  EXPECT_GE(kept[1].score(0), kept[2].score(0));
}

TEST(GreedyDetectionNmsTest, EmptyInputReturnsEmpty) {
  std::vector<Detection> dets;
  auto kept = GreedyDetectionNms(dets, 0.5f, false);
  EXPECT_TRUE(kept.empty());
}

TEST(GreedyDetectionNmsTest, SingleDetectionKept) {
  std::vector<Detection> dets = {Det(0.8f, 0, 0.1f, 0.1f, 0.3f, 0.3f)};
  auto kept = GreedyDetectionNms(dets, 0.5f, false);
  ASSERT_EQ(kept.size(), 1u);
  EXPECT_NEAR(kept[0].score(0), 0.8f, 1e-6f);
}

TEST(GreedyDetectionNmsTest, HandlesMissingScoreAndLabel) {
  // `a` is a normal detection; `b` shares the same box but has neither a score
  // nor a label_id. The guarded code reads `b` as score 0.0 and label -1, so it
  // sorts last and is treated as a distinct class -> not suppressed by `a`.
  // Without the score_size()/label_id_size() guards this would be an
  // out-of-bounds proto read (DCHECK-fatal in debug, UB in opt).
  Detection a = Det(0.9f, 1, 0.0f, 0.0f, 0.5f, 0.5f);
  Detection b;  // no score, no label_id, same box as `a`.
  *b.mutable_location_data() = a.location_data();
  auto kept = GreedyDetectionNms({a, b}, /*iou_threshold=*/0.5f,
                                 /*class_agnostic=*/false);
  EXPECT_EQ(kept.size(), 2u);
}

// -------------------------------------------------------------------------
// GreedyOrientedDetectionNms tests
// -------------------------------------------------------------------------

OrientedDetection Obb(float score, int label_id, float cx, float cy, float w,
                      float h, float rot) {
  OrientedDetection d;
  d.set_cx(cx);
  d.set_cy(cy);
  d.set_width(w);
  d.set_height(h);
  d.set_rotation(rot);
  d.add_score(score);
  d.add_label_id(label_id);
  return d;
}

TEST(GreedyOrientedDetectionNmsTest, BasicDedupSameClass) {
  std::vector<OrientedDetection> dets = {
      Obb(0.7f, 0, 0.5f, 0.5f, 0.4f, 0.4f, 0.3f),
      Obb(0.9f, 0, 0.5f, 0.5f, 0.4f, 0.4f, 0.3f),
  };
  auto kept = GreedyOrientedDetectionNms(dets, /*iou_threshold=*/0.5f,
                                         /*class_agnostic=*/false);
  ASSERT_EQ(kept.size(), 1u);
  EXPECT_NEAR(kept[0].score(0), 0.9f, 1e-6f);
}

TEST(GreedyOrientedDetectionNmsTest, RotationSeparatesCrossedBoxes) {
  // Two thin boxes crossing at ~90 degrees: rotated IoU is
  // (0.1*0.1)/(2*0.6*0.1 - 0.01) ~= 0.09, far below the threshold.
  std::vector<OrientedDetection> dets = {
      Obb(0.9f, 0, 0.5f, 0.5f, 0.6f, 0.1f, 0.0f),
      Obb(0.8f, 0, 0.5f, 0.5f, 0.6f, 0.1f, 1.5708f),
  };
  auto kept = GreedyOrientedDetectionNms(dets, 0.5f, false);
  EXPECT_EQ(kept.size(), 2u);
}

TEST(GreedyOrientedDetectionNmsTest, PerClassDefaultKeepsDifferentClasses) {
  std::vector<OrientedDetection> dets = {
      Obb(0.9f, 0, 0.5f, 0.5f, 0.4f, 0.4f, 0.0f),
      Obb(0.8f, 1, 0.5f, 0.5f, 0.4f, 0.4f, 0.0f),
  };
  EXPECT_EQ(GreedyOrientedDetectionNms(dets, 0.5f, false).size(), 2u);
  EXPECT_EQ(GreedyOrientedDetectionNms(dets, 0.5f, true).size(), 1u);
}

TEST(GreedyOrientedDetectionNmsTest, OutputIsDescendingScore) {
  std::vector<OrientedDetection> dets = {
      Obb(0.4f, 0, 0.1f, 0.1f, 0.1f, 0.1f, 0.0f),
      Obb(0.9f, 0, 0.8f, 0.8f, 0.1f, 0.1f, 0.0f),
      Obb(0.7f, 0, 0.5f, 0.5f, 0.1f, 0.1f, 0.0f),
  };
  auto kept = GreedyOrientedDetectionNms(dets, 0.5f, false);
  ASSERT_EQ(kept.size(), 3u);
  EXPECT_GE(kept[0].score(0), kept[1].score(0));
  EXPECT_GE(kept[1].score(0), kept[2].score(0));
}

}  // namespace
}  // namespace mediapipe
