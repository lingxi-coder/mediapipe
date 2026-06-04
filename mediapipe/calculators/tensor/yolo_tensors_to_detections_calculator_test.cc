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
#include <memory>
#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using BatchDetections = std::vector<std::vector<Detection>>;

// Builds a single-tensor input packet from row-major float data.
std::unique_ptr<std::vector<Tensor>> MakeTensor(
    const Tensor::Shape& shape, const std::vector<float>& data) {
  auto tensor = Tensor(Tensor::ElementType::kFloat32, shape);
  auto write = tensor.GetCpuWriteView();
  std::copy(data.begin(), data.end(), write.buffer<float>());
  auto v = std::make_unique<std::vector<Tensor>>();
  v->push_back(std::move(tensor));
  return v;
}

TEST(YoloTensorsToDetectionsCalculatorTest, ChannelsFirstSingleClass) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 1
        conf_threshold: 0.25
      }
    }
  )pb"));

  // Shape [N=1, C=5, A=2]; channels = cx,cy,w,h,score; row-major c*A + a.
  // Anchor0: cx .5 cy .5 w .2 h .4 score .9  -> kept
  // Anchor1: cx .25 cy .75 w .1 h .1 score .1 -> dropped (< 0.25)
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 5, 2},
                       {0.5f, 0.25f, 0.5f, 0.75f, 0.2f, 0.1f, 0.4f, 0.1f,
                        0.9f, 0.1f})
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Tag("DETECTIONS").packets;
  ASSERT_EQ(out.size(), 1);
  const BatchDetections& batch = out[0].Get<BatchDetections>();
  ASSERT_EQ(batch.size(), 1);          // one batch row
  ASSERT_EQ(batch[0].size(), 1);       // one kept detection
  const Detection& d = batch[0][0];
  ASSERT_EQ(d.score_size(), 1);
  EXPECT_NEAR(d.score(0), 0.9f, 1e-5);
  ASSERT_EQ(d.label_id_size(), 1);
  EXPECT_EQ(d.label_id(0), 0);
  const auto& bb = d.location_data().relative_bounding_box();
  EXPECT_NEAR(bb.xmin(), 0.4f, 1e-5);   // cx - w/2
  EXPECT_NEAR(bb.ymin(), 0.3f, 1e-5);   // cy - h/2
  EXPECT_NEAR(bb.width(), 0.2f, 1e-5);
  EXPECT_NEAR(bb.height(), 0.4f, 1e-5);
}

TEST(YoloTensorsToDetectionsCalculatorTest, ChannelsLastSameResult) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_LAST
        num_classes: 1
        conf_threshold: 0.25
      }
    }
  )pb"));

  // Shape [N=1, A=2, C=5]; row-major a*C + c. Same two anchors as Task 2.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 2, 5},
                       {0.5f, 0.5f, 0.2f, 0.4f, 0.9f,
                        0.25f, 0.75f, 0.1f, 0.1f, 0.1f})
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<
          std::vector<std::vector<Detection>>>();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 1);
  const auto& bb = batch[0][0].location_data().relative_bounding_box();
  EXPECT_NEAR(bb.xmin(), 0.4f, 1e-5);
  EXPECT_NEAR(bb.ymin(), 0.3f, 1e-5);
  EXPECT_NEAR(bb.width(), 0.2f, 1e-5);
  EXPECT_NEAR(bb.height(), 0.4f, 1e-5);
}

TEST(YoloTensorsToDetectionsCalculatorTest, MultiClassArgmaxAndThreshold) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 3
        conf_threshold: 0.5
      }
    }
  )pb"));

  // Shape [N=1, C=7, A=2]; channels = cx,cy,w,h, s0,s1,s2 ; index c*A + a.
  // Anchor0: box(.5,.5,.2,.2) scores [.1,.8,.3] -> class 1, score .8 (kept)
  // Anchor1: box(.5,.5,.2,.2) scores [.4,.2,.1] -> max .4 < .5 (dropped)
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 7, 2},
                       {0.5f, 0.5f,   // cx
                        0.5f, 0.5f,   // cy
                        0.2f, 0.2f,   // w
                        0.2f, 0.2f,   // h
                        0.1f, 0.4f,   // s0
                        0.8f, 0.2f,   // s1
                        0.3f, 0.1f})  // s2
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<
          std::vector<std::vector<Detection>>>();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 1);
  EXPECT_EQ(batch[0][0].label_id(0), 1);
  EXPECT_NEAR(batch[0][0].score(0), 0.8f, 1e-5);
}

TEST(YoloTensorsToDetectionsCalculatorTest, BatchNativeTwoRows) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 1
        conf_threshold: 0.25
      }
    }
  )pb"));

  // Shape [N=2, C=5, A=1]; per row index ((n*C)+c)*A + a, A=1 so = n*5 + c.
  // Row0 score .9 (kept); Row1 score .1 (dropped) -> outer size 2, sizes {1,0}.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{2, 5, 1},
                       {0.5f, 0.5f, 0.2f, 0.4f, 0.9f,    // row 0
                        0.5f, 0.5f, 0.2f, 0.4f, 0.1f})   // row 1
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<
          std::vector<std::vector<Detection>>>();
  ASSERT_EQ(batch.size(), 2);       // one inner vector per batch row
  EXPECT_EQ(batch[0].size(), 1);
  EXPECT_EQ(batch[1].size(), 0);    // empty row still present
}

TEST(YoloTensorsToDetectionsCalculatorTest, TopKKeepsHighestScores) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 1
        conf_threshold: 0.1
        max_detections_before_nms: 2
      }
    }
  )pb"));

  // Shape [N=1, C=5, A=3]; index c*A + a. Three anchors, scores .3,.9,.6.
  // top-2 by score -> keep .9 and .6, drop .3.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 5, 3},
                       {0.5f, 0.5f, 0.5f,    // cx
                        0.5f, 0.5f, 0.5f,    // cy
                        0.2f, 0.2f, 0.2f,    // w
                        0.2f, 0.2f, 0.2f,    // h
                        0.3f, 0.9f, 0.6f})   // score
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<
          std::vector<std::vector<Detection>>>();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 2);
  EXPECT_NEAR(batch[0][0].score(0), 0.9f, 1e-5);  // sorted desc
  EXPECT_NEAR(batch[0][1].score(0), 0.6f, 1e-5);
}

// ---------------------------------------------------------------------------
// Tile-local NMS tests
// ---------------------------------------------------------------------------

// Two highly-overlapping same-class boxes + one non-overlapping box.
// With NMS threshold=0.5 the duplicate should be removed; the separate box kept.
TEST(YoloTensorsToDetectionsCalculatorTest, TileLocalNmsRemovesWithinRowDuplicate) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 1
        conf_threshold: 0.1
        tile_local_nms_iou_threshold: 0.5
      }
    }
  )pb"));

  // Shape [N=1, C=5, A=3]; CHANNELS_FIRST layout: index c*A + a.
  // Anchor0: cx=0.5 cy=0.5 w=0.4 h=0.4 score=0.9  (high scorer, kept)
  // Anchor1: cx=0.5 cy=0.5 w=0.4 h=0.4 score=0.6  (duplicate of anchor0, suppressed)
  // Anchor2: cx=0.1 cy=0.1 w=0.1 h=0.1 score=0.7  (separate box, kept)
  // Anchor0 and Anchor1 are identical boxes -> IoU=1.0 > 0.5, so anchor1 suppressed.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 5, 3},
                       {0.5f, 0.5f, 0.1f,    // cx: a0, a1, a2
                        0.5f, 0.5f, 0.1f,    // cy
                        0.4f, 0.4f, 0.1f,    // w
                        0.4f, 0.4f, 0.1f,    // h
                        0.9f, 0.6f, 0.7f})   // score (class 0)
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<BatchDetections>();
  ASSERT_EQ(batch.size(), 1);
  // Expect 2 detections: anchor0 (score 0.9) and anchor2 (score 0.7).
  // Anchor1 (score 0.6, same box as anchor0) must be suppressed.
  ASSERT_EQ(batch[0].size(), 2);
  EXPECT_NEAR(batch[0][0].score(0), 0.9f, 1e-5);
  EXPECT_NEAR(batch[0][1].score(0), 0.7f, 1e-5);
}

// Two overlapping boxes of DIFFERENT classes. With per-class NMS (default),
// both should be kept because they have different label_ids.
TEST(YoloTensorsToDetectionsCalculatorTest, TileLocalNmsIsPerClassByDefault) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 2
        conf_threshold: 0.1
        tile_local_nms_iou_threshold: 0.5
        tile_local_nms_class_agnostic: false
      }
    }
  )pb"));

  // Shape [N=1, C=6, A=2]; channels = cx,cy,w,h,s0,s1; index c*A + a.
  // Anchor0: box(0.5,0.5,0.4,0.4), class0=0.9, class1=0.2 -> class 0, score 0.9
  // Anchor1: box(0.5,0.5,0.4,0.4), class0=0.1, class1=0.8 -> class 1, score 0.8
  // Same geometry, IoU=1.0 > 0.5, but different classes -> both kept (per-class).
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 6, 2},
                       {0.5f, 0.5f,    // cx
                        0.5f, 0.5f,    // cy
                        0.4f, 0.4f,    // w
                        0.4f, 0.4f,    // h
                        0.9f, 0.1f,    // s0
                        0.2f, 0.8f})   // s1
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<BatchDetections>();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 2);  // both kept: different classes
  EXPECT_EQ(batch[0][0].label_id(0), 0);
  EXPECT_EQ(batch[0][1].label_id(0), 1);
}

// Same two overlapping different-class boxes, but with class_agnostic=true.
// The lower-scoring box must be suppressed regardless of class.
TEST(YoloTensorsToDetectionsCalculatorTest,
     TileLocalNmsClassAgnosticSuppressesAcrossClasses) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 2
        conf_threshold: 0.1
        tile_local_nms_iou_threshold: 0.5
        tile_local_nms_class_agnostic: true
      }
    }
  )pb"));

  // Same tensor as TileLocalNmsIsPerClassByDefault.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 6, 2},
                       {0.5f, 0.5f,    // cx
                        0.5f, 0.5f,    // cy
                        0.4f, 0.4f,    // w
                        0.4f, 0.4f,    // h
                        0.9f, 0.1f,    // s0
                        0.2f, 0.8f})   // s1
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<BatchDetections>();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 1);  // lower-scoring suppressed across classes
  EXPECT_EQ(batch[0][0].label_id(0), 0);   // class 0 had score 0.9 (highest)
  EXPECT_NEAR(batch[0][0].score(0), 0.9f, 1e-5);
}

// NMS operates WITHIN each row only. Two rows each with a box at the same
// coordinates; NMS enabled -> both survive (one per row).
TEST(YoloTensorsToDetectionsCalculatorTest, TileLocalNmsDoesNotCrossRows) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 1
        conf_threshold: 0.1
        tile_local_nms_iou_threshold: 0.5
      }
    }
  )pb"));

  // Shape [N=2, C=5, A=1]; CHANNELS_FIRST, A=1 so index (n*C + c)*1 = n*5+c.
  // Row 0: cx=0.5,cy=0.5,w=0.4,h=0.4,score=0.9
  // Row 1: cx=0.5,cy=0.5,w=0.4,h=0.4,score=0.8
  // Both rows have a box at the same location, but NMS is per-row -> both kept.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{2, 5, 1},
                       {0.5f, 0.5f, 0.4f, 0.4f, 0.9f,    // row 0
                        0.5f, 0.5f, 0.4f, 0.4f, 0.8f})   // row 1
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<BatchDetections>();
  ASSERT_EQ(batch.size(), 2);
  EXPECT_EQ(batch[0].size(), 1);  // row 0: 1 detection
  EXPECT_EQ(batch[1].size(), 1);  // row 1: 1 detection (not suppressed by row 0)
  EXPECT_NEAR(batch[0][0].score(0), 0.9f, 1e-5);
  EXPECT_NEAR(batch[1][0].score(0), 0.8f, 1e-5);
}

// max_detections_after_tile_nms caps the result to the top N highest-scoring.
TEST(YoloTensorsToDetectionsCalculatorTest, MaxDetectionsAfterTileNmsCaps) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 1
        conf_threshold: 0.1
        tile_local_nms_iou_threshold: 0.5
        max_detections_after_tile_nms: 2
      }
    }
  )pb"));

  // Shape [N=1, C=5, A=4]; four non-overlapping boxes (different positions),
  // scores 0.3, 0.9, 0.5, 0.7. None overlap -> NMS keeps all 4, then cap to 2.
  // Expected survivors: score 0.9 and 0.7.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 5, 4},
                       {0.1f, 0.5f, 0.9f, 0.3f,    // cx
                        0.1f, 0.5f, 0.9f, 0.3f,    // cy
                        0.1f, 0.1f, 0.1f, 0.1f,    // w
                        0.1f, 0.1f, 0.1f, 0.1f,    // h
                        0.3f, 0.9f, 0.5f, 0.7f})   // score
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<BatchDetections>();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 2);
  EXPECT_NEAR(batch[0][0].score(0), 0.9f, 1e-5);
  EXPECT_NEAR(batch[0][1].score(0), 0.7f, 1e-5);
}

// With tile_local_nms_iou_threshold=0 (default), two overlapping boxes both
// survive — current behavior is preserved (no NMS applied).
TEST(YoloTensorsToDetectionsCalculatorTest, DisabledTileLocalNmsIsNoOp) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        layout: CHANNELS_FIRST
        num_classes: 1
        conf_threshold: 0.1
      }
    }
  )pb"));

  // Two identical boxes; with NMS disabled both should survive.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 5, 2},
                       {0.5f, 0.5f,    // cx
                        0.5f, 0.5f,    // cy
                        0.4f, 0.4f,    // w
                        0.4f, 0.4f,    // h
                        0.9f, 0.6f})   // score
                .release())
          .At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const auto& batch =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<BatchDetections>();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 2);  // both boxes kept (NMS disabled)
  EXPECT_NEAR(batch[0][0].score(0), 0.9f, 1e-5);
  EXPECT_NEAR(batch[0][1].score(0), 0.6f, 1e-5);
}

TEST(YoloTensorsToDetectionsCalculatorTest, PixelSpaceBoxesNormalizedByInputDims) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:dets"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        num_classes: 1
        conf_threshold: 0.25
        input_width: 640
        input_height: 480
      }
    }
  )pb"));
  // CHANNELS_FIRST [1, 5, 1]: rows = cx,cy,w,h,score0 ; one anchor.
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 5, 1},
                       {320.0f, 240.0f, 64.0f, 48.0f, 0.9f})
                .release())
          .At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out =
      runner.Outputs().Tag("DETECTIONS").packets[0].Get<std::vector<std::vector<Detection>>>();
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].size(), 1u);
  const auto& bb = out[0][0].location_data().relative_bounding_box();
  // (cx-w/2)/W = (320-32)/640 = 0.45 ; (cy-h/2)/H = (240-24)/480 = 0.45
  EXPECT_NEAR(bb.xmin(), 0.45f, 1e-5);
  EXPECT_NEAR(bb.ymin(), 0.45f, 1e-5);
  EXPECT_NEAR(bb.width(), 64.0f / 640.0f, 1e-5);   // 0.1
  EXPECT_NEAR(bb.height(), 48.0f / 480.0f, 1e-5);  // 0.1
}

TEST(YoloTensorsToDetectionsCalculatorTest, AllowClassesFiltersByIndex) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:dets"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        num_classes: 2 conf_threshold: 0.25 allow_classes: 1
      }
    }
  )pb"));
  // [1, 4+2=6, 2]: cx,cy,w,h, s0,s1 ; anchor0 argmax class0(0.9), anchor1 class1(0.8).
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 6, 2},
                       {0.5f,0.5f, 0.5f,0.5f, 0.2f,0.2f, 0.2f,0.2f,
                        0.9f,0.1f, 0.1f,0.8f}).release()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Tag("DETECTIONS").packets[0]
                        .Get<std::vector<std::vector<Detection>>>();
  ASSERT_EQ(out[0].size(), 1u);          // only the class-1 anchor survives
  EXPECT_EQ(out[0][0].label_id(0), 1);
}
TEST(YoloTensorsToDetectionsCalculatorTest, IgnoreClassesDropsByIndex) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:dets"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        num_classes: 2 conf_threshold: 0.25 ignore_classes: 0
      }
    }
  )pb"));
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 6, 2},
                       {0.5f,0.5f, 0.5f,0.5f, 0.2f,0.2f, 0.2f,0.2f,
                        0.9f,0.1f, 0.1f,0.8f}).release()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Tag("DETECTIONS").packets[0]
                        .Get<std::vector<std::vector<Detection>>>();
  ASSERT_EQ(out[0].size(), 1u);
  EXPECT_EQ(out[0][0].label_id(0), 1);   // class 0 dropped
}

}  // namespace
}  // namespace mediapipe
