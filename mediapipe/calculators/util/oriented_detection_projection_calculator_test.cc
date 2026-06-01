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

#include <algorithm>
#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

OrientedDetection Obb(float cx, float cy, float w, float h, float rot) {
  OrientedDetection d;
  d.set_cx(cx); d.set_cy(cy); d.set_width(w); d.set_height(h);
  d.set_rotation(rot); d.add_score(0.9f); d.add_label_id(0);
  return d;
}

// Row-major 4x4 affine; only m[0],m[1],m[3] (x) and m[4],m[5],m[7] (y) used.
std::unique_ptr<std::array<float, 16>> Matrix(float sx, float sy, float tx,
                                              float ty) {
  auto m = std::make_unique<std::array<float, 16>>();
  m->fill(0.0f);
  (*m)[0] = sx;  (*m)[3] = tx;
  (*m)[5] = sy;  (*m)[7] = ty;
  (*m)[10] = 1;  (*m)[15] = 1;
  return m;
}

std::unique_ptr<std::vector<OrientedDetection>> Dets(OrientedDetection d) {
  auto v = std::make_unique<std::vector<OrientedDetection>>();
  v->push_back(std::move(d));
  return v;
}

const std::vector<OrientedDetection>& RunCalc(
    CalculatorRunner* runner, std::unique_ptr<std::vector<OrientedDetection>> in,
    std::unique_ptr<std::array<float, 16>> m) {
  runner->MutableInputs()->Tag("ORIENTED_DETECTIONS").packets.push_back(
      Adopt(in.release()).At(Timestamp(0)));
  runner->MutableInputs()->Tag("PROJECTION_MATRIX").packets.push_back(
      Adopt(m.release()).At(Timestamp(0)));
  MP_EXPECT_OK(runner->Run());
  return runner->Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
      .Get<std::vector<OrientedDetection>>();
}

CalculatorRunner MakeRunner() {
  return CalculatorRunner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "OrientedDetectionProjectionCalculator"
    input_stream: "ORIENTED_DETECTIONS:in"
    input_stream: "PROJECTION_MATRIX:matrix"
    output_stream: "ORIENTED_DETECTIONS:out"
  )pb"));
}

TEST(OrientedDetectionProjectionCalculatorTest, IdentityPreservesBox) {
  auto runner = MakeRunner();
  const auto& out = RunCalc(&runner, Dets(Obb(0.5f, 0.5f, 0.4f, 0.2f, 0.0f)),
                        Matrix(1, 1, 0, 0));
  ASSERT_EQ(out.size(), 1);
  EXPECT_NEAR(out[0].cx(), 0.5f, 1e-4);
  EXPECT_NEAR(out[0].cy(), 0.5f, 1e-4);
  // minAreaRect may return width/height transposed; assert the unordered set.
  const float w = out[0].width(), h = out[0].height();
  const float lo = std::min(w, h), hi = std::max(w, h);
  EXPECT_NEAR(lo, 0.2f, 1e-3);
  EXPECT_NEAR(hi, 0.4f, 1e-3);
}

TEST(OrientedDetectionProjectionCalculatorTest, ScaleTranslateAxisAligned) {
  auto runner = MakeRunner();
  const auto& out = RunCalc(&runner, Dets(Obb(0.6f, 0.4f, 0.4f, 0.2f, 0.0f)),
                        Matrix(0.5f, 0.5f, 0.1f, 0.0f));
  ASSERT_EQ(out.size(), 1);
  EXPECT_NEAR(out[0].cx(), 0.4f, 1e-3);   // 0.5*0.6 + 0.1
  EXPECT_NEAR(out[0].cy(), 0.2f, 1e-3);   // 0.5*0.4
  const float w = out[0].width(), h = out[0].height();
  const float lo = std::min(w, h), hi = std::max(w, h);
  EXPECT_NEAR(lo, 0.1f, 1e-3);            // 0.2*0.5
  EXPECT_NEAR(hi, 0.2f, 1e-3);            // 0.4*0.5
}

}  // namespace
}  // namespace mediapipe
