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

#include <set>
#include <thread>
#include <vector>

#include "BoTSORT.h"
#include "DataType.h"
#include "GlobalMotionCompensation.h"
#include "TrackerParams.h"
#include "gtest/gtest.h"
#include "opencv2/core.hpp"

namespace {

TrackerParams TestParams() {
  TrackerParams params;
  params.track_high_thresh = 0.5F;
  params.track_low_thresh = 0.1F;
  params.new_track_thresh = 0.5F;
  params.match_thresh = 0.8F;
  params.track_buffer = 30;
  return params;
}

Detection Box(float x, float y, float w = 20.0F, float h = 40.0F) {
  Detection d;
  d.bbox_tlwh = cv::Rect_<float>(x, y, w, h);
  d.class_id = 1;
  d.confidence = 0.9F;
  return d;
}

// Proves the vendored motion-only BoTSORT builds, links (no TensorRT/ONNX/
// INIReader), and tracks the SAME object across frames with GMC + ReID off.
//
// This asserts PERSISTENCE, not mere survival: it captures the output track's
// id (`Track::track_id`, a public int) on each frame and requires it to be
// STABLE across consecutive output frames. A tracker that spawned a fresh id
// every frame would fail this. We feed 3 deterministic frames of the same box
// drifting 2px/frame; BoT-SORT may keep a freshly-seen track tentative before
// confirming it, so we assert that the last two OUTPUT frames agree on the id.
TEST(BotsortSmokeTest, KeepsStableTrackIdAcrossFrames) {
  TrackerParams params = TestParams();
  BoTSORT tracker(params);

  // Single static black frame, reused each step (motion comes from the box).
  cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));

  Detection d;
  d.bbox_tlwh = cv::Rect_<float>(50.f, 50.f, 20.f, 40.f);
  d.class_id = 1;
  d.confidence = 0.9f;

  // Deterministic drift: same object moves 2px to the right each frame.
  const float kStartX = 50.f;
  const float kDriftPerFrame = 2.f;
  const int kNumFrames = 3;

  std::vector<std::shared_ptr<Track>> tracks;
  std::vector<int> output_track_ids;  // id reported on each frame that output one
  for (int i = 0; i < kNumFrames; ++i) {
    Detection step = d;
    step.bbox_tlwh.x = kStartX + kDriftPerFrame * static_cast<float>(i);
    tracks = tracker.track({step}, frame);
    if (!tracks.empty()) {
      output_track_ids.push_back(tracks[0]->track_id);
    }
  }

  // Last frame must produce an output track for the (still-present) object.
  ASSERT_FALSE(tracks.empty());
  EXPECT_EQ(tracks[0]->get_class_id(), 1);

  // Persistence: at least two output frames, and the last two agree on the id
  // (the same object kept the same track_id rather than re-spawning).
  ASSERT_GE(output_track_ids.size(), 2u)
      << "Expected the track to appear in at least two output frames";
  EXPECT_EQ(output_track_ids[output_track_ids.size() - 1],
            output_track_ids[output_track_ids.size() - 2])
      << "Track id changed between consecutive output frames; the tracker is "
         "re-spawning ids instead of persisting the same track";
}

TEST(BotsortSmokeTest, SkipPredictsAndRedetectionKeepsId) {
  BoTSORT tracker(TestParams());
  cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));

  auto detected = tracker.track({Box(50.0F, 50.0F)}, frame);
  ASSERT_EQ(detected.size(), 1u);
  const int id = detected[0]->track_id;

  for (int i = 0; i < 3; ++i) {
    auto predicted = tracker.predict_only(frame);
    ASSERT_EQ(predicted.size(), 1u);
    EXPECT_EQ(predicted[0]->track_id, id);
  }

  auto redetected = tracker.track_observed({Box(50.0F, 50.0F)}, frame,
                                            {cv::Rect(0, 0, frame.cols,
                                                      frame.rows)});
  ASSERT_EQ(redetected.size(), 1u);
  EXPECT_EQ(redetected[0]->track_id, id);
}

TEST(BotsortSmokeTest, PartialObservationAppliesNegativeEvidenceLocally) {
  BoTSORT tracker(TestParams());
  cv::Mat frame(200, 100, CV_8UC3, cv::Scalar(0, 0, 0));

  auto detected = tracker.track(
      {Box(10.0F, 50.0F), Box(70.0F, 50.0F)}, frame);
  ASSERT_EQ(detected.size(), 2u);
  const int left_id = detected[0]->track_id;
  const int right_id = detected[1]->track_id;

  auto partial = tracker.track_observed({}, frame, {cv::Rect(0, 0, 50, 200)});
  ASSERT_EQ(partial.size(), 1u);
  EXPECT_EQ(partial[0]->track_id, right_id);
  EXPECT_NE(partial[0]->track_id, left_id);
}

TEST(BotsortSmokeTest,
     PartialObservationDoesNotConsumeFutureLostTrackRetention) {
  TrackerParams params = TestParams();
  params.track_buffer = 3;
  BoTSORT tracker(params);
  cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));

  auto detected = tracker.track({Box(140.0F, 50.0F)}, frame);
  ASSERT_EQ(detected.size(), 1u);
  const int id = detected[0]->track_id;

  // These refreshes only observe the left side of the frame. They must advance
  // the active track's clock without treating the unobserved right-side track
  // as lost or consuming its future lost-track retention window.
  for (int i = 0; i < 5; ++i) {
    auto partial =
        tracker.track_observed({}, frame, {cv::Rect(0, 0, 100, 200)});
    ASSERT_EQ(partial.size(), 1u);
    EXPECT_EQ(partial[0]->track_id, id);
  }

  EXPECT_TRUE(tracker.track_observed(
                         {}, frame, {cv::Rect(0, 0, frame.cols, frame.rows)})
                  .empty());
  EXPECT_TRUE(tracker.predict_only(frame).empty());

  auto refound = tracker.track_observed(
      {Box(140.0F, 50.0F)}, frame,
      {cv::Rect(0, 0, frame.cols, frame.rows)});
  ASSERT_EQ(refound.size(), 1u);
  EXPECT_EQ(refound[0]->track_id, id);
}

TEST(BotsortSmokeTest, LostTracksCanBeRefoundAfterPredictionOnlyFrames) {
  BoTSORT tracker(TestParams());
  cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));

  auto detected = tracker.track({Box(50.0F, 50.0F)}, frame);
  ASSERT_EQ(detected.size(), 1u);
  const int id = detected[0]->track_id;

  EXPECT_TRUE(tracker.track_observed(
                              {}, frame,
                              {cv::Rect(0, 0, frame.cols, frame.rows)})
                  .empty());
  EXPECT_TRUE(tracker.predict_only(frame).empty());

  auto refound = tracker.track_observed(
      {Box(50.0F, 50.0F)}, frame,
      {cv::Rect(0, 0, frame.cols, frame.rows)});
  ASSERT_EQ(refound.size(), 1u);
  EXPECT_EQ(refound[0]->track_id, id);
}

TEST(BotsortSmokeTest, ForegroundMaskUsesExactDownscaledHalfOpenBounds) {
  const cv::Mat mask = botsort_internal::BuildForegroundMask(
      cv::Size(5, 4), 2.0F,
      {cv::Rect_<float>(1.0F, 3.0F, 4.0F, 3.0F),
       cv::Rect_<float>(-10.0F, -10.0F, 1.0F, 1.0F)});

  ASSERT_EQ(mask.type(), CV_8UC1);
  EXPECT_EQ(mask.at<unsigned char>(0, 0), 255);
  EXPECT_EQ(mask.at<unsigned char>(1, 0), 0);
  EXPECT_EQ(mask.at<unsigned char>(2, 2), 0);
  EXPECT_EQ(mask.at<unsigned char>(3, 2), 255);
  EXPECT_EQ(cv::countNonZero(mask), 14);
}

TEST(BotsortSmokeTest, FullyMaskedForegroundFallsBackToIdentityMask) {
  const cv::Mat mask = botsort_internal::BuildForegroundMask(
      cv::Size(4, 4), 2.0F,
      {cv::Rect_<float>(-1.0F, -1.0F, 10.0F, 10.0F)});
  EXPECT_EQ(cv::countNonZero(mask), 0);
}

TEST(BotsortSmokeTest,
     SparseGmcReturnsIdentityWhenCurrentFrameHasNoBackgroundFeatures) {
  SparseOptFlow_Params params;
  params.downscale = 1.0F;
  SparseOptFlow_GMC gmc(params);

  cv::Mat previous(96, 96, CV_8UC3);
  cv::RNG rng(12345);
  rng.fill(previous, cv::RNG::UNIFORM, 0, 255);
  EXPECT_TRUE(gmc.apply(previous, {}, {}).isApprox(
      HomographyMatrix::Identity(), 1e-6));

  cv::Mat current;
  const cv::Mat translation =
      (cv::Mat_<double>(2, 3) << 1.0, 0.0, 6.0, 0.0, 1.0, 0.0);
  cv::warpAffine(previous, current, translation, previous.size());
  const HomographyMatrix transform = gmc.apply(
      current, {},
      {cv::Rect_<float>(0.0F, 0.0F, static_cast<float>(current.cols),
                        static_cast<float>(current.rows))});

  EXPECT_TRUE(transform.isApprox(HomographyMatrix::Identity(), 1e-6))
      << transform;
}

TEST(BotsortSmokeTest, TrackIdsAreUniqueAcrossConcurrentAllocators) {
  constexpr int kThreads = 4;
  constexpr int kIdsPerThread = 128;
  std::vector<int> ids(kThreads * kIdsPerThread);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int thread = 0; thread < kThreads; ++thread) {
    workers.emplace_back([&, thread] {
      for (int i = 0; i < kIdsPerThread; ++i) {
        ids[thread * kIdsPerThread + i] = Track::next_id();
      }
    });
  }
  for (auto& worker : workers) worker.join();

  const std::set<int> unique_ids(ids.begin(), ids.end());
  EXPECT_EQ(unique_ids.size(), ids.size());
}

}  // namespace
