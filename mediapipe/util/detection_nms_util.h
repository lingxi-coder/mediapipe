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

#ifndef MEDIAPIPE_UTIL_DETECTION_NMS_UTIL_H_
#define MEDIAPIPE_UTIL_DETECTION_NMS_UTIL_H_

#include <vector>

#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/oriented_detection.pb.h"

namespace mediapipe {

// Axis-aligned IoU of two detections' relative_bounding_box. 0 if either box
// has non-positive area or the union is empty.
float DetectionRelativeIoU(const Detection& a, const Detection& b);

// Greedy axis-aligned NMS over relative-bbox detections. Sorts by score(0)
// descending (stable), keeps higher-scoring boxes, suppresses a lower-scoring
// box when IoU > iou_threshold. When !class_agnostic, only boxes with the same
// label_id(0) suppress each other. Returns kept detections in descending-score
// order. Requires each Detection to carry >=1 score and >=1 label_id.
std::vector<Detection> GreedyDetectionNms(std::vector<Detection> dets,
                                          float iou_threshold,
                                          bool class_agnostic);

// Rotated IoU of two oriented detections, computed in the coordinate space
// the OrientedDetection fields are defined in (frame-normalized; see
// oriented_detection.proto — on non-square frames that space is anisotropic
// w.r.t. pixels). 0 if the boxes don't intersect or the union is empty.
float OrientedDetectionIoU(const OrientedDetection& a,
                           const OrientedDetection& b);

// Greedy rotated-IoU NMS over OrientedDetections. Sorts by score(0)
// descending (stable), keeps higher-scoring boxes, suppresses a lower-scoring
// box when IoU >= iou_threshold (note: >=, matching
// RotatedNonMaxSuppressionCalculator, while the axis-aligned util uses >).
// When !class_agnostic, only boxes with the same label_id(0) suppress each
// other. A missing score reads as 0.0 and a missing label_id as -1. Returns
// kept detections in descending-score order.
std::vector<OrientedDetection> GreedyOrientedDetectionNms(
    std::vector<OrientedDetection> dets, float iou_threshold,
    bool class_agnostic);

}  // namespace mediapipe
#endif  // MEDIAPIPE_UTIL_DETECTION_NMS_UTIL_H_
