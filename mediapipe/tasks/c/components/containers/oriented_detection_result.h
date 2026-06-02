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

#ifndef MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_ORIENTED_DETECTION_RESULT_H_
#define MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_ORIENTED_DETECTION_RESULT_H_

#include <stdint.h>

#include "mediapipe/tasks/c/components/containers/category.h"

#ifdef __cplusplus
extern "C" {
#endif

// One oriented (rotated) bounding-box detection, in original-image PIXEL units.
struct MpOrientedDetection {
  struct MpCategory* categories;
  uint32_t categories_count;
  float cx;        // box center x, pixels
  float cy;        // box center y, pixels
  float width;     // pixels
  float height;    // pixels
  float rotation;  // radians, counter-clockwise
};

struct MpOrientedDetectionResult {
  struct MpOrientedDetection* detections;
  uint32_t detections_count;
};

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_ORIENTED_DETECTION_RESULT_H_
