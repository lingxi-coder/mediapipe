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

#include "mediapipe/tasks/c/components/containers/oriented_detection_result_converter.h"

#include <cstddef>
#include <cstdlib>

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/category_converter.h"
#include "mediapipe/tasks/c/components/containers/oriented_detection_result.h"
#include "mediapipe/tasks/cc/components/containers/oriented_object_detection_result.h"

namespace mediapipe::tasks::c::components::containers {

namespace {
using CppOrientedDetection =
    ::mediapipe::tasks::components::containers::OrientedObjectDetection;
using CppOrientedDetectionResult =
    ::mediapipe::tasks::components::containers::OrientedObjectDetectionResult;
}  // namespace

void CppConvertToOrientedDetection(const CppOrientedDetection& in,
                                   MpOrientedDetection* out) {
  out->categories_count = in.categories.size();
  out->categories = new MpCategory[out->categories_count];
  for (size_t i = 0; i < out->categories_count; ++i) {
    CppConvertToCategory(in.categories[i], &out->categories[i]);
  }
  out->cx = in.cx;
  out->cy = in.cy;
  out->width = in.width;
  out->height = in.height;
  out->rotation = in.rotation;
  out->track_id =
      in.track_id.has_value() ? strdup(in.track_id->c_str()) : nullptr;
}

void CppConvertToOrientedDetectionResult(const CppOrientedDetectionResult& in,
                                         MpOrientedDetectionResult* out) {
  out->detections_count = in.detections.size();
  out->detections = new MpOrientedDetection[out->detections_count];
  for (size_t i = 0; i < out->detections_count; ++i) {
    CppConvertToOrientedDetection(in.detections[i], &out->detections[i]);
  }
}

void CppCloseOrientedDetection(MpOrientedDetection* in) {
  for (size_t i = 0; i < in->categories_count; ++i) {
    CppCloseCategory(&in->categories[i]);
  }
  delete[] in->categories;
  in->categories = nullptr;
  free(const_cast<char*>(in->track_id));
  in->track_id = nullptr;
}

void CppCloseOrientedDetectionResult(MpOrientedDetectionResult* in) {
  for (size_t i = 0; i < in->detections_count; ++i) {
    CppCloseOrientedDetection(&in->detections[i]);
  }
  delete[] in->detections;
  in->detections = nullptr;
}

}  // namespace mediapipe::tasks::c::components::containers
