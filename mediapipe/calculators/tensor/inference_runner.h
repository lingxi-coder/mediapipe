#ifndef MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_RUNNER_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_RUNNER_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/formats/inference_metadata.pb.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe {

// Common interface to implement inference runners in MediaPipe.
class InferenceRunner {
 public:
  virtual ~InferenceRunner() = default;
  virtual absl::StatusOr<std::vector<Tensor>> Run(
      CalculatorContext* cc, const TensorSpan& tensor_span) = 0;

  // Returns the TfLite model's input/output tensor names. This enables tensor
  // name based I/O mapping in the InferenceCalculator base class.
  virtual const InputOutputTensorNames& GetInputOutputTensorNames() const = 0;

  // Returns static, model-level metadata read from the loaded instance.
  // Non-pure with a default so backends opt in by overriding. Callers needing
  // metadata must use a backend that overrides this (CPU/TFLite does).
  virtual absl::StatusOr<InferenceMetadata> GetModelMetadata() const {
    return absl::UnimplementedError(
        "GetModelMetadata is not implemented for this inference backend");
  }
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_RUNNER_H_
