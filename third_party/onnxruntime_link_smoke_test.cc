// Verifies the Homebrew onnxruntime dependency compiles and links.
#include <string>

#include "mediapipe/framework/port/gtest.h"
#include "onnxruntime_cxx_api.h"  // from @macos_onnxruntime//:onnxruntime

namespace mediapipe {
namespace {

TEST(OnnxRuntimeLinkSmokeTest, VersionStringIsNonEmpty) {
  const std::string version = Ort::GetVersionString();
  EXPECT_FALSE(version.empty()) << "ONNX Runtime version string should be set";
}

}  // namespace
}  // namespace mediapipe
