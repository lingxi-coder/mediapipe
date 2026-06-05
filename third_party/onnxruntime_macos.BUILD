# Description:
#   ONNX Runtime (CPU) for the OnnxInferenceCalculator on macOS, consumed from a
#   Homebrew install. Mirrors third_party/opencv_macos.BUILD.
#
# Setup: `brew install onnxruntime`. The version-independent Homebrew symlink
#   /opt/homebrew/opt/onnxruntime -> Cellar/onnxruntime/<version>
# means this needs no edit after `brew upgrade onnxruntime`.

load("@bazel_skylib//lib:paths.bzl", "paths")

licenses(["notice"])  # MIT

# Apple-Silicon Homebrew symlink prefix. Intel Homebrew would be "opt/onnxruntime"
# under "/usr/local" (set in the WORKSPACE new_local_repository path).
PREFIX = "opt/onnxruntime"

cc_library(
    name = "onnxruntime",
    srcs = glob([paths.join(PREFIX, "lib/libonnxruntime.dylib")]),
    hdrs = glob([paths.join(PREFIX, "include/onnxruntime/*.h")]),
    includes = [paths.join(PREFIX, "include/onnxruntime")],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
