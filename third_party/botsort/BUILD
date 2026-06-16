# Vendored, motion-only BoTSORT-cpp (https://github.com/viplix3/BoTSORT-cpp).
# MIT-licensed. ReID/TensorRT/ONNXRuntime/INIReader paths are removed; this
# builds CPU-only against the already-vendored OpenCV+Eigen.

licenses(["notice"])  # MIT

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

cc_library(
    name = "botsort",
    srcs = glob(["src/*.cpp"]),
    hdrs = glob(["include/*.h"]),
    copts = ["-w"],  # silence third-party warnings
    includes = ["include"],
    deps = [
        "//third_party:opencv",
        "@eigen//:eigen3",
    ],
)

cc_test(
    name = "botsort_smoke_test",
    srcs = ["botsort_smoke_test.cc"],
    deps = [
        ":botsort",
        "//third_party:opencv",
        "@com_google_googletest//:gtest_main",
    ],
)
