# Description:
#   OpenCV libraries for video/image processing on MacOS

load("@bazel_skylib//lib:paths.bzl", "paths")

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

# Example configurations:
#
# # OpenCV 3
# To configure OpenCV 3, obtain the path of OpenCV 3 from Homebrew. The
# following commands show the output of the command with version 3.4.16_10:
#
# $ brew ls opencv@3 | grep version.hpp
# $ /opt/homebrew/Cellar/opencv@3/3.4.16_10/include/opencv2/core/version.hpp
#
# Then set path in "macos_opencv" rule in the WORKSPACE file to
# "/opt/homebrew/Cellar" and the PREFIX below to "opencv/<version>" (e.g.
# "opencv/3.4.16_10" for the example above).
#
# # OpenCV 4
# To configure OpenCV 4, obtain the path of OpenCV 4 from Homebrew. The
# following commands show the output of the command with version 4.10.0_12:
#
# $ brew ls opencv | grep version.hpp
# $ /opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4/opencv2/core/version.hpp
# $ /opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4/opencv2/dnn/version.hpp
#
# Then set path in "macos_opencv" rule in the WORKSPACE file to
# "/opt/homebrew/Cellar" and the PREFIX below to "opencv/<version>" (e.g.
# "opencv/4.10.0_12" for the example above). For OpenCV 4, you will also need to
# adjust the include paths. The header search path should be
# "include/opencv4/opencv2/**/*.h*" and the include prefix needs to be set to
# "include/opencv4".

# Version-independent Homebrew symlink (opt/opencv -> Cellar/opencv/<version>),
# so this does not need editing after `brew upgrade opencv`. For OpenCV 4 the
# headers live under include/opencv4/ (handled below).
PREFIX = "opt/opencv"

cc_library(
    name = "opencv",
    srcs = glob(
        [
            paths.join(PREFIX, "lib/libopencv_core.dylib"),
            paths.join(PREFIX, "lib/libopencv_imgproc.dylib"),
            paths.join(PREFIX, "lib/libopencv_imgcodecs.dylib"),
            # util/tracking (MotionBox PnP homography) needs solvePnP/
            # projectPoints/undistortPoints/Rodrigues from calib3d.
            paths.join(PREFIX, "lib/libopencv_calib3d.dylib"),
            # util/tracking optical-flow (RegionFlowComputation) needs
            # calcOpticalFlowPyrLK / buildOpticalFlowPyramid from video and
            # FastFeatureDetector from features2d.
            # Use the static archives (.a) instead of .dylib here: the dylib
            # variants transitively load libopencv_dnn, which loads the
            # Homebrew libprotobuf.34 dylib and causes a protobuf ODR crash
            # against the statically-linked protobuf that Bazel embeds in test
            # binaries. The .a archives contain only cv:: symbols and standard-
            # library references — no protobuf chain.
            paths.join(PREFIX, "lib/libopencv_video.a"),
            paths.join(PREFIX, "lib/libopencv_features2d.a"),
            # kleidicv HAL + thread are required transitive deps of the static
            # video archive (provides NEON-optimised Scharr derivatives on arm64).
            paths.join(PREFIX, "lib/libkleidicv.a"),
            paths.join(PREFIX, "lib/libkleidicv_hal.a"),
            paths.join(PREFIX, "lib/libkleidicv_thread.a"),
        ],
    ),
    hdrs = glob([paths.join(PREFIX, "include/opencv4/opencv2/**/*.h*")]),
    includes = [paths.join(PREFIX, "include/opencv4/")],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
