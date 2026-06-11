# Tasks API 静态 Tiled 检测(子项目 A)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用户通过 `YoloObjectDetector` / `OrientedObjectDetector` 的 C++ Tasks options 配置静态 tiling(网格或显式 rect),检测器内部切 tile → 批推理 → 合并回帧坐标 → 全局 NMS。保持现有 Tasks public API 坐标单位不变:graph 内部 merge 输出帧归一化坐标,但 C++ result 仍是 pixel units。

**Architecture:** subgraph 中间层(共享 `TiledDetectionFrontGraph` + 两个 Merge 后半 subgraph),推理留给调用方;Tasks graph builder 在 tiling 启用时走 tiled 分支。tiled 分支不接 projection 矩阵,merge 直接输出源帧归一化坐标;随后 OBB 保持归一化 proto 输出(容器转换阶段乘 image size 得到 pixel result),YOLO/axis-aligned 必须继续接 `DetectionTransformationCalculator` 转 pixel bbox 再输出。Spec: `docs/superpowers/specs/2026-06-11-tasks-tiled-detection-design.md`(实现前先通读)。

**Tech Stack:** MediaPipe Bazel(C++20),api2 calculator/builder,proto2。所有测试命令均带 `--define MEDIAPIPE_DISABLE_GPU=1`。commit 信息结尾必须带 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。**每个任务严格 TDD:先写测试、跑一次确认按预期原因失败、再实现。**

**已存在可直接复用的组件(不要重写)**:`TileSpecToTilePlanCalculator`、`StreamingTilesToTensorBatchCalculator`、`YoloTensorsToDetectionsCalculator`/`YoloObbTensorsToOrientedDetectionsCalculator`(含 tile-local NMS 选项)、`MergeTileDetectionsAccumulatorCalculator`(OBB)、`MergeTileBoxDetectionsAccumulatorCalculator`、`RotatedNonMaxSuppressionCalculator`、`TiledFrameSuppressionCalculator`、上游 `FromImageCalculator`(mediapipe/calculators/util)、`ImagePropertiesCalculator`(mediapipe/calculators/image)与 `ClipDetectionVectorSizeCalculator`(target `//mediapipe/calculators/core:clip_vector_size_calculator`)。

**坐标/预处理硬约束:** merge graph 输出的是 frame-normalized proto;Tasks public C++ result 必须继续是 pixel units。当前 `StreamingTilesToTensorBatchCalculator` 只实现 uint8 RGB → float32 `[0,1]`(`/255`)预处理,Tasks tiled builder 构建期校验判据:**输入张量 float32 + 4D 硬性;TFLite Metadata 的 NormalizationOptions 存在且不等价 (mean 0, std 255) → InvalidArgument;缺失 → 按 /255 假设放行**(fork 导出脚本不写 normalization metadata,硬卡"必须存在"会拒掉自家 e2e fixtures)。不要静默绕过 `ImagePreprocessingGraph` 的 normalization 语义。

---

### Task 1: TileGridCalculator

**Files:**
- Create: `mediapipe/calculators/tensor/tile_grid_calculator.proto`
- Create: `mediapipe/calculators/tensor/tile_grid_calculator.cc`
- Create: `mediapipe/calculators/tensor/tile_grid_calculator_test.cc`
- Modify: `mediapipe/calculators/tensor/BUILD`(三个新 target:proto/cc_library/cc_test,模式照抄同文件里 `tile_spec_to_tile_plan_calculator` 的三件套)

- [ ] **Step 1: 写 proto**

```proto
// mediapipe/calculators/tensor/tile_grid_calculator.proto
syntax = "proto2";
package mediapipe;
import "mediapipe/framework/calculator.proto";

message TileGridCalculatorOptions {
  extend .mediapipe.CalculatorOptions {
    optional TileGridCalculatorOptions ext = 471230012;
  }
  // 均匀网格:rows x cols 个 tile,相邻 tile 重叠 overlap_fraction(占 tile
  // 边长比例,[0,1))。rows=cols=1(默认)输出单个全帧 tile。
  optional int32 rows = 1 [default = 1];
  optional int32 cols = 2 [default = 1];
  optional float overlap_fraction = 3 [default = 0.0];

  // 显式 tile 列表(帧归一化中心+宽高)。非空时与网格参数互斥
  // (rows*cols>1 同时设置 -> Open() 报错)。
  message TileRect {
    optional float x_center = 1;
    optional float y_center = 2;
    optional float width = 3;
    optional float height = 4;
  }
  repeated TileRect explicit_tiles = 4;
}
```

- [ ] **Step 2: 写失败测试**(关键用例;`Rect` 辅助函数照抄 `video_tile_scheduler_calculator_test.cc` 顶部的同名 helper)

```cpp
// mediapipe/calculators/tensor/tile_grid_calculator_test.cc
// includes: calculator_runner.h, rect.pb.h, gtest, parse_text_proto,
// status_matchers, tile_grid_calculator.pb.h
namespace mediapipe { namespace {

std::vector<NormalizedRect> Run(const std::string& options_body, int ticks) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      absl::StrCat(R"(calculator: "TileGridCalculator"
        input_stream: "TICK:tick" output_stream: "TILES:tiles"
        options { [mediapipe.TileGridCalculatorOptions.ext] { )",
                   options_body, "} }")));
  for (int t = 0; t < ticks; ++t)
    runner.MutableInputs()->Tag("TICK").packets.push_back(
        MakePacket<int>(0).At(Timestamp(t)));
  ABSL_CHECK_OK(runner.Run());
  return runner.Outputs().Tag("TILES").packets.back()
      .Get<std::vector<NormalizedRect>>();
}

TEST(TileGridTest, DefaultIsSingleFullFrameTile) {
  auto tiles = Run("", 1);
  ASSERT_EQ(tiles.size(), 1u);
  EXPECT_NEAR(tiles[0].x_center(), 0.5f, 1e-6);
  EXPECT_NEAR(tiles[0].width(), 1.0f, 1e-6);
  EXPECT_NEAR(tiles[0].height(), 1.0f, 1e-6);
}

TEST(TileGridTest, TwoColsWithOverlap) {
  // tile_w = 1/(cols-(cols-1)*o) = 1/(2-0.2) = 1/1.8; stride = tile_w*(1-o).
  auto tiles = Run("cols: 2 overlap_fraction: 0.2", 1);
  ASSERT_EQ(tiles.size(), 2u);
  const float w = 1.0f / 1.8f;
  EXPECT_NEAR(tiles[0].width(), w, 1e-5);
  EXPECT_NEAR(tiles[0].x_center(), w / 2, 1e-5);
  EXPECT_NEAR(tiles[1].x_center(), w * 0.8f + w / 2, 1e-5);
  EXPECT_NEAR(tiles[1].x_center() + tiles[1].width() / 2, 1.0f, 1e-5);  // 贴右边
  EXPECT_NEAR(tiles[0].height(), 1.0f, 1e-5);  // rows=1 -> 全高
}

TEST(TileGridTest, RowsColsGrid) {
  auto tiles = Run("rows: 2 cols: 3", 1);
  ASSERT_EQ(tiles.size(), 6u);              // 行优先:r*cols+c
  EXPECT_NEAR(tiles[0].width(), 1.0f / 3, 1e-5);
  EXPECT_NEAR(tiles[0].height(), 0.5f, 1e-5);
  EXPECT_NEAR(tiles[5].x_center(), 1.0f - 1.0f / 6, 1e-5);
  EXPECT_NEAR(tiles[5].y_center(), 0.75f, 1e-5);
}

TEST(TileGridTest, ExplicitTilesPassThroughEveryTick) {
  auto tiles = Run(
      "explicit_tiles { x_center: 0.3 y_center: 0.4 width: 0.2 height: 0.6 }",
      2);  // 两个 tick,取最后一帧
  ASSERT_EQ(tiles.size(), 1u);
  EXPECT_NEAR(tiles[0].x_center(), 0.3f, 1e-6);
  EXPECT_NEAR(tiles[0].height(), 0.6f, 1e-6);
}

TEST(TileGridTest, GridAndExplicitMutuallyExclusive) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "TileGridCalculator"
    input_stream: "TICK:tick" output_stream: "TILES:tiles"
    options { [mediapipe.TileGridCalculatorOptions.ext] {
      cols: 2 explicit_tiles { x_center:.5 y_center:.5 width:1 height:1 } } }
  )pb"));
  runner.MutableInputs()->Tag("TICK").packets.push_back(
      MakePacket<int>(0).At(Timestamp(0)));
  EXPECT_FALSE(runner.Run().ok());
}

TEST(TileGridTest, InvalidParamsFailOpen) {
  for (const char* bad : {"cols: 0", "overlap_fraction: 1.0",
                          "explicit_tiles { width: 0 height: 1 }"}) { /* 同上模式,逐个 EXPECT_FALSE(ok) */ }
}
}}  // namespaces
```

(`InvalidParamsFailOpen` 写成完整循环体:每个 bad 串构造 runner、喂一个 tick、`EXPECT_FALSE(runner.Run().ok())`。)

- [ ] **Step 3: 跑测试确认失败**:`bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:tile_grid_calculator_test` → 预期 FAILED TO BUILD(calculator 不存在)。

- [ ] **Step 4: 实现**

```cpp
// mediapipe/calculators/tensor/tile_grid_calculator.cc(api2 Node)
// includes: tile_grid_calculator.pb.h, api2/node.h, calculator_framework.h,
// formats/rect.pb.h, ret_check.h, absl/status
namespace mediapipe { namespace api2 {

// 每个 TICK 输出一份静态 tile 列表(Open 时算一次)。网格公式:
// tile_w = 1/(cols-(cols-1)*overlap);stride = tile_w*(1-overlap)。
class TileGridCalculator : public Node {
 public:
  static constexpr Input<AnyType> kTick{"TICK"};
  static constexpr Output<std::vector<NormalizedRect>> kTiles{"TILES"};
  MEDIAPIPE_NODE_CONTRACT(kTick, kTiles);

  absl::Status Open(CalculatorContext* cc) override {
    const auto& o = cc->Options<mediapipe::TileGridCalculatorOptions>();
    RET_CHECK_GE(o.rows(), 1); RET_CHECK_GE(o.cols(), 1);
    RET_CHECK_GE(o.overlap_fraction(), 0.0f);
    RET_CHECK_LT(o.overlap_fraction(), 1.0f);
    if (!o.explicit_tiles().empty()) {
      RET_CHECK(o.rows() == 1 && o.cols() == 1)
          << "explicit_tiles is mutually exclusive with a rows/cols grid";
      for (const auto& t : o.explicit_tiles()) {
        RET_CHECK(t.width() > 0 && t.height() > 0);
        NormalizedRect r;
        r.set_x_center(t.x_center()); r.set_y_center(t.y_center());
        r.set_width(t.width()); r.set_height(t.height());
        tiles_.push_back(std::move(r));
      }
      return absl::OkStatus();
    }
    const float o_f = o.overlap_fraction();
    const float tw = 1.0f / (o.cols() - (o.cols() - 1) * o_f);
    const float th = 1.0f / (o.rows() - (o.rows() - 1) * o_f);
    const float sx = tw * (1.0f - o_f), sy = th * (1.0f - o_f);
    for (int r = 0; r < o.rows(); ++r)
      for (int c = 0; c < o.cols(); ++c) {
        NormalizedRect t;
        t.set_x_center(c * sx + tw / 2); t.set_y_center(r * sy + th / 2);
        t.set_width(tw); t.set_height(th);
        tiles_.push_back(std::move(t));
      }
    return absl::OkStatus();
  }
  absl::Status Process(CalculatorContext* cc) override {
    kTiles(cc).Send(tiles_);
    return absl::OkStatus();
  }
 private:
  std::vector<NormalizedRect> tiles_;
};
MEDIAPIPE_REGISTER_NODE(TileGridCalculator);
}}  // namespaces
```

- [ ] **Step 5: 跑测试确认通过**(同 Step 3 命令 → PASSED)。
- [ ] **Step 6: Commit** `feat(tiling): TileGridCalculator — static grid/explicit tiles per tick`。

---

### Task 2: batcher options 携带模型元数据(METADATA side packet 可选化)

**Files:**
- Modify: `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.proto`(字段 12-16)
- Modify: `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator.cc`(kSideMeta 改 Optional + Open() 装配)
- Test: `mediapipe/calculators/tensor/streaming_tiles_to_tensor_batch_calculator_test.cc`

- [ ] **Step 1: 失败测试**(加在文件末 namespace 关闭前;`Meta`/`WhiteFrame`/`TwoTiles` helper 已在该文件)

```cpp
// 元数据来自 options(METADATA side packet 不连接)时行为与 side packet 等价。
TEST(StreamingTilesTest, OptionsBorneMetadataWorksWithoutSidePacket) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image"
    input_stream: "TILE_PLAN:plan"
    output_stream: "TENSORS:tensors"
    output_stream: "BATCH_INFO:info"
    options { [mediapipe.StreamingTilesToTensorBatchCalculatorOptions.ext] {
      metadata_batch_capacity: 4 metadata_input_height: 8
      metadata_input_width: 8 metadata_input_channels: 3 } }
  )pb"));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      Adopt(WhiteFrame(16, 16).release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(TwoTiles()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& tensors =
      runner.Outputs().Tag("TENSORS").packets[0].Get<std::vector<Tensor>>();
  EXPECT_EQ(tensors[0].shape().dims[0], 4);  // 固定批=cap, padding
  EXPECT_EQ(tensors[0].shape().dims[1], 8);
  const auto& info =
      runner.Outputs().Tag("BATCH_INFO").packets[0].Get<TensorBatchInfo>();
  EXPECT_EQ(info.valid_count, 2);
}

// 两者都缺 -> Open 失败(沿用既有 RET_CHECK)。
TEST(StreamingTilesTest, MissingMetadataEverywhereFailsOpen) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "StreamingTilesToTensorBatchCalculator"
    input_stream: "IMAGE:image" input_stream: "TILE_PLAN:plan"
    output_stream: "TENSORS:tensors" output_stream: "BATCH_INFO:info"
  )pb"));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      Adopt(WhiteFrame(16, 16).release()).At(Timestamp(0)));
  runner.MutableInputs()->Tag("TILE_PLAN").packets.push_back(
      MakePacket<TilePlan>(TwoTiles()).At(Timestamp(0)));
  EXPECT_FALSE(runner.Run().ok());
}
```

- [ ] **Step 2: 跑确认失败**(第一条 parse 崩在未知字段;第二条因 kSideMeta 必选直接 graph init 失败——两条都 FAIL 即可)。
- [ ] **Step 3: 实现**——proto 加(注释照 spec 3.2):

```proto
  optional int32 metadata_batch_capacity = 12 [default = 0];
  optional int32 metadata_input_height = 13 [default = 0];
  optional int32 metadata_input_width = 14 [default = 0];
  optional int32 metadata_input_channels = 15 [default = 0];
  optional bool metadata_is_dynamic_batch = 16 [default = false];
```

cc:`static constexpr SideInput<InferenceMetadata>::Optional kSideMeta{"METADATA"};`,Open() 开头替换 `meta_ = kSideMeta(cc).Get();` 为:

```cpp
    if (kSideMeta(cc).IsConnected()) {
      meta_ = kSideMeta(cc).Get();
      if (options_.metadata_batch_capacity() > 0) {
        ABSL_LOG(WARNING) << "Both METADATA side packet and options-borne "
                             "metadata set; side packet wins.";
      }
    } else {
      meta_.set_batch_capacity(options_.metadata_batch_capacity());
      meta_.set_input_height(options_.metadata_input_height());
      meta_.set_input_width(options_.metadata_input_width());
      meta_.set_input_channels(options_.metadata_input_channels());
      meta_.set_is_dynamic_batch(options_.metadata_is_dynamic_batch());
    }
```

注意 `options_ = cc->Options<...>()` 必须先于此块(把现有赋值行上移即可);`#include "absl/log/absl_log.h"`。既有 RET_CHECK_GT(meta_.…) 不动(它们就是"都缺则失败"的实现)。

- [ ] **Step 4: 全量回归**:`bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_test //mediapipe/calculators/tensor:video_tile_scheduler_pipeline_test //mediapipe/calculators/tensor:tiled_obb_pipeline_test` → PASS。
- [ ] **Step 5: Commit** `feat(tiling): options-borne model metadata for StreamingTilesToTensorBatch`。

---

### Task 3: tiled_detection subgraph 包骨架 + options proto

**Files:**
- Create: `mediapipe/graphs/tiled_detection/BUILD`
- Create: `mediapipe/graphs/tiled_detection/tiled_detection_graphs.proto`

- [ ] **Step 1: proto**(无测试,纯声明;编译即验证)

```proto
syntax = "proto2";
package mediapipe;
import "mediapipe/framework/calculator.proto";
import "mediapipe/calculators/tensor/tile_grid_calculator.proto";

message TiledDetectionFrontGraphOptions {
  extend .mediapipe.CalculatorOptions {
    optional TiledDetectionFrontGraphOptions ext = 471230013;
  }
  // 整体透传给内部 TileGridCalculator。
  optional TileGridCalculatorOptions tile_grid = 1;
  // 模型输入元数据,透传给 StreamingTilesToTensorBatch 的 options 元数据字段。
  optional int32 batch_capacity = 2 [default = 0];
  optional int32 input_height = 3 [default = 0];
  optional int32 input_width = 4 [default = 0];
  optional int32 input_channels = 5 [default = 0];
  optional bool is_dynamic_batch = 6 [default = false];
}

message TiledObbMergeGraphOptions {
  extend .mediapipe.CalculatorOptions {
    optional TiledObbMergeGraphOptions ext = 471230014;
  }
  optional float iou_threshold = 1 [default = 0.45];
  optional int32 max_detections = 2 [default = -1];
  optional bool class_agnostic = 3 [default = false];
}

message TiledBoxMergeGraphOptions {
  extend .mediapipe.CalculatorOptions {
    optional TiledBoxMergeGraphOptions ext = 471230015;
  }
  optional float iou_threshold = 1 [default = 0.45];
  optional bool class_agnostic = 2 [default = false];
  optional int32 max_detections = 3 [default = -1];
}
```

- [ ] **Step 2: BUILD**:`mediapipe_proto_library(name="tiled_detection_graphs_proto", deps=[..calculator_proto, "//mediapipe/calculators/tensor:tile_grid_calculator_proto"])`(头部 `load` 与 `package(default_visibility=["//visibility:public"])` 照抄 `mediapipe/calculators/tensor/BUILD` 顶部模式)。`bazel build //mediapipe/graphs/tiled_detection:tiled_detection_graphs_cc_proto` → 通过。
- [ ] **Step 3: Commit** `feat(tiling): tiled_detection subgraph package + options protos`。

---

### Task 4: TiledDetectionFrontGraph(+ graph 级测试)

**Files:**
- Create: `mediapipe/graphs/tiled_detection/tiled_detection_front_graph.cc`
- Create: `mediapipe/graphs/tiled_detection/tiled_detection_graphs_test.cc`(本任务先放 Front 用例,Task 5/6 追加)
- Modify: `mediapipe/graphs/tiled_detection/BUILD`

- [ ] **Step 1: 失败测试**(CalculatorGraph 级;helper 同 video_tile_scheduler_pipeline_test 的 WhiteFrame)

```cpp
TEST(TiledDetectionFrontGraphTest, EmitsBatchesWithInfo) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "image"
    output_stream: "tensors"
    output_stream: "info"
    node {
      calculator: "mediapipe.tiled_detection.TiledDetectionFrontGraph"
      input_stream: "IMAGE:image"
      output_stream: "TENSORS:tensors"
      output_stream: "BATCH_INFO:info"
      options { [mediapipe.TiledDetectionFrontGraphOptions.ext] {
        tile_grid { cols: 2 }
        batch_capacity: 2 input_height: 8 input_width: 8 input_channels: 3 } }
    }
  )pb");
  std::vector<Packet> tensors, info;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("tensors", /*push 到 tensors*/));
  MP_ASSERT_OK(graph.ObserveOutputStream("info", /*push 到 info*/));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image", Adopt(WhiteFrame(64, 48).release()).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  ASSERT_EQ(tensors.size(), 1u);  // 2 tiles / cap 2 -> 1 批
  const auto& t = tensors[0].Get<std::vector<Tensor>>()[0];
  EXPECT_EQ(t.shape().dims[0], 2);
  EXPECT_EQ(t.shape().dims[1], 8);
  EXPECT_EQ(t.shape().dims[3], 3);
  EXPECT_EQ(info[0].Get<TensorBatchInfo>().valid_count, 2);
}
```

(ObserveOutputStream lambda 写完整:`[&](const Packet& p){ tensors.push_back(p); return absl::OkStatus(); }`。)

- [ ] **Step 2: 跑确认失败**(`bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/graphs/tiled_detection:tiled_detection_graphs_test` → 找不到 subgraph 注册名)。
- [ ] **Step 3: 实现**

```cpp
// tiled_detection_front_graph.cc
// includes: subgraph.h, api2/builder.h, calculator.pb.h,
// tiled_detection_graphs.pb.h, tile_grid_calculator.pb.h,
// streaming_tiles_to_tensor_batch_calculator.pb.h, image_frame.h
namespace mediapipe {
namespace tiled_detection {

// IMAGE(ImageFrame) -> TileGrid -> TileSpecToTilePlan ->
// StreamingTilesToTensorBatch -> TENSORS(逐批) + BATCH_INFO。
class TiledDetectionFrontGraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    const auto& options = sc->Options<TiledDetectionFrontGraphOptions>();
    api2::builder::Graph graph;
    auto image = graph.In("IMAGE").Cast<ImageFrame>();

    auto& grid = graph.AddNode("TileGridCalculator");
    grid.GetOptions<TileGridCalculatorOptions>() = options.tile_grid();
    image >> grid.In("TICK");

    auto& plan = graph.AddNode("TileSpecToTilePlanCalculator");
    grid.Out("TILES") >> plan.In("TILES");

    auto& batcher = graph.AddNode("StreamingTilesToTensorBatchCalculator");
    auto& bo =
        batcher.GetOptions<StreamingTilesToTensorBatchCalculatorOptions>();
    bo.set_metadata_batch_capacity(options.batch_capacity());
    bo.set_metadata_input_height(options.input_height());
    bo.set_metadata_input_width(options.input_width());
    bo.set_metadata_input_channels(options.input_channels());
    bo.set_metadata_is_dynamic_batch(options.is_dynamic_batch());
    image >> batcher.In("IMAGE");
    plan.Out("TILE_PLAN") >> batcher.In("TILE_PLAN");

    batcher.Out("TENSORS") >> graph.Out("TENSORS");
    batcher.Out("BATCH_INFO") >> graph.Out("BATCH_INFO");
    return graph.GetConfig();
  }
};
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tiled_detection::TiledDetectionFrontGraph);

}  // namespace tiled_detection
}  // namespace mediapipe
```

BUILD:`cc_library(name="tiled_detection_front_graph", alwayslink=1, deps=[graphs proto cc, tile_grid_calculator, tile_spec_to_tile_plan_calculator, streaming_tiles_to_tensor_batch_calculator + 其 cc_proto, framework:subgraph, api2:builder, formats:image_frame])`;test target deps 加 front graph + tiling_types + tensor + image_frame + gtest_main 等。

- [ ] **Step 4: 跑确认通过。**
- [ ] **Step 5: Commit** `feat(tiling): TiledDetectionFrontGraph subgraph`。

---

### Task 5: TiledObbMergeGraph

**Files:**
- Create: `mediapipe/graphs/tiled_detection/tiled_obb_merge_graph.cc`
- Modify: `mediapipe/graphs/tiled_detection/tiled_detection_graphs_test.cc`、BUILD

(**设计决定:不做 NUM_TILES/bypass_single_tile**——tile-local NMS 在 TilingOptions 默认关闭,单显式 tile 配置下 bypass 会把未去重的原始检测返回给用户;而对已 tile-local NMS 的集合再跑同阈值贪心 NMS 是幂等 no-op,省它不值得改两个 calculator 契约。全局 NMS 恒执行。)

- [ ] **Step 1: 失败测试**(合成两批、各 1 tile;几何构造照抄 `merge_tile_detections_accumulator_calculator_test.cc` 的 `MakeGeom`/`Obb` helper——graph 级测试用 `AddPacketToInputStream` 喂 `std::vector<std::vector<OrientedDetection>>` 与 `TensorBatchInfo`)

```cpp
TEST(TiledObbMergeGraphTest, MergesBatchesAndRunsGlobalNms) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "dets" input_stream: "info"
    output_stream: "merged"
    node {
      calculator: "mediapipe.tiled_detection.TiledObbMergeGraph"
      input_stream: "ORIENTED_DETECTIONS:dets"
      input_stream: "BATCH_INFO:info"
      output_stream: "ORIENTED_DETECTIONS:merged"
      options { [mediapipe.TiledObbMergeGraphOptions.ext] {
        iou_threshold: 0.5 } }
    }
  )pb");
  // 两批(total_batches=2):批0 一个 ship@(0.25,0.5),批1 在 overlap 区域
  // 重复同一目标(投影后 IoU>0.5)-> 全局 NMS 后只剩 1 个。
  // 断言:1 个输出包、Timestamp==src_ts、size()==1。
}
```

(测试体写全:左右两半 tile 的 geom、两个 info(batch_index 0/1)、dets packet;重复目标用全帧中心同一框即可——两 tile overlap 区内。)

追加单 tile 用例:`SingleTileStillRunsGlobalNms`:一个 source frame `valid_count=1,total_batches=1`,该 tile row 内放两个会被 rotated NMS 互相 suppress 的 OBB(模拟 tile-local NMS 关闭的默认配置)→ 输出 **1 个**。钉死"全局 NMS 恒执行,单 tile 不 bypass"的设计决定。

- [ ] **Step 2: 跑确认失败。**
- [ ] **Step 3: 实现**:`TiledObbMergeGraph` GetConfig:`MergeTileDetectionsAccumulatorCalculator` → `RotatedNonMaxSuppressionCalculator`,NMS options 从 subgraph options 三字段(iou_threshold/max_detections/class_agnostic)透传;In/Out tag 同测试 pbtxt。不修改任何既有 calculator。
- [ ] **Step 4: 跑确认通过;Commit** `feat(tiling): TiledObbMergeGraph subgraph`。

---

### Task 6: TiledBoxMergeGraph

**Files:**
- Create: `mediapipe/graphs/tiled_detection/tiled_box_merge_graph.cc`
- Modify: 同上 test/BUILD

- [ ] **Step 1: 失败测试**:同 Task 5 形态,Detection 版(helper 照抄 `merge_tile_box_detections_accumulator_calculator_test.cc` 的 `Box`/`MakeGeom`);额外用例 `MaxDetectionsCaps`:3 个互不重叠 det + `max_detections: 2` → 输出 2 个且为最高分两个。追加 `SingleTileStillRunsGlobalNms`:一个 tile row 内两个会被 NMS suppress 的框 → 输出 **1 个**(全局 NMS 恒执行,理由同 Task 5)。
- [ ] **Step 2: 跑确认失败。**
- [ ] **Step 3: 实现**:
  - `TiledBoxMergeGraph`: `MergeTileBoxDetectionsAccumulatorCalculator` → `TiledFrameSuppressionCalculator`(`NUM_TILES`/`TRACKER_DETECTIONS` 均不连接,`bypass_single_tile` 保持默认 false → 恒执行全局 NMS);iou_threshold/class_agnostic 透传。不修改任何既有 calculator。
  - `options.max_detections()>=0` 时追加 `ClipDetectionVectorSizeCalculator` 节点(`GetOptions<mediapipe::ClipVectorSizeCalculatorOptions>().set_max_vec_size(...)`,target `//mediapipe/calculators/core:clip_vector_size_calculator`)。
- [ ] **Step 4: 跑确认通过;Commit** `feat(tiling): TiledBoxMergeGraph subgraph`。

---

### Task 7: OrientedObjectDetector Tasks options + 转换器

**Files:**
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/proto/oriented_object_detector_options.proto`(加 `TilingOptions tiling = 11;`,message 定义照 spec 3.4)
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector.h`(struct 加字段)与 `.cc`(`ConvertOrientedObjectDetectorOptionsToProto` 同步映射,函数在 .h:118 声明)
- Test: `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_test.cc`

- [ ] **Step 1: 失败测试**(转换器单测,加在 e2e 文件里):

```cpp
TEST(OrientedObjectDetectorOptionsTest, TilingOptionsConvertToProto) {
  auto options = std::make_unique<OrientedObjectDetectorOptions>();
  options->tiling.tile_rows = 2;
  options->tiling.tile_cols = 3;
  options->tiling.tile_overlap_fraction = 0.2f;
  options->tiling.tile_local_nms_iou_threshold = 0.5f;
  auto proto = ConvertOrientedObjectDetectorOptionsToProto(options.get());
  EXPECT_EQ(proto->tiling().tile_rows(), 2);
  EXPECT_EQ(proto->tiling().tile_cols(), 3);
  EXPECT_NEAR(proto->tiling().tile_overlap_fraction(), 0.2f, 1e-6);
  EXPECT_NEAR(proto->tiling().tile_local_nms_iou_threshold(), 0.5f, 1e-6);
}
```

- [ ] **Step 2: 跑确认失败**(字段不存在编译失败)。
- [ ] **Step 3: 实现**:h 加

```cpp
  struct TilingOptions {
    int tile_rows = 1;
    int tile_cols = 1;
    float tile_overlap_fraction = 0.0f;
    // 每项 {x_center, y_center, width, height},帧归一化;与网格互斥。
    std::vector<std::array<float, 4>> explicit_tiles;
    float tile_local_nms_iou_threshold = 0.0f;
    int max_detections_after_tile_nms = 0;
  };
  TilingOptions tiling;
```

converter 加逐字段映射(explicit_tiles 循环 `auto* t = tiling->add_explicit_tiles(); t->set_x_center(e[0]); ...`)。proto `TilingOptions` 按 spec 3.4 原文。

- [ ] **Step 4: 跑确认通过;Commit** `feat(tiling): OrientedObjectDetector tiling options + converter`。

---

### Task 8: OBB graph builder tiled 分支 + 真模型 e2e

**Files:**
- Modify: `mediapipe/tasks/cc/vision/oriented_object_detector/oriented_object_detector_graph.cc`
- Test: `oriented_object_detector_test.cc`(真模型 e2e)
- Modify: 两处 BUILD(graph deps 加三个 subgraph + FromImageCalculator)

- [ ] **Step 1: 失败 e2e**(镜像同文件既有 boats.jpg 用例的 harness——加载方式/模型路径/Detect 调用照抄相邻测试,fixtures 缺失 GTEST_SKIP 同既有写法):

```cpp
TEST_F(/*同文件既有 fixture*/, TiledGridDetectsShipsOnBoats) {
  auto options = /* 照抄相邻用例的 options 构造(模型/num_classes/阈值) */;
  options->tiling.tile_cols = 2;
  options->tiling.tile_overlap_fraction = 0.2f;
  // Create + Detect(boats.jpg) 同相邻用例。
  // 硬断言(spec §5.3):
  ASSERT_GE(result.detections.size(), 1u);
  for (const auto& d : result.detections) {
    EXPECT_GE(d.cx - d.width / 2, -1e-3);
    EXPECT_LE(d.cx + d.width / 2, image.width() + 1e-3);
    EXPECT_GE(d.cy - d.height / 2, -1e-3);
    EXPECT_LE(d.cy + d.height / 2, image.height() + 1e-3);
  }
  // 与单图路径对照:N_single 来自同文件相邻用例同配置再跑一次(无 tiling),
  // EXPECT_GE(tiled_n, N_single - 1); EXPECT_LE(tiled_n, N_single + 3);
  // 类别断言:至少一个 detection 的 category 名含 "ship"(同既有用例写法)。
}
```

(检测结果字段名以 `OrientedObjectDetectionResult` 实际定义为准——实现者先读 `tasks/cc/components/containers/oriented_object_detection_result.h`。注意 public result 是 pixel units,不是 `[0,1]`:断言 `cx/width/height` 在 image pixel bounds 内,不要用归一化范围断言。)

追加 negative 用例:tiling 启用但模型输入 dtype/normalization 不受支持时 `Create()` 返回 InvalidArgument;如果当前 test harness 能传 ROI/NORM_RECT,tiling+ROI 也必须返回 InvalidArgument。

- [ ] **Step 2: 跑确认失败**(tiling 字段未被 builder 消费 → 行为同单图;对照断言不致失败,**所以失败信号靠 Step 3 前的中间断言**:在 builder 实现前,tiled 用例应当与单图结果完全一致——为获得真 RED,先断言 `tiled_n != N_single || HasTiledNodes(graph)` 不可行;改用更可靠的 RED:**builder 在 tiling 启用但未实现时应显式报错**。即 Step 3 实现前,先在 builder 加 `if (tiling enabled) return InvalidArgument("tiling not implemented")`?——不,直接以"未知 proto 字段"为 RED:Step 1 的转换器已在 Task 7 落地,字段存在;因此本任务 RED = e2e 在 builder 未消费 tiling 时输出与单图全等,而用例断言 `result 来自 tiled 分支独有的性质`。最实际的 RED 信号:**先只写用例骨架 + `FAIL() << "tiled branch not wired"` 哨兵**,Step 3 完成后删哨兵补全断言再跑 GREEN。)
- [ ] **Step 3: 实现 builder 分支 + wrapper 三处改动**:
  - **wrapper 层(`oriented_object_detector.cc`,三处;它目前无条件声明 NORM_RECT graph input 且每次 Detect 发包,见 :77-90/:195)**:
    (a) `CreateGraphConfig`(或同职责函数)接收 `tiling_enabled`:tiled 时不声明、不连接 NORM_RECT graph input(否则 wrapper 对 task subgraph 的 `In(kNormRectTag)` 连接会在图展开期失败);
    (b) `Detect`/`DetectForVideo`/`DetectAsync` 的输入 packet map:tiled 时只发 image 包,不发 norm_rect 包;
    (c) 三个 Detect 变体开头:`image_processing_options` 含 region_of_interest 且 tiling 启用 → `InvalidArgument("tiling and ROI are mutually exclusive")`。
  - **task subgraph(graph builder)**:在 `GetConfig()` 里先读取 `task_options` 并计算 `tiling_enabled`,再决定是否引用 `NORM_RECT` 输入(api2 builder 不引用即不声明)。非 tiled 路径保持现状;不要使用不存在的 `Source::IsConnected()` 伪 API。
  - 在 `BuildOrientedObjectDetectorTask`(含 AddInference 的函数)开头计算:

```cpp
    const auto& tiling = task_options.tiling();
    const bool tiling_enabled = tiling.tile_rows() * tiling.tile_cols() > 1 ||
                                !tiling.explicit_tiles().empty();
```

非 tiled 路径原封不动包进 `if (!tiling_enabled) { ... 既有代码 ... } else { tiled 分支 }`。tiled 分支:

```cpp
    // 模型输入维度 [N,H,W,C]。本轮 tiled front 只支持 float32 BHWC。
    // normalization 判据(见 spec §2 预处理事实):float32 + 4D 硬性;
    // BuildInputImageTensorSpecs(model_resources) 的 normalization_options
    // 存在且不等价 (mean 0, std 255) -> InvalidArgument;缺失 -> 按 /255
    // 假设放行(fork 导出脚本不写 normalization metadata;与单图路径对
    // 此类模型的隐含假设一致)。
    const tflite::Model* model = model_resources.GetTfLiteModel();
    const tflite::SubGraph* sg = model->subgraphs()->Get(0);
    const auto* input_tensor = sg->tensors()->Get(sg->inputs()->Get(0));
    const auto* dims = input_tensor->shape();
    RET_CHECK(dims != nullptr && dims->size() == 4)
        << "tiled mode expects a [N,H,W,C] image input tensor";
    RET_CHECK_EQ(input_tensor->type(), tflite::TensorType_FLOAT32)
        << "tiled mode currently supports float32 image input only";

    auto& to_frame = graph.AddNode("FromImageCalculator");
    image_in >> to_frame.In("IMAGE");        // FromImageCalculator: IMAGE(Image)
    // 输出 tag 为 "IMAGE_CPU"(见 mediapipe/calculators/util/from_image_calculator.cc)

    auto& front = graph.AddNode(
        "mediapipe.tiled_detection.TiledDetectionFrontGraph");
    auto& fo = front.GetOptions<TiledDetectionFrontGraphOptions>();
    auto* tg = fo.mutable_tile_grid();
    tg->set_rows(tiling.tile_rows()); tg->set_cols(tiling.tile_cols());
    tg->set_overlap_fraction(tiling.tile_overlap_fraction());
    for (const auto& e : tiling.explicit_tiles()) {
      auto* t = tg->add_explicit_tiles();
      t->set_x_center(e.x_center()); t->set_y_center(e.y_center());
      t->set_width(e.width()); t->set_height(e.height());
    }
    fo.set_batch_capacity(dims->Get(0)); fo.set_input_height(dims->Get(1));
    fo.set_input_width(dims->Get(2));    fo.set_input_channels(dims->Get(3));
    fo.set_is_dynamic_batch(input_tensor->shape_signature() != nullptr &&
                            input_tensor->shape_signature()->size() > 0 &&
                            input_tensor->shape_signature()->Get(0) == -1);
    to_frame.Out("IMAGE_CPU") >> front.In("IMAGE");

    auto& inference = AddInference(
        model_resources, task_options.base_options().acceleration(), graph);
    front.Out("TENSORS") >> inference.In(kTensorTag);

    // decoder 节点与非 tiled 分支同构(num_classes/layout/conf/allow-deny 同),
    // 额外写入 tile-local NMS 两参:
    //   opts.set_tile_local_nms_iou_threshold(tiling.tile_local_nms_iou_threshold());
    //   opts.set_max_detections_after_tile_nms(tiling.max_detections_after_tile_nms());
    inference.Out(kTensorTag) >> obb_decode.In(kTensorTag);

    auto& merge = graph.AddNode("mediapipe.tiled_detection.TiledObbMergeGraph");
    auto& mo = merge.GetOptions<TiledObbMergeGraphOptions>();
    mo.set_iou_threshold(task_options.iou_threshold());
    mo.set_class_agnostic(task_options.class_agnostic_nms());
    mo.set_max_detections(task_options.max_results());
    obb_decode.Out(kOrientedDetectionsTag) >> merge.In(kOrientedDetectionsTag);
    front.Out("BATCH_INFO") >> merge.In("BATCH_INFO");
    // merge 输出 frame-normalized OrientedDetection,直接接 label_id_to_text。
    // 跳过 flatten/NMS/projection 三个单图节点;public C++ result 的 pixel
    // conversion 仍由 oriented_object_detector.cc 的
    // ConvertToOrientedObjectDetectionResult(dets, image_size) 完成。
```

(实现者注意:既有单图路径里 `flatten`/`nms`/`projection` 三节点只属于非 tiled 分支;`label_id_to_text` 可共享,但不要把 YOLO 的 pixel transformation 规则套到 OBB proto 上。tiled 分支的 `output_streams.image` 可直接用原始 `image_in`。)BUILD deps:graph target 加 `//mediapipe/graphs/tiled_detection:tiled_detection_front_graph`、`:tiled_obb_merge_graph`、`//mediapipe/calculators/util:from_image_calculator` 及相应 cc_proto。

- [ ] **Step 4: 删哨兵,跑 e2e GREEN**:`bazel test --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test`。
- [ ] **Step 5: Commit** `feat(tiling): OrientedObjectDetector tiled branch + real-model e2e`。

---

### Task 9: YoloObjectDetector 同构集成(options + builder + e2e)

**Files:**
- Modify: `mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.proto`(`TilingOptions tiling = 10;`,message 文本与 Task 7 相同——按 spec 原文重抄,勿引用)
- Modify: `yolo_object_detector.h/.cc`(struct + `ConvertYoloObjectDetectorOptionsToProto`,函数在 .cc:97;**wrapper 三处改动与 Task 8 同型**:CreateGraphConfig 按 tiling_enabled 分支不声明 NORM_RECT、三个 Detect 变体 packet map 不发 norm_rect、tiling+ROI → InvalidArgument)
- Modify: `yolo_object_detector_graph.cc`(tiled 分支:`ImagePropertiesCalculator` 取原图 `SIZE`;`FromImageCalculator` → FrontGraph → AddInference → 既有 yolo decoder(写入 tile-local NMS 两参)→ `mediapipe.tiled_detection.TiledBoxMergeGraph`(iou/class_agnostic/max_results 透传)→ 既有 label 解析尾部 → `DetectionTransformationCalculator` 将 frame-normalized `relative_bounding_box` 转为 pixel `bounding_box` → `DetectionsDeduplicateCalculator`;同样跳过单图路径的 batch-flatten/NMS/projection 节点,但不能跳过 pixel transformation/dedup 尾部)
- Test: `yolo_object_detector_test.cc`(转换器单测 + 真模型 e2e,断言与 Task 8 同构但检查 public `DetectionResult.bounding_box` 为 pixel units,类别名按该模型 COCO 标签——"boat")

步骤同 Task 7+8(失败转换器测试 → 实现 → 失败 e2e 哨兵 → builder → GREEN → Commit)。结构代码与 Task 8 同形,Detection 版差异:merge subgraph 名 `TiledBoxMergeGraph`、流 tag `DETECTIONS`;merge 输出是 frame-normalized `RELATIVE_BOUNDING_BOX`,必须用原图 `SIZE` 接 `DetectionTransformationCalculator` 输出 `PIXEL_DETECTIONS`,再接 `DetectionsDeduplicateCalculator` 后作为 graph 输出,以保持 `YoloObjectDetector` 现有 pixel-unit public API。Commit ×2:`feat(tiling): YoloObjectDetector tiling options + converter`、`feat(tiling): YoloObjectDetector tiled branch + real-model e2e`。

---

### Task 10: 全量回归 + 收尾

- [ ] **Step 1: CPU 全量**:

```bash
bazel test --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/calculators/tensor:all //mediapipe/graphs/tiled_detection:all \
  //mediapipe/calculators/util:rotated_non_max_suppression_calculator_test \
  //mediapipe/calculators/util:oriented_detection_projection_calculator_test \
  //mediapipe/tasks/cc/vision/yolo_object_detector:yolo_object_detector_test \
  //mediapipe/tasks/cc/vision/oriented_object_detector:oriented_object_detector_test
```

(`tensor:all` 中 GPU-only target 在 DISABLE_GPU 下自动空化;若个别 target 不兼容该 define,改用本计划各任务列出的明确 target 列表。)
- [ ] **Step 2: Metal 回归**(无 define):`bazel test //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_calculator_metal_test //mediapipe/calculators/tensor:streaming_to_inference_metal_zero_copy_test //mediapipe/calculators/tensor:tiled_zero_copy_detection_metal_test //mediapipe/calculators/tensor:streaming_tiles_to_tensor_batch_metal_test`(Task 2 改了 batcher,必须验证 Metal 路径)。
- [ ] **Step 3: Commit**(若有零散修正)`test(tiling): full regression for tasks tiled detection`。

---

## Self-review 记录

- Spec 覆盖:§3.1→Task 1;§3.2→Task 2;§3.3→Task 3-6;§3.4→Task 7-9;§4 错误处理分散在 Task 1(互斥/参数)、Task 8(NORM_RECT、维度 RET_CHECK);§5 测试→各任务 + Task 10。
- 类型一致性:`TileGridCalculatorOptions.TileRect` 仅定义一次(Task 1),FrontGraph options 整体嵌 `tile_grid`(Task 3),Tasks proto 的 `TilingOptions.TileRect` 是独立嵌套消息(Task 7/9,按 spec 原文)。
- 输出单位一致性:merge graph 内部输出 frame-normalized proto;OBB public result 由容器转换成 pixel units;YOLO tiled 分支必须显式接 `DetectionTransformationCalculator` + dedup 后再输出 pixel bbox。
- NMS 决定:全局 NMS 恒执行、不做 NUM_TILES bypass(tile-local NMS 默认关闭,bypass 会让单显式 tile 配置返回未去重结果;对已去重集合重跑 NMS 是幂等 no-op)。两个 merge subgraph 都有 `SingleTileStillRunsGlobalNms` 测试钉住此决定。
- 已知实现期需就地确认的点(非占位,均给了查证路径):`FromImageCalculator` 输出 tag(`IMAGE_CPU`,见其源文件)、`ModelResources::GetTfLiteModel()` 准确签名(model_resources.h)、检测结果容器字段名(oriented_object_detection_result.h)、模型 normalization helper 的准确入口(image_tensor_specs/metadata extractor)。
