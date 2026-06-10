# Tasks API 静态 Tiled 检测(子项目 A)设计

> 状态:已与用户确认的设计(2026-06-11)。本 spec 是「生产级 tiled 检测入口」三个子项目中的第一个;后续 C(BoxTracker 接入)、B(VIDEO/LIVE_STREAM 运动调度接线)各自单独立 spec。
>
> 决策记录:入口层级 = **直接集成 Tasks API**;检测器范围 = **OBB + axis-aligned 两个都做**;配置面 = **网格参数 + 显式 rect 列表(互斥)**;实现途径 = **独立 subgraph 中间层**(Tasks 是第一个消费者,subgraph 可独立复用);scheduler/suppression/tracker = **不在本子项目**(C、B 轮),但本设计的后半 subgraph 预留其接入口。

## 1. 目标

用户通过 `YoloObjectDetector` / `OrientedObjectDetector` 的公开 C++ Tasks options 配置 tiling(均匀网格或显式 tile 列表),检测器内部把每帧切成多个 tile、批量推理、把各 tile 检测合并回整帧坐标并做全局 NMS。所有 running mode(IMAGE/VIDEO/LIVE_STREAM)本轮均为**静态 tiling**(每帧全量 tile)。

非目标(明确范围外):C/Python 绑定(归既有 Plan B 后续)、letterbox、Metal zero-copy 进 Tasks 路径(保持 calculator 级)、运动调度与 tracker(C/B 轮)、NORM_RECT(ROI)与 tiling 的组合。

## 2. 架构总览

```
Tasks detector graph(tiling 启用时的分支):
  Image ──FromImageCalculator──▶ ImageFrame
        ──▶ TiledDetectionFrontGraph ──TENSORS(逐批)──▶ AddInference(不动)
                       │                                   │
                       └────────BATCH_INFO────────┐        ▼
                                                  │   decoder(既有节点,
                                                  │   含 tile-local NMS 选项)
                                                  ▼        │
                              TiledObbMergeGraph / TiledBoxMergeGraph
                                                  │
                                                  ▼
                              label_id_to_text ──▶ 帧级检测输出(帧归一化坐标)
```

关键坐标事实:单图路径在模型输入空间检测、靠 PROJECTION_MATRIX 投回原图;tiled 路径的 merge accumulator **直接输出源帧归一化坐标**(几何随 BATCH_INFO 携带),因此 **tiled 分支没有 projection 步骤**。OrientedDetection 的归一化坐标约定见 `oriented_detection.proto`(P1.2 已文档化)。

推理留给调用方(Tasks 用 `AddInference`/ModelResources):subgraph 中间层只做「前半」(切 tile → 批张量)和「后半」(逐批检测 → 帧级合并+NMS),避免在 subgraph 里重新发明模型传递。

## 3. 新组件

### 3.1 `TileGridCalculator`(新,~60 行)

- 位置:`mediapipe/calculators/tensor/tile_grid_calculator.{cc,proto}` + 单测。
- 契约:`Input<AnyType> TICK`(仅作节拍,任意类型;Tasks 接 IMAGE 流),`Output<std::vector<NormalizedRect>> TILES`。
- options(ext = **471230012**):
  - `int32 rows = 1 [default = 1]`、`int32 cols = 2 [default = 1]`、`float overlap_fraction = 3 [default = 0.0]`(相邻 tile 重叠占 tile 边长比例,[0,1));
  - `repeated TileRect explicit_tiles = 4`,`TileRect { x_center, y_center, width, height }`(帧归一化)。
- 语义:`explicit_tiles` 非空时直接逐帧输出该列表(与网格参数互斥,同时设置 → Open() 报错);否则按 rows×cols 生成均匀网格:tile 尺寸 `w=1/(cols-(cols-1)*overlap)` 的等价展开式——具体公式:`stride_x=(1-tile_w)/(cols-1)`(cols>1),`tile_w=1/(cols-(cols-1)*overlap_fraction)`;rows 同理。rows=cols=1 输出单个全帧 tile。Open() 计算一次,Process 每 tick 发同一列表。
- 校验:rows/cols ≥ 1;overlap_fraction ∈ [0,1);explicit_tiles 的每项 width/height > 0。

### 3.2 `StreamingTilesToTensorBatchCalculatorOptions` 增加元数据字段(加法)

METADATA side packet **缺席**时,从 options 读取(两者都给 → side packet 优先,Open() 打 LOG(WARNING);都缺 → 现有 RET_CHECK 报错路径):

```proto
  // 当 METADATA side packet 未连接时,模型输入元数据从这里读取
  //(subgraph 场景:由 graph builder 在构建期填入)。
  optional int32 metadata_batch_capacity = 12 [default = 0];
  optional int32 metadata_input_height   = 13 [default = 0];
  optional int32 metadata_input_width    = 14 [default = 0];
  optional int32 metadata_input_channels = 15 [default = 0];
  optional bool  metadata_is_dynamic_batch = 16 [default = false];
```

(字段 6/8/9/10 已 reserved,11 = metal_direct_delegate_input,故从 12 起。)同时 `kSideMeta` 由必选改为 `SideInput<...>::Optional`,Open() 内统一装配成 `meta_`,校验逻辑不变。

### 3.3 三个 subgraph(新目录 `mediapipe/graphs/tiled_detection/`)

均为 C++ `Subgraph`(`REGISTER_MEDIAPIPE_GRAPH`),options proto 共置一文件 `tiled_detection_graphs.proto`。

**`TiledDetectionFrontGraph`**(OBB/YOLO 共享)
- 输入流:`IMAGE`(ImageFrame);输出流:`TENSORS`(`std::vector<Tensor>`,合成时间戳逐批)、`BATCH_INFO`(`TensorBatchInfo`)。
- options(ext = **471230013**):tile 网格/显式列表(与 3.1 同形态,GetConfig 透传给内部 TileGridCalculator)+ 模型元数据四元组(透传给 batcher 的 3.2 字段)。
- 内部:`TileGridCalculator(TICK=IMAGE)` → `TileSpecToTilePlanCalculator` → `StreamingTilesToTensorBatchCalculator(IMAGE, TILE_PLAN)`。

**`TiledObbMergeGraph`**
- 输入流:`ORIENTED_DETECTIONS`(`std::vector<std::vector<OrientedDetection>>`,逐批)、`BATCH_INFO`;输出流:`ORIENTED_DETECTIONS`(`std::vector<OrientedDetection>`,帧级,源帧时间戳)。
- options(ext = **471230014**):`iou_threshold`(default 0.45)、`max_detections`(default -1)、`class_agnostic`(default false)→ 透传 RotatedNMS。
- 内部:`MergeTileDetectionsAccumulatorCalculator` → `RotatedNonMaxSuppressionCalculator`。

**`TiledBoxMergeGraph`**
- 输入流:`DETECTIONS`(`std::vector<std::vector<Detection>>`,逐批)、`BATCH_INFO`;输出流:`DETECTIONS`(`std::vector<Detection>`,帧级)。
- options(ext = **471230015**):`iou_threshold`(default 0.45)、`class_agnostic`(default false)→ 透传 TiledFrameSuppression;`max_detections`(default -1)→ ≥0 时 GetConfig 追加一个上游既有的 `ClipDetectionVectorSizeCalculator` 节点截断(NMS 输出为分数降序,截断即保最高分)。
- 内部:`MergeTileBoxDetectionsAccumulatorCalculator` → `TiledFrameSuppressionCalculator`(TRACKER_DETECTIONS/NUM_TILES 不连接 → 纯全局 NMS;**B/C 轮的 tracker 即从这两个既有可选输入接入,这就是预留的接口**)→ [可选 Clip]。

### 3.4 Tasks options 与 graph builder 集成(两个检测器对称)

**proto**(`yolo_object_detector_options.proto` 加 `TilingOptions tiling = 10;`,`oriented_object_detector_options.proto` 加 `= 11;`):

```proto
message TilingOptions {
  optional int32 tile_rows = 1 [default = 1];
  optional int32 tile_cols = 2 [default = 1];
  optional float tile_overlap_fraction = 3 [default = 0.0];
  message TileRect { optional float x_center=1; optional float y_center=2;
                     optional float width=3; optional float height=4; }
  repeated TileRect explicit_tiles = 4;
  // 进入 decoder 的 tile-local NMS(P2.2 新能力)透传:
  optional float tile_local_nms_iou_threshold = 5 [default = 0.0];
  optional int32 max_detections_after_tile_nms = 6 [default = 0];
}
```

**C++ struct**(`yolo_object_detector.h` / `oriented_object_detector.h`):对应 `struct TilingOptions { int tile_rows=1; int tile_cols=1; float tile_overlap_fraction=0; std::vector<std::array<float,4>> explicit_tiles; float tile_local_nms_iou_threshold=0; int max_detections_after_tile_nms=0; }` 字段,转换器(既有 `Convert*OptionsToProto`)同步映射。

**启用判定**:`tile_rows*tile_cols > 1 || !explicit_tiles.empty()`。未启用 → 现有单图路径**逐字节不变**。

**graph builder 分支**(`*_detector_graph.cc`):
1. 启用 tiling 且 NORM_RECT 输入被连接 → 构建期返回 InvalidArgument("tiling 与 NORM_RECT/ROI 暂不支持组合")。
2. `Image` → `FromImageCalculator` → ImageFrame。
3. FrontGraph 节点:tile 配置来自 TilingOptions;模型元数据(batch_capacity/H/W/C)从 ModelResources 的输入张量形状就地读取(与现有 num_classes 推导同处)。
4. `AddInference`(现状不动)← FrontGraph 的 TENSORS。
5. 既有 decoder 节点(YoloObb…/Yolo…)——把 TilingOptions 里的 tile-local NMS 两参写进 decoder options。
6. 对应 MergeGraph ← decoder 输出 + FrontGraph 的 BATCH_INFO;NMS 参数沿用检测器既有 `iou_threshold`/`class_agnostic_nms`/`max_results`。
7. `*LabelIdToText`(现状)→ 输出。**不接 projection 节点**。

## 4. 错误处理

| 场景 | 行为 |
|---|---|
| 网格参数(rows*cols>1)与 explicit_tiles 同时设置 | TileGridCalculator Open() RET_CHECK(构建后首帧前即失败);Tasks 转换器层同样预校验,报 InvalidArgument |
| rows/cols<1、overlap∉[0,1)、explicit tile 宽高≤0 | 同上(Open()/转换器双层) |
| explicit tile 与帧不相交 | 沿用 TileSpecToTilePlan 既有 RET_CHECK |
| tiling + NORM_RECT | 构建期 InvalidArgument |
| METADATA side packet 与 options 元数据都缺 | batcher Open() RET_CHECK(现有 meta_ 校验) |
| 空帧/缺包(T==0 等) | P0 守卫已覆盖(merge 发空结果) |

## 5. 测试

1. `tile_grid_calculator_test`:1×1 全帧;2×3 无 overlap 几何断言;overlap 公式断言(2 列 overlap=0.2 → tile_w=1/1.8);显式列表透传;互斥/非法参数 Open 失败。
2. `tiled_detection_graphs_test`(CalculatorGraph 级,合成输入):FrontGraph 用 8×8 元数据白图 → 断言批数/形状/BATCH_INFO;两个 MergeGraph 用合成逐批检测 → 断言帧级合并+NMS 结果与源帧时间戳。
3. **真模型 e2e ×2**(CPU,DISABLE_GPU=1,fixtures 缺失则 GTEST_SKIP):`yolo_object_detector_test` / `oriented_object_detector_test` 各加 tiled 用例——boats.jpg,2×1 网格 overlap 0.2。硬断言:检出 ≥1 个船类目标、类别名正确、所有框坐标 ∈ [0,1](帧归一化)、无重复(任意两框 IoU < NMS 阈值);对照断言:同图单图路径检出数 N,tiled 检出数 ∈ [N-1, N+3](tiling 对小目标只增不减,上界容忍 overlap 引入的边界重复残留)。
4. 全量既有套件回归(19 CPU + 4 Metal)。

## 6. 文件清单

- 新:`mediapipe/calculators/tensor/tile_grid_calculator.{cc,proto}` + test;`mediapipe/graphs/tiled_detection/{BUILD, tiled_detection_graphs.proto, tiled_detection_front_graph.cc, tiled_obb_merge_graph.cc, tiled_box_merge_graph.cc}` + graphs test。
- 改:`streaming_tiles_to_tensor_batch_calculator.{proto,cc}`(3.2);两个 tasks options proto + 两个 detector `.h/.cc` 转换器 + 两个 `*_detector_graph.cc`;两个 detector test 加 e2e;相关 BUILD。

## 7. 后续衔接(C、B 轮,本轮不实现)

- C(BoxTracker):tracker 输出 `TRACKER_DETECTIONS` → `TiledBoxMergeGraph` 内 TiledFrameSuppression 的既有可选输入;OBB 侧需要 OrientedDetection 版 suppression(复用 `GreedyOrientedDetectionNms` + `TileFrameAccumulator` 同款共享模式)。
- B(调度):`TileGridCalculator` 的 TILES 输出与 `VideoTileSchedulerCalculator` 的 TILES 输入同型(`std::vector<NormalizedRect>`),scheduler 插在 TileGrid 与 TileSpecToTilePlan 之间即可;SKIP 帧结果 = tracker 推进框(用户已决策)。
