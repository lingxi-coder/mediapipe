# Detector Category Names + Allowlist/Denylist — Plan A (C++ in-graph core)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In-graph category-name mapping + `category_allowlist`/`category_denylist` filtering for the YOLO and OBB C++ detectors, reusing upstream MediaPipe mechanisms.

**Architecture:** Filtering is by class index inside the two Yolo decoders (new default-empty `allow_classes`/`ignore_classes`, applied before the NMS/`max_results` cap). Names come from the model-metadata label file: YOLO uses the stock `DetectionLabelIdToTextCalculator` (`keep_label_id=true`); OBB uses a new `OrientedDetectionLabelIdToTextCalculator` (mirrors the stock one for `OrientedDetection`). Each graph reads labels + resolves allow/deny names→indices via a shared helper. Defaults empty ⇒ byte-identical behavior.

**Tech Stack:** C++ (api2 + legacy calculators, MediaPipe Tasks vision graphs), proto2, `mediapipe/util/label_map_util`, Bazel (`--define MEDIAPIPE_DISABLE_GPU=1`), GoogleTest/CalculatorRunner.

**Spec:** `docs/superpowers/specs/2026-06-04-detector-label-names-and-category-filter-design.md`.

**Scope note:** This is Plan A of two. Plan A is the C++ core (fully CPU-verifiable here). Plan B (separate) adds the OBB **C API + Python** option surface + Python fallback migration + Python tests (build-deferred — needs the blocked `libmediapipe.so`). Plan A leaves the C++ task fully working with names + filtering.

---

## Reference facts (verified)

- **Stock `DetectionLabelIdToTextCalculator`** (`mediapipe/calculators/util/detection_label_id_to_text_calculator.{cc,proto}`): legacy calc, `Inputs().Index(0).Set<std::vector<Detection>>()` / same out. Options: `label_map_path` | `label` (repeated string) | `label_items` (`Map<int64,LabelMapItem>`) | `keep_label_id` (bool, default false). Process: for each `label_id` present in the map, `add_label(name)` + `add_display_name` if any; **clears `label_id` unless `keep_label_id` is true** → so the graph MUST set `keep_label_id=true` (else `ConvertToDetectionResult` sees no `label_id` and emits `index = -1`).
- **`OrientedDetection` proto** (`mediapipe/framework/formats/oriented_detection.proto`): `cx,cy,width,height,rotation`, `repeated string label=6`, `repeated int32 label_id=7`, `repeated float score=8`, `repeated string display_name=9`.
- **YOLO decoder** `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.{cc,proto}`: `DecodeRow` computes argmax `best` then `if (best_score < conf_threshold) continue;`. proto fields 1–9 used (next free = **10**).
- **OBB decoder** `mediapipe/calculators/tensor/yolo_obb_tensors_to_oriented_detections_calculator.{cc,proto}`: `DecodeRow` computes argmax `best`, writes `OrientedDetection`. proto fields 1–7 used (`layout=1,num_classes=2,conf_threshold=3,max_detections_before_nms=4,tile_local_nms_iou_threshold=5,max_detections_after_tile_nms=6,tile_local_nms_class_agnostic=7`; next free = **8**).
- **OBB options proto** `oriented_object_detector/proto/oriented_object_detector_options.proto`: fields 1–7 used (next free = **8**). C++ `OrientedObjectDetectorOptions` (`.h`) lacks `display_names_locale`/`category_allowlist`/`category_denylist`. YOLO already has them.
- **OBB result conversion** `oriented_object_detection_result.cc`: `ConvertToOrientedObjectDetectionResult` sets `category_name=std::nullopt`, `display_name=std::nullopt` — must read `d.label(i)`/`d.display_name(i)`.
- **Label map from metadata:** `model_resources.GetMetadataExtractor()` → find the OUTPUT tensor's associated file of type `TENSOR_AXIS_LABELS` → `GetAssociatedFile(name)` → `mediapipe::BuildLabelMapFromFiles(labels, "")` (`mediapipe/util/label_map_util.h`) → `Map<int64,LabelMapItem>`. (Pattern in `detection_postprocessing_graph.cc` `GetLabelItemsIfAny` / `GetAllowOrDenyCategoryIndicesIfAny`.) The export scripts attach this file.
- Build/test: `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 <target> --test_output=errors`. A piped `exit 0` is NOT success — read the real output.

## Task order / file map
1. YOLO decoder allow/deny — `yolo_tensors_to_detections_calculator.{proto,cc,_test}`
2. OBB decoder allow/deny — `yolo_obb_tensors_to_oriented_detections_calculator.{proto,cc,_test}`
3. New `OrientedDetectionLabelIdToTextCalculator` — `mediapipe/calculators/util/oriented_detection_label_id_to_text_calculator.{cc,proto}` + BUILD + `_test`
4. OBB result container reads label/display_name — `oriented_object_detection_result.{cc,_test}`
5. Shared metadata→labels helper — `mediapipe/tasks/cc/vision/utils/detection_label_resolution.{h,cc}` + BUILD + `_test`
6. OBB C++ options surface — `oriented_object_detector_options.proto`, `oriented_object_detector.{h,cc}` + passthrough test
7. YOLO graph wiring + integration test
8. OBB graph wiring + integration test

---

### Task 1: YOLO decoder `allow_classes` / `ignore_classes`

**Files:** `mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.{proto,cc}`, `..._test.cc`.

- [ ] **Step 1: proto fields.** After field 9 (`input_height`) in `message YoloTensorsToDetectionsCalculatorOptions`:
```proto
  // Class-index filter applied during decode (before conf-threshold / cap).
  // Mutually exclusive. Empty (default) = no filtering (unchanged). Mirrors
  // TensorsToDetectionsCalculatorOptions.allow_classes/ignore_classes.
  repeated int32 allow_classes = 10 [packed = true];
  repeated int32 ignore_classes = 11 [packed = true];
```

- [ ] **Step 2: failing test** — add to `yolo_tensors_to_detections_calculator_test.cc` (CHANNELS_FIRST `[1,5,2]`, num_classes:1 → use 2 anchors of different classes via a 2-class tensor). Use a `[1,6,2]` num_classes:2 tensor: anchors {class0 @0.9, class1 @0.8}; `allow_classes: 1` keeps only the class-1 anchor:
```cpp
TEST(YoloTensorsToDetectionsCalculatorTest, AllowClassesFiltersByIndex) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:dets"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        num_classes: 2 conf_threshold: 0.25 allow_classes: 1
      }
    }
  )pb"));
  // [1, 4+2=6, 2]: cx,cy,w,h, s0,s1 ; anchor0 argmax class0(0.9), anchor1 class1(0.8).
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 6, 2},
                       {0.5f,0.5f, 0.5f,0.5f, 0.2f,0.2f, 0.2f,0.2f,
                        0.9f,0.1f, 0.1f,0.8f}).release()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Tag("DETECTIONS").packets[0]
                        .Get<std::vector<std::vector<Detection>>>();
  ASSERT_EQ(out[0].size(), 1u);          // only the class-1 anchor survives
  EXPECT_EQ(out[0][0].label_id(0), 1);
}
TEST(YoloTensorsToDetectionsCalculatorTest, IgnoreClassesDropsByIndex) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "DETECTIONS:dets"
    options {
      [mediapipe.YoloTensorsToDetectionsCalculatorOptions.ext] {
        num_classes: 2 conf_threshold: 0.25 ignore_classes: 0
      }
    }
  )pb"));
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 6, 2},
                       {0.5f,0.5f, 0.5f,0.5f, 0.2f,0.2f, 0.2f,0.2f,
                        0.9f,0.1f, 0.1f,0.8f}).release()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Tag("DETECTIONS").packets[0]
                        .Get<std::vector<std::vector<Detection>>>();
  ASSERT_EQ(out[0].size(), 1u);
  EXPECT_EQ(out[0][0].label_id(0), 1);   // class 0 dropped
}
```

- [ ] **Step 3: run, expect FAIL** (fields/logic absent). `bazel test ... :yolo_tensors_to_detections_calculator_test`.

- [ ] **Step 4: implement.** In `Open()`, after the existing checks:
```cpp
    RET_CHECK(options_.allow_classes().empty() || options_.ignore_classes().empty())
        << "allow_classes and ignore_classes are mutually exclusive";
    for (int c : options_.allow_classes()) allow_classes_.insert(c);
    for (int c : options_.ignore_classes()) ignore_classes_.insert(c);
```
Add members: `absl::flat_hash_set<int> allow_classes_; absl::flat_hash_set<int> ignore_classes_;` (include `"absl/container/flat_hash_set.h"`).
In `DecodeRow`, right after the argmax computes `best`/`best_score` and BEFORE `if (best_score < conf_threshold) continue;`:
```cpp
      if (!allow_classes_.empty() && !allow_classes_.contains(best)) continue;
      if (ignore_classes_.contains(best)) continue;
```

- [ ] **Step 5: run, expect PASS** (incl. all existing tests — empty sets ⇒ unchanged). `bazel test ... :yolo_tensors_to_detections_calculator_test --test_output=errors`.

- [ ] **Step 6: commit.**
```bash
git add mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.proto \
        mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator.cc \
        mediapipe/calculators/tensor/yolo_tensors_to_detections_calculator_test.cc \
        mediapipe/calculators/tensor/BUILD
git commit -m "feat(yolo-decode): default-off allow_classes/ignore_classes index filter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
(Add `@com_google_absl//absl/container:flat_hash_set` to the calculator's BUILD deps if not present.)

---

### Task 2: OBB decoder `allow_classes` / `ignore_classes`

**Files:** `mediapipe/calculators/tensor/yolo_obb_tensors_to_oriented_detections_calculator.{proto,cc}`, `..._test.cc`.

- [ ] **Step 1: proto fields.** After field 7 (`tile_local_nms_class_agnostic`) in `message YoloObbTensorsToOrientedDetectionsCalculatorOptions`:
```proto
  // Class-index filter during decode (before conf-threshold / cap). Mutually
  // exclusive; empty (default) = no filtering.
  repeated int32 allow_classes = 8 [packed = true];
  repeated int32 ignore_classes = 9 [packed = true];
```

- [ ] **Step 2: failing test** — add to `yolo_obb_tensors_to_oriented_detections_calculator_test.cc` (mirror Task 1 but `[1, 4+2+1=7, 2]`, angle at channel 6):
```cpp
TEST(YoloObbCalculatorTest, AllowClassesFiltersByIndex) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloObbTensorsToOrientedDetectionsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "ORIENTED_DETECTIONS:dets"
    options {
      [mediapipe.YoloObbTensorsToOrientedDetectionsCalculatorOptions.ext] {
        num_classes: 2 conf_threshold: 0.25 allow_classes: 1
      }
    }
  )pb"));
  // [1,7,2]: cx,cy,w,h, s0,s1, angle ; anchor0 class0(0.9), anchor1 class1(0.8).
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      Adopt(MakeTensor(Tensor::Shape{1, 7, 2},
                       {0.5f,0.5f, 0.5f,0.5f, 0.2f,0.2f, 0.2f,0.2f,
                        0.9f,0.1f, 0.1f,0.8f, 0.3f,0.4f}).release()).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& batch = runner.Outputs().Tag("ORIENTED_DETECTIONS").packets[0]
                          .Get<BatchOrientedDetections>();
  ASSERT_EQ(batch[0].size(), 1u);
  EXPECT_EQ(batch[0][0].label_id(0), 1);
}
```
(Add a parallel `IgnoreClassesDropsByIndex` with `ignore_classes: 0`.)

- [ ] **Step 3: run, expect FAIL.**
- [ ] **Step 4: implement** — same as Task 1 Step 4 (Open mutual-exclusion + sets; in `DecodeRow`, after `best` argmax and before the conf-threshold `continue`, add the two skip lines). Add the `flat_hash_set` members + include + BUILD dep.
- [ ] **Step 5: run, expect PASS** (incl. existing OBB cases).
- [ ] **Step 6: commit** `feat(yolo-obb-decode): default-off allow_classes/ignore_classes index filter`.

---

### Task 3: `OrientedDetectionLabelIdToTextCalculator`

**Files:** Create `mediapipe/calculators/util/oriented_detection_label_id_to_text_calculator.{cc,proto}`; modify `mediapipe/calculators/util/BUILD`; create `..._test.cc`.

Mirror the stock `detection_label_id_to_text_calculator` exactly, swapping `Detection` → `OrientedDetection`.

- [ ] **Step 1: proto** `oriented_detection_label_id_to_text_calculator.proto` (copy the stock proto, rename the message):
```proto
syntax = "proto2";
package mediapipe;
import "mediapipe/framework/calculator.proto";
import "mediapipe/util/label_map.proto";
message OrientedDetectionLabelIdToTextCalculatorOptions {
  extend CalculatorOptions {
    optional OrientedDetectionLabelIdToTextCalculatorOptions ext = 471230020;
  }
  optional string label_map_path = 1;
  repeated string label = 2;
  optional bool keep_label_id = 3;
  map<int64, LabelMapItem> label_items = 4;
}
```

- [ ] **Step 2: `.cc`** — copy `detection_label_id_to_text_calculator.cc`, replace `Detection`→`OrientedDetection`, the options type, and the include (`oriented_detection.pb.h`). Keep `Open()` (builds `local_label_map_` from `label_map_path` via `PathToResourceAsFile`+`ParseDetectionLabels`/`BuildLabelMapFromFiles`, or from `label`), `GetContract` (`Index(0).Set<std::vector<OrientedDetection>>()`), and the Process loop (add label/display_name per `label_id`; clear `label_id` unless `keep_label_id_`). Register `REGISTER_CALCULATOR(OrientedDetectionLabelIdToTextCalculator);`.

- [ ] **Step 3: BUILD** — add a `mediapipe_proto_library` for the proto and a `cc_library` for the calculator (mirror the stock `detection_label_id_to_text_calculator` targets; deps: `//mediapipe/framework/formats:oriented_detection_cc_proto`, `//mediapipe/util:label_map_cc_proto`, `//mediapipe/util:label_map_util`, framework, `//mediapipe/framework/port:status`, the proto). Add a `cc_test`.

- [ ] **Step 4: test** `oriented_detection_label_id_to_text_calculator_test.cc`:
```cpp
TEST(OrientedDetectionLabelIdToTextCalculatorTest, MapsLabelAndKeepsId) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "OrientedDetectionLabelIdToTextCalculator"
    input_stream: "in"
    output_stream: "out"
    options {
      [mediapipe.OrientedDetectionLabelIdToTextCalculatorOptions.ext] {
        label: "cat" label: "dog" keep_label_id: true
      }
    }
  )pb"));
  std::vector<OrientedDetection> in(1);
  in[0].add_label_id(1); in[0].add_score(0.9f);
  runner.MutableInputs()->Index(0).packets.push_back(
      MakePacket<std::vector<OrientedDetection>>(in).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Index(0).packets[0]
                        .Get<std::vector<OrientedDetection>>();
  ASSERT_EQ(out[0].label_size(), 1);
  EXPECT_EQ(out[0].label(0), "dog");
  ASSERT_EQ(out[0].label_id_size(), 1);   // kept
  EXPECT_EQ(out[0].label_id(0), 1);
}
TEST(OrientedDetectionLabelIdToTextCalculatorTest, ClearsIdWhenNotKept) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "OrientedDetectionLabelIdToTextCalculator"
    input_stream: "in"
    output_stream: "out"
    options { [mediapipe.OrientedDetectionLabelIdToTextCalculatorOptions.ext] {
      label: "cat" label: "dog" } }
  )pb"));
  std::vector<OrientedDetection> in(1);
  in[0].add_label_id(0); in[0].add_score(0.5f);
  runner.MutableInputs()->Index(0).packets.push_back(
      MakePacket<std::vector<OrientedDetection>>(in).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& out = runner.Outputs().Index(0).packets[0]
                        .Get<std::vector<OrientedDetection>>();
  EXPECT_EQ(out[0].label(0), "cat");
  EXPECT_EQ(out[0].label_id_size(), 0);   // cleared
}
```

- [ ] **Step 5: build + run.** `bazel test ... :oriented_detection_label_id_to_text_calculator_test` → PASS.
- [ ] **Step 6: commit** `feat(util): OrientedDetectionLabelIdToTextCalculator (mirrors the Detection one)`.

---

### Task 4: OBB result container reads label/display_name

**Files:** `mediapipe/tasks/cc/components/containers/oriented_object_detection_result.cc`, `..._test.cc`.

- [ ] **Step 1: failing test** — in `oriented_object_detection_result_test.cc`, build an `OrientedDetection` with `add_label("ship")` + `add_label_id(1)` + `add_score(0.9)` and assert the converted struct's `categories[0].category_name == "ship"` and `index == 1`.
- [ ] **Step 2: run, expect FAIL** (currently `category_name == std::nullopt`).
- [ ] **Step 3: implement** — in `ConvertToOrientedObjectDetectionResult`, replace the `category_name`/`display_name` `std::nullopt` with reads of the proto (mirror `ConvertToDetectionResult`):
```cpp
      od.categories.push_back(
          {/* index= */ d.label_id_size() > i ? d.label_id(i) : kDefaultCategoryIndex,
           /* score= */ d.score(i),
           /* category_name= */ d.label_size() > i
               ? std::make_optional(d.label(i)) : std::nullopt,
           /* display_name= */ d.display_name_size() > i
               ? std::make_optional(d.display_name(i)) : std::nullopt});
```
- [ ] **Step 4: run, expect PASS** (+ existing container tests).
- [ ] **Step 5: commit** `feat(tasks-obb): populate OrientedDetection category_name/display_name from proto`.

---

### Task 5: Shared metadata→labels resolution helper

**Files:** Create `mediapipe/tasks/cc/vision/utils/detection_label_resolution.{h,cc}` + BUILD `cc_library` + `..._test.cc`.

Provides (reused by both graphs):
- `absl::StatusOr<proto_ns::Map<int64_t, LabelMapItem>> GetLabelItemsFromMetadata(const core::ModelResources& model_resources, absl::string_view display_names_locale)` — finds the output-tensor `TENSOR_AXIS_LABELS` associated file (+ optional locale display-names file), `BuildLabelMapFromFiles`. Empty map if no label file.
- `absl::StatusOr<absl::flat_hash_set<int>> ResolveCategoryIndices(const proto_ns::Map<int64_t,LabelMapItem>& label_items, const google::protobuf::RepeatedPtrField<std::string>& allowlist, const google::protobuf::RepeatedPtrField<std::string>& denylist)` — mutual-exclusion already checked by caller; if both empty → empty set (no filter); if non-empty but `label_items` empty → `CreateStatusWithPayload(kInvalidArgument, ..., kMetadataMissingLabelsError)`; else map names→indices (ignore unknown/duplicate). (Lift the logic from `detection_postprocessing_graph.cc` `GetLabelItemsIfAny`/`GetAllowOrDenyCategoryIndicesIfAny`.)

- [ ] **Step 1: write `.h`** with the two signatures above (+ includes: `model_resources.h`, `label_map.pb.h`, absl).
- [ ] **Step 2: failing test** `detection_label_resolution_test.cc`:
```cpp
TEST(DetectionLabelResolutionTest, ResolvesNamesToIndices) {
  proto_ns::Map<int64_t, LabelMapItem> items;
  items[0].set_name("cat"); items[1].set_name("dog"); items[2].set_name("bird");
  google::protobuf::RepeatedPtrField<std::string> allow, deny;
  *allow.Add() = "dog"; *allow.Add() = "bird"; *allow.Add() = "unknown";  // unknown ignored
  MP_ASSERT_OK_AND_ASSIGN(auto idx, ResolveCategoryIndices(items, allow, deny));
  EXPECT_THAT(idx, ::testing::UnorderedElementsAre(1, 2));
}
TEST(DetectionLabelResolutionTest, AllUnknownAllowlistIsEmptyNoOp) {
  proto_ns::Map<int64_t, LabelMapItem> items; items[0].set_name("cat");
  google::protobuf::RepeatedPtrField<std::string> allow, deny; *allow.Add() = "zzz";
  MP_ASSERT_OK_AND_ASSIGN(auto idx, ResolveCategoryIndices(items, allow, deny));
  EXPECT_TRUE(idx.empty());  // no-op, NOT "drop all"
}
TEST(DetectionLabelResolutionTest, MissingLabelsWithFilterIsError) {
  proto_ns::Map<int64_t, LabelMapItem> empty;
  google::protobuf::RepeatedPtrField<std::string> allow, deny; *allow.Add() = "dog";
  EXPECT_FALSE(ResolveCategoryIndices(empty, allow, deny).ok());
}
```
- [ ] **Step 3: run, expect FAIL** (not implemented).
- [ ] **Step 4: implement `.cc`** per the signatures (the `GetLabelItemsFromMetadata` metadata-reading mirrors `detection_postprocessing_graph.cc`; `ResolveCategoryIndices` mirrors `GetAllowOrDenyCategoryIndicesIfAny`). Note `ResolveCategoryIndices` is metadata-free → directly unit-testable; `GetLabelItemsFromMetadata` is exercised by the integration tests (Tasks 7/8).
- [ ] **Step 5: run, expect PASS.**
- [ ] **Step 6: commit** `feat(tasks-vision): shared metadata label-items + category-index resolution helper`.

---

### Task 6: OBB C++ options surface (`display_names_locale`, allowlist, denylist)

**Files:** `oriented_object_detector/proto/oriented_object_detector_options.proto`, `oriented_object_detector.{h,cc}`, `oriented_object_detector_test.cc` (passthrough). (C API + Python = Plan B.)

- [ ] **Step 1: proto** — after `num_classes = 7`:
```proto
  optional string display_names_locale = 8 [default = "en"];
  repeated string category_allowlist = 9;
  repeated string category_denylist = 10;
```
- [ ] **Step 2: C++ options** — in `oriented_object_detector.h` `struct OrientedObjectDetectorOptions`, add (mirror YOLO's):
```cpp
  std::string display_names_locale = "en";
  std::vector<std::string> category_allowlist = {};
  std::vector<std::string> category_denylist = {};
```
- [ ] **Step 3: options→proto conversion** — in `oriented_object_detector.cc` (where the options proto is populated), copy the three fields (mirror `yolo_object_detector.cc`):
```cpp
  options_proto->set_display_names_locale(options->display_names_locale);
  for (const std::string& c : options->category_allowlist)
    options_proto->add_category_allowlist(c);
  for (const std::string& c : options->category_denylist)
    options_proto->add_category_denylist(c);
```
- [ ] **Step 4: passthrough test** — add a C++ test asserting the three fields round-trip into the proto, AND that the graph rejects allowlist+denylist both set (mirror YOLO's mutual-exclusion `RET_CHECK`; ensure the OBB graph has/gets it — Task 8). A pure options→proto unit test:
```cpp
TEST(OrientedObjectDetectorOptionsTest, CopiesCategoryFieldsToProto) {
  auto opts = std::make_unique<OrientedObjectDetectorOptions>();
  opts->display_names_locale = "fr";
  opts->category_allowlist = {"ship", "plane"};
  // ... call the conversion (expose/observe the proto), assert it carries them.
}
```
(If the conversion is private, assert via building the task with a fake model OR factor a small testable converter — keep it minimal; the integration tests in Task 8 also exercise it.)
- [ ] **Step 5: build + test** the OBB task lib + the passthrough test → PASS. `bazel build ... :oriented_object_detector` ; `bazel test ... :oriented_object_detector_test`.
- [ ] **Step 6: commit** `feat(tasks-obb): expose display_names_locale + category_allowlist/denylist (C++)`.

---

### Task 7: YOLO graph wiring + integration test

**Files:** `yolo_object_detector_graph.cc` + BUILD; `yolo_object_detector_test.cc`.

- [ ] **Step 1: wire names + filtering.** In `yolo_object_detector_graph.cc`:
  (a) get label items: `MP_ASSIGN_OR_RETURN(auto label_items, GetLabelItemsFromMetadata(model_resources, task_options.display_names_locale()));`
  (b) resolve allow/deny → decoder options (in the `YoloTensorsToDetectionsCalculator` opts block):
```cpp
      MP_ASSIGN_OR_RETURN(auto allow_idx, ResolveCategoryIndices(
          label_items, task_options.category_allowlist(),
          task_options.category_denylist()));
      if (!task_options.category_allowlist().empty())
        for (int c : allow_idx) opts.add_allow_classes(c);
      else
        for (int c : allow_idx) opts.add_ignore_classes(c);
```
  (c) after `NonMaxSuppressionCalculator`, before `DetectionProjectionCalculator`, insert:
```cpp
    auto& id_to_text = graph.AddNode("DetectionLabelIdToTextCalculator");
    auto& id_opts = id_to_text.GetOptions<
        ::mediapipe::DetectionLabelIdToTextCalculatorOptions>();
    id_opts.set_keep_label_id(true);
    *id_opts.mutable_label_items() = label_items;
    nms.Out("") >> id_to_text.In("");
    // then feed id_to_text.Out("") into detection_projection instead of nms.Out("")
```
  Add BUILD deps: the helper (`//mediapipe/tasks/cc/vision/utils:detection_label_resolution`), `//mediapipe/calculators/util:detection_label_id_to_text_calculator` + its `_cc_proto`.

- [ ] **Step 2: build the graph** → success.

- [ ] **Step 3: harden the integration test.** In `yolo_object_detector_test.cc` `DetectOnImage`, after the existing assertions add: every `det.categories[0].category_name` is set and in the COCO names; the cat/dog detections have `category_name == "cat"`/`"dog"` with `index` 15/16; and add a test that `category_allowlist = {"dog"}` returns only dogs (index 16, name "dog"), and `category_denylist = {"dog"}` excludes dogs. (Derive exact names from the oracle as in the verify plan.)
```cpp
  for (const auto& det : result.detections)
    EXPECT_TRUE(det.categories[0].category_name.has_value());
```
- [ ] **Step 4: run** `bazel test ... :yolo_object_detector_test --test_output=all` → runs (fixture present) + passes; names populated, allow/deny filter works, `index` preserved.
- [ ] **Step 5: commit** `feat(tasks-yolo): in-graph category names + allowlist/denylist`.

---

### Task 8: OBB graph wiring + integration test

**Files:** `oriented_object_detector_graph.cc` + BUILD; `oriented_object_detector_test.cc`.

- [ ] **Step 1: add OBB mutual-exclusion + wiring.** In `oriented_object_detector_graph.cc`:
  (a) RET_CHECK allowlist+denylist not both set (mirror YOLO).
  (b) `GetLabelItemsFromMetadata` + `ResolveCategoryIndices` → set `allow_classes`/`ignore_classes` on the `YoloObbTensorsToOrientedDetectionsCalculator` opts (same shape as Task 7 (b)).
  (c) after `RotatedNonMaxSuppressionCalculator`, before `OrientedDetectionProjectionCalculator`, insert an `OrientedDetectionLabelIdToTextCalculator` with `keep_label_id=true` + `label_items` (same shape as Task 7 (c)); rewire the projection input to its output.
  BUILD deps: the helper, `//mediapipe/calculators/util:oriented_detection_label_id_to_text_calculator` + `_cc_proto`.
- [ ] **Step 2: build the graph** → success.
- [ ] **Step 3: harden the OBB integration test.** In `oriented_object_detector_test.cc`: assert `det.categories[0].category_name == "ship"` (index 1 preserved) on boats.jpg; add `category_allowlist = {"ship"}` returns only ships; `category_denylist = {"ship"}` excludes ships.
- [ ] **Step 4: run** `bazel test ... :oriented_object_detector_test --test_output=all` → runs + passes.
- [ ] **Step 5: commit** `feat(tasks-obb): in-graph category names + allowlist/denylist`.

---

## Self-review checklist
- New decoder options + calculator + OBB options all default to empty/unset ⇒ existing behavior (tiled pipeline, prior tests) byte-identical.
- `keep_label_id=true` set in BOTH label calculators (else `index` → -1).
- allow/deny resolved names→indices via the shared helper; unknown/dup ignored; all-unknown = no-op; missing-labels-with-filter = error (locked by Task 5 tests).
- `category_name`/`index` both populated (names added, `label_id` preserved).
- OBB result container reads `label`/`display_name`.

## Final verification
- [ ] `bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1` green for: both decoder tests, the new label-calc test, the container test, the helper test, both task integration tests.
- [ ] `git status` clean except gitignored model fixtures.

## Out of scope (→ Plan B, build-deferred)
OBB **C API** struct/conversion; OBB **Python** dataclass/ctypes; the Python `_load_label_map()`/`_enrich_with_label_map()` fallback migration; Python task integration tests. (All need `libmediapipe.so`, blocked here.) Also: iOS/Java/Web; letterbox; score calibration.
