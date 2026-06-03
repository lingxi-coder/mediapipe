# Video tile scheduler — real-graph integration notes (M9 / Phase 4)

These are forward-looking notes for whoever assembles the production video-mode
graph around the calculators built in M9. They are NOT part of the scheduler's
own contract — the scheduler, suppression, and decoder calculators are complete
and tested in isolation + integration (`video_tile_scheduler_*_test`,
`tiled_frame_suppression_calculator_test`, `detection_nms_util_test`,
`yolo_tensors_to_detections_calculator_test`).

Plan: `docs/superpowers/plans/2026-06-03-video-tile-scheduler.md`.
Design: `docs/superpowers/specs/2026-06-03-video-tile-scheduler-design.md`.

## Calculators delivered by M9
- `VideoTileSchedulerCalculator` — per-frame DETECT/SKIP from `FlowPackager` `TrackingData` + prior detections; emits scheduled `TILES` (rects) + `REFRESH` (bool). FlowPackager-driven (no frame-cadence clock); `TRACKING` optional (cadence-free fallback DETECTs when absent).
- `YoloTensorsToDetectionsCalculator` — gained optional **tile-local NMS** (within a batch row only) + `max_detections_after_tile_nms`; default-off.
- `detection_nms_util` — shared `DetectionRelativeIoU` + `GreedyDetectionNms` (used by the decoder's tile-local NMS and the suppression stage).
- `TiledFrameSuppressionCalculator` — concatenates fresh + tracker detections and runs ONE global NMS, with the safe single-tile/no-tracker pass-through bypass.

## Production wiring checklist
- Keep `MotionAnalysisCalculator -> FlowPackagerCalculator` **per-frame**; feed its `TRACKING:TrackingData` to the scheduler. `FlowPackager`'s internal `frame_idx_` is chunk metadata only — do not treat it as a refresh clock.
- Feed prior final detections back to the scheduler's `PRIOR_DETECTIONS` via `PreviousLoopbackCalculator` (`MAIN` = a per-frame tick, `LOOP` = final detections, declared `input_stream_info { tag_index: "LOOP" back_edge: true }`).
- Scheduler `TILES` → `TileSpecToTilePlanCalculator` → `StreamingTilesToTensorBatchCalculator`. When scheduled tiles exceed model `batch_capacity`, the streaming calc already emits multiple frame-local inference batches at synthetic timestamps (proven: `video_tile_scheduler_pipeline_test` `T=5,B=2`).
- Decoder (`YoloTensorsToDetectionsCalculator`) runs `conf_threshold` → top-K → optional tile-local NMS (per row) → optional post-cap. Tile-local NMS never compares across tiles.
- Merge tile detections to one frame-relative vector, then `TiledFrameSuppressionCalculator` with `TRACKER_DETECTIONS` (from the tracker branch) and `NUM_TILES` for the single-tile bypass decision.
- SKIP-frame detections come from the tracker branch (`BoxTrackerCalculator` + `TrackedDetectionManagerCalculator`), NOT from scheduler-copied priors. Use `PassThroughOrEmptyDetectionVectorCalculator` to materialize an empty fresh-detection vector for SKIP frames so the suppressor's inputs stay timestamp-aligned.
- Use `FlowLimiterCalculator` only around the expensive tiled-detection branch if the graph needs mobile-style backpressure; keep the motion/tracker branch per-frame.
- Keep all detection streams in `RELATIVE_BOUNDING_BOX` (relative coords) through scheduler → merge → tracker → suppression; convert to pixel `BOUNDING_BOX` only in the public Tasks layer, after the final suppression.
- The scheduler is metadata-only: it never inspects pixels or maps image/tensor packets to CPU, so it does not break GPU zero-copy.

## Invariant
Exactly one frame-level suppression per source frame. `TiledFrameSuppressionCalculator` runs one global NMS over fresh + tracker detections, except the documented safe bypass (one valid tile row, no tracker detections, fresh already tile-local-NMSed), which is opt-in via `bypass_single_tile` (default false ⇒ always run global NMS).
