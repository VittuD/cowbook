# Job Execution

Cowbook exposes [structured execution state](reference/execution.md) so callers can monitor a run without scraping logs. Core pieces include `PipelineRunner` for direct synchronous runs, `RunResult` and `JobRun` for run summaries, `JobEvent` as the event stream element, `StageProgressReporter` and `TrackingProgressReporter` as shared milestone reporters, and `CancellationToken` for cooperative cancellation.

The execution layer emits structured events and keeps execution state separate from presentation concerns. It describes a run without imposing any particular user interface. At the coarse level this includes job, group, processing, merge, export, and video stage events. It can also emit milestone progress events for masking, tracking, per-camera processing, frame rendering, and video assembly.

Tracking-internal milestone events use a single shape:

- `tracking_stage_started`
- `tracking_stage_progress`
- `tracking_stage_completed`

Their payload carries the stage details instead of multiplying event names:

- `tracking_mode`
- `stage_name`
- `video_path`
- `camera_nr`
- `frame_current`
- `frame_total`
- `progress_fraction`

Human-readable tracking progress is separate and opt-in through `log_progress` / `--log-progress`.

Other long-running stages use the same pattern with stage-specific prefixes:

- `masking_stage_started` / `masking_stage_progress` / `masking_stage_completed`
- `processing_stage_started` / `processing_stage_progress` / `processing_stage_completed`
- `video_stage_started` / `video_stage_progress` / `video_stage_completed`

Those payloads always include `stage_name`, may include path metadata such as `video_path`, `base_filename`, or `output_video_path`, and use `current`, `total`, and `progress_fraction` for milestone updates.

Cancellation is cooperative rather than forceful: the engine can stop between stages and in selected inner loops, but it is not a hard process kill mechanism.
