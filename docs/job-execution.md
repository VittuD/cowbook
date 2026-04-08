# Job Execution

Cowbook exposes [structured execution state](reference/execution.md) so callers can monitor a run without scraping logs. Core pieces include `PipelineRunner` for direct synchronous runs, `JobRun` as the aggregated run snapshot, `JobEvent` as the event stream element, `TrackingProgressReporter` as the shared milestone reporter for tracking stages, and `CancellationToken` for cooperative cancellation.

The execution layer emits structured events and keeps execution state separate from presentation concerns. It describes a run without imposing any particular user interface. At the coarse level this includes job, group, processing, merge, export, and video stage events. During tracking it can also emit shared milestone events for direct tracking and cleanup tracking internals.

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

Cancellation is cooperative rather than forceful: the engine can stop between stages and in selected inner loops, but it is not a hard process kill mechanism.
