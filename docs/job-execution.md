# Job Execution

Cowbook exposes [structured execution state](reference/execution.md) so callers can monitor a run without scraping logs. Core pieces include `PipelineRunner` for direct synchronous runs, `JobRun` as the aggregated run snapshot, `JobEvent` as the event stream element, and `CancellationToken` for cooperative cancellation.

The execution layer emits structured events and keeps execution state separate from presentation concerns. It describes a run without imposing any particular user interface.

Cancellation is cooperative rather than forceful: the engine can stop between stages and in selected inner loops, but it is not a hard process kill mechanism.
