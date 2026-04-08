# Execution Reference

The execution layer provides the run-state and event primitives used by the
pipeline. It does not schedule or persist jobs. It defines:

- the event stream emitted during a run
- the aggregated run snapshot derived from those events
- observer helpers that consume the stream
- shared progress helpers that translate long-running stages into events and logs
- cooperative cancellation primitives

## Run Models

::: cowbook.execution.results.RunResult

::: cowbook.execution.models.JobRun

::: cowbook.execution.models.JobEvent

::: cowbook.execution.models.JobArtifact

## Observers and Reporting

::: cowbook.execution.observers.JobObserver

::: cowbook.execution.observers.JobReporter

::: cowbook.execution.observers.CompositeObserver

::: cowbook.execution.observers.InMemoryJobStore

## Progress

::: cowbook.execution.progress.StageProgressReporter

::: cowbook.execution.progress.TrackingProgressReporter

## Cancellation

::: cowbook.execution.error_codes

::: cowbook.execution.control.JobCancelledError

::: cowbook.execution.control.CancellationToken
