# Package Boundaries

There are two supported entry routes: the CLI with `python -m cowbook --config ...`, and the Python package imported from `cowbook` or [`cowbook.runtime`](reference/runtime.md#cowbook.runtime).

For Python callers, the stable surface is the [runtime layer](python-usage.md): [`run_pipeline()`](reference/runtime.md#cowbook.runtime.run_pipeline), [`run_pipeline_request()`](reference/runtime.md#cowbook.runtime.run_pipeline_request), [`load_pipeline_config()`](reference/runtime.md#cowbook.runtime.load_pipeline_config), [`load_pipeline_config_object()`](reference/runtime.md#cowbook.runtime.load_pipeline_config_object), [`materialize_pipeline_config()`](reference/runtime.md#cowbook.runtime.materialize_pipeline_config), [`PipelineRunner`](reference/runtime.md#cowbook.app.pipeline.PipelineRunner), [`PipelineConfig`](reference/runtime.md#cowbook.core.contracts.PipelineConfig), [`RunRequest`](reference/runtime.md#cowbook.core.contracts.RunRequest), [`RunResult`](reference/execution.md#cowbook.execution.results.RunResult), plus the [execution models and cancellation primitives](reference/execution.md) re-exported at the package root.

External callers should use the public runtime surface. Internal modules remain free to evolve unless they are explicitly promoted into that surface.
