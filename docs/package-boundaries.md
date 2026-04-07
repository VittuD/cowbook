# Package Boundaries

There are two supported entry routes: the CLI with `python -m cowbook --config ...`, and the Python package imported from `cowbook` or [`cowbook.runtime`](reference/runtime.md#cowbook.runtime).

For Python callers, the stable surface is the [runtime layer](python-usage.md): [`run_pipeline()`](reference/runtime.md#cowbook.runtime.run_pipeline), [`load_pipeline_config()`](reference/runtime.md#cowbook.runtime.load_pipeline_config), [`PipelineRunner`](reference/runtime.md#cowbook.app.pipeline.PipelineRunner), [`PipelineConfig`](reference/runtime.md#cowbook.core.contracts.PipelineConfig), plus the [execution models and cancellation primitives](reference/execution.md) re-exported at the package root.

External callers should use the public runtime surface. Internal modules remain free to evolve unless they are explicitly promoted into that surface.
