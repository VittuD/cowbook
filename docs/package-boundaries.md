# Package Boundaries

There are two supported entry routes: the CLI with `python -m cowbook --config ...`, and the Python package imported from `cowbook` or `cowbook.runtime`.

For Python callers, the stable surface is the [runtime layer](python-usage.md): `run_pipeline()`, `load_pipeline_config()`, `PipelineRunner`, `PipelineConfig`, plus the [execution models](job-execution.md) and cancellation types re-exported at the package root.

External callers should use the public runtime surface. Internal modules remain free to evolve unless they are explicitly promoted into that surface.
