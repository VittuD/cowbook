# Architecture

`cowbook` is the engine layer for tracking, [projection](calibration.md), rendering, and run orchestration.

It takes explicit runtime inputs and produces explicit runtime outputs: config loading, run-folder preparation, tracking and projection workflows, exported artifacts, and structured execution state.

It deliberately stops at the package boundary. Long-lived application concerns are outside the scope of this repository.

## Package Map

```text
config file / Python call / CLI
              |
              v
      cowbook.runtime
              |
              v
        cowbook.app
              |
    +---------+---------+
    |                   |
    v                   v
cowbook.execution   cowbook.workflows
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
     cowbook.io   cowbook.vision  cowbook.core
          |             |             |
          +------+------+-------------+
                 |
                 v
         run artifacts under var/
```

The package map is meant to be read top-down: [`cowbook.runtime`](reference/runtime.md#cowbook.runtime) is the public entrypoint, [`cowbook.app`](reference/runtime.md#cowbook.app.pipeline.PipelineRunner) drives a run, [`cowbook.workflows`](pipeline.md) connects the group-level stages, and the lower layers provide the execution, IO, vision, and contract machinery those stages depend on.

## Internal Layout

The internal package layout follows responsibilities rather than scripts:

- [`cowbook.runtime`](reference/runtime.md#cowbook.runtime): public package surface and stable imports
- [`cowbook.app`](reference/runtime.md#cowbook.app.pipeline.PipelineRunner): synchronous orchestration centered on [`PipelineRunner`](reference/runtime.md#cowbook.app.pipeline.PipelineRunner), plus CLI glue and thin service adapters
- [`cowbook.execution`](reference/execution.md): structured run state, observers, shared progress reporting, and [execution reference](reference/execution.md)
- [`cowbook.core`](reference/runtime.md#cowbook.core.contracts.PipelineConfig): typed contracts, including [`PipelineConfig`](reference/runtime.md#cowbook.core.contracts.PipelineConfig), plus shared transforms
- `cowbook.io`: [config loading](configuration.md), file-based inputs, and [runtime artifacts](runtime-artifacts.md)
- `cowbook.vision`: tracking, [projection](calibration.md), and rendering
- `cowbook.workflows`: [group-level pipeline flow](pipeline.md)

Top-level `tools/` and `scripts/` remain outside the package boundary. They are repository utilities, not part of the supported runtime surface.

## Design Rule

The public runtime surface should remain small and explicit, and the engine should continue to run against explicit runtime artifacts. That keeps runs reproducible and the CLI independent.
