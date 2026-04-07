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

`cowbook.runtime` is the public entrypoint. `cowbook.app` coordinates a run, `cowbook.workflows` handles group-level flow, `cowbook.vision` owns tracking and projection, `cowbook.io` owns file-based inputs and outputs, `cowbook.core` provides contracts and pure transforms, and `cowbook.execution` carries structured run state and events.

## Internal Layout

The internal package layout follows responsibilities rather than scripts: [`cowbook.runtime`](package-boundaries.md) is the public runtime surface, `cowbook.app` contains orchestration entrypoints, [`cowbook.execution`](job-execution.md) handles jobs and progress, `cowbook.core` holds typed contracts and transforms, `cowbook.io` handles file-based IO, `cowbook.vision` contains the vision-specific code, and `cowbook.workflows` ties those parts together at the group level.

## Design Rule

The public runtime surface should remain small and explicit, and the engine should continue to run against explicit runtime artifacts. That keeps runs reproducible and the CLI independent.
