# Python Usage

The stable public package surface is `cowbook` itself, plus [`cowbook.runtime`](reference/runtime.md#cowbook.runtime) for callers that want a dedicated module path. In normal code, prefer importing from the package root:

```python
from cowbook import (
    RunRequest,
    load_pipeline_config,
    load_pipeline_config_object,
    materialize_pipeline_config,
    run_pipeline,
    run_pipeline_request,
)
```

Direct synchronous run:

```python
from cowbook import run_pipeline

result = run_pipeline("configs/smoke.json")
job_run = result.job_run
```

Validated typed config from a JSON file:

```python
from cowbook import load_pipeline_config

config = load_pipeline_config("configs/smoke.json", overrides={"run_name": "demo"})
```

`load_pipeline_config(...)` raises `FileNotFoundError` for missing files, `json.JSONDecodeError` for invalid JSON, and `ValueError` for invalid runtime config values.

Validated typed config from an in-memory object:

```python
from cowbook import load_pipeline_config_object

config = load_pipeline_config_object(
    {
        "model_path": "models/best.pt",
        "video_groups": [[{"path": "sample_data/videos/Ch1_60.mp4", "camera_nr": 1}]],
    },
    overrides={"run_name": "demo"},
)
```

`load_pipeline_config_object(...)` applies the same runtime validation rules and does not mutate the input object.

At execution time, `tracking_concurrency=1` uses the same tracking/reporting contract as the multi-process path, but runs tracking inline to avoid single-worker process overhead. Within that inline path, Cowbook may reuse one YOLO model instance per `(model_path, tracking mode)` during a group run. Direct tracking and cleanup tracking stay isolated from each other so tracker state does not bleed across modes.

Typed programmatic submission:

```python
from cowbook import RunRequest, run_pipeline_request

request = RunRequest(
    config={
        "model_path": "models/best.pt",
        "video_groups": [[{"path": "sample_data/videos/Ch1_60.mp4", "camera_nr": 1}]],
    },
    overrides={"run_name": "demo"},
)

snapshot = run_pipeline_request(request)
tracking_jsons = snapshot.tracking_json_paths
```

Materialize a normalized config for reproducible reruns:

```python
from cowbook import materialize_pipeline_config

path = materialize_pipeline_config(config, "var/tmp/materialized.json")
```

The request-based runtime entrypoints return [`RunResult`](reference/execution.md#cowbook.execution.results.RunResult), which wraps the underlying `JobRun` plus a normalized artifact summary.

`PipelineRunner` also supports the same shapes through [`run()`](reference/runtime.md#cowbook.app.pipeline.PipelineRunner.run), [`run_config()`](reference/runtime.md#cowbook.app.pipeline.PipelineRunner.run_config), and [`run_request()`](reference/runtime.md#cowbook.app.pipeline.PipelineRunner.run_request).

External callers should not depend on deep internal modules unless those modules are explicitly promoted into the [public runtime surface](package-boundaries.md).
