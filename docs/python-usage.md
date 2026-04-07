# Python Usage

The stable public package surface is `cowbook` itself, plus `cowbook.runtime` for callers that want a dedicated module path. In normal code, prefer importing from the package root:

```python
from cowbook import load_pipeline_config, run_pipeline
```

Direct synchronous run:

```python
from cowbook import run_pipeline

snapshot = run_pipeline("configs/smoke.json")
```

Validated typed config:

```python
from cowbook import load_pipeline_config

config = load_pipeline_config("configs/smoke.json", overrides={"run_name": "demo"})
```

External callers should not depend on deep internal modules unless those modules are explicitly promoted into the [public runtime surface](package-boundaries.md).
