# Runtime Artifacts

By default, each run lives under:

```text
var/runs/<run_name>/
├── frames/
├── json/
└── videos/
```

Within that tree, you will usually see raw tracking JSON, [processed JSON](pipeline.md), merged group-level JSON, CSV exports, rendered frame images, and the final projection video. Masked-video preprocessing uses a separate cache under `var/cache/masked_videos`.

This directory layout defines the run output structure. Runs execute from explicit config and produce explicit artifacts under this tree.

The cleanup benchmark image follows the same artifact pattern, but its current defaults point at `/scratch/vet/var/...` on the target GPU machine instead of the repo-local `var/` tree.
