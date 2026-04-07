# CLI Usage

The CLI is intentionally small. It runs the engine from a config file plus a small set of practical overrides.

Supported entrypoint:

```bash
python -m cowbook
```

Most runs use one of these forms:

```bash
python -m cowbook
python -m cowbook --config configs/smoke.json
python -m cowbook configs/full.cpu.json
```

Supported overrides cover only operational concerns: frame rate, output filename, image format, plotting workers, tracking workers, projection-video creation, frame cleanup, and whether masking runs before inference.

Typical examples:

```bash
python -m cowbook --config configs/full.cpu.json --fps 12
python -m cowbook --config configs/full.cpu.json --mask-videos
python -m cowbook --config configs/full.cpu.json --no-clean-frames-after-video
```

The CLI is a thin wrapper around the package runtime surface. For direct Python use, see [Python Usage](python-usage.md).
