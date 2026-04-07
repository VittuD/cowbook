# Getting Started

Runtime install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Development checks only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Documentation tooling only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[docs]"
```

Full contributor install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

Run the default local config:

```bash
python -m cowbook
```

Or choose an explicit example config:

```bash
python -m cowbook --config configs/smoke.json
python -m cowbook configs/full.cpu.json
```

Serve the documentation locally with MkDocs:

```bash
mkdocs serve
```

Build the static site:

```bash
mkdocs build
```
