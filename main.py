# main.py

from _package_bootstrap import ensure_src_path

ensure_src_path()

from cowbook.cli import main as cli_main
from cowbook.pipeline import PipelineRunner


def main(
    config_path: str,
    overrides: dict[str, object] | None = None,
) -> None:
    PipelineRunner().run(config_path, overrides=overrides)


if __name__ == "__main__":
    raise SystemExit(cli_main())
