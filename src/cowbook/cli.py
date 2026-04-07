from __future__ import annotations

import argparse

from cowbook.pipeline import PipelineRunner
from cowbook.runtime import ensure_repo_root_on_path

CLI_DESCRIPTION = "Process videos and create projection video."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument(
        "config", nargs="?", help="Path to the configuration file (positional, optional)"
    )
    parser.add_argument("--config", dest="config_opt", type=str, help="Path to the configuration file")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_config_path(args: argparse.Namespace) -> str:
    return args.config_opt or args.config or "config.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve_config_path(args)
    ensure_repo_root_on_path()
    PipelineRunner().run(config_path)
    return 0


def entrypoint() -> None:
    raise SystemExit(main())
