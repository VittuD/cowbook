from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


CLI_DESCRIPTION = "Process videos and create projection video."


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_repo_root_on_path() -> Path:
    root = repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


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


def load_legacy_main_module():
    ensure_repo_root_on_path()
    return importlib.import_module("main")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve_config_path(args)
    legacy_main = load_legacy_main_module()
    legacy_main.main(config_path)
    return 0


def entrypoint() -> None:
    raise SystemExit(main())
