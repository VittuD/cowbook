# main.py

import argparse
from _package_bootstrap import ensure_src_path

ensure_src_path()

from cowbook.pipeline import PipelineRunner


def main(
    config_path: str,
) -> None:
    PipelineRunner().run(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and create projection video.")
    parser.add_argument(
        "config", nargs="?", help="Path to the configuration file (positional, optional)"
    )
    parser.add_argument("--config", dest="config_opt", type=str, help="Path to the configuration file")

    args = parser.parse_args()
    config_path = args.config_opt or args.config or "config.json"
    main(
        config_path,
    )
