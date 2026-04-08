from __future__ import annotations

import argparse

from cowbook.app.pipeline import PipelineRunner
from cowbook.core.runtime import ensure_repo_root_on_path

CLI_DESCRIPTION = "Process videos and create projection video."
CLI_OVERRIDE_FIELDS = (
    "fps",
    "output_video_filename",
    "output_image_format",
    "num_plot_workers",
    "tracking_concurrency",
    "log_progress",
    "create_projection_video",
    "clean_frames_after_video",
    "mask_videos",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument(
        "config", nargs="?", help="Path to the configuration file (positional, optional)"
    )
    parser.add_argument("--config", dest="config_opt", type=str, help="Path to the configuration file")
    parser.add_argument("--fps", type=int, help="Override frames-per-second for rendered video output")
    parser.add_argument(
        "--output-video-filename",
        type=str,
        help="Override the combined projection video filename",
    )
    parser.add_argument(
        "--output-image-format",
        choices=("jpg", "png"),
        help="Override the intermediate frame image format",
    )
    parser.add_argument(
        "--num-plot-workers",
        type=int,
        help="Override the number of workers used to render projection frames",
    )
    parser.add_argument(
        "--tracking-concurrency",
        dest="tracking_concurrency",
        type=int,
        help="Override how many videos may be tracked concurrently within a group",
    )
    parser.add_argument(
        "--log-progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable human-readable progress logs during long-running stages",
    )
    parser.add_argument(
        "--create-projection-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable final projection video assembly",
    )
    parser.add_argument(
        "--clean-frames-after-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable cleanup of intermediate rendered frames",
    )
    parser.add_argument(
        "--mask-videos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable preprocessing videos with masks before inference",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_config_path(args: argparse.Namespace) -> str:
    return args.config_opt or args.config or "config.json"


def resolve_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for field_name in CLI_OVERRIDE_FIELDS:
        value = getattr(args, field_name, None)
        if value is not None:
            overrides[field_name] = value
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve_config_path(args)
    overrides = resolve_overrides(args)
    ensure_repo_root_on_path()
    PipelineRunner().run(config_path, overrides=overrides)
    return 0


def entrypoint() -> None:
    raise SystemExit(main())
