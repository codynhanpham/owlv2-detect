import argparse
import logging
import json
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from commands.common import BaseConfig


KNOWN_SUBCOMMANDS = {"general", "black-marks"}


def build_parser() -> argparse.ArgumentParser:
    from commands.black_marks import register_parser as register_black_marks_parser
    from commands.common import SortedHelpFormatter
    from commands.general import register_parser as register_general_parser

    examples = """Examples:
    Default general detection:
        owlv2-detect -i /path/to/image.png -t "human" "cat" "car"
        Output: [{"file": "...", "objects": [{"label": "human", "bbox": [x1, y1, x2, y2], "score": 0.98}]}]

    Explicit general subcommand:
        owlv2-detect general -i /path/to/image.png -t "human" "cat" "car"
        Output: [{"file": "...", "objects": [{"label": "human", "bbox": [x1, y1, x2, y2], "score": 0.98}]}]

    General detection help:
        owlv2-detect general -h

    Black-marks detection pipeline:
        owlv2-detect black-marks -i /path/to/file.mp4
        Output: [{"file": "...", "markers": [[x, y], ...]}]
"""
    parser = argparse.ArgumentParser(
        description="Run OWLv2 zero-shot object detection, with optional specialized pipelines via subcommands.",
        epilog=examples,
        formatter_class=SortedHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", title="pipelines")
    register_general_parser(subparsers)
    register_black_marks_parser(subparsers)

    return parser


def normalize_cli_args(argv: list[str]) -> list[str]:
    if not argv:
        return ["general"]
    if argv[0] in KNOWN_SUBCOMMANDS or argv[0] in {"-h", "--help"}:
        return argv
    return ["general", *argv]


def _log_active_arguments(namespace: argparse.Namespace, config: "BaseConfig", show_progress: bool, save_preprocessed_debug: bool) -> None:
    logging_payload = {
        "subcommand": namespace.subcommand,
        "show_progress": show_progress,
        "save_preprocessed_debug": save_preprocessed_debug,
        "config": asdict(config),
    }
    logging.debug("Active arguments:\n%s", json.dumps(logging_payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    from commands.common import parse_adjustments

    argv = normalize_cli_args(sys.argv[1:])
    namespace = parser.parse_args(argv)

    if namespace.subcommand is None:
        namespace.subcommand = "general"
    elif namespace.subcommand not in {"general", "black-marks"}:
        parser.error("A valid subcommand is required")

    crop = tuple(float(value) for value in namespace.crop)
    try:
        namespace.filter = parse_adjustments(namespace.filter)
    except ValueError as exc:
        parser.error(str(exc))

    if namespace.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")
    if namespace.detection_threshold < 0:
        parser.error("--detection-threshold must be >= 0")
    if any(value < 0 for value in crop):
        parser.error("--crop values must be >= 0")
    if crop[0] + crop[2] >= 1 or crop[1] + crop[3] >= 1:
        parser.error("--crop values remove the whole image; top+bottom and right+left must each be < 1")

    if namespace.subcommand == "black-marks":
        duration_range: tuple[float, float] = (
            float(namespace.duration_range[0]),
            float(namespace.duration_range[1]),
        )
        if namespace.sample_frame_count <= 0:
            parser.error("--sample-frame-count must be greater than 0")
        if duration_range[0] < 0 or duration_range[1] > 1:
            parser.error("--duration-range values must be between 0 and 1")
        if duration_range[0] >= duration_range[1]:
            parser.error("--duration-range start must be less than end")
        if namespace.top_n_marks <= 0:
            parser.error("--top-n-marks must be greater than 0")
    return namespace


def cli() -> int:
    from commands.black_marks import build_black_marks_config, run_black_marks
    from commands.common import configure_runtime
    from commands.general import build_general_config, run_general

    namespace = parse_args()
    show_progress, save_preprocessed_debug = configure_runtime(namespace.log_level)

    if namespace.subcommand == "black-marks":
        black_marks_config = build_black_marks_config(namespace)
        if namespace.log_level == "debug":
            _log_active_arguments(namespace, black_marks_config, show_progress, save_preprocessed_debug)
        results = run_black_marks(black_marks_config, show_progress, save_preprocessed_debug)
    else:
        general_config = build_general_config(namespace)
        if namespace.log_level == "debug":
            _log_active_arguments(namespace, general_config, show_progress, save_preprocessed_debug)
        results = run_general(general_config, show_progress, save_preprocessed_debug)

    print()
    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())