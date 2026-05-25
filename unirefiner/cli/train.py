"""Command-line entrypoint for UniRefiner training."""

from __future__ import annotations

import argparse
import sys

from unirefiner.config import dump_config, load_config
from unirefiner.training import run_training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UniRefiner with a structured config.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override experiment.output_dir.")
    parser.add_argument("--resume", default=None, help="Override runtime.resume.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a nested config field, e.g. optimizer.lr=5e-5",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = list(args.override)
    if args.output_dir:
        overrides.append(f"experiment.output_dir={args.output_dir}")
    if args.resume:
        overrides.append(f"runtime.resume={args.resume}")

    config = load_config(args.config, overrides=overrides)
    dump_config(config, f"{config.experiment.output_dir}/{config.experiment.name}/resolved_config.yaml")
    run_training(config)


if __name__ == "__main__":
    main(sys.argv[1:])
