"""CLI entry point for training and tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config.training_config import TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nMELTS training CLI")
    parser.add_argument("command", choices=["train", "tune"], help="Action to perform")
    parser.add_argument("--config", required=True, help="Path to training.yaml")
    parser.add_argument("--stage", required=True, help="Training stage name")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    defaults_path = Path(__file__).parent / "config" / "defaults.yaml"
    config = TrainingConfig.from_yaml(args.config, str(defaults_path))
    print(f"Loaded config for stage '{args.stage}' with command '{args.command}'.")
    print("TODO: wire training and tuning entry points.")


if __name__ == "__main__":
    main()
