#!/usr/bin/env python3
"""Utility to launch SustainVision training with a specific config file."""

from __future__ import annotations

import argparse
from pathlib import Path

from sustainvision.config import ConfigManager
from sustainvision.training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SustainVision training with a given YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the YAML config file (defaults to ~/.config/sustainvision/config.yaml).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("outputs"),
        help="Directory where reports/checkpoints should be written (default: ./outputs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if config_path is not None:
        cm = ConfigManager(config_path)
    else:
        cm = ConfigManager()
    config = cm.load()

    project_root = args.project_root
    if not project_root.is_absolute():
        project_root = (Path.cwd() / project_root).resolve()
    project_root.mkdir(parents=True, exist_ok=True)

    summary = train_model(config, project_root=project_root)
    print("\n[run summary]")
    print(f"- report: {summary.report_path}")
    if summary.emissions_kg is not None:
        print(f"- emissions (kg CO₂e): {summary.emissions_kg:.6f}")
    if summary.energy_kwh is not None:
        print(f"- energy (kWh): {summary.energy_kwh:.6f}")
    if summary.duration_seconds is not None:
        print(f"- duration (s): {summary.duration_seconds:.2f}")


if __name__ == "__main__":
    main()

