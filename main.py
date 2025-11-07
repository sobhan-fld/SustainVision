"""SustainVision entrypoint with training menu."""

from __future__ import annotations

from pathlib import Path

import questionary

from sustainvision.config import ConfigManager
from sustainvision.data import prompt_and_download
from sustainvision.tui import run_config_tui
from sustainvision.training import MissingDependencyError, TrainingRunSummary, train_model


def _start_training(cm: ConfigManager) -> None:
    """Kick off a training run and display summary output."""

    config = cm.load()
    try:
        summary: TrainingRunSummary = train_model(config, project_root=Path.cwd())
    except MissingDependencyError as exc:
        print(f"\n[error] {exc}\n")
        return
    except Exception as exc:  # pragma: no cover - runtime protection
        print(f"\n[error] Training failed: {exc}\n")
        return

    print("\nTraining summary:")
    print(f"- report: {summary.report_path}")
    if summary.emissions_kg is not None:
        print(f"- emissions (kg CO₂e): {summary.emissions_kg:.6f}")
    if summary.energy_kwh is not None:
        print(f"- energy (kWh): {summary.energy_kwh:.6f}")
    if summary.duration_seconds is not None:
        print(f"- duration (s): {summary.duration_seconds:.2f}")


def main() -> None:
    """Interactive menu for configuring and training SustainVision pipelines."""

    cm = ConfigManager()
    cm.load()

    while True:
        choice = questionary.select(
            "Select an action:",
            choices=[
                "Start training",
                "Configure settings",
                "Download databases",
                "Show current config",
                "Exit",
            ],
            default="Start training",
        ).ask()

        if choice == "Configure settings":
            run_config_tui(cm.path)
            cm.load()  # refresh in-memory config after edits
        elif choice == "Start training":
            _start_training(cm)
        elif choice == "Download databases":
            dataset_path = prompt_and_download(Path.cwd())
            if dataset_path is not None:
                try:
                    relative_path = dataset_path.relative_to(Path.cwd())
                except ValueError:
                    relative_path = dataset_path
                cm.set(database=str(relative_path))
                cm.save()
                print("Configuration updated with new dataset path.")
        elif choice == "Show current config":
            cfg = cm.load()
            print("\nCurrent configuration:")
            print(cfg)
        elif choice == "Exit" or choice is None:
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()