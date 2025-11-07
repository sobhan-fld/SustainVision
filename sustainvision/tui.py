"""Text-based user interface (TUI) for configuring SustainVision.

This TUI lets users review and edit the configuration and saves it to disk.
It relies on the ConfigManager to handle reading/writing YAML and device detection.
"""

from __future__ import annotations

from typing import Any, Dict, List

import questionary

from .config import ConfigManager, TrainingConfig


def _ask_model(current: str) -> str:
    """Prompt the user to choose (or enter) a model identifier."""
    presets = [
        "resnet18",
        "resnet34",
        "vit-base",
        "bert-base-uncased",
        "custom",
    ]
    choice = questionary.select(
        "Choose model:",
        choices=[questionary.Choice(title=m, value=m) for m in presets]
        + [questionary.Choice(title="Other...", value="__other__")],
        default=current if current in presets else "custom",
    ).ask()

    if choice in ("custom", "__other__"):
        # Allow arbitrary model identifiers
        return questionary.text("Enter model name:", default=current).ask()
    return choice


def _ask_database(current: str) -> str:
    """Prompt the user to choose (or enter) a dataset/path."""
    presets = ["imagenet", "cifar10", "custom_path"]
    choice = questionary.select(
        "Choose database:",
        choices=[questionary.Choice(title=d, value=d) for d in presets]
        + [questionary.Choice(title="Other...", value="__other__")],
        default=current if current in presets else "custom_path",
    ).ask()

    if choice in ("custom_path", "__other__"):
        # Accept any filesystem path or dataset alias
        return questionary.text("Enter dataset path or name:", default=current).ask()
    return choice


def _ask_device(current: str, devices: List[str]) -> str:
    """Prompt the user to select a device from detected devices or current value."""
    # Ensure current device is selectable even if not detected
    if current not in devices:
        devices = devices + [current]
    return questionary.select("Choose device:", choices=devices, default=current).ask()


def _ask_hyperparameters(hp: Dict[str, Any]) -> Dict[str, Any]:
    """Prompt the user for basic hyperparameters with safe type conversion."""
    batch_size = questionary.text("batch_size:", default=str(hp.get("batch_size", 32))).ask()
    lr = questionary.text("learning rate:", default=str(hp.get("lr", 1e-3))).ask()
    epochs = questionary.text("epochs:", default=str(hp.get("epochs", 3))).ask()

    def as_int(value: str, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def as_float(value: str, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    return {
        "batch_size": as_int(batch_size, hp.get("batch_size", 32)),
        "lr": as_float(lr, hp.get("lr", 1e-3)),
        "epochs": as_int(epochs, hp.get("epochs", 3)),
    }


def run_config_tui(config_path: str | None = None) -> TrainingConfig:
    """Run the interactive configuration flow and persist the result.

    Returns the updated configuration (as a dataclass instance).
    """
    cm = ConfigManager(config_path)
    cfg = cm.load()

    # Present current configuration for quick review
    print("\nCurrent config:")
    print(f"- model: {cfg.model}")
    print(f"- database: {cfg.database}")
    print(f"- device: {cfg.device}")
    print(f"- hyperparameters: {cfg.hyperparameters}")

    action = questionary.select(
        "What would you like to do?",
        choices=["Edit and Save", "Keep and Exit"],
        default="Edit and Save",
    ).ask()

    if action == "Keep and Exit":
        return cfg

    new_model = _ask_model(cfg.model)
    new_db = _ask_database(cfg.database)
    new_device = _ask_device(cfg.device, cm.available_devices)
    new_hp = _ask_hyperparameters(cfg.hyperparameters)

    cm.set(model=new_model, database=new_db, device=new_device, hyperparameters=new_hp)
    cm.save()
    print("\nConfig saved.")
    return cm.config


