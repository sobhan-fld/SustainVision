"""CSV reporting utilities for training metrics and emissions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from .config import TrainingConfig
from .types import EpochMetrics


def write_report_csv(
    path: Path,
    epochs: List[EpochMetrics],
    *,
    emissions_kg: Optional[float],
    energy_kwh: Optional[float],
    duration_seconds: Optional[float],
    config: TrainingConfig,
    append: bool = False,
) -> None:
    """Write per-epoch metrics and sustainability data to CSV."""
    fieldnames = [
        "epoch",
        "phase",
        "loss_name",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "learning_rate",
        "contrastive_train_loss",
        "contrastive_val_loss",
        "classifier_train_loss",
        "classifier_val_loss",
        "emissions_kg",
        "energy_kwh",
        "duration_seconds",
        "model",
        "database",
        "device",
        "optimizer",
        "loss_function",
        "weight_decay",
        "scheduler",
        "seed",
        "temperature",
    ]

    last_lr = epochs[-1].learning_rate if epochs else None

    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    mode = "a" if append and file_exists else "w"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not (append and file_exists):
            writer.writeheader()
        for metrics in epochs:
            loss_name = metrics.loss_name or config.loss_function
            loss_mode = (metrics.loss_mode or "").lower()
            is_contrastive = loss_mode in {"simclr", "supcon"} or loss_name.lower() in {"simclr", "supcon"}
            optimizer_name = metrics.optimizer_name or config.optimizer

            row = {
                "epoch": metrics.epoch,
                "phase": metrics.phase,
                "loss_name": loss_name,
                "train_loss": f"{metrics.train_loss:.6f}",
                "train_accuracy": f"{metrics.train_accuracy:.6f}",
                "val_loss": f"{metrics.val_loss:.6f}",
                "val_accuracy": f"{metrics.val_accuracy:.6f}",
                "learning_rate": f"{metrics.learning_rate:.6f}",
                "contrastive_train_loss": "",
                "contrastive_val_loss": "",
                "classifier_train_loss": "",
                "classifier_val_loss": "",
                "emissions_kg": "",
                "energy_kwh": "",
                "duration_seconds": "",
                "model": config.model,
                "database": config.database,
                "device": config.device,
                "optimizer": optimizer_name,
                "loss_function": config.loss_function,
                "weight_decay": config.weight_decay,
                "scheduler": config.scheduler,
                "seed": config.seed,
                "temperature": config.hyperparameters.get("temperature"),
            }

            if is_contrastive:
                row["contrastive_train_loss"] = f"{metrics.train_loss:.6f}"
                row["contrastive_val_loss"] = f"{metrics.val_loss:.6f}"
            else:
                row["classifier_train_loss"] = f"{metrics.train_loss:.6f}"
                row["classifier_val_loss"] = f"{metrics.val_loss:.6f}"

            writer.writerow(row)
        
        # Only write summary row if not appending (summary written at end of all phases)
        if not append:
            summary_optimizer = config.optimizer
            if epochs:
                summary_optimizer = epochs[-1].optimizer_name or config.optimizer
            writer.writerow(
                {
                    "epoch": "summary",
                    "phase": "summary",
                    "loss_name": "",
                    "train_loss": "",
                    "train_accuracy": "",
                    "val_loss": "",
                    "val_accuracy": "",
                    "learning_rate": f"{last_lr:.6f}" if last_lr is not None else "",
                    "contrastive_train_loss": "",
                    "contrastive_val_loss": "",
                    "classifier_train_loss": "",
                    "classifier_val_loss": "",
                    "loss_function": config.loss_function,
                    "emissions_kg": f"{emissions_kg:.6f}" if emissions_kg is not None else "",
                    "energy_kwh": f"{energy_kwh:.6f}" if energy_kwh is not None else "",
                    "duration_seconds": f"{duration_seconds:.2f}" if duration_seconds is not None else "",
                    "model": config.model,
                    "database": config.database,
                    "device": config.device,
                    "optimizer": summary_optimizer,
                    "loss_function": config.loss_function,
                    "weight_decay": config.weight_decay,
                    "scheduler": config.scheduler,
                    "seed": config.seed,
                    "temperature": config.hyperparameters.get("temperature"),
                }
            )

