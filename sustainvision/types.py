"""Shared type definitions for SustainVision training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class EpochMetrics:
    """Container for per-epoch training/validation metrics."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    phase: str = "train"
    loss_name: str = ""
    loss_mode: str = ""
    optimizer_name: str = ""
    weight_decay: Optional[float] = None
    projector_hidden_dim: Optional[int] = None


@dataclass
class TrainingRunSummary:
    """Summary of a training run, including emissions information."""

    report_path: Path
    emissions_kg: Optional[float]
    energy_kwh: Optional[float]
    duration_seconds: Optional[float]
    epochs: List[EpochMetrics]

