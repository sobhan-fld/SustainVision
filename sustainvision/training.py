"""Training utilities for SustainVision.

This module provides a lightweight reference training loop that:

- Uses PyTorch (if available) to train a simple classifier on a synthetic dataset
  derived from the current configuration. The implementation is intentionally
  minimal so it can be swapped out for a real project-specific training routine.
- Wraps the run in a CodeCarbon tracker to estimate energy usage and emissions.
- Logs per-epoch metrics and final sustainability data to a CSV file in the
  project root. If a file with the desired name already exists, an index is
  appended to avoid overwriting previous runs.

It is meant to be a template: replace the dataset/model sections with the logic
that matches your project once you connect SustainVision to real workloads.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import TrainingConfig


try:  # Optional imports handled gracefully in `_ensure_dependencies`
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore

try:
    from codecarbon import EmissionsTracker
except Exception:  # pragma: no cover - handled at runtime
    EmissionsTracker = None  # type: ignore


@dataclass
class EpochMetrics:
    """Container for per-epoch training/validation metrics."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


@dataclass
class TrainingRunSummary:
    """Summary of a training run, including emissions information."""

    report_path: Path
    emissions_kg: Optional[float]
    energy_kwh: Optional[float]
    duration_seconds: Optional[float]
    epochs: List[EpochMetrics]


class MissingDependencyError(RuntimeError):
    """Raised when optional training dependencies are not installed."""


def _ensure_dependencies() -> None:
    """Ensure that PyTorch and CodeCarbon are available before training."""

    if torch is None or nn is None or DataLoader is None:
        raise MissingDependencyError(
            "PyTorch is required for training. Install it or adjust the "
            "training implementation to match your stack."
        )
    if EmissionsTracker is None:
        raise MissingDependencyError(
            "CodeCarbon is required to record emissions. Install it with "
            "`pip install codecarbon`."
        )


def _resolve_device(preferred: str) -> "torch.device":  # type: ignore[name-defined]
    """Resolve the requested device, falling back to CPU if unavailable."""

    assert torch is not None  # for type checkers
    if preferred == "cpu" or not torch.cuda.is_available():
        if preferred != "cpu":
            print("[warn] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    try:
        device = torch.device(preferred)
        _ = torch.zeros(1, device=device)  # probe availability
        return device
    except Exception:
        print(f"[warn] Device '{preferred}' unavailable. Using CPU instead.")
        return torch.device("cpu")


def _prepare_synthetic_dataset(
    *,
    num_samples: int = 2000,
    input_dim: int = 32,
    num_classes: int = 4,
) -> "tuple[TensorDataset, TensorDataset]":  # type: ignore[name-defined]
    """Create a reproducible synthetic classification dataset."""

    assert torch is not None and TensorDataset is not None
    generator = torch.Generator().manual_seed(42)
    features = torch.randn(num_samples, input_dim, generator=generator)

    # Construct deterministic labels by applying a fixed linear transform
    weights = torch.randn(input_dim, num_classes, generator=generator)
    logits = features @ weights
    targets = torch.argmax(logits, dim=1)

    split_idx = int(num_samples * 0.8)
    train_ds = TensorDataset(features[:split_idx], targets[:split_idx])
    val_ds = TensorDataset(features[split_idx:], targets[split_idx:])
    return train_ds, val_ds


def _unique_report_path(base_dir: Path, desired_name: str) -> Path:
    """Return a unique path for the report file, avoiding overwrites."""

    sanitized = desired_name.strip() or "training_report.csv"
    target = base_dir / sanitized
    if target.suffix.lower() != ".csv":
        target = target.with_suffix(".csv")

    counter = 1
    unique_target = target
    while unique_target.exists():
        unique_target = target.with_stem(f"{target.stem}_{counter}")
        counter += 1
    return unique_target


def train_model(
    config: TrainingConfig,
    *,
    project_root: Optional[Path] = None,
) -> TrainingRunSummary:
    """Train a toy model using the provided configuration.

    Returns
    -------
    TrainingRunSummary
        Includes the path to the CSV report, emissions information, and per-epoch metrics.

    Notes
    -----
    - Replace the synthetic dataset/model with your real training pipeline.
    - The CSV report contains one row per epoch plus a summary row with emissions data.
    """

    _ensure_dependencies()
    assert torch is not None and nn is not None and DataLoader is not None

    project_dir = project_root or Path.cwd()
    project_dir.mkdir(parents=True, exist_ok=True)
    report_path = _unique_report_path(project_dir, config.report_filename)

    device = _resolve_device(config.device)

    batch_size = int(config.hyperparameters.get("batch_size", 32))
    learning_rate = float(config.hyperparameters.get("lr", 1e-3))
    epochs = int(config.hyperparameters.get("epochs", 3))

    train_ds, val_ds = _prepare_synthetic_dataset()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = nn.Sequential(
        nn.Linear(train_ds.tensors[0].shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, len(torch.unique(train_ds.tensors[1]))),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    tracker = EmissionsTracker(measure_power_secs=1)  # type: ignore[call-arg]
    tracker.start()

    epoch_metrics: List[EpochMetrics] = []

    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = epoch_loss / total
        train_acc = correct / total if total else 0.0

        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss_accum += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_accum / val_total if val_total else 0.0
        val_acc = val_correct / val_total if val_total else 0.0

        epoch_metrics.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
        )

        print(
            f"Epoch {epoch:02d}/{epochs} - "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    emissions_kg = tracker.stop()
    emissions_data = getattr(tracker, "final_emissions_data", None)

    duration = time.perf_counter() - start_time
    energy_kwh = None
    if emissions_data is not None:
        energy_kwh = emissions_data.energy_consumed
        if emissions_data.duration is not None:
            duration = emissions_data.duration

    _write_report_csv(
        report_path,
        epoch_metrics,
        emissions_kg=emissions_kg,
        energy_kwh=energy_kwh,
        duration_seconds=duration,
        config=config,
    )

    print(f"\nTraining complete. Report saved to {report_path}")

    return TrainingRunSummary(
        report_path=report_path,
        emissions_kg=emissions_kg,
        energy_kwh=energy_kwh,
        duration_seconds=duration,
        epochs=epoch_metrics,
    )


def _write_report_csv(
    path: Path,
    epochs: List[EpochMetrics],
    *,
    emissions_kg: Optional[float],
    energy_kwh: Optional[float],
    duration_seconds: Optional[float],
    config: TrainingConfig,
) -> None:
    """Write per-epoch metrics and sustainability data to CSV."""

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "emissions_kg",
        "energy_kwh",
        "duration_seconds",
        "model",
        "database",
        "device",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in epochs:
            writer.writerow(
                {
                    "epoch": metrics.epoch,
                    "train_loss": f"{metrics.train_loss:.6f}",
                    "train_accuracy": f"{metrics.train_accuracy:.6f}",
                    "val_loss": f"{metrics.val_loss:.6f}",
                    "val_accuracy": f"{metrics.val_accuracy:.6f}",
                    "emissions_kg": "",
                    "energy_kwh": "",
                    "duration_seconds": "",
                    "model": config.model,
                    "database": config.database,
                    "device": config.device,
                }
            )

        writer.writerow(
            {
                "epoch": "summary",
                "train_loss": "",
                "train_accuracy": "",
                "val_loss": "",
                "val_accuracy": "",
                "emissions_kg": f"{emissions_kg:.6f}" if emissions_kg is not None else "",
                "energy_kwh": f"{energy_kwh:.6f}" if energy_kwh is not None else "",
                "duration_seconds": f"{duration_seconds:.2f}" if duration_seconds is not None else "",
                "model": config.model,
                "database": config.database,
                "device": config.device,
            }
        )


