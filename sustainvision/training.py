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

import copy
import csv
import random
import time
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import TrainingConfig
from .data import DatasetPreparationError, build_classification_dataloaders


try:  # Optional imports handled gracefully in `_ensure_dependencies`
    import torch
    from torch import nn
    from torch.cuda import amp
    import torch.nn.functional as F
    from torchvision import models as tv_models
except Exception:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore
    nn = None  # type: ignore
    amp = None  # type: ignore
    F = None  # type: ignore
    tv_models = None  # type: ignore

try:
    from codecarbon import EmissionsTracker
except Exception:  # pragma: no cover - handled at runtime
    EmissionsTracker = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover - runtime safeguard
    np = None  # type: ignore


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


@dataclass
class LossSpec:
    """Description of the selected loss function."""

    name: str
    mode: str  # 'logits', 'simclr', 'supcon'
    criterion: Optional[Any] = None


class MissingDependencyError(RuntimeError):
    """Raised when optional training dependencies are not installed."""


def _ensure_dependencies() -> None:
    """Ensure that PyTorch and CodeCarbon are available before training."""

    if torch is None or nn is None:
        raise MissingDependencyError(
            "PyTorch is required for training. Install it or adjust the "
            "training implementation to match your stack."
        )
    if EmissionsTracker is None:
        raise MissingDependencyError(
            "CodeCarbon is required to record emissions. Install it with "
            "`pip install codecarbon`."
        )


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


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
    assert torch is not None and nn is not None

    _set_seed(config.seed)

    project_dir = project_root or Path.cwd()
    project_dir.mkdir(parents=True, exist_ok=True)
    report_path = _unique_report_path(project_dir, config.report_filename)

    device = _resolve_device(config.device)

    batch_size = int(config.hyperparameters.get("batch_size", 32))
    learning_rate = float(config.hyperparameters.get("lr", 1e-3))
    epochs = int(config.hyperparameters.get("epochs", 3))
    momentum = float(config.hyperparameters.get("momentum", 0.9))
    temperature = float(config.hyperparameters.get("temperature", 0.1))
    num_workers = int(config.hyperparameters.get("num_workers", 2))
    val_split = float(config.hyperparameters.get("val_split", 0.1))
    image_size = int(config.hyperparameters.get("image_size", 224))
    projection_dim = int(config.hyperparameters.get("projection_dim", 128))

    loss_spec = _build_loss(config.loss_function)
    contrastive_mode = loss_spec.mode != "logits"

    try:
        train_loader, val_loader, num_classes = build_classification_dataloaders(
            config.database,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            seed=config.seed,
            project_root=project_dir,
            image_size=image_size,
            contrastive=contrastive_mode,
        )
    except DatasetPreparationError as exc:
        raise MissingDependencyError(str(exc)) from exc

    model = _build_model(
        config.model,
        num_classes=num_classes,
        image_size=image_size,
        projection_dim=projection_dim,
    ).to(device)
    try:
        model_device = next(model.parameters()).device
        print(f"[info] Training on device: {model_device}")
    except StopIteration:
        print("[info] Training on device: unknown (model has no parameters)")

    optimizer = _build_optimizer(
        config.optimizer,
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
        momentum=momentum,
    )
    scheduler = _build_scheduler(config.scheduler, optimizer)

    use_amp = bool(config.mixed_precision) and device.type == "cuda" and amp is not None
    scaler = amp.GradScaler(enabled=use_amp) if amp is not None else None

    logging.getLogger("codecarbon").setLevel(logging.WARNING)
    tracker = EmissionsTracker(
        measure_power_secs=5,
        project_name=report_path.stem,
        log_level="warning",
    )  # type: ignore[call-arg]
    tracker.start()

    epoch_metrics: List[EpochMetrics] = []
    emissions_kg: Optional[float] = None
    emissions_data: Any = None

    start_time = time.perf_counter()

    early_cfg = config.early_stopping or {}
    early_enabled = bool(early_cfg.get("enabled", False))
    raw_patience = early_cfg.get("patience", 5)
    try:
        early_patience = max(1, int(raw_patience))
    except (TypeError, ValueError):
        early_patience = 5
    early_metric = str(early_cfg.get("metric", "val_loss")).lower()
    early_mode = str(early_cfg.get("mode", "min")).lower()
    if early_metric not in {"val_loss", "val_accuracy", "train_loss", "train_accuracy"}:
        early_metric = "val_loss"
    if early_mode not in {"min", "max"}:
        early_mode = "min"
    best_metric: Optional[float] = None
    patience_counter = 0
    best_state: Optional[Dict[str, Any]] = None
    early_stopped = False
    best_epoch = 0

    try:
        epoch_iterable = range(1, epochs + 1)
        use_tqdm = False
        try:
            from tqdm.auto import tqdm
            epoch_iterable = tqdm(epoch_iterable, desc="Epochs", unit="epoch")
            use_tqdm = True
        except Exception:
            print("[warn] tqdm is not installed. Using range instead.")
            pass

        for epoch in epoch_iterable:
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            batch_iterable = train_loader
            if use_tqdm:
                from tqdm.auto import tqdm  # type: ignore

                batch_iterable = tqdm(
                    train_loader,
                    desc=f"Train {epoch:02d}/{epochs}",
                    leave=False,
                    unit="batch",
                )

            for inputs, labels in batch_iterable:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if use_amp and scaler is not None:
                    with amp.autocast():  # type: ignore[attr-defined]
                        logits, embeddings = model(inputs)
                        loss = _compute_loss(loss_spec, logits, embeddings, labels, temperature)
                    scaler.scale(loss).backward()
                    if config.gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, embeddings = model(inputs)
                    loss = _compute_loss(loss_spec, logits, embeddings, labels, temperature)
                    loss.backward()
                    if config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            if use_tqdm:
                try:
                    batch_iterable.close()  # type: ignore[attr-defined]
                except Exception:
                    pass

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
                    logits, embeddings = model(inputs)
                    loss = _compute_loss(loss_spec, logits, embeddings, labels, temperature)
                    val_loss_accum += loss.item() * inputs.size(0)
                    preds = logits.argmax(dim=1)
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

            message = (
                f"Epoch {epoch:02d}/{epochs} - "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            if use_tqdm and hasattr(epoch_iterable, "write"):
                epoch_iterable.write(message)
            else:
                print(message)

            metrics_map = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
            current_metric = metrics_map.get(early_metric, val_loss)
            if early_enabled:
                improved = False
                if best_metric is None:
                    improved = True
                elif early_mode == "min":
                    improved = current_metric < best_metric - 1e-6
                else:
                    improved = current_metric > best_metric + 1e-6

                if improved:
                    best_metric = current_metric
                    patience_counter = 0
                    best_state = {
                        "model": copy.deepcopy(model.state_dict()),
                        "optimizer": copy.deepcopy(optimizer.state_dict()),
                    }
                    best_epoch = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= early_patience:
                        early_stopped = True
                        notice = (
                            f"[info] Early stopping triggered after {epoch} epochs "
                            f"(no improvement in {early_patience} epochs)."
                        )
                        if use_tqdm and hasattr(epoch_iterable, "write"):
                            epoch_iterable.write(notice)
                        else:
                            print(notice)
                        break

            if scheduler is not None:
                scheduler.step()
        if use_tqdm and hasattr(epoch_iterable, "close"):
            epoch_iterable.close()

        emissions_kg = tracker.stop()
        emissions_data = getattr(tracker, "final_emissions_data", None)
    finally:
        if emissions_kg is None:
            emissions_kg = tracker.stop()
            emissions_data = getattr(tracker, "final_emissions_data", None)

    if early_enabled and best_state is not None:
        model.load_state_dict(best_state["model"])
        optimizer.load_state_dict(best_state["optimizer"])
        if best_metric is not None:
            print(
                f"[info] Restored best model from epoch {best_epoch} "
                f"({early_metric}={best_metric:.4f})."
            )

    if config.save_model:
        artifact_dir = Path(config.save_model_path or "artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{report_path.stem}_model.pt"
        model_path = artifact_dir / base_name
        counter = 1
        while model_path.exists():
            model_path = artifact_dir / f"{report_path.stem}_model_{counter}.pt"
            counter += 1
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": asdict(config),
            },
            model_path,
        )
        print(f"[info] Model checkpoint saved to {model_path}")

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
        "optimizer",
        "loss_function",
        "weight_decay",
        "scheduler",
        "seed",
        "temperature",
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
                    "optimizer": config.optimizer,
                    "loss_function": config.loss_function,
                    "weight_decay": config.weight_decay,
                    "scheduler": config.scheduler,
                    "seed": config.seed,
                    "temperature": config.hyperparameters.get("temperature"),
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
                "optimizer": config.optimizer,
                "loss_function": config.loss_function,
                "weight_decay": config.weight_decay,
                "scheduler": config.scheduler,
                "seed": config.seed,
                "temperature": config.hyperparameters.get("temperature"),
            }
        )


def _build_loss(name: str) -> LossSpec:
    assert nn is not None
    key = (name or "cross_entropy").lower()
    mapping = {
        "cross_entropy": nn.CrossEntropyLoss,
        "mse": nn.MSELoss,
        "l1": nn.L1Loss,
        "smooth_l1": nn.SmoothL1Loss,
        "binary_cross_entropy": nn.BCEWithLogitsLoss,
    }
    if key in mapping:
        mode = "bce" if key == "binary_cross_entropy" else "logits"
        return LossSpec(name=key, mode=mode, criterion=mapping[key]())
    if key == "simclr":
        return LossSpec(name=key, mode="simclr")
    if key in {"supcon", "supervised_contrastive"}:
        return LossSpec(name="supcon", mode="supcon")
    print(f"[warn] Unsupported loss '{name}'. Falling back to CrossEntropyLoss.")
    return LossSpec(name="cross_entropy", mode="logits", criterion=nn.CrossEntropyLoss())


def _build_optimizer(
    name: str,
    params,
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
):
    assert torch is not None
    key = (name or "adam").lower()

    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if key == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if key == "lion":
        try:
            from lion_pytorch import Lion  # type: ignore

            return Lion(params, lr=lr, weight_decay=weight_decay)
        except Exception:
            print("[warn] lion optimizer requested but lion-pytorch is not installed. Using Adam instead.")
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    print(f"[warn] Unsupported optimizer '{name}'. Falling back to Adam.")
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def _build_scheduler(config: Dict[str, Any], optimizer) -> Optional[Any]:
    if not isinstance(config, dict):
        return None
    sched_type = (config.get("type") or "none").lower()
    params = config.get("params", {}) or {}

    if sched_type in {"none", ""}:
        return None

    if sched_type == "step_lr":
        step_size = int(params.get("step_size", 10))
        gamma = float(params.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if sched_type == "cosine_annealing":
        t_max = int(params.get("t_max", 10))
        eta_min = float(params.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    if sched_type == "exponential":
        gamma = float(params.get("gamma", 0.95))
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    print(f"[warn] Unsupported scheduler '{config.get('type')}'. Scheduler disabled.")
    return None


def _compute_loss(
    spec: LossSpec,
    logits: "torch.Tensor",
    embeddings: "torch.Tensor",
    labels: "torch.Tensor",
    temperature: float,
) -> "torch.Tensor":  # type: ignore[name-defined]
    if spec.mode == "logits":
        assert spec.criterion is not None
        return spec.criterion(logits, labels)
    if spec.mode == "bce":
        assert spec.criterion is not None
        target = labels
        if target.shape != logits.shape:
            if target.dim() == 1 and logits.dim() == 2:
                num_classes = logits.size(1)
                target = F.one_hot(target.long(), num_classes=num_classes)
                target = target.to(dtype=logits.dtype)
            else:
                target = target.to(dtype=logits.dtype).view_as(logits)
        else:
            target = target.to(dtype=logits.dtype)
        return spec.criterion(logits, target)
    if spec.mode == "simclr":
        return _simclr_loss(embeddings, temperature)
    if spec.mode == "supcon":
        return _supcon_loss(embeddings, labels, temperature)
    # Fallback safety
    assert spec.criterion is not None
    return spec.criterion(logits, labels)


def _simclr_loss(embeddings: "torch.Tensor", temperature: float) -> "torch.Tensor":  # type: ignore[name-defined]
    batch_size = embeddings.size(0)
    if batch_size < 2:
        return embeddings.new_zeros(())

    noise = torch.randn_like(embeddings) * 0.01
    z1 = F.normalize(embeddings + noise, dim=1)
    z2 = F.normalize(embeddings - noise, dim=1)
    representations = torch.cat([z1, z2], dim=0)

    logits = torch.matmul(representations, representations.T) / max(temperature, 1e-6)
    mask = torch.eye(2 * batch_size, device=embeddings.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, -1e9)

    targets = torch.arange(2 * batch_size, device=embeddings.device)
    targets = (targets + batch_size) % (2 * batch_size)

    return F.cross_entropy(logits, targets)


def _supcon_loss(
    embeddings: "torch.Tensor",
    labels: "torch.Tensor",
    temperature: float,
) -> "torch.Tensor":  # type: ignore[name-defined]
    device = embeddings.device
    labels = labels.contiguous().view(-1, 1)
    batch_size = embeddings.size(0)

    if batch_size < 2:
        return embeddings.new_zeros(())

    mask = torch.eq(labels, labels.T).float().to(device)
    anchor_dot_contrast = torch.div(
        torch.matmul(embeddings, embeddings.T),
        max(temperature, 1e-6),
    )

    # For numerical stability
    logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    mask_sum = mask.sum(dim=1)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / torch.clamp(mask_sum, min=1.0)

    loss = -temperature * mean_log_prob_pos
    return loss.mean()


class ProjectionModel(nn.Module):  # type: ignore[name-defined]
    """Wrap a backbone with classification and projection heads."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        projection_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, max(feature_dim, projection_dim)),
            nn.ReLU(),
            nn.Linear(max(feature_dim, projection_dim), projection_dim),
        )

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:  # type: ignore[name-defined]
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        embeddings = F.normalize(self.projector(features), dim=1) if F is not None else features
        return logits, embeddings


def _build_linear_model(
    num_classes: int,
    image_size: int,
    projection_dim: int,
) -> ProjectionModel:
    in_features = image_size * image_size * 3
    feature_dim = 512
    backbone = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, feature_dim),
        nn.ReLU(),
    )
    return ProjectionModel(backbone, feature_dim, num_classes, projection_dim)


def _build_model(
    name: str,
    *,
    num_classes: int,
    image_size: int,
    projection_dim: int,
) -> nn.Module:
    key = (name or "resnet18").lower()

    if tv_models is None:
        print("[warn] torchvision models unavailable. Using simple MLP classifier.")
        return _build_linear_model(num_classes, image_size, projection_dim)

    try:
        if key == "resnet18":
            backbone = tv_models.resnet18(weights=None)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            return ProjectionModel(backbone, feature_dim, num_classes, projection_dim)
        if key == "resnet34":
            backbone = tv_models.resnet34(weights=None)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            return ProjectionModel(backbone, feature_dim, num_classes, projection_dim)
        if key == "resnet50":
            backbone = tv_models.resnet50(weights=None)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            return ProjectionModel(backbone, feature_dim, num_classes, projection_dim)
        if key == "mobilenet_v3_small":
            backbone = tv_models.mobilenet_v3_small(weights=None)
            feature_dim = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Identity()
            return ProjectionModel(backbone, feature_dim, num_classes, projection_dim)
        if key == "efficientnet_b0":
            backbone = tv_models.efficientnet_b0(weights=None)
            feature_dim = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Identity()
            return ProjectionModel(backbone, feature_dim, num_classes, projection_dim)
        if key in {"vit_b_16", "vit-b-16", "vit"}:
            backbone = tv_models.vit_b_16(weights=None)
            feature_dim = backbone.heads.head.in_features
            backbone.heads.head = nn.Identity()
            return ProjectionModel(backbone, feature_dim, num_classes, projection_dim)
    except Exception as exc:  # pragma: no cover - model construction safeguard
        print(f"[warn] Failed to build model '{name}': {exc}. Falling back to MLP classifier.")

    print(f"[warn] Unsupported model '{name}'. Using simple MLP classifier instead.")
    return _build_linear_model(num_classes, image_size, projection_dim)