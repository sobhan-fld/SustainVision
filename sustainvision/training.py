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
import time
from dataclasses import dataclass, asdict
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import TrainingConfig
from .data import DatasetPreparationError, build_classification_dataloaders
from .utils import resolve_device, set_seed, unique_report_path
from .losses import LossSpec, build_loss, compute_loss
from .optimizers import build_optimizer, build_scheduler
from .models import build_model
from .reporting import write_report_csv
from .types import EpochMetrics, TrainingRunSummary


try:  # Optional imports handled gracefully in `_ensure_dependencies`
    import torch
    from torch import nn
    from torch.cuda import amp
except Exception:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore
    nn = None  # type: ignore
    amp = None  # type: ignore

try:
    from codecarbon import EmissionsTracker
except Exception:  # pragma: no cover - handled at runtime
    EmissionsTracker = None  # type: ignore







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




def train_model(
    config: TrainingConfig,
    *,
    project_root: Optional[Path] = None,
) -> TrainingRunSummary:
    """Main training entry point that handles contrastive schedule or standard training.
    
    If simclr_schedule is enabled, runs alternating pretrain/finetune cycles.
    Otherwise, runs a single training phase.
    """
    schedule_cfg = config.simclr_schedule or {}
    schedule_enabled = bool(schedule_cfg.get("enabled", False))
    project_root = project_root or (Path.cwd() / "outputs")
    if not schedule_enabled:
        # Standard single-phase training
        summary, _ = _execute_training_phase(
            config,
            project_root=project_root,
            phase_label="train",
            write_outputs=True,
        )
        return summary
    
    # Run contrastive learning alternating schedule (lazy import to avoid circular dependency)
    from .schedule import run_contrastive_schedule
    return run_contrastive_schedule(config, project_root=project_root)


def _execute_training_phase(
    config: TrainingConfig,
    *,
    project_root: Optional[Path] = None,
    report_path: Optional[Path] = None,
    phase_label: str = "train",
    initial_state: Optional[dict] = None,
    write_outputs: bool = True,
    loss_function_override: Optional[str] = None,
    epochs_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    optimizer_override: Optional[str] = None,
    weight_decay_override: Optional[float] = None,
    freeze_backbone_override: Optional[bool] = None,
    skip_emissions_tracker: bool = False,
    reset_classifier: bool = False,
) -> Tuple[TrainingRunSummary, dict]:
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

    set_seed(config.seed)

    project_dir = project_root or (Path.cwd() / "outputs")
    project_dir.mkdir(parents=True, exist_ok=True)
    report_path = unique_report_path(project_dir, config.report_filename)

    device = resolve_device(config.device)

    batch_size = int(config.hyperparameters.get("batch_size", 32))
    learning_rate = float(lr_override if lr_override is not None else config.hyperparameters.get("lr", 1e-3))
    epochs = int(epochs_override if epochs_override is not None else config.hyperparameters.get("epochs", 3))
    momentum = float(config.hyperparameters.get("momentum", 0.9))
    temperature = float(config.hyperparameters.get("temperature", 0.1))
    num_workers = int(config.hyperparameters.get("num_workers", 2))
    val_split = float(config.hyperparameters.get("val_split", 0.1))
    image_size = int(config.hyperparameters.get("image_size", 224))
    projection_dim = int(config.hyperparameters.get("projection_dim", 128))
    projection_hidden_raw = config.hyperparameters.get("projection_hidden_dim")
    projection_hidden_dim = (
        int(projection_hidden_raw)
        if projection_hidden_raw not in (None, "", "none")
        else None
    )
    projection_use_bn = bool(config.hyperparameters.get("projection_use_bn", False))
    use_gaussian_blur = bool(config.hyperparameters.get("use_gaussian_blur", False))
    lars_eta = float(config.hyperparameters.get("lars_eta", 0.001))
    lars_eps = float(config.hyperparameters.get("lars_eps", 1e-9))
    lars_exclude_bias_n_norm = bool(config.hyperparameters.get("lars_exclude_bias_n_norm", True))

    loss_function = loss_function_override if loss_function_override is not None else config.loss_function
    loss_spec = build_loss(loss_function)
    contrastive_mode = loss_spec.mode in {"simclr", "supcon"}
    classification_mode = not contrastive_mode

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
            use_gaussian_blur=use_gaussian_blur if contrastive_mode else False,
        )
    except DatasetPreparationError as exc:
        raise MissingDependencyError(str(exc)) from exc

    model = build_model(
        config.model,
        num_classes=num_classes,
        image_size=image_size,
        projection_dim=projection_dim,
        projection_hidden_dim=projection_hidden_dim,
        projection_use_bn=projection_use_bn,
    ).to(device)

    if initial_state is not None:
        model.load_state_dict(initial_state, strict=False)
    elif config.checkpoint_path:
        checkpoint_file = Path(config.checkpoint_path)
        if not checkpoint_file.is_absolute():
            checkpoint_file = project_dir / checkpoint_file
        if checkpoint_file.exists():
            print(f"[info] Loading checkpoint from {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=device)
            if "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"], strict=False)
                print("[info] Loaded model weights from checkpoint")
            elif "model" in checkpoint:
                model.load_state_dict(checkpoint["model"], strict=False)
                print("[info] Loaded model weights from checkpoint")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print("[info] Loaded model weights from checkpoint")
        else:
            raise MissingDependencyError(f"Checkpoint file not found: {checkpoint_file}")

    if reset_classifier and hasattr(model, "classifier"):
        classifier = getattr(model, "classifier")
        if hasattr(classifier, "reset_parameters"):
            classifier.reset_parameters()  # type: ignore[attr-defined]
        else:
            for module in classifier.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()  # type: ignore[attr-defined]
        print("[info] Classifier head reset for new evaluation phase.")

    freeze_backbone = freeze_backbone_override if freeze_backbone_override is not None else config.freeze_backbone
    if freeze_backbone and hasattr(model, "backbone"):
        model.backbone.eval()
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("[info] Backbone frozen - only classifier head will be trained")
    elif freeze_backbone:
        print("[warn] freeze_backbone requested but model has no 'backbone' attribute")

    try:
        model_device = next(model.parameters()).device
        print(f"[info] Training on device: {model_device}")
    except StopIteration:
        print("[info] Training on device: unknown (model has no parameters)")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = optimizer_override if optimizer_override else config.optimizer
    weight_decay_value = (
        config.weight_decay if weight_decay_override is None else float(weight_decay_override)
    )

    optimizer = build_optimizer(
        optimizer_name,
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay_value,
        momentum=momentum,
        lars_eta=lars_eta,
        lars_eps=lars_eps,
        lars_exclude_bias_n_norm=lars_exclude_bias_n_norm,
    )
    scheduler = build_scheduler(config.scheduler, optimizer)

    use_amp = bool(config.mixed_precision) and device.type == "cuda" and amp is not None
    scaler = amp.GradScaler(enabled=use_amp) if amp is not None else None

    tracker = None
    if not skip_emissions_tracker:
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
    if loss_spec.mode in {"simclr", "supcon"} and early_metric == "val_accuracy":
        early_metric = "train_loss"
        early_mode = "min"
        print(
            "[info] Contrastive loss detected: using train_loss for early stopping "
            "(val_accuracy not meaningful for representation learning)."
        )
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
            loss_normalizer = 0.0
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
                if contrastive_mode and isinstance(inputs, (list, tuple)):
                    if len(inputs) != 2:
                        raise ValueError("Contrastive batches must provide two augmented views.")
                    view1, view2 = inputs
                    inputs = torch.cat([view1, view2], dim=0)
                    if loss_spec.mode == "supcon":
                        labels = torch.cat([labels, labels], dim=0)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if use_amp and scaler is not None:
                    with amp.autocast():  # type: ignore[attr-defined]
                        logits, embeddings = model(inputs)
                        loss = compute_loss(loss_spec, logits, embeddings, labels, temperature)
                    scaler.scale(loss).backward()
                    if config.gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, embeddings = model(inputs)
                    loss = compute_loss(loss_spec, logits, embeddings, labels, temperature)
                    loss.backward()
                    if config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    optimizer.step()

                batch_weight = float(inputs.size(0)) if contrastive_mode else 1.0
                epoch_loss += loss.item() * batch_weight
                loss_normalizer += batch_weight
                if classification_mode:
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            if use_tqdm:
                try:
                    batch_iterable.close()  # type: ignore[attr-defined]
                except Exception:
                    pass

            train_loss = epoch_loss / max(loss_normalizer, 1.0)
            train_acc = correct / total if total else 0.0

            model.eval()
            val_loss_accum = 0.0
            val_loss_normalizer = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    if contrastive_mode and isinstance(inputs, (list, tuple)):
                        if len(inputs) != 2:
                            raise ValueError("Contrastive batches must provide two augmented views.")
                        view1, view2 = inputs
                        inputs = torch.cat([view1, view2], dim=0)
                        if loss_spec.mode == "supcon":
                            labels = torch.cat([labels, labels], dim=0)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    logits, embeddings = model(inputs)
                    loss = compute_loss(loss_spec, logits, embeddings, labels, temperature)
                    batch_weight = float(inputs.size(0)) if contrastive_mode else 1.0
                    val_loss_accum += loss.item() * batch_weight
                    val_loss_normalizer += batch_weight
                    if classification_mode:
                        preds = logits.argmax(dim=1)
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)

            val_loss = val_loss_accum / max(val_loss_normalizer, 1.0)
            val_acc = val_correct / val_total if val_total else 0.0
            current_lr = optimizer.param_groups[0].get("lr", 0.0)

            epoch_metrics.append(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_acc,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    learning_rate=current_lr,
                    phase=phase_label,
                    loss_name=loss_function,
                    loss_mode=loss_spec.mode,
                    optimizer_name=optimizer_name,
                    weight_decay=weight_decay_value,
                )
            )

            message = (
                f"Epoch {epoch:02d}/{epochs} - "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"lr={current_lr:.6f}"
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

        if tracker is not None:
            emissions_kg = tracker.stop()
            emissions_data = getattr(tracker, "final_emissions_data", None)
    finally:
        if tracker is not None and emissions_kg is None:
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
        config_yaml_path = model_path.with_suffix('.yaml')
        with config_yaml_path.open('w', encoding='utf-8') as cfg_handle:
            yaml.safe_dump(asdict(config), cfg_handle, sort_keys=False)

        print(f"[info] Model checkpoint saved to {model_path}")
        print(f"[info] Config snapshot written to {config_yaml_path}")

    duration = time.perf_counter() - start_time
    energy_kwh = None
    if emissions_data is not None:
        energy_kwh = emissions_data.energy_consumed
        if emissions_data.duration is not None:
            duration = emissions_data.duration

    if write_outputs:
        write_report_csv(
            report_path,
            epoch_metrics,
            emissions_kg=emissions_kg,
            energy_kwh=energy_kwh,
            duration_seconds=duration,
            config=config,
            append=False,
        )
        print(f"\nPhase {phase_label} complete. Report saved to {report_path}")
    else:
        print(f"\nPhase {phase_label} complete.")

    return TrainingRunSummary(
        report_path=report_path,
        emissions_kg=emissions_kg,
        energy_kwh=energy_kwh,
        duration_seconds=duration,
        epochs=epoch_metrics,
    ), {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

