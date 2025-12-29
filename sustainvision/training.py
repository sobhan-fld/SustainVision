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
from typing import Any, Dict, List, Optional, Tuple, Set

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
    import torch.ao.quantization as ao_quant
except Exception:  # pragma: no cover - handled at runtime
    ao_quant = None  # type: ignore

try:
    from codecarbon import EmissionsTracker
except Exception:  # pragma: no cover - handled at runtime
    EmissionsTracker = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore







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


def _resolve_quant_module_types(module_names: Any) -> "Set[type]":
    """Convert user-specified module names into actual nn.Module classes."""

    resolved: Set[type] = set()
    if nn is None or module_names is None:
        return resolved

    if not isinstance(module_names, (list, tuple, set)):
        module_iterable = [module_names]
    else:
        module_iterable = module_names

    for entry in module_iterable:
        if entry is None:
            continue
        if isinstance(entry, str):
            candidate = getattr(nn, entry, None)  # type: ignore[attr-defined]
            if isinstance(candidate, type) and issubclass(candidate, nn.Module):  # type: ignore[arg-type]
                resolved.add(candidate)  # type: ignore[arg-type]
        elif isinstance(entry, type) and issubclass(entry, nn.Module):  # type: ignore[arg-type]
            resolved.add(entry)  # type: ignore[arg-type]
    return resolved


def _export_quantized_artifact(
    model: "nn.Module",  # type: ignore[name-defined]
    config: TrainingConfig,
    quant_cfg: Dict[str, Any],
    *,
    artifact_dir: Path,
    report_path: Path,
    image_size: int,
) -> Optional[Path]:
    """Apply dynamic quantization and export an artifact for inference."""

    if torch is None or nn is None or ao_quant is None:
        print("[warn] Quantization requested but torch.ao.quantization is unavailable.")
        return None

    if not quant_cfg.get("enabled"):
        return None

    approach = str(quant_cfg.get("approach") or quant_cfg.get("mode") or "dynamic").lower()
    if approach not in {"dynamic"}:
        print(f"[warn] Quantization approach '{approach}' is not supported. Falling back to dynamic quantization.")
        approach = "dynamic"

    backend = str(quant_cfg.get("backend", "qnnpack")).lower()
    if hasattr(torch, "backends") and hasattr(torch.backends, "quantized"):
        try:
            if backend in {"qnnpack", "fbgemm"}:
                torch.backends.quantized.engine = backend  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - backend guard
            print(f"[warn] Unable to set quantized backend '{backend}': {exc}")

    dtype_map = {
        "qint8": torch.qint8,
        "quint8": torch.quint8,
        "float16": torch.float16,
    }
    dtype = dtype_map.get(str(quant_cfg.get("dtype", "qint8")).lower(), torch.qint8)

    module_types = _resolve_quant_module_types(quant_cfg.get("modules") or ["Linear"])
    if not module_types:
        module_types = {nn.Linear}

    quant_model = copy.deepcopy(model).cpu().eval()
    try:
        quantized = ao_quant.quantize_dynamic(quant_model, module_types, dtype=dtype)
    except Exception as exc:
        print(f"[warn] Dynamic quantization failed: {exc}")
        return None

    export_format = str(quant_cfg.get("export_format", "torchscript")).lower()
    base_name = quant_cfg.get("artifact_name") or f"{report_path.stem}_quantized"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    quant_path: Optional[Path] = None
    metadata = {
        "quantization": quant_cfg,
        "config": asdict(config),
    }

    if export_format == "state_dict":
        quant_path = artifact_dir / f"{base_name}.pt"
        torch.save(
            {
                **metadata,
                "model_state": quantized.state_dict(),
            },
            quant_path,
        )
    elif export_format == "module":
        quant_path = artifact_dir / f"{base_name}.pt"
        torch.save({**metadata, "model": quantized}, quant_path)
    else:
        example = torch.randn(1, 3, image_size, image_size)
        try:
            scripted = torch.jit.trace(quantized, example)
            quant_path = artifact_dir / f"{base_name}.ts"
            scripted.save(str(quant_path))
        except Exception as exc:
            fallback = artifact_dir / f"{base_name}.pt"
            torch.save({**metadata, "model": quantized}, fallback)
            quant_path = fallback
            print(
                "[warn] TorchScript export failed "
                f"({exc}). Saved a pickled quantized module instead: {fallback}"
            )

    print(f"[info] Quantized artifact saved to {quant_path}")
    return quant_path



def _log_resource_and_energy_snapshot(
    *,
    artifact_dir: Path,
    epoch: int,
    emissions_data: Any,
    elapsed_seconds: float,
    device: Any,
) -> None:
    """Append a CSV row with energy + CPU/RAM/GPU stats every N epochs.

    The log is written next to model checkpoints (save_model_path), so each run keeps
    its own resource log.
    """

    # Best-effort: if psutil is not available we still try to log what we can.
    cpu_percent = None
    ram_percent = None
    if psutil is not None:
        try:
            cpu_percent = float(psutil.cpu_percent(interval=None))
            ram_percent = float(psutil.virtual_memory().percent)
        except Exception:
            cpu_percent = None
            ram_percent = None

    gpu_util = None
    gpu_mem = None
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available() and device is not None:
        try:
            gpu_index = getattr(device, "index", None)
            if gpu_index is None and isinstance(device, torch.device):
                gpu_index = device.index
            if gpu_index is None and isinstance(device, str) and device.startswith("cuda:"):
                try:
                    gpu_index = int(device.split(":", 1)[1])
                except Exception:
                    gpu_index = 0
            if gpu_index is None:
                gpu_index = 0

            handle = torch.cuda.device(gpu_index)
            with handle:
                util = torch.cuda.utilization(handle) if hasattr(torch.cuda, "utilization") else None  # type: ignore[attr-defined]
                mem_alloc = torch.cuda.memory_allocated(gpu_index)
                mem_total = torch.cuda.get_device_properties(gpu_index).total_memory

            gpu_util = float(util) if util is not None else None
            gpu_mem = float(mem_alloc) / float(mem_total) * 100.0 if mem_total else None
        except Exception:
            gpu_util = None
            gpu_mem = None

    # Try to use the provided emissions data from CodeCarbon for this block.
    energy_kwh = None
    cpu_energy_kwh = None
    gpu_energy_kwh = None
    ram_energy_kwh = None
    if emissions_data is not None:
        try:
            energy_kwh = getattr(emissions_data, "energy_consumed", None)
            cpu_energy_kwh = getattr(emissions_data, "cpu_energy", None)
            gpu_energy_kwh = getattr(emissions_data, "gpu_energy", None)
            ram_energy_kwh = getattr(emissions_data, "ram_energy", None)
        except Exception:
            energy_kwh = None
            cpu_energy_kwh = None
            gpu_energy_kwh = None
            ram_energy_kwh = None

    elapsed = float(elapsed_seconds)

    log_path = artifact_dir / "resource_energy_log.csv"
    header = (
        "epoch,elapsed_seconds,"
        "energy_kwh,total_cpu_energy_kwh,total_gpu_energy_kwh,total_ram_energy_kwh,"
        "cpu_percent,ram_percent,gpu_util_percent,gpu_mem_percent\n"
    )
    line = (
        f"{epoch},{elapsed:.2f},"
        f"{'' if energy_kwh is None else energy_kwh},"
        f"{'' if cpu_energy_kwh is None else cpu_energy_kwh},"
        f"{'' if gpu_energy_kwh is None else gpu_energy_kwh},"
        f"{'' if ram_energy_kwh is None else ram_energy_kwh},"
        f"{'' if cpu_percent is None else cpu_percent},"
        f"{'' if ram_percent is None else ram_percent},"
        f"{'' if gpu_util is None else gpu_util},"
        f"{'' if gpu_mem is None else gpu_mem}\n"
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with log_path.open("w", encoding="utf-8") as f:
            f.write(header)
            f.write(line)
    else:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)



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
    if project_root:
        try:
            print("\n[info] Training configuration snapshot:")
            print(f"  - model: {config.model}")
            print(f"  - loss: {config.loss_function}")
            print(f"  - optimizer: {config.optimizer} (lr={config.hyperparameters.get('lr')})")
            if config.scheduler:
                print(f"  - scheduler: {config.scheduler.get('type')} {config.scheduler.get('params')}")
        except Exception:
            pass
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
    scheduler_config_override: Optional[Dict[str, Any]] = None,
    simclr_recipe_override: Optional[bool] = None,
    subset_per_class_override: Optional[int] = None,
    subset_seed_override: Optional[int] = None,
    scheduler_state: Optional[dict] = None,
    cumulative_scheduler_steps: Optional[int] = None,
    total_t_max: Optional[int] = None,
    total_cycles: Optional[int] = None,
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
    artifact_dir = Path(config.save_model_path or "artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

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
    simclr_recipe = (
        bool(simclr_recipe_override)
        if simclr_recipe_override is not None
        else bool(config.hyperparameters.get("simclr_reference_transforms", False))
    )
    quant_cfg = config.quantization or {}
    subset_per_class = subset_per_class_override
    if subset_per_class is None:
        subset_per_class = config.hyperparameters.get("linear_subset_per_class")
    if isinstance(subset_per_class, str):
        subset_per_class = subset_per_class.strip()
        subset_per_class = int(subset_per_class) if subset_per_class else None
    elif subset_per_class is not None:
        try:
            subset_per_class = int(subset_per_class)
        except Exception:
            subset_per_class = None
    subset_seed = subset_seed_override
    if subset_seed is None:
        subset_seed = int(config.hyperparameters.get("linear_subset_seed", config.seed))
    lars_eta = float(config.hyperparameters.get("lars_eta", 0.001))
    lars_eps = float(config.hyperparameters.get("lars_eps", 1e-9))
    lars_exclude_bias_n_norm = bool(config.hyperparameters.get("lars_exclude_bias_n_norm", True))

    loss_function = loss_function_override if loss_function_override is not None else config.loss_function
    loss_spec = build_loss(loss_function)
    contrastive_mode = loss_spec.mode in {"simclr", "supcon"}
    classification_mode = not contrastive_mode

    data_root = Path.cwd()
    
    # Check if we should use M-per-class sampler for SupCon
    use_m_per_class_sampler = False
    m_per_class = None
    if contrastive_mode and loss_spec.mode == "supcon":
        m_per_class = config.hyperparameters.get("m_per_class")
        use_m_per_class_sampler = config.hyperparameters.get("use_m_per_class_sampler", False)
        if m_per_class is not None and m_per_class > 0:
            use_m_per_class_sampler = True
    
    try:
        train_loader, val_loader, num_classes = build_classification_dataloaders(
            config.database,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            seed=config.seed,
            project_root=data_root,
            image_size=image_size,
            contrastive=contrastive_mode,
            use_gaussian_blur=use_gaussian_blur if contrastive_mode else False,
            simclr_recipe=simclr_recipe,
            subset_per_class=subset_per_class if (subset_per_class and classification_mode) else None,
            subset_seed=subset_seed,
            use_m_per_class_sampler=use_m_per_class_sampler,
            m_per_class=m_per_class,
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
        # Also freeze projector head during linear evaluation (it's only for contrastive learning)
        if hasattr(model, "projector"):
            for param in model.projector.parameters():
                param.requires_grad = False
        print("[info] Backbone and projector frozen - only classifier head will be trained")
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
    scheduler_config = scheduler_config_override if scheduler_config_override is not None else config.scheduler
    scheduler_cfg_copy = copy.deepcopy(scheduler_config) if isinstance(scheduler_config, dict) else None
    scheduler_step_on_batch = False
    scheduler_type = ""
    if isinstance(scheduler_cfg_copy, dict):
        scheduler_type = str(scheduler_cfg_copy.get("type", "") or "").lower()
        scheduler_step_on_batch = bool(scheduler_cfg_copy.pop("step_on_batch", False))
        scheduler_params = scheduler_cfg_copy.get("params", {}) or {}
        t_max_strategy = scheduler_params.pop("t_max_strategy", None)
        steps_per_epoch = max(1, len(train_loader))
        total_units = epochs * (steps_per_epoch if scheduler_step_on_batch else 1)
        warmup_epochs_raw = scheduler_params.pop("warmup_epochs", 0)
        try:
            warmup_epochs_val = int(warmup_epochs_raw)
        except Exception:
            warmup_epochs_val = 0
        warmup_units = warmup_epochs_val * (steps_per_epoch if scheduler_step_on_batch else 1)
        
        # If total_t_max is provided (from previous cycles), use it directly
        # This ensures consistent t_max across all cycles
        if total_t_max is not None and t_max_strategy == "per_batch":
            scheduler_params["t_max"] = total_t_max
        # First cycle: calculate total_t_max for ALL cycles
        elif total_t_max is None and total_cycles is not None and total_cycles > 1 and t_max_strategy == "per_batch":
            # Calculate total steps across all cycles
            total_steps_all_cycles = total_units * total_cycles
            scheduler_params["t_max"] = max(1, total_steps_all_cycles - max(0, warmup_units))
        # If cumulative_scheduler_steps is provided but no total_t_max, calculate it
        elif cumulative_scheduler_steps is not None and t_max_strategy == "per_batch":
            # Calculate total steps: cumulative + remaining steps for this phase
            cumulative_total = cumulative_scheduler_steps + total_units
            scheduler_params["t_max"] = max(1, cumulative_total - max(0, warmup_units))
        elif t_max_strategy == "per_batch":
            scheduler_params["t_max"] = max(1, total_units - max(0, warmup_units))
        
        if scheduler_type == "warmup_cosine":
            scheduler_params["warmup_steps"] = max(0, warmup_units)
            scheduler_params.setdefault("warmup_start_factor", 0.0)
        scheduler_cfg_copy["params"] = scheduler_params
    scheduler = build_scheduler(scheduler_cfg_copy, optimizer) if scheduler_cfg_copy else None
    
    # Load scheduler state if provided (for continuing across cycles)
    if scheduler is not None and scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
            print(f"[info] Loaded scheduler state to continue from previous cycle")
        except Exception as e:
            print(f"[warn] Failed to load scheduler state: {e}. Starting with fresh scheduler.")

    # AMP can be numerically fragile for contrastive objectives (SimCLR/SupCon),
    # so we only enable it for standard classification losses by default.
    use_amp = (
        bool(config.mixed_precision)
        and device.type == "cuda"
        and amp is not None
        and not contrastive_mode
    )
    scaler = None
    autocast_factory = None
    if use_amp:
        if torch is not None and hasattr(torch, "amp"):
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            autocast_factory = lambda: torch.amp.autocast("cuda")  # type: ignore[call-arg]
        elif amp is not None:
            scaler = amp.GradScaler(enabled=True)
            autocast_factory = getattr(amp, "autocast", None)

    epoch_metrics: List[EpochMetrics] = []
    emissions_kg_total: Optional[float] = None
    energy_kwh_total: Optional[float] = None

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

        # We track energy in 10-epoch blocks. Each block gets its own CodeCarbon tracker
        # so we can log per-block emissions and then aggregate a total at the end.
        block_tracker: Any = None

        def _start_block_tracker() -> Any:
            """Start a new CodeCarbon tracker for a 10-epoch block.

            Note: We disable CodeCarbon's own CSV logging (`save_to_file=False`)
            so that we don't create a global `emissions.csv`. Per-block and
            total energy/emissions are instead captured via the tracker object
            and our own `resource_energy_log.csv`.
            """
            if skip_emissions_tracker or EmissionsTracker is None:
                return None
            logging.getLogger("codecarbon").setLevel(logging.WARNING)
            local_tracker = EmissionsTracker(
                measure_power_secs=5,
                project_name=report_path.stem,
                log_level="warning",
                save_to_file=False,
            )  # type: ignore[call-arg]
            local_tracker.start()
            return local_tracker

        def _stop_block_tracker(current_epoch: int) -> None:
            """Stop the current block tracker, log stats, and update totals."""
            nonlocal block_tracker, emissions_kg_total, energy_kwh_total
            if block_tracker is None:
                return
            try:
                block_kg = block_tracker.stop()
                block_data = getattr(block_tracker, "final_emissions_data", None)
            except Exception:
                block_kg = None
                block_data = None
            block_tracker = None

            # Aggregate totals
            if block_kg is not None:
                emissions_kg_total = (emissions_kg_total or 0.0) + float(block_kg)
            if block_data is not None:
                try:
                    block_energy = getattr(block_data, "energy_consumed", None)
                    if block_energy is not None:
                        energy_kwh_total = (energy_kwh_total or 0.0) + float(block_energy)
                except Exception:
                    pass

            # Log this block as a snapshot for the last epoch of the block
            try:
                _log_resource_and_energy_snapshot(
                    artifact_dir=artifact_dir,
                    epoch=current_epoch,
                    emissions_data=block_data,
                    elapsed_seconds=time.perf_counter() - start_time,
                    device=device,
                )
            except Exception as exc:  # pragma: no cover - best-effort logging
                print(f"[warn] Failed to record resource/energy snapshot at epoch {current_epoch}: {exc}")

        for epoch in epoch_iterable:
            # Start a new 10-epoch energy tracking block at epochs 1, 11, 21, ...
            if not skip_emissions_tracker and ((epoch - 1) % 10 == 0) and block_tracker is None:
                block_tracker = _start_block_tracker()

            model.train()
            # If backbone is frozen, keep it in eval mode (for proper BatchNorm behavior)
            if freeze_backbone and hasattr(model, "backbone"):
                model.backbone.eval()
            epoch_loss = 0.0
            loss_normalizer = 0.0
            correct = 0
            total = 0
            epoch_had_optimizer_step = False

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
                optimizer_step_done = False

                if use_amp and scaler is not None and autocast_factory is not None:
                    with autocast_factory():  # type: ignore[call-arg]
                        logits, embeddings = model(inputs)
                        loss = compute_loss(loss_spec, logits, embeddings, labels, temperature)
                    scaler.scale(loss).backward()
                    if config.gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_step_done = True
                else:
                    logits, embeddings = model(inputs)
                    loss = compute_loss(loss_spec, logits, embeddings, labels, temperature)
                    loss.backward()
                    if config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    optimizer.step()
                    optimizer_step_done = True

                if scheduler is not None and scheduler_step_on_batch and optimizer_step_done:
                    scheduler.step()
                if optimizer_step_done:
                    epoch_had_optimizer_step = True

                batch_weight = float(inputs.size(0)) if contrastive_mode else 1.0
                epoch_loss += loss.item() * batch_weight
                loss_normalizer += batch_weight
                if classification_mode:
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            if scheduler is not None and not scheduler_step_on_batch and epoch_had_optimizer_step:
                scheduler.step()

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
                        # End current block before breaking
                        _stop_block_tracker(epoch)
                        break

            # End of 10-epoch block or final epoch: stop tracker and log snapshot.
            if epoch % 10 == 0 or epoch == epochs:
                _stop_block_tracker(epoch)

        if use_tqdm and hasattr(epoch_iterable, "close"):
            epoch_iterable.close()
    finally:
        # Ensure any active block tracker is stopped if something goes wrong.
        try:
            if 'block_tracker' in locals() and block_tracker is not None:
                _stop_block_tracker(epoch)  # type: ignore[name-defined]
        except Exception:
            pass

    if early_enabled and best_state is not None:
        model.load_state_dict(best_state["model"])
        optimizer.load_state_dict(best_state["optimizer"])
        if best_metric is not None:
            print(
                f"[info] Restored best model from epoch {best_epoch} "
                f"({early_metric}={best_metric:.4f})."
            )

    quantized_artifact: Optional[Path] = None
    if config.save_model:
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

    if quant_cfg.get("enabled") and write_outputs:
        quantized_artifact = _export_quantized_artifact(
            model,
            config,
            quant_cfg,
            artifact_dir=artifact_dir,
            report_path=report_path,
            image_size=image_size,
        )

    duration = time.perf_counter() - start_time
    energy_kwh = energy_kwh_total

    if write_outputs:
        write_report_csv(
            report_path,
            epoch_metrics,
            emissions_kg=emissions_kg_total,
            energy_kwh=energy_kwh,
            duration_seconds=duration,
            config=config,
            append=False,
        )
        print(f"\nPhase {phase_label} complete. Report saved to {report_path}")
    else:
        print(f"\nPhase {phase_label} complete.")

    # Extract t_max from scheduler if it exists (for passing to next cycle)
    final_total_t_max = None
    if scheduler is not None and hasattr(scheduler, 'T_max'):
        final_total_t_max = scheduler.T_max
    
    return TrainingRunSummary(
        report_path=report_path,
        emissions_kg=emissions_kg_total,
        energy_kwh=energy_kwh,
        duration_seconds=duration,
        epochs=epoch_metrics,
        quantized_model_path=quantized_artifact,
    ), {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "total_t_max": final_total_t_max,
    }

