"""Text-based user interface (TUI) for configuring SustainVision.

This TUI lets users review and edit the configuration and saves it to disk.
It relies on the ConfigManager to handle reading/writing YAML and device detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import questionary

from .config import ConfigManager, TrainingConfig
def _derive_run_name(cfg: TrainingConfig) -> str:
    """Infer a default run/output name from the current report filename."""
    name = (cfg.report_filename or "training_report.csv").strip()
    if not name:
        return "training_run"
    return Path(name).stem or "training_run"


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
    momentum = questionary.text("momentum (SGD-based optimizers):", default=str(hp.get("momentum", 0.9))).ask()
    temperature = questionary.text(
        "contrastive temperature:",
        default=str(hp.get("temperature", 0.1)),
    ).ask()
    num_workers = questionary.text(
        "dataloader workers:",
        default=str(hp.get("num_workers", 2)),
    ).ask()
    val_split = questionary.text(
        "validation split (0-1):",
        default=str(hp.get("val_split", 0.1)),
    ).ask()
    image_size = questionary.text(
        "input image size:",
        default=str(hp.get("image_size", 224)),
    ).ask()
    projection_dim = questionary.text(
        "projection head dimension:",
        default=str(hp.get("projection_dim", 128)),
    ).ask()
    projection_hidden_dim = questionary.text(
        "projection hidden dimension (enter 'none' for default):",
        default=str(hp.get("projection_hidden_dim", "none")),
    ).ask()
    use_bn_projection = questionary.confirm(
        "Use BatchNorm in projection head?",
        default=bool(hp.get("projection_use_bn", False)),
    ).ask()
    use_gaussian_blur = questionary.confirm(
        "Add Gaussian blur to contrastive augmentations?",
        default=bool(hp.get("use_gaussian_blur", False)),
    ).ask()
    use_reference_transforms = questionary.confirm(
        "Use CIFAR-10 SimCLR reference transforms?",
        default=bool(hp.get("simclr_reference_transforms", False)),
    ).ask()
    subset_default = hp.get("linear_subset_per_class")
    subset_prompt = "Linear-eval subset per class (0 = entire dataset):"
    subset_value = questionary.text(
        subset_prompt,
        default=str(subset_default or 0),
    ).ask()
    subset_seed_value = questionary.text(
        "Linear-eval subset seed:",
        default=str(hp.get("linear_subset_seed", 42)),
    ).ask()

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

    def as_subset_int(value: Optional[str], current: Optional[int]) -> Optional[int]:
        if value is None:
            return current if isinstance(current, int) and current > 0 else None
        try:
            parsed = int(str(value).strip())
        except Exception:
            return current if isinstance(current, int) and current > 0 else None
        return parsed if parsed > 0 else None

    return {
        "batch_size": as_int(batch_size, hp.get("batch_size", 32)),
        "lr": as_float(lr, hp.get("lr", 1e-3)),
        "epochs": as_int(epochs, hp.get("epochs", 3)),
        "momentum": as_float(momentum, hp.get("momentum", 0.9)),
        "temperature": as_float(temperature, hp.get("temperature", 0.1)),
        "num_workers": as_int(num_workers, hp.get("num_workers", 2)),
        "val_split": as_float(val_split, hp.get("val_split", 0.1)),
        "image_size": as_int(image_size, hp.get("image_size", 224)),
        "projection_dim": as_int(projection_dim, hp.get("projection_dim", 128)),
        "projection_hidden_dim": None
        if projection_hidden_dim is None or str(projection_hidden_dim).strip().lower() in {"", "none"}
        else as_int(projection_hidden_dim, hp.get("projection_hidden_dim", 0) or 0),
        "projection_use_bn": bool(use_bn_projection),
        "use_gaussian_blur": bool(use_gaussian_blur),
        "simclr_reference_transforms": bool(use_reference_transforms),
        "linear_subset_per_class": as_subset_int(subset_value, hp.get("linear_subset_per_class")),
        "linear_subset_seed": as_int(subset_seed_value, hp.get("linear_subset_seed", 42)),
    }


def _ask_seed(current: int) -> int:
    seed = questionary.text("Random seed:", default=str(current)).ask()
    try:
        return int(seed)
    except Exception:
        return current


def _ask_optimizer(current: str) -> str:
    options = ["adam", "adamw", "sgd", "rmsprop", "lion", "lars", "custom"]
    choice = questionary.select(
        "Optimizer:",
        choices=options,
        default=current if current in options else "custom",
    ).ask()
    if choice == "custom":
        return questionary.text("Enter optimizer name:", default=current).ask()
    return choice


def _ask_loss_function(current: str) -> str:
    options = [
        "cross_entropy",
        "mse",
        "l1",
        "smooth_l1",
        "binary_cross_entropy",
        "simclr",
        "supcon",
        "custom",
    ]
    choice = questionary.select(
        "Loss function:",
        choices=options,
        default=current if current in options else "custom",
    ).ask()
    if choice == "custom":
        return questionary.text("Enter loss function:", default=current).ask()
    return choice


def _ask_weight_decay(current: float) -> float:
    value = questionary.text("Weight decay:", default=str(current)).ask()
    try:
        return float(value)
    except Exception:
        return current


def _ask_scheduler(scheduler: Dict[str, Any]) -> Dict[str, Any]:
    sched_type = scheduler.get("type", "none")
    options = ["none", "step_lr", "cosine_annealing", "exponential", "custom"]
    choice = questionary.select(
        "Scheduler:",
        choices=options,
        default=sched_type if sched_type in options else "custom",
    ).ask()

    if choice == "none":
        step_each_batch = questionary.confirm(
            "Update scheduler every batch?",
            default=bool(scheduler.get("step_on_batch", False)),
        ).ask()
        return {"type": "none", "params": {}, "step_on_batch": bool(step_each_batch)}

    if choice == "custom":
        custom_type = questionary.text("Scheduler name:", default=sched_type).ask()
        params_text = questionary.text(
            "Scheduler params as key=value pairs (comma separated):",
            default=", ".join(f"{k}={v}" for k, v in (scheduler.get("params", {}) or {}).items()),
        ).ask()
        step_each_batch = questionary.confirm(
            "Update scheduler every batch?",
            default=bool(scheduler.get("step_on_batch", False)),
        ).ask()
        return {"type": custom_type, "params": _parse_params(params_text), "step_on_batch": bool(step_each_batch)}

    params_defaults = {
        "step_lr": {"step_size": 10, "gamma": 0.1},
        "cosine_annealing": {"t_max": 10, "eta_min": 0.0},
        "exponential": {"gamma": 0.95},
    }
    defaults = params_defaults.get(choice, {})
    params = {}
    for key, default in defaults.items():
        value = questionary.text(f"{choice}::{key}", default=str(scheduler.get("params", {}).get(key, default))).ask()
        params[key] = _safe_number(value, default)
    if choice == "cosine_annealing":
        auto_tmax = questionary.confirm(
            "Set cosine T_max based on total train steps (per-batch SimCLR style)?",
            default=str(scheduler.get("params", {}).get("t_max_strategy", "")).lower() == "per_batch",
        ).ask()
        if auto_tmax:
            params.pop("t_max", None)
            params["t_max_strategy"] = "per_batch"
        else:
            params.pop("t_max_strategy", None)
    step_each_batch = questionary.confirm(
        "Update scheduler every batch?",
        default=bool(scheduler.get("step_on_batch", False)),
    ).ask()
    return {"type": choice, "params": params, "step_on_batch": bool(step_each_batch)}


def _parse_params(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    result: Dict[str, Any] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        result[key] = _safe_number(value, value)
    return result


def _safe_number(value: str, default: Any) -> Any:
    try:
        value_str = str(value)
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except Exception:
        return default


def _ask_gradient_clip(current: Optional[float]) -> Optional[float]:
    default_value = "none" if current is None else str(current)
    answer = questionary.text(
        "Gradient clip norm (enter 'none' to disable):",
        default=default_value,
    ).ask()
    if answer is None:
        return current
    answer = answer.strip().lower()
    if answer in {"none", "", "off"}:
        return None
    try:
        return float(answer)
    except Exception:
        return current


def _ask_mixed_precision(current: bool) -> bool:
    answer = questionary.confirm(
        "Enable automatic mixed precision (AMP)?",
        default=current,
    ).ask()
    if answer is None:
        return current
    return bool(answer)


def _safe_positive_int(value: Optional[str], default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _ask_simclr_schedule(schedule: Dict[str, Any]) -> Dict[str, Any]:
    """Prompt the user for contrastive learning alternating schedule configuration."""
    enabled = questionary.confirm(
        "Enable contrastive learning alternating schedule (pretrain + finetune cycles)?",
        default=schedule.get("enabled", False),
    ).ask()
    if enabled is None:
        enabled = schedule.get("enabled", False)
    
    if not enabled:
        return {**schedule, "enabled": False}
    
    cycles = questionary.text(
        "Number of cycles (pretrain + finetune pairs):",
        default=str(schedule.get("cycles", 8)),
    ).ask()
    cycles_value = _safe_positive_int(cycles, schedule.get("cycles", 8))
    
    pretrain_epochs = questionary.text(
        "Pretrain epochs per cycle (contrastive learning):",
        default=str(schedule.get("pretrain_epochs", 50)),
    ).ask()
    pretrain_value = _safe_positive_int(pretrain_epochs, schedule.get("pretrain_epochs", 50))
    
    finetune_epochs = questionary.text(
        "Finetune epochs per cycle (cross-entropy):",
        default=str(schedule.get("finetune_epochs", 20)),
    ).ask()
    finetune_value = _safe_positive_int(finetune_epochs, schedule.get("finetune_epochs", 20))
    
    pretrain_loss = questionary.select(
        "Pretrain loss function:",
        choices=["simclr", "supcon"],
        default=schedule.get("pretrain_loss", "simclr"),
    ).ask()
    if pretrain_loss is None:
        pretrain_loss = schedule.get("pretrain_loss", "simclr")
    
    finetune_loss = questionary.select(
        "Finetune loss function:",
        choices=["cross_entropy", "mse", "l1"],
        default=schedule.get("finetune_loss", "cross_entropy"),
    ).ask()
    if finetune_loss is None:
        finetune_loss = schedule.get("finetune_loss", "cross_entropy")
    
    finetune_lr_text = questionary.text(
        "Finetune learning rate (leave empty to use same as pretrain):",
        default=str(schedule.get("finetune_lr", "")) if schedule.get("finetune_lr") is not None else "",
    ).ask()
    finetune_lr = None
    if finetune_lr_text and finetune_lr_text.strip():
        try:
            finetune_lr = float(finetune_lr_text)
        except (TypeError, ValueError):
            finetune_lr = None
    
    finetune_optimizer = questionary.select(
        "Finetune optimizer (linear head):",
        choices=["inherit"] + ["adam", "adamw", "sgd", "rmsprop", "lion", "lars"],
        default=schedule.get("finetune_optimizer", None) or "inherit",
    ).ask()
    if finetune_optimizer in ("inherit", None):
        finetune_optimizer = None
    
    finetune_weight_decay_text = questionary.text(
        "Finetune weight decay (enter 'inherit' to reuse main setting):",
        default=(
            str(schedule.get("finetune_weight_decay", "inherit"))
            if schedule.get("finetune_weight_decay") is not None
            else "inherit"
        ),
    ).ask()
    finetune_weight_decay: Optional[float] = None
    if finetune_weight_decay_text:
        finetune_weight_decay_text = finetune_weight_decay_text.strip().lower()
        if finetune_weight_decay_text not in {"inherit", "none", ""}:
            try:
                finetune_weight_decay = float(finetune_weight_decay_text)
            except (TypeError, ValueError):
                finetune_weight_decay = schedule.get("finetune_weight_decay")
        elif finetune_weight_decay_text in {"none", ""}:
            finetune_weight_decay = 0.0
        else:
            finetune_weight_decay = None
    
    freeze_backbone = questionary.confirm(
        "Freeze backbone during finetune (linear evaluation)?",
        default=schedule.get("freeze_backbone", False),
    ).ask()
    if freeze_backbone is None:
        freeze_backbone = schedule.get("freeze_backbone", False)
    
    optimizer_reset = questionary.confirm(
        "Reset optimizer between phases?",
        default=schedule.get("optimizer_reset", True),
    ).ask()
    if optimizer_reset is None:
        optimizer_reset = schedule.get("optimizer_reset", True)

    use_reference = questionary.confirm(
        "Use SimCLR reference transforms for CIFAR-10?",
        default=schedule.get("use_reference_transforms", False),
    ).ask()
    subset_linear = questionary.text(
        "Linear-eval subset per class (0 = full dataset):",
        default=str(schedule.get("linear_subset_per_class") or 0),
    ).ask()
    subset_linear_seed = questionary.text(
        "Linear-eval subset seed:",
        default=str(schedule.get("linear_subset_seed", 42)),
    ).ask()

    def _subset_value(raw: Optional[str], current):
        if raw is None:
            return current if isinstance(current, int) and current > 0 else None
        try:
            parsed = int(str(raw).strip())
        except Exception:
            return current if isinstance(current, int) and current > 0 else None
        return parsed if parsed > 0 else None
    
    return {
        "enabled": enabled,
        "cycles": cycles_value,
        "pretrain_epochs": pretrain_value,
        "finetune_epochs": finetune_value,
        "pretrain_loss": pretrain_loss,
        "finetune_loss": finetune_loss,
        "finetune_lr": finetune_lr,
        "finetune_optimizer": finetune_optimizer,
        "finetune_weight_decay": finetune_weight_decay,
        "freeze_backbone": freeze_backbone,
        "optimizer_reset": optimizer_reset,
        "use_reference_transforms": bool(use_reference),
        "linear_subset_per_class": _subset_value(subset_linear, schedule.get("linear_subset_per_class")),
        "linear_subset_seed": (
            schedule.get("linear_subset_seed", 42)
            if subset_linear_seed is None or subset_linear_seed.strip() == ""
            else _safe_positive_int(subset_linear_seed, schedule.get("linear_subset_seed", 42))
        ),
    }


def _ask_quantization_settings(current: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Prompt for optional quantization export settings."""

    current = current or {}
    enabled = questionary.confirm(
        "Export a quantized model artifact after training?",
        default=bool(current.get("enabled", False)),
    ).ask()
    if enabled is None:
        enabled = current.get("enabled", False)

    if not enabled:
        return {**current, "enabled": False, "custom": False}

    customize = questionary.confirm(
        "Customize quantization parameters (experimental)?",
        default=bool(current.get("custom", False)),
    ).ask()
    if customize is None:
        customize = current.get("custom", False)

    quant_cfg: Dict[str, Any] = {**current, "enabled": True, "custom": bool(customize)}

    if not quant_cfg["custom"]:
        # Keep defaults; just ensure backend/dtype/module defaults are populated.
        quant_cfg.setdefault("approach", "dynamic")
        quant_cfg.setdefault("dtype", "qint8")
        quant_cfg.setdefault("modules", ["Linear"])
        quant_cfg.setdefault("backend", "qnnpack")
        quant_cfg.setdefault("export_format", "torchscript")
        quant_cfg.setdefault("artifact_name", None)
        return quant_cfg

    approach = questionary.select(
        "Quantization approach:",
        choices=["dynamic"],
        default=str(current.get("approach", "dynamic")),
    ).ask()

    dtype = questionary.select(
        "Quantization dtype:",
        choices=["qint8", "float16"],
        default=str(current.get("dtype", "qint8")),
    ).ask()

    backend = questionary.select(
        "Quantized backend:",
        choices=["qnnpack", "fbgemm"],
        default=str(current.get("backend", "qnnpack")),
    ).ask()

    modules_default = ", ".join(current.get("modules", ["Linear"]))
    modules_text = questionary.text(
        "Module types to quantize (comma separated, e.g., Linear,Conv2d):",
        default=modules_default,
    ).ask()

    export_format = questionary.select(
        "Quantized artifact format:",
        choices=[
            questionary.Choice(title="TorchScript (.ts)", value="torchscript"),
            questionary.Choice(title="State dict (.pt)", value="state_dict"),
            questionary.Choice(title="Pickled module (.pt)", value="module"),
        ],
        default=str(current.get("export_format", "torchscript")),
    ).ask()

    artifact_name = questionary.text(
        "Custom file name (leave empty to auto-generate):",
        default=str(current.get("artifact_name", "") or ""),
    ).ask()

    modules = [m.strip() for m in (modules_text or "").split(",") if m and m.strip()]
    if not modules:
        modules = ["Linear"]

    quant_cfg.update(
        {
            "approach": approach or "dynamic",
            "dtype": dtype or "qint8",
            "backend": backend or "qnnpack",
            "modules": modules,
            "export_format": export_format or "torchscript",
            "artifact_name": (artifact_name.strip() or None) if artifact_name else None,
        }
    )
    return quant_cfg


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
    print(f"- seed: {cfg.seed}")
    print(f"- optimizer: {cfg.optimizer}")
    print(f"- loss_function: {cfg.loss_function}")
    print(f"- weight_decay: {cfg.weight_decay}")
    print(f"- scheduler: {cfg.scheduler}")
    print(f"- gradient_clip_norm: {cfg.gradient_clip_norm}")
    print(f"- mixed_precision: {cfg.mixed_precision}")
    print(f"- quantization: {getattr(cfg, 'quantization', {})}")
    print(f"- report_filename: {cfg.report_filename}")
    print(f"- simclr_schedule: {cfg.simclr_schedule}")
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
    new_seed = _ask_seed(cfg.seed)
    new_optimizer = _ask_optimizer(cfg.optimizer)
    new_loss = _ask_loss_function(cfg.loss_function)
    new_weight_decay = _ask_weight_decay(cfg.weight_decay)
    new_scheduler = _ask_scheduler(cfg.scheduler)
    new_clip = _ask_gradient_clip(cfg.gradient_clip_norm)
    new_amp = _ask_mixed_precision(cfg.mixed_precision)
    save_model = questionary.confirm(
        "Persist trained model weights after each run?",
        default=cfg.save_model,
    ).ask()
    if save_model is None:
        save_model = cfg.save_model
    default_run_name = _derive_run_name(cfg)
    run_name_answer = questionary.text(
        "Run / artifact name (stored under outputs/<name>):",
        default=default_run_name,
    ).ask()
    run_name = (run_name_answer or default_run_name or "training_run").strip() or "training_run"
    report_name = str(Path(run_name) / f"{run_name}.csv")
    save_model_path = str(Path("outputs") / run_name)
    checkpoint_path = questionary.text(
        "Checkpoint path for fine-tuning (leave empty for training from scratch):",
        default=cfg.checkpoint_path or "",
    ).ask()
    if not checkpoint_path or checkpoint_path.strip() == "":
        checkpoint_path = None
    else:
        checkpoint_path = checkpoint_path.strip()
    freeze_backbone = False
    if checkpoint_path is not None:
        freeze_backbone = questionary.confirm(
            "Freeze backbone (only train classifier head)?",
            default=cfg.freeze_backbone,
        ).ask()
        if freeze_backbone is None:
            freeze_backbone = cfg.freeze_backbone
    early_defaults = cfg.early_stopping or {}
    early_enabled = questionary.confirm(
        "Enable early stopping?",
        default=early_defaults.get("enabled", False),
    ).ask()
    if early_enabled is None:
        early_enabled = early_defaults.get("enabled", False)

    patience_value = early_defaults.get("patience", 5)
    early_metric = early_defaults.get("metric", "val_loss")
    early_mode = early_defaults.get("mode", "min")

    if early_enabled:
        early_patience = questionary.text(
            "Early stopping patience (epochs):",
            default=str(early_defaults.get("patience", 5)),
        ).ask()
        patience_value = _safe_positive_int(early_patience, early_defaults.get("patience", 5))

        metric_answer = questionary.select(
            "Monitor metric:",
            choices=["val_loss", "val_accuracy", "train_loss", "train_accuracy"],
            default=early_defaults.get("metric", "val_loss"),
        ).ask()
        if metric_answer is not None:
            early_metric = metric_answer

        mode_answer = questionary.select(
            "Early stopping mode:",
            choices=["min", "max"],
            default=early_defaults.get("mode", "min"),
        ).ask()
        if mode_answer is not None:
            early_mode = mode_answer
    new_simclr_schedule = _ask_simclr_schedule(cfg.simclr_schedule)
    new_quantization = _ask_quantization_settings(getattr(cfg, "quantization", {}))

    cm.set(
        model=new_model,
        database=new_db,
        device=new_device,
        seed=new_seed,
        optimizer=new_optimizer,
        loss_function=new_loss,
        weight_decay=new_weight_decay,
        scheduler=new_scheduler,
        gradient_clip_norm=new_clip,
        mixed_precision=new_amp,
        save_model=save_model,
        save_model_path=save_model_path,
        checkpoint_path=checkpoint_path,
        freeze_backbone=freeze_backbone,
        early_stopping={
            "enabled": early_enabled,
            "patience": patience_value,
            "metric": early_metric,
            "mode": early_mode,
        },
        report_filename=report_name,
        quantization=new_quantization,
        simclr_schedule=new_simclr_schedule,
        hyperparameters=new_hp,
    )
    cm.save()
    print("\nConfig saved.")
    return cm.config


