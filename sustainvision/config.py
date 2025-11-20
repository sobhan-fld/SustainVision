"""Configuration manager for SustainVision.

This module provides a YAML-backed configuration manager with:
- Sensible defaults for model, database, device, and hyperparameters
- Device detection (CPU and CUDA indices, if PyTorch is installed)
- Load/merge behavior that preserves new defaults while honoring existing user config

The default config file path follows XDG conventions:
  ~/.config/sustainvision/config.yaml (or $XDG_CONFIG_HOME/sustainvision/config.yaml)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import yaml


_UNSET = object()


def _default_config_path() -> str:
    """Return the default path for the YAML configuration file.

    Uses $XDG_CONFIG_HOME when available; falls back to ~/.config.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = xdg if xdg else os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "sustainvision", "config.yaml")


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory for the given path if it does not exist."""
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)


def _detect_devices() -> List[str]:
    """Detect available compute devices.

    - Always include "cpu"
    - If PyTorch is installed and CUDA is available, include cuda:0..cuda:N-1
    - Fail closed (return just CPU) if detection errors occur
    """
    devices: List[str] = ["cpu"]
    try:
        import torch  # optional dependency for device discovery

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
    except Exception:
        # Silently ignore any detection error and stick to CPU
        pass
    return devices


@dataclass
class TrainingConfig:
    """Top-level training configuration persisted to YAML.

    - model: name or identifier (e.g., "resnet18", "bert-base-uncased", custom string)
    - database: dataset name or filesystem path
    - device: "cpu" or "cuda:{index}"
    - hyperparameters: batch size, learning rate, epochs, etc.
    - report_filename: CSV file name for training metrics + emissions
    - seed: reproducibility control for PyTorch/Numpy
    - optimizer / loss_function: training algorithm choices
    - weight_decay: L2 regularization strength
    - scheduler: learning-rate scheduler configuration
    - gradient_clip_norm: optional gradient clipping threshold
    - mixed_precision: enable/disable automatic mixed precision (AMP)
    - save_model/save_model_path: whether and where to persist trained weights
    """

    model: str = "resnet18"
    database: str = "databases/sample_dataset"
    device: str = "cpu"
    seed: int = 42
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"
    weight_decay: float = 0.0
    scheduler: Dict[str, Any] = field(
        default_factory=lambda: {"type": "none", "params": {}}
    )
    gradient_clip_norm: Optional[float] = None
    mixed_precision: bool = False
    save_model: bool = False
    save_model_path: str = "outputs/checkpoints"
    checkpoint_path: Optional[str] = None
    freeze_backbone: bool = False
    early_stopping: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": False, "patience": 5, "metric": "val_loss", "mode": "min"}
    )
    simclr_schedule: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "cycles": 8,
            "pretrain_epochs": 50,
            "finetune_epochs": 20,
            "pretrain_loss": "simclr",
            "finetune_loss": "cross_entropy",
            "finetune_lr": None,  # None means use same LR as pretrain
            "finetune_optimizer": None,
            "finetune_weight_decay": None,
            "freeze_backbone": False,
            "optimizer_reset": True,
        }
    )
    report_filename: str = "training_report.csv"
    hyperparameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": 3,
            "momentum": 0.9,
            "temperature": 0.1,
            "num_workers": 2,
            "val_split": 0.1,
            "image_size": 224,
            "projection_dim": 128,
            "lars_eta": 0.001,
            "lars_eps": 1e-9,
            "lars_exclude_bias_n_norm": True,
            "projection_hidden_dim": None,
            "projection_use_bn": False,
            "use_gaussian_blur": False,
        }
    )


class ConfigManager:
    """YAML-backed configuration manager with merge-friendly loads.

    Example:
        cm = ConfigManager()
        cfg = cm.load()
        cm.set(device="cuda:0", hyperparameters={"lr": 3e-4})
        cm.save()
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or _default_config_path()
        self._config: TrainingConfig = TrainingConfig()

    @property
    def available_devices(self) -> List[str]:
        """Return a list of available devices (e.g., ["cpu", "cuda:0", ...])."""
        return _detect_devices()

    @property
    def config(self) -> TrainingConfig:
        """Return the current in-memory configuration object."""
        return self._config

    def load(self) -> TrainingConfig:
        """Load configuration from YAML, merging with current defaults.

        - If the file does not exist, keep defaults in memory and return them.
        - If the file exists, shallow-merge top-level keys into defaults.
        - Hyperparameters are deep-merged (existing keys preserved unless overridden).
        """
        if os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}

            default_dict = asdict(TrainingConfig())
            merged = {**default_dict, **data}

            # Merge nested hyperparameters while preserving defaults
            default_hp = default_dict.get("hyperparameters", {})
            incoming_hp = merged.get("hyperparameters") or {}
            merged["hyperparameters"] = {**default_hp, **incoming_hp}

            default_sched = default_dict.get("scheduler", {}) or {}
            incoming_sched = merged.get("scheduler") or {}
            merged_sched = {**default_sched, **incoming_sched}
            default_sched_params = default_sched.get("params", {}) if isinstance(default_sched, dict) else {}
            incoming_sched_params = incoming_sched.get("params", {}) if isinstance(incoming_sched, dict) else {}
            merged_sched["params"] = {**default_sched_params, **incoming_sched_params}
            merged["scheduler"] = merged_sched

            # Merge nested simclr_schedule while preserving defaults
            default_schedule = default_dict.get("simclr_schedule", {}) or {}
            incoming_schedule = merged.get("simclr_schedule") or {}
            merged["simclr_schedule"] = {**default_schedule, **incoming_schedule}

            self._config = TrainingConfig(**merged)
        return self._config

    def save(self) -> None:
        """Persist the current configuration to the YAML file on disk."""
        _ensure_parent_dir(self.path)
        with open(self.path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(asdict(self._config), handle, sort_keys=False)

    def set(
        self,
        *,
        model: Optional[str] = None,
        database: Optional[str] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        optimizer: Optional[str] = None,
        loss_function: Optional[str] = None,
        weight_decay: Optional[float] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        gradient_clip_norm: Any = _UNSET,
        mixed_precision: Optional[bool] = None,
        save_model: Optional[bool] = None,
        save_model_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: Optional[bool] = None,
        report_filename: Optional[str] = None,
        early_stopping: Optional[Dict[str, Any]] = None,
        simclr_schedule: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update selected fields in the in-memory configuration.

        Only provided arguments are applied; others remain unchanged.
        """
        if model is not None:
            self._config.model = model
        if database is not None:
            self._config.database = database
        if device is not None:
            self._config.device = device
        if seed is not None:
            self._config.seed = int(seed)
        if optimizer is not None:
            self._config.optimizer = optimizer
        if loss_function is not None:
            self._config.loss_function = loss_function
        if weight_decay is not None:
            self._config.weight_decay = float(weight_decay)
        if scheduler is not None:
            current_sched = self._config.scheduler if isinstance(self._config.scheduler, dict) else {}
            current_type = current_sched.get("type")
            incoming_type = scheduler.get("type") if isinstance(scheduler, dict) else None
            incoming_params_present = isinstance(scheduler, dict) and "params" in scheduler
            base_params = current_sched.get("params", {}) if isinstance(current_sched.get("params"), dict) else {}
            incoming_params = scheduler.get("params", {}) if incoming_params_present else {}

            if incoming_type and incoming_type != current_type:
                base_sched = {"type": incoming_type}
            else:
                base_sched = {**current_sched}

            sched = {**base_sched, **scheduler}

            if incoming_params_present:
                merged_params = dict(incoming_params or {})
            elif incoming_type and incoming_type != current_type:
                merged_params = {}
            else:
                merged_params = dict(base_params)

            sched["params"] = merged_params
            self._config.scheduler = sched
        if gradient_clip_norm is not _UNSET:
            self._config.gradient_clip_norm = gradient_clip_norm
        if mixed_precision is not None:
            self._config.mixed_precision = bool(mixed_precision)
        if save_model is not None:
            self._config.save_model = bool(save_model)
        if save_model_path is not None:
            self._config.save_model_path = save_model_path
        if checkpoint_path is not None:
            self._config.checkpoint_path = checkpoint_path
        if freeze_backbone is not None:
            self._config.freeze_backbone = bool(freeze_backbone)
        if early_stopping is not None:
            merged = {**self._config.early_stopping, **early_stopping}
            self._config.early_stopping = merged
        if report_filename is not None:
            self._config.report_filename = report_filename
        if simclr_schedule is not None:
            merged = {**self._config.simclr_schedule, **simclr_schedule}
            self._config.simclr_schedule = merged
        if hyperparameters is not None:
            # Update (not replace) to preserve non-provided keys
            self._config.hyperparameters.update(hyperparameters)


