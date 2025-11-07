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
        gradient_clip_norm: Optional[float] = None,
        mixed_precision: Optional[bool] = None,
        report_filename: Optional[str] = None,
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
            sched = {**self._config.scheduler, **scheduler}
            default_params = self._config.scheduler.get("params", {}) if isinstance(self._config.scheduler, dict) else {}
            incoming_params = scheduler.get("params", {}) if isinstance(scheduler, dict) else {}
            sched["params"] = {**default_params, **incoming_params}
            self._config.scheduler = sched
        if gradient_clip_norm is not None:
            self._config.gradient_clip_norm = gradient_clip_norm
        if mixed_precision is not None:
            self._config.mixed_precision = bool(mixed_precision)
        if report_filename is not None:
            self._config.report_filename = report_filename
        if hyperparameters is not None:
            # Update (not replace) to preserve non-provided keys
            self._config.hyperparameters.update(hyperparameters)


