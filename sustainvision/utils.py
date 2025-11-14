"""Utility functions for training setup and configuration."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    import numpy as np
except Exception:
    np = None  # type: ignore


def set_seed(seed: int) -> None:
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


def resolve_device(preferred: str) -> "torch.device":  # type: ignore[name-defined]
    """Resolve the requested device, falling back to CPU if unavailable."""
    if torch is None:
        raise RuntimeError("PyTorch is required for device resolution")
    
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


def unique_report_path(base_dir: Path, desired_name: str) -> Path:
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

