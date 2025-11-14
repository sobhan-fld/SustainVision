"""Optimizer and scheduler builders."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import torch
except Exception:
    torch = None  # type: ignore


def build_optimizer(
    name: str,
    params,
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
):
    """Build an optimizer from name and parameters."""
    if torch is None:
        raise RuntimeError("PyTorch is required for optimizers")
    
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


def build_scheduler(config: Dict[str, Any], optimizer) -> Optional[Any]:
    """Build a learning rate scheduler from configuration."""
    if torch is None:
        raise RuntimeError("PyTorch is required for schedulers")
    
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

