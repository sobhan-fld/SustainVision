"""Optimizer and scheduler builders."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import torch
except Exception:
    torch = None  # type: ignore


if torch is not None:
    class LARS(torch.optim.Optimizer):  # type: ignore[attr-defined]
        """Layer-wise Adaptive Rate Scaling optimizer.

        Minimal implementation inspired by SimCLR / ImageNet training recipes.
        """

        def __init__(
            self,
            params,
            lr: float = 0.1,
            momentum: float = 0.9,
            weight_decay: float = 0.0,
            eta: float = 0.001,
            eps: float = 1e-9,
            exclude_bias_n_norm: bool = True,
        ):
            defaults = dict(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                eta=eta,
                eps=eps,
                exclude_bias_n_norm=exclude_bias_n_norm,
            )
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                eta = group["eta"]
                eps = group["eps"]
                exclude_bias_n_norm = group["exclude_bias_n_norm"]

                for param in group["params"]:
                    if param.grad is None:
                        continue
                    grad = param.grad
                    if grad.is_sparse:
                        raise RuntimeError("LARS does not support sparse gradients.")

                    grad = grad.detach()
                    param_norm = torch.norm(param)

                    apply_weight_decay = weight_decay != 0.0 and not (
                        exclude_bias_n_norm and param.ndim == 1
                    )
                    if apply_weight_decay:
                        grad = grad + weight_decay * param

                    grad_norm = torch.norm(grad)
                    if exclude_bias_n_norm and param.ndim == 1:
                        trust_ratio = 1.0
                    else:
                        trust_ratio = 1.0
                        if param_norm > 0.0 and grad_norm > 0.0:
                            trust_ratio = eta * param_norm / (grad_norm + eps)

                    scaled_lr = trust_ratio * lr
                    if momentum > 0:
                        state = self.state.setdefault(param, {})
                        buf = state.get("momentum_buffer")
                        if buf is None:
                            buf = grad.clone()
                            state["momentum_buffer"] = buf
                        else:
                            buf.mul_(momentum).add_(grad)
                        update = buf
                    else:
                        update = grad

                    param.add_(update, alpha=-scaled_lr)

            return loss

else:  # pragma: no cover - torch missing path
    LARS = None  # type: ignore


def build_optimizer(
    name: str,
    params,
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    lars_eta: float = 0.001,
    lars_eps: float = 1e-9,
    lars_exclude_bias_n_norm: bool = True,
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
    if key == "lars":
        if LARS is None:
            print("[warn] LARS optimizer requested but PyTorch is unavailable. Falling back to SGD.")
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        return LARS(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=float(lars_eta),
            eps=float(lars_eps),
            exclude_bias_n_norm=bool(lars_exclude_bias_n_norm),
        )

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

