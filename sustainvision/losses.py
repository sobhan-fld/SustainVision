"""Loss function implementations and builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    from torch import nn
    import torch.nn.functional as F
    import torch
except Exception:
    nn = None  # type: ignore
    F = None  # type: ignore
    torch = None  # type: ignore


@dataclass
class LossSpec:
    """Description of the selected loss function."""

    name: str
    mode: str  # 'logits', 'simclr', 'supcon', 'bce'
    criterion: Optional[Any] = None


def build_loss(name: str) -> LossSpec:
    """Build a loss function specification from name."""
    if nn is None:
        raise RuntimeError("PyTorch is required for loss functions")
    
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


def compute_loss(
    spec: LossSpec,
    logits: "torch.Tensor",
    embeddings: "torch.Tensor",
    labels: "torch.Tensor",
    temperature: float,
) -> "torch.Tensor":  # type: ignore[name-defined]
    """Compute loss based on the loss specification."""
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for loss computation")
    
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
        return simclr_loss(embeddings, temperature)
    if spec.mode == "supcon":
        return supcon_loss(embeddings, labels, temperature)
    # Fallback safety
    assert spec.criterion is not None
    return spec.criterion(logits, labels)


def simclr_loss(embeddings: "torch.Tensor", temperature: float) -> "torch.Tensor":  # type: ignore[name-defined]
    """Compute SimCLR contrastive loss."""
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for SimCLR loss")
    
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


def supcon_loss(
    embeddings: "torch.Tensor",
    labels: "torch.Tensor",
    temperature: float,
) -> "torch.Tensor":  # type: ignore[name-defined]
    """Compute Supervised Contrastive (SupCon) loss."""
    if torch is None:
        raise RuntimeError("PyTorch is required for SupCon loss")
    
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

