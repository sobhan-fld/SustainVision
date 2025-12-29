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
    """Compute SimCLR contrastive loss using two augmented views per sample."""
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for SimCLR loss")
    
    if embeddings.size(0) % 2 != 0:
        raise ValueError("SimCLR embeddings must contain an even number of samples (two views per item).")

    batch_size = embeddings.size(0) // 2
    if batch_size == 0:
        return embeddings.new_zeros(())

    representations = F.normalize(embeddings, dim=1)
    similarity_matrix = torch.matmul(representations, representations.T) / max(temperature, 1e-6)
    mask = torch.eye(2 * batch_size, device=embeddings.device, dtype=torch.bool)
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

    targets = torch.arange(2 * batch_size, device=embeddings.device)
    targets = (targets + batch_size) % (2 * batch_size)

    return F.cross_entropy(similarity_matrix, targets)


def supcon_loss(
    embeddings: "torch.Tensor",
    labels: "torch.Tensor",
    temperature: float,
) -> "torch.Tensor":  # type: ignore[name-defined]
    """Compute Supervised Contrastive (SupCon) loss with two views per sample.
    
    Implementation follows the standard SupCon formula:
    - Same sample's two views are always positive
    - Different samples of the same class are positive
    - Different classes are negative
    
    Features come in as [view1_all_samples, view2_all_samples] and are interleaved
    to match the mask structure which expects [view1_sample1, view2_sample1, view1_sample2, view2_sample2, ...]
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for SupCon loss")
    
    if embeddings.size(0) % 2 != 0:
        raise ValueError("SupCon embeddings must contain an even number of samples (two views per item).")
    
    device = embeddings.device
    n_views = 2
    batch_size = embeddings.size(0) // n_views
    if batch_size == 0:
        return embeddings.new_zeros(())

    # Normalize embeddings
    features = F.normalize(embeddings, dim=1)
    
    # Extract labels: [batch_size * 2] -> [batch_size] (one label per original sample)
    # Labels come as [labels_all, labels_all], so we take first half
    labels_flat = labels[:batch_size]
    
    # Interleave features to match mask structure
    # Input: [view1_sample1, view1_sample2, ..., view2_sample1, view2_sample2, ...]
    # Output: [view1_sample1, view2_sample1, view1_sample2, view2_sample2, ...]
    # This ensures same-sample pairs are at positions (i, i+1) for i in [0, 2, 4, ...]
    indices = torch.arange(batch_size * n_views, device=device)
    indices = indices.view(n_views, batch_size).t().contiguous().view(-1)
    features = features[indices]

    # Compute similarity matrix: [batch_size * n_views, batch_size * n_views]
    logits = torch.matmul(features, features.T) / max(temperature, 1e-6)
    
    # Create mask for positive pairs
    # 1. Same sample's two views (adjacent positions after interleaving)
    # After interleaving, positions (2i, 2i+1) are the two views of sample i
    same_sample_mask = torch.zeros(batch_size * n_views, batch_size * n_views, device=device)
    for i in range(batch_size):
        idx1 = i * n_views
        idx2 = i * n_views + 1
        same_sample_mask[idx1, idx2] = 1.0
        same_sample_mask[idx2, idx1] = 1.0
    
    # 2. Same class across different samples
    labels_expanded = labels_flat.repeat_interleave(n_views)  # [view1_label1, view2_label1, view1_label2, ...]
    class_mask = torch.eq(labels_expanded.unsqueeze(1), labels_expanded.unsqueeze(0)).float()
    
    # Combine: positive if same sample OR same class
    if hasattr(torch, 'maximum'):
        mask = torch.maximum(same_sample_mask, class_mask)
    else:
        mask = torch.max(same_sample_mask, class_mask)
    
    # Exclude self-similarity (diagonal)
    mask = mask * (1.0 - torch.eye(batch_size * n_views, device=device))
    
    # Compute log probabilities
    exp_logits = torch.exp(logits) * (1.0 - torch.eye(batch_size * n_views, device=device))
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    
    # Compute mean log prob of positive pairs for each anchor
    mask_sum = mask.sum(dim=1)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / torch.clamp(mask_sum, min=1.0)
    
    # Average over all anchors
    loss = -mean_log_prob_pos.mean()
    return loss

