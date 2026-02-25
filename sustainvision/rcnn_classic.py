"""Experimental classic R-CNN style detection pipeline.

This is an intentionally simple, slower two-stage detector:
- generate region proposals (grid + jittered GT during training)
- crop/resize ROI patches from the image
- run a backbone on each ROI patch
- predict class + refined box for each ROI

It is designed as an evaluation/experimentation path, not a production detector.
"""

from __future__ import annotations

import math
import csv
import hashlib
import json
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - runtime safeguard
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from torchvision.ops import roi_align  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - runtime safeguard
    roi_align = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from codecarbon import EmissionsTracker  # type: ignore
except Exception:  # pragma: no cover - runtime safeguard
    EmissionsTracker = None  # type: ignore

from .config import TrainingConfig
from .data import (
    COCO_CONTIGUOUS_TO_ID,
    VOC_CIFAR10_SUBSET,
    build_detection_dataloaders,
)
from .detection_metrics import (
    _get_coco_api_from_dataset,
    _xyxy_to_xywh,
    apply_nms,
    compute_coco_map,
    compute_iou,
    compute_voc_map50,
)
from .detection_models import (
    _extract_backbone_features,
    build_model,
    infer_feature_dim,
    load_pretrained_backbone,
)
from .training import _log_resource_and_energy_snapshot
from .utils import resolve_device, set_seed


def _clamp_boxes_xyxy(boxes: "torch.Tensor", h: int, w: int) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required")
    out = boxes.clone()
    out[:, 0::2] = out[:, 0::2].clamp(min=0.0, max=float(w - 1))
    out[:, 1::2] = out[:, 1::2].clamp(min=0.0, max=float(h - 1))
    # enforce valid ordering and minimum size
    x1 = torch.minimum(out[:, 0], out[:, 2])
    y1 = torch.minimum(out[:, 1], out[:, 3])
    x2 = torch.maximum(out[:, 0], out[:, 2])
    y2 = torch.maximum(out[:, 1], out[:, 3])
    x2 = torch.maximum(x2, x1 + 1.0)
    y2 = torch.maximum(y2, y1 + 1.0)
    out = torch.stack([x1, y1, x2, y2], dim=-1)
    out[:, 0::2] = out[:, 0::2].clamp(min=0.0, max=float(w - 1))
    out[:, 1::2] = out[:, 1::2].clamp(min=0.0, max=float(h - 1))
    return out


def _generate_grid_proposals(
    h: int,
    w: int,
    *,
    max_props: int = 96,
) -> "torch.Tensor":
    """Generate simple multi-scale grid proposals in pixel xyxy."""
    if torch is None:
        raise RuntimeError("PyTorch is required")

    proposals: list[list[float]] = []
    scales = [0.2, 0.35, 0.5, 0.7]
    ratios = [0.5, 1.0, 2.0]
    centers_y = [0.2, 0.4, 0.6, 0.8]
    centers_x = [0.2, 0.4, 0.6, 0.8]

    for s in scales:
        base = s * min(h, w)
        for ar in ratios:
            box_h = base * math.sqrt(ar)
            box_w = base / math.sqrt(ar)
            for cyf in centers_y:
                for cxf in centers_x:
                    cy = cyf * h
                    cx = cxf * w
                    x1 = cx - box_w / 2
                    y1 = cy - box_h / 2
                    x2 = cx + box_w / 2
                    y2 = cy + box_h / 2
                    proposals.append([x1, y1, x2, y2])
                    if len(proposals) >= max_props:
                        break
                if len(proposals) >= max_props:
                    break
            if len(proposals) >= max_props:
                break
        if len(proposals) >= max_props:
            break

    if not proposals:
        proposals = [[0.0, 0.0, float(max(w - 1, 1)), float(max(h - 1, 1))]]
    return _clamp_boxes_xyxy(torch.tensor(proposals, dtype=torch.float32), h, w)


def _jitter_boxes(
    boxes: "torch.Tensor",
    h: int,
    w: int,
    *,
    num_jitters: int = 2,
    scale_noise: float = 0.15,
    center_noise: float = 0.1,
) -> "torch.Tensor":
    """Create jittered proposals around GT boxes (pixel xyxy)."""
    if torch is None:
        raise RuntimeError("PyTorch is required")
    if boxes.numel() == 0 or num_jitters <= 0:
        return boxes.new_zeros((0, 4))

    out: list[torch.Tensor] = []
    for _ in range(num_jitters):
        xy1 = boxes[:, :2]
        xy2 = boxes[:, 2:]
        wh = (xy2 - xy1).clamp(min=1.0)
        ctr = (xy1 + xy2) / 2.0
        ctr = ctr + (torch.rand_like(ctr) * 2 - 1) * center_noise * wh
        wh = wh * (1.0 + (torch.rand_like(wh) * 2 - 1) * scale_noise)
        x1y1 = ctr - wh / 2.0
        x2y2 = ctr + wh / 2.0
        out.append(torch.cat([x1y1, x2y2], dim=1))
    return _clamp_boxes_xyxy(torch.cat(out, dim=0), h, w)


_SELECTIVE_SEARCH_WARNED = False


def _image_tensor_to_uint8_bgr(image: "torch.Tensor") -> Any:
    """Convert CHW tensor (possibly ImageNet-normalized) to uint8 HWC BGR for OpenCV."""
    if torch is None:
        raise RuntimeError("PyTorch is required")
    img = image.detach().cpu()
    if img.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape={tuple(img.shape)}")
    if img.size(0) == 1:
        img = img.repeat(3, 1, 1)
    if img.size(0) != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {img.size(0)}")
    img = img.float()
    # Heuristic: if values are outside [0,1], assume ImageNet normalization was applied.
    try:
        vmin = float(img.min().item())
        vmax = float(img.max().item())
    except Exception:
        vmin, vmax = -1.0, 2.0
    if vmin < -0.1 or vmax > 1.1:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype).view(3, 1, 1)
        img = img * std + mean
    img = img.clamp(0.0, 1.0)
    rgb = (img.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    # OpenCV expects BGR
    return rgb[:, :, ::-1].copy()


def _run_selective_search(
    image: "torch.Tensor",
    *,
    max_props: int,
    mode: str = "fast",
    min_box_size: int = 8,
) -> "torch.Tensor":
    """Run OpenCV selective search on a resized image tensor and return pixel xyxy boxes."""
    if torch is None:
        raise RuntimeError("PyTorch is required")
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is not installed (required for selective search).") from exc

    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "segmentation"):
        raise RuntimeError(
            "OpenCV ximgproc selective search is unavailable. Install opencv-contrib-python."
        )

    bgr = _image_tensor_to_uint8_bgr(image)
    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(bgr)
    mode_norm = str(mode or "fast").strip().lower()
    if mode_norm in {"quality", "slow"}:
        ss.switchToSelectiveSearchQuality()
    else:
        ss.switchToSelectiveSearchFast()
    rects = ss.process()

    proposals: List[List[float]] = []
    seen: set[tuple[int, int, int, int]] = set()
    max_props = max(1, int(max_props))
    min_box_size = max(1, int(min_box_size))
    for rect in rects:
        try:
            x, y, rw, rh = [int(v) for v in rect]
        except Exception:
            continue
        if rw < min_box_size or rh < min_box_size:
            continue
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w - 1, x + rw)
        y2 = min(h - 1, y + rh)
        if x2 <= x1 or y2 <= y1:
            continue
        key = (x1, y1, x2, y2)
        if key in seen:
            continue
        seen.add(key)
        proposals.append([float(x1), float(y1), float(x2), float(y2)])
        if len(proposals) >= max_props:
            break

    if not proposals:
        proposals = [[0.0, 0.0, float(max(w - 1, 1)), float(max(h - 1, 1))]]
    return _clamp_boxes_xyxy(torch.tensor(proposals, dtype=torch.float32), h, w)


def _proposal_cache_key(
    *,
    image: "torch.Tensor",
    target: Dict[str, Any],
    split_name: str,
    method: str,
    mode: str,
    max_props: int,
) -> str:
    """Build a stable cache key for proposal generation."""
    image_id = target.get("image_id", None)
    try:
        if hasattr(image_id, "item"):
            image_id = int(image_id.item())
        elif image_id is not None:
            image_id = int(image_id)
        else:
            image_id = None
    except Exception:
        image_id = None

    if image_id is not None and image_id >= 0:
        base = f"id{image_id}"
    else:
        # Fallback for datasets that do not expose a stable image_id (e.g., some VOC layouts).
        img = image.detach().cpu().contiguous()
        digest = hashlib.sha1(img.numpy().tobytes()).hexdigest()[:16]
        base = f"sha1_{digest}"

    h = int(image.shape[-2])
    w = int(image.shape[-1])
    return f"{split_name}_{method}_{mode}_{base}_{h}x{w}_n{int(max_props)}.pt"


def _generate_base_proposals_for_image(
    *,
    image: "torch.Tensor",
    target: Dict[str, Any],
    proposal_method: str,
    max_grid_proposals: int,
    proposal_cache_dir: Optional[Path],
    proposal_cache_split: str,
    selective_search_mode: str,
    selective_search_min_box_size: int,
) -> "torch.Tensor":
    """Generate (or load cached) base proposals before train-time GT/jitter augmentation."""
    global _SELECTIVE_SEARCH_WARNED
    if torch is None:
        raise RuntimeError("PyTorch is required")

    h = int(image.shape[1])
    w = int(image.shape[2])
    method = str(proposal_method or "grid").strip().lower()
    if method not in {"grid", "selective_search", "ss"}:
        method = "grid"
    if method == "ss":
        method = "selective_search"

    if method == "grid":
        return _generate_grid_proposals(h, w, max_props=max_grid_proposals)

    cache_path: Optional[Path] = None
    if proposal_cache_dir is not None:
        try:
            proposal_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_name = _proposal_cache_key(
                image=image,
                target=target,
                split_name=proposal_cache_split,
                method=method,
                mode=selective_search_mode,
                max_props=max_grid_proposals,
            )
            cache_path = proposal_cache_dir / cache_name
            if cache_path.exists():
                cached = torch.load(cache_path, map_location="cpu")
                if isinstance(cached, torch.Tensor) and cached.ndim == 2 and cached.size(-1) == 4:
                    return _clamp_boxes_xyxy(cached.float(), h, w)
        except Exception:
            cache_path = None

    try:
        proposals = _run_selective_search(
            image,
            max_props=max_grid_proposals,
            mode=selective_search_mode,
            min_box_size=selective_search_min_box_size,
        )
    except Exception as exc:
        if not _SELECTIVE_SEARCH_WARNED:
            print(f"[warn] Selective search unavailable ({exc}); falling back to grid proposals.")
            _SELECTIVE_SEARCH_WARNED = True
        proposals = _generate_grid_proposals(h, w, max_props=max_grid_proposals)

    if cache_path is not None:
        try:
            torch.save(proposals.detach().cpu(), cache_path)
        except Exception as exc:
            print(f"[warn] Failed to cache proposals at {cache_path}: {exc}")

    return proposals


def _boxes_to_rois(boxes_list: Sequence["torch.Tensor"], device: "torch.device") -> "torch.Tensor":
    """Convert list[boxes_xyxy] to torchvision ROI format [N,5] with batch idx."""
    if torch is None:
        raise RuntimeError("PyTorch is required")
    rois = []
    for batch_idx, boxes in enumerate(boxes_list):
        if boxes is None or boxes.numel() == 0:
            continue
        batch_col = torch.full((boxes.size(0), 1), float(batch_idx), dtype=boxes.dtype, device=boxes.device)
        rois.append(torch.cat([batch_col, boxes], dim=1))
    if not rois:
        return torch.zeros((0, 5), dtype=torch.float32, device=device)
    return torch.cat(rois, dim=0).to(device)


class ClassicRCNNModel(nn.Module):  # type: ignore[name-defined]
    """Classic R-CNN style model that processes ROI crops independently."""

    def __init__(
        self,
        roi_backbone: "nn.Module",
        *,
        image_size: int,
        num_classes: int,
        freeze_backbone: bool = True,
        roi_size: int = 96,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required")

        self.backbone = roi_backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.num_classes = int(num_classes)
        self.roi_size = int(roi_size)
        self.image_size = int(image_size)

        for p in self.backbone.parameters():
            p.requires_grad = not self.freeze_backbone
        if self.freeze_backbone:
            self.backbone.eval()

        feature_dim = infer_feature_dim(self.backbone, self.roi_size)
        # Add 4D normalized proposal coordinates as a positional cue.
        mlp_in = feature_dim + 4
        self.classifier = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_classes + 1),  # +1 background
        )
        self.box_regressor = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def _extract_roi_features(
        self,
        images: "torch.Tensor",
        rois: "torch.Tensor",
    ) -> "torch.Tensor":
        if torch is None or roi_align is None:
            raise RuntimeError("PyTorch + torchvision.ops.roi_align are required for classic RCNN")
        if rois.numel() == 0:
            return torch.zeros((0, self.classifier[0].in_features - 4), device=images.device)

        # Crop from input images directly (classic R-CNN style per-ROI processing).
        patches = roi_align(
            images,
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        grad_enabled = (not self.freeze_backbone) and self.training
        grad_ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with grad_ctx:
            feat = _extract_backbone_features(self.backbone, patches)
            feat = F.normalize(feat, dim=1) if F is not None and feat.numel() > 0 else feat
        return feat

    def forward(
        self,
        images: "torch.Tensor",
        proposals_per_image: Sequence["torch.Tensor"],
    ) -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError("PyTorch is required")

        rois = _boxes_to_rois(proposals_per_image, images.device)
        if rois.numel() == 0:
            empty_logits = torch.zeros((0, self.num_classes + 1), device=images.device)
            empty_boxes = torch.zeros((0, 4), device=images.device)
            counts = [0 for _ in proposals_per_image]
            return {
                "logits": empty_logits,
                "pred_boxes_norm": empty_boxes,
                "roi_batch_indices": torch.zeros((0,), dtype=torch.long, device=images.device),
                "proposal_boxes": torch.zeros((0, 4), device=images.device),
                "proposal_counts": counts,
            }

        features = self._extract_roi_features(images, rois)
        proposal_boxes = rois[:, 1:5]
        batch_idx = rois[:, 0].long()

        # Proposal coords normalized by the corresponding image size.
        norm_boxes = []
        for bidx, box in zip(batch_idx.tolist(), proposal_boxes):
            h = float(images[bidx].shape[1])
            w = float(images[bidx].shape[2])
            norm_boxes.append(
                torch.tensor(
                    [box[0] / w, box[1] / h, box[2] / w, box[3] / h],
                    dtype=box.dtype,
                    device=box.device,
                )
            )
        proposal_boxes_norm = torch.stack(norm_boxes, dim=0) if norm_boxes else proposal_boxes.new_zeros((0, 4))

        head_in = torch.cat([features, proposal_boxes_norm], dim=1)
        logits = self.classifier(head_in)

        # Predict refined boxes directly in normalized xyxy and clamp to valid range.
        pred_boxes_raw = torch.sigmoid(self.box_regressor(head_in))
        x1 = torch.minimum(pred_boxes_raw[:, 0], pred_boxes_raw[:, 2])
        y1 = torch.minimum(pred_boxes_raw[:, 1], pred_boxes_raw[:, 3])
        x2 = torch.maximum(pred_boxes_raw[:, 0], pred_boxes_raw[:, 2])
        y2 = torch.maximum(pred_boxes_raw[:, 1], pred_boxes_raw[:, 3])
        pred_boxes_norm = torch.stack([x1, y1, x2, y2], dim=-1)

        counts = [int(p.size(0)) for p in proposals_per_image]
        return {
            "logits": logits,
            "pred_boxes_norm": pred_boxes_norm,
            "roi_batch_indices": batch_idx,
            "proposal_boxes": proposal_boxes,
            "proposal_counts": counts,
        }


def build_rcnn_classic_model(
    *,
    config: TrainingConfig,
    checkpoint_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Path] = None,
    num_classes: int,
) -> ClassicRCNNModel:
    """Build classic R-CNN model using a SustainVision backbone or fresh backbone."""
    if torch is None:
        raise RuntimeError("PyTorch is required")

    image_size = int(config.hyperparameters.get("image_size", 224))
    backbone_image_size = int(config.hyperparameters.get("backbone_image_size", image_size))
    projection_dim = int(config.hyperparameters.get("projection_dim", 128))
    projection_hidden_dim = config.hyperparameters.get("projection_hidden_dim")
    projection_use_bn = bool(config.hyperparameters.get("projection_use_bn", False))

    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if not ckpt.is_absolute():
            base = Path(project_root) if project_root is not None else Path.cwd()
            ckpt = (base / ckpt).resolve()
        backbone = load_pretrained_backbone(
            ckpt,
            config.model,
            image_size=backbone_image_size,
            projection_dim=projection_dim,
            projection_hidden_dim=projection_hidden_dim,
            projection_use_bn=projection_use_bn,
        )
    else:
        full_model = build_model(
            config.model,
            num_classes=10,
            image_size=backbone_image_size,
            projection_dim=projection_dim,
            projection_hidden_dim=projection_hidden_dim,
            projection_use_bn=projection_use_bn,
        )
        backbone = full_model.backbone if hasattr(full_model, "backbone") else full_model

    eval_cfg = getattr(config, "evaluation", {}) or {}
    freeze_backbone = bool(eval_cfg.get("freeze_backbone", checkpoint_path is not None))
    roi_size = int(eval_cfg.get("rcnn_classic_roi_size", 96))
    hidden_dim = int(eval_cfg.get("hidden_dim", 256))

    return ClassicRCNNModel(
        backbone,
        image_size=image_size,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        roi_size=roi_size,
        hidden_dim=hidden_dim,
    )


@dataclass
class _RoiBatchTargets:
    cls_labels: "torch.Tensor"  # [R] in [0..C] (0=background)
    box_targets_norm: "torch.Tensor"  # [R,4] normalized xyxy
    positive_mask: "torch.Tensor"  # [R] bool


def _build_training_targets_for_rois(
    *,
    proposals_per_image: Sequence["torch.Tensor"],
    targets: Sequence[Dict[str, Any]],
    images: "torch.Tensor",
    num_classes: int,
    pos_iou: float = 0.5,
    neg_iou: float = 0.3,
) -> _RoiBatchTargets:
    if torch is None:
        raise RuntimeError("PyTorch is required")

    cls_labels_parts: list[torch.Tensor] = []
    box_targets_parts: list[torch.Tensor] = []
    positive_mask_parts: list[torch.Tensor] = []

    for img_idx, proposals in enumerate(proposals_per_image):
        device = proposals.device
        n = proposals.size(0)
        if n == 0:
            continue
        target = targets[img_idx]
        gt_boxes = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).to(device)
        gt_labels = target.get("labels", torch.zeros((0,), dtype=torch.long)).to(device)
        cls_labels = torch.zeros((n,), dtype=torch.long, device=device)  # background by default
        box_targets = torch.zeros((n, 4), dtype=torch.float32, device=device)
        positive_mask = torch.zeros((n,), dtype=torch.bool, device=device)

        if gt_boxes.numel() > 0:
            iou = compute_iou(proposals, gt_boxes)
            best_iou, best_gt = iou.max(dim=1)
            pos_mask = best_iou >= float(pos_iou)
            ignore_mask = (best_iou >= float(neg_iou)) & ~pos_mask

            # Internal classification labels: 0 background, 1..C foreground classes.
            valid_gt_labels = gt_labels[best_gt].clamp(min=0, max=max(num_classes - 1, 0))
            cls_labels[pos_mask] = valid_gt_labels[pos_mask] + 1
            cls_labels[ignore_mask] = -1  # ignore in CE
            positive_mask = pos_mask

            h = float(images[img_idx].shape[1])
            w = float(images[img_idx].shape[2])
            gt_pos = gt_boxes[best_gt]
            if gt_pos.numel() > 0:
                box_targets = torch.stack(
                    [
                        gt_pos[:, 0] / w,
                        gt_pos[:, 1] / h,
                        gt_pos[:, 2] / w,
                        gt_pos[:, 3] / h,
                    ],
                    dim=1,
                )

        cls_labels_parts.append(cls_labels)
        box_targets_parts.append(box_targets)
        positive_mask_parts.append(positive_mask)

    if not cls_labels_parts:
        dev = images.device
        return _RoiBatchTargets(
            cls_labels=torch.zeros((0,), dtype=torch.long, device=dev),
            box_targets_norm=torch.zeros((0, 4), dtype=torch.float32, device=dev),
            positive_mask=torch.zeros((0,), dtype=torch.bool, device=dev),
        )
    return _RoiBatchTargets(
        cls_labels=torch.cat(cls_labels_parts, dim=0),
        box_targets_norm=torch.cat(box_targets_parts, dim=0),
        positive_mask=torch.cat(positive_mask_parts, dim=0),
    )


def _generate_proposals_for_batch(
    *,
    images: "torch.Tensor",
    targets: Sequence[Dict[str, Any]],
    training: bool,
    max_grid_proposals: int,
    jitter_per_gt: int,
    proposal_method: str = "grid",
    proposal_cache_dir: Optional[Path] = None,
    proposal_cache_split: str = "train",
    selective_search_mode: str = "fast",
    selective_search_min_box_size: int = 8,
) -> List["torch.Tensor"]:
    if torch is None:
        raise RuntimeError("PyTorch is required")

    proposals_per_image: List[torch.Tensor] = []
    for i in range(images.size(0)):
        h = int(images[i].shape[1])
        w = int(images[i].shape[2])
        proposals = _generate_base_proposals_for_image(
            image=images[i],
            target=targets[i],
            proposal_method=proposal_method,
            max_grid_proposals=max_grid_proposals,
            proposal_cache_dir=proposal_cache_dir,
            proposal_cache_split=proposal_cache_split,
            selective_search_mode=selective_search_mode,
            selective_search_min_box_size=selective_search_min_box_size,
        ).to(images.device)
        if training:
            gt_boxes = targets[i].get("boxes", torch.zeros((0, 4))).to(images.device)
            if gt_boxes.numel() > 0:
                jittered = _jitter_boxes(gt_boxes, h, w, num_jitters=jitter_per_gt).to(images.device)
                proposals = torch.cat([proposals, gt_boxes, jittered], dim=0)
                proposals = _clamp_boxes_xyxy(proposals, h, w)
        proposals_per_image.append(proposals)
    return proposals_per_image


def _decode_predictions_for_batch(
    *,
    outputs: Dict[str, Any],
    images: "torch.Tensor",
    score_threshold: float,
    nms_iou_threshold: float,
) -> List[Dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required")

    logits = outputs["logits"]
    pred_boxes_norm = outputs["pred_boxes_norm"]
    batch_indices = outputs["roi_batch_indices"]
    proposal_counts = outputs["proposal_counts"]

    probs = torch.softmax(logits, dim=1) if logits.numel() > 0 else logits.new_zeros((0, 1))
    scores, labels_internal = probs.max(dim=1) if logits.numel() > 0 else (
        logits.new_zeros((0,)),
        logits.new_zeros((0,), dtype=torch.long),
    )

    preds_by_image: List[Dict[str, Any]] = []
    for img_idx in range(len(proposal_counts)):
        mask = batch_indices == img_idx if batch_indices.numel() > 0 else torch.zeros((0,), dtype=torch.bool, device=images.device)
        if mask.numel() == 0 or not bool(mask.any().item()):
            preds_by_image.append({"boxes": [], "scores": [], "labels": []})
            continue

        labels_i = labels_internal[mask]
        scores_i = scores[mask]
        boxes_norm_i = pred_boxes_norm[mask]

        # Drop background (class 0).
        keep = (labels_i > 0) & (scores_i >= float(score_threshold))
        if not keep.any():
            preds_by_image.append({"boxes": [], "scores": [], "labels": []})
            continue

        labels_i = labels_i[keep] - 1  # back to contiguous [0..C-1]
        scores_i = scores_i[keep]
        boxes_norm_i = boxes_norm_i[keep]

        h = int(images[img_idx].shape[1])
        w = int(images[img_idx].shape[2])
        boxes_xyxy = torch.stack(
            [
                boxes_norm_i[:, 0] * w,
                boxes_norm_i[:, 1] * h,
                boxes_norm_i[:, 2] * w,
                boxes_norm_i[:, 3] * h,
            ],
            dim=1,
        )
        boxes_xyxy = _clamp_boxes_xyxy(boxes_xyxy, h, w)
        boxes_xyxy, scores_i, labels_i = apply_nms(
            boxes_xyxy, scores_i, labels_i, iou_threshold=float(nms_iou_threshold)
        )
        preds_by_image.append(
            {
                "boxes": boxes_xyxy.detach().cpu().tolist(),
                "scores": scores_i.detach().cpu().tolist(),
                "labels": labels_i.detach().cpu().tolist(),
            }
        )
    return preds_by_image


def evaluate_rcnn_classic(
    config: TrainingConfig,
    checkpoint_path: Optional[Union[str, Path]] = None,
    *,
    project_root: Optional[Path] = None,
    report_path: Optional[Path] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Train/evaluate the classic RCNN variant on COCO/VOC/toy_detection."""
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required")
    if roi_align is None:
        raise RuntimeError("torchvision.ops.roi_align is required for rcnn_classic")

    set_seed(config.seed)
    device = resolve_device(config.device)
    eval_cfg = getattr(config, "evaluation", {}) or {}
    image_size = int(config.hyperparameters.get("image_size", 224))
    root = Path(project_root or Path.cwd())

    # Optional checkpoint/CSV logging similar to the other detection loops.
    save_model = bool(getattr(config, "save_model", False))
    save_model_path = getattr(config, "save_model_path", None) or "output_object_detection/rcnn_classic_run"
    save_dir = Path(save_model_path if isinstance(save_model_path, str) else str(save_model_path))
    if not save_dir.is_absolute():
        save_dir = (root / save_dir).resolve()
    if save_model:
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            import yaml  # type: ignore

            cfg_path = save_dir / "config.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(getattr(config, "__dict__", {}), f, sort_keys=False)
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"[warn] Failed to write classic RCNN config snapshot: {exc}")

    if report_path is None:
        report_filename = getattr(config, "report_filename", None)
        if isinstance(report_filename, str) and report_filename:
            report_path = Path(report_filename)
            if not report_path.is_absolute():
                report_path = (root / report_path).resolve()
        else:
            report_path = save_dir / "detection_metrics.csv"
    else:
        report_path = Path(report_path)
        if not report_path.is_absolute():
            report_path = (root / report_path).resolve()

    metrics_rows: List[Dict[str, Any]] = []
    checkpoint_interval = int(config.hyperparameters.get("checkpoint_interval", 50) or 50)

    proposal_method = str(eval_cfg.get("rcnn_classic_proposal_method", "grid")).strip().lower()
    selective_search_mode = str(eval_cfg.get("rcnn_classic_selective_search_mode", "fast")).strip().lower()
    selective_search_min_box_size = int(eval_cfg.get("rcnn_classic_selective_search_min_box_size", 8))
    cache_proposals = bool(eval_cfg.get("rcnn_classic_cache_proposals", False))
    proposal_cache_dir: Optional[Path] = None
    if cache_proposals:
        cache_dir_cfg = eval_cfg.get("rcnn_classic_proposal_cache_dir")
        if isinstance(cache_dir_cfg, str) and cache_dir_cfg.strip():
            proposal_cache_dir = Path(cache_dir_cfg.strip())
            if not proposal_cache_dir.is_absolute():
                proposal_cache_dir = (root / proposal_cache_dir).resolve()
        else:
            proposal_cache_dir = (save_dir / "proposal_cache").resolve()
        proposal_cache_dir.mkdir(parents=True, exist_ok=True)

    filter_classes = None
    class_remap = None
    if eval_cfg.get("filter_coco_to_cifar100", False) and str(config.database).lower() == "coco":
        from .data import get_coco_classes_for_cifar100

        filter_classes = get_coco_classes_for_cifar100()

    if "voc" in str(config.database).lower():
        voc_subset = eval_cfg.get("voc_subset_classes")
        if isinstance(voc_subset, str):
            voc_subset_key = voc_subset.strip().lower()
            if voc_subset_key in {"cifar10", "cifar"}:
                voc_subset = VOC_CIFAR10_SUBSET
            else:
                voc_subset = None
        if isinstance(voc_subset, (list, tuple)) and len(voc_subset) > 0:
            from .data import resolve_voc_class_id

            ordered_ids: List[int] = []
            for entry in voc_subset:
                if isinstance(entry, int):
                    resolved = entry
                else:
                    resolved = resolve_voc_class_id(str(entry))
                if resolved is None:
                    continue
                if resolved not in ordered_ids:
                    ordered_ids.append(int(resolved))
            if ordered_ids:
                filter_classes = set(ordered_ids)
                class_remap = {old_id: new_idx for new_idx, old_id in enumerate(ordered_ids)}

    train_loader, val_loader, num_classes = build_detection_dataloaders(
        config.database,
        batch_size=int(config.hyperparameters.get("batch_size", 2)),
        num_workers=int(config.hyperparameters.get("num_workers", 0)),
        val_split=float(config.hyperparameters.get("val_split", 0.0)),
        seed=int(config.seed),
        project_root=project_root,
        image_size=image_size,
        max_train_images=config.hyperparameters.get("max_train_images"),
        max_val_images=config.hyperparameters.get("max_val_images"),
        filter_classes=filter_classes,
        class_remap=class_remap,
        # Classic R-CNN path uses roi_align over a single batched tensor, so we
        # require fixed-size images from the dataloader. Keep targets in pixel
        # coordinates, but on the resized image geometry.
        resize_images=True,
        normalize_images=True,
        box_coordinate_format="pixel",
        pin_memory=(device.type == "cuda"),
    )

    model = build_rcnn_classic_model(
        config=config,
        checkpoint_path=checkpoint_path,
        project_root=project_root,
        num_classes=num_classes,
    ).to(device)

    freeze_backbone = bool(getattr(model, "freeze_backbone", True))
    if freeze_backbone:
        optim_params = list(model.classifier.parameters()) + list(model.box_regressor.parameters())
    else:
        optim_params = list(model.parameters())

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(config.hyperparameters.get("lr", 1e-3)),
        weight_decay=float(config.weight_decay),
    )
    cls_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    bbox_criterion = nn.SmoothL1Loss()

    epochs = int(config.hyperparameters.get("epochs", 5))
    max_train_batches = config.hyperparameters.get("max_train_batches")
    max_val_batches = config.hyperparameters.get("max_val_batches")
    try:
        max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    except Exception:
        max_train_batches = None
    try:
        max_val_batches = int(max_val_batches) if max_val_batches is not None else None
    except Exception:
        max_val_batches = None

    gradient_clip_norm = config.gradient_clip_norm
    if gradient_clip_norm is None and "gradient_clip_norm" in (config.hyperparameters or {}):
        # Backward-compatible fallback for older configs while standardizing on top-level config.
        try:
            gradient_clip_norm = float(config.hyperparameters.get("gradient_clip_norm"))
            print("[warn] `hyperparameters.gradient_clip_norm` is deprecated for detection; use top-level `gradient_clip_norm`.")
        except Exception:
            gradient_clip_norm = None

    max_grid_props = int(eval_cfg.get("rcnn_classic_max_grid_proposals", 96))
    jitter_per_gt = int(eval_cfg.get("rcnn_classic_jitter_per_gt", 2))
    score_threshold = float(config.hyperparameters.get("score_threshold", 0.05))
    nms_iou_threshold = float(eval_cfg.get("nms_threshold", config.hyperparameters.get("nms_iou_threshold", 0.5)))

    best_val_loss = float("inf")
    best_epoch = 0
    final_train_loss = float("nan")
    final_val_loss = float("nan")
    track_emissions = bool(config.hyperparameters.get("track_emissions", True))
    emissions_kg_total: Optional[float] = None
    energy_kwh_total: Optional[float] = None
    emissions_data: Any = None
    tracker: Any = None
    if track_emissions and EmissionsTracker is not None:
        try:
            import logging

            logging.getLogger("codecarbon").setLevel(logging.WARNING)
            tracker = EmissionsTracker(
                measure_power_secs=5,
                project_name=f"rcnn_classic_{str(config.database)}",
                log_level="warning",
                save_to_file=False,
            )  # type: ignore[call-arg]
            tracker.start()
        except Exception as exc:  # pragma: no cover - best-effort
            tracker = None
            print(f"[warn] Failed to start CodeCarbon tracker: {exc}")
    elif track_emissions and EmissionsTracker is None:
        print("[warn] CodeCarbon is not installed; continuing without emissions tracking.")
    start_time = time.perf_counter()
    run_started_at = datetime.now().astimezone()
    if proposal_method in {"selective_search", "ss"}:
        cache_info = f", cache={proposal_cache_dir}" if proposal_cache_dir is not None else ""
        print(f"[info] Classic RCNN proposals: selective_search ({selective_search_mode}){cache_info}")
    else:
        print(f"[info] Classic RCNN proposals: grid (max={max_grid_props})")

    for epoch in range(epochs):
        model.train()
        if freeze_backbone and hasattr(model, "backbone"):
            model.backbone.eval()
        train_loss_sum = 0.0
        train_steps = 0

        try:
            from tqdm.auto import tqdm

            train_iterable = tqdm(
                train_loader,
                desc=f"Train {epoch+1}/{epochs} (rcnn_classic)",
                leave=False,
                unit="batch",
            )
        except Exception:
            train_iterable = train_loader

        for batch_idx, (images, targets) in enumerate(train_iterable):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            images = images.to(device)
            normalized_targets = []
            for t in targets:
                t_out: Dict[str, Any] = {}
                for k, v in t.items():
                    if k in {"boxes", "labels"} and hasattr(v, "to"):
                        t_out[k] = v.to(device)
                    else:
                        t_out[k] = v
                normalized_targets.append(t_out)
            targets = normalized_targets

            proposals_per_image = _generate_proposals_for_batch(
                images=images,
                targets=targets,
                training=True,
                max_grid_proposals=max_grid_props,
                jitter_per_gt=jitter_per_gt,
                proposal_method=proposal_method,
                proposal_cache_dir=proposal_cache_dir,
                proposal_cache_split="train",
                selective_search_mode=selective_search_mode,
                selective_search_min_box_size=selective_search_min_box_size,
            )
            outputs = model(images, proposals_per_image)
            roi_targets = _build_training_targets_for_rois(
                proposals_per_image=proposals_per_image,
                targets=targets,
                images=images,
                num_classes=num_classes,
            )
            if roi_targets.cls_labels.numel() == 0:
                continue

            logits = outputs["logits"]
            pred_boxes_norm = outputs["pred_boxes_norm"]
            cls_labels = roi_targets.cls_labels
            pos_mask = roi_targets.positive_mask

            ce_loss = cls_criterion(logits, cls_labels)
            if pos_mask.any():
                reg_loss = bbox_criterion(pred_boxes_norm[pos_mask], roi_targets.box_targets_norm[pos_mask])
            else:
                reg_loss = pred_boxes_norm.sum() * 0.0
            loss = ce_loss + float(eval_cfg.get("rcnn_classic_bbox_loss_weight", 5.0)) * reg_loss

            optimizer.zero_grad()
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_steps += 1

        final_train_loss = train_loss_sum / max(train_steps, 1)

        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        preds_all: List[Dict[str, Any]] = []
        gts_all: List[Dict[str, Any]] = []
        image_counter = 0

        try:
            from tqdm.auto import tqdm

            val_iterable = tqdm(
                val_loader,
                desc=f"Val {epoch+1}/{epochs} (rcnn_classic)",
                leave=False,
                unit="batch",
            )
        except Exception:
            val_iterable = val_loader

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_iterable):
                if max_val_batches is not None and batch_idx >= max_val_batches:
                    break
                images = images.to(device)
                normalized_targets = []
                for t in targets:
                    t_out: Dict[str, Any] = {}
                    for k, v in t.items():
                        if k in {"boxes", "labels"} and hasattr(v, "to"):
                            t_out[k] = v.to(device)
                        else:
                            t_out[k] = v
                    normalized_targets.append(t_out)
                targets = normalized_targets

                proposals_per_image = _generate_proposals_for_batch(
                    images=images,
                    targets=targets,
                    training=False,
                    max_grid_proposals=max_grid_props,
                    jitter_per_gt=0,
                    proposal_method=proposal_method,
                    proposal_cache_dir=proposal_cache_dir,
                    proposal_cache_split="val",
                    selective_search_mode=selective_search_mode,
                    selective_search_min_box_size=selective_search_min_box_size,
                )
                outputs = model(images, proposals_per_image)
                roi_targets = _build_training_targets_for_rois(
                    proposals_per_image=proposals_per_image,
                    targets=targets,
                    images=images,
                    num_classes=num_classes,
                )

                if roi_targets.cls_labels.numel() > 0:
                    logits = outputs["logits"]
                    pred_boxes_norm = outputs["pred_boxes_norm"]
                    ce_loss = cls_criterion(logits, roi_targets.cls_labels)
                    if roi_targets.positive_mask.any():
                        reg_loss = bbox_criterion(
                            pred_boxes_norm[roi_targets.positive_mask],
                            roi_targets.box_targets_norm[roi_targets.positive_mask],
                        )
                    else:
                        reg_loss = pred_boxes_norm.sum() * 0.0
                    loss = ce_loss + float(eval_cfg.get("rcnn_classic_bbox_loss_weight", 5.0)) * reg_loss
                    val_loss_sum += float(loss.item())
                    val_steps += 1

                batch_preds = _decode_predictions_for_batch(
                    outputs=outputs,
                    images=images,
                    score_threshold=score_threshold,
                    nms_iou_threshold=nms_iou_threshold,
                )

                for local_idx, (pred, tgt, img) in enumerate(zip(batch_preds, targets, images)):
                    image_id_raw = tgt.get("image_id", image_counter)
                    if hasattr(image_id_raw, "item"):
                        image_id = int(image_id_raw.item())
                    else:
                        image_id = int(image_id_raw)
                    preds_all.append(
                        {
                            "image_id": image_id,
                            "boxes": pred["boxes"],
                            "scores": pred["scores"],
                            "labels": pred["labels"],
                        }
                    )

                    gt_boxes = tgt.get("boxes", torch.zeros((0, 4), device=img.device))
                    gt_labels = tgt.get("labels", torch.zeros((0,), dtype=torch.long, device=img.device))
                    gts_all.append(
                        {
                            "image_id": image_id,
                            "boxes": gt_boxes.detach().cpu().tolist(),
                            "labels": gt_labels.detach().cpu().tolist(),
                        }
                    )
                    image_counter += 1

        final_val_loss = val_loss_sum / max(val_steps, 1)
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_epoch = epoch + 1

        print(
            f"Epoch {epoch+1}/{epochs} (rcnn_classic) - "
            f"train_loss={final_train_loss:.4f} val_loss={final_val_loss:.4f}"
        )

        metrics_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
                "best_val_loss_so_far": best_val_loss,
            }
        )

        if save_model and checkpoint_interval > 0 and (
            (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs
        ):
            try:
                ckpt_path = save_dir / f"epoch_{epoch+1:04d}.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": getattr(config, "__dict__", {}),
                        "train_loss": final_train_loss,
                        "val_loss": final_val_loss,
                    },
                    ckpt_path,
                )
            except Exception as exc:  # pragma: no cover - best-effort
                print(f"[warn] Failed to save classic RCNN checkpoint at epoch {epoch+1}: {exc}")

        if save_model and best_epoch == (epoch + 1):
            try:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": getattr(config, "__dict__", {}),
                        "train_loss": final_train_loss,
                        "val_loss": final_val_loss,
                    },
                    save_dir / "best_model.pt",
                )
            except Exception as exc:  # pragma: no cover - best-effort
                print(f"[warn] Failed to save best classic RCNN checkpoint at epoch {epoch+1}: {exc}")

    duration_seconds = time.perf_counter() - start_time
    finished_at = datetime.now().astimezone()
    finished_at_iso = finished_at.isoformat()
    finished_at_local = finished_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    if tracker is not None:
        try:
            emissions_kg_total = tracker.stop()
            emissions_data = getattr(tracker, "final_emissions_data", None)
            energy_kwh_total = getattr(emissions_data, "energy_consumed", None)
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"[warn] Failed to stop CodeCarbon tracker cleanly: {exc}")
    results: Dict[str, Any] = {
        "head_type": "rcnn_classic",
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "duration_seconds": duration_seconds,
        "emissions_kg": float(emissions_kg_total) if emissions_kg_total is not None else None,
        "energy_kwh": float(energy_kwh_total) if energy_kwh_total is not None else None,
        "proposal_method": proposal_method,
        "proposal_cache_dir": str(proposal_cache_dir) if proposal_cache_dir is not None else None,
        "started_at": run_started_at.isoformat(),
        "finished_at": finished_at_iso,
        "note": "Experimental classic R-CNN style pipeline using proposals + ROI crops.",
    }

    coco_gt = _get_coco_api_from_dataset(getattr(val_loader, "dataset", None))
    if coco_gt is not None:
        try:
            coco_predictions: List[Dict[str, Any]] = []
            for pred in preds_all:
                for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
                    coco_cat = COCO_CONTIGUOUS_TO_ID.get(int(label))
                    if coco_cat is None:
                        continue
                    box_xywh = _xyxy_to_xywh(torch.tensor([box], dtype=torch.float32))[0].tolist()
                    coco_predictions.append(
                        {
                            "image_id": int(pred["image_id"]),
                            "category_id": int(coco_cat),
                            "bbox": [float(x) for x in box_xywh],
                            "score": float(score),
                        }
                    )
            results.update(compute_coco_map(coco_gt, coco_predictions))
        except Exception as exc:
            results["note"] += f" COCO mAP failed: {exc}."
    elif "voc" in str(config.database).lower():
        try:
            results.update(compute_voc_map50(gts_all, preds_all, num_classes=num_classes))
        except Exception as exc:
            results["note"] += f" VOC mAP failed: {exc}."

    if save_model:
        # Save a one-row resource/energy snapshot at run end (best-effort).
        try:
            _log_resource_and_energy_snapshot(
                artifact_dir=save_dir,
                epoch=epochs,
                emissions_data=emissions_data,
                elapsed_seconds=duration_seconds,
                device=device,
            )
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"[warn] Failed to write resource_energy_log.csv: {exc}")

        # Persist final results in both JSON and YAML for easy scripting/manual review.
        try:
            with (save_dir / "results.json").open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, sort_keys=True, default=str)
        except Exception as exc:
            print(f"[warn] Failed to write classic RCNN results.json: {exc}")
        try:
            import yaml  # type: ignore

            with (save_dir / "results.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(results, f, sort_keys=False)
        except Exception as exc:
            print(f"[warn] Failed to write classic RCNN results.yaml: {exc}")

        # Save predictions/GTs from the final validation epoch for debugging metric failures.
        try:
            with (save_dir / "val_predictions_last_epoch.json").open("w", encoding="utf-8") as f:
                json.dump(preds_all, f, indent=2)
            with (save_dir / "val_ground_truth_last_epoch.json").open("w", encoding="utf-8") as f:
                json.dump(gts_all, f, indent=2)
        except Exception as exc:
            print(f"[warn] Failed to write final validation predictions/GT JSON: {exc}")

        # Epoch metrics + summary row (with finish timestamp and sustainability fields).
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "best_val_loss_so_far",
                    "best_epoch",
                    "final_train_loss",
                    "final_val_loss",
                    "map50",
                    "emissions_kg",
                    "energy_kwh",
                    "duration_seconds",
                    "proposal_method",
                    "proposal_cache_dir",
                    "started_at",
                    "finished_at",
                    "finished_at_local",
                    "note",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in metrics_rows:
                    writer.writerow(
                        {
                            **row,
                            "best_epoch": "",
                            "final_train_loss": "",
                            "final_val_loss": "",
                            "map50": "",
                            "emissions_kg": "",
                            "energy_kwh": "",
                            "duration_seconds": "",
                            "proposal_method": proposal_method,
                            "proposal_cache_dir": str(proposal_cache_dir) if proposal_cache_dir is not None else "",
                            "started_at": "",
                            "finished_at": "",
                            "finished_at_local": "",
                            "note": "",
                        }
                    )
                writer.writerow(
                    {
                        "epoch": "summary",
                        "train_loss": "",
                        "val_loss": "",
                        "best_val_loss_so_far": results.get("best_val_loss", ""),
                        "best_epoch": results.get("best_epoch", ""),
                        "final_train_loss": results.get("final_train_loss", ""),
                        "final_val_loss": results.get("final_val_loss", ""),
                        "map50": results.get("map50", ""),
                        "emissions_kg": results.get("emissions_kg", ""),
                        "energy_kwh": results.get("energy_kwh", ""),
                        "duration_seconds": results.get("duration_seconds", ""),
                        "proposal_method": proposal_method,
                        "proposal_cache_dir": str(proposal_cache_dir) if proposal_cache_dir is not None else "",
                        "started_at": results.get("started_at", ""),
                        "finished_at": results.get("finished_at", ""),
                        "finished_at_local": finished_at_local,
                        "note": results.get("note", ""),
                    }
                )
        except Exception as exc:
            print(f"[warn] Failed to write classic RCNN CSV report to {report_path}: {exc}")

    return results
