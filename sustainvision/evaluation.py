"""Evaluation module for multi-head evaluation of pretrained feature spaces.

This module allows loading a pretrained backbone and evaluating it with
different heads (classification, object detection) without retraining.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    DataLoader = None  # type: ignore

from .config import TrainingConfig
from .models import build_model
from .data import build_classification_dataloaders, build_detection_dataloaders, COCO_CONTIGUOUS_TO_ID
from .training import _log_resource_and_energy_snapshot

try:
    from codecarbon import EmissionsTracker  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EmissionsTracker = None  # type: ignore


class DetectionHead(nn.Module):  # type: ignore[name-defined]
    """Object detection head for bounding box prediction.
    
    Takes backbone features and predicts:
    - Class logits for each anchor/region
    - Bounding box coordinates in **normalized** \(x1, y1, x2, y2\) format for each slot.

    Note:
        This is a simplified, slot-based head (fixed number of predictions per image)
        built on a global feature vector. It is intended for *evaluation* of
        representation quality (linear / shallow heads), not as a production detector.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_anchors: int = 9,  # 3 scales × 3 aspect ratios
        hidden_dim: int = 256,
        use_bn: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification head: predict class for each anchor
        cls_layers = [nn.Linear(feature_dim, hidden_dim)]
        if use_bn:
            cls_layers.append(nn.BatchNorm1d(hidden_dim))
        cls_layers.append(nn.ReLU())
        if dropout > 0:
            cls_layers.append(nn.Dropout(dropout))
        cls_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            cls_layers.append(nn.BatchNorm1d(hidden_dim))
        cls_layers.append(nn.ReLU())
        if dropout > 0:
            cls_layers.append(nn.Dropout(dropout))
        cls_layers.append(nn.Linear(hidden_dim, num_anchors * num_classes))
        self.cls_head = nn.Sequential(*cls_layers)
        
        # Regression head: predict bbox offsets (dx, dy, dw, dh) for each anchor
        bbox_layers = [nn.Linear(feature_dim, hidden_dim)]
        if use_bn:
            bbox_layers.append(nn.BatchNorm1d(hidden_dim))
        bbox_layers.append(nn.ReLU())
        if dropout > 0:
            bbox_layers.append(nn.Dropout(dropout))
        bbox_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            bbox_layers.append(nn.BatchNorm1d(hidden_dim))
        bbox_layers.append(nn.ReLU())
        if dropout > 0:
            bbox_layers.append(nn.Dropout(dropout))
        bbox_layers.append(nn.Linear(hidden_dim, num_anchors * 4))  # 4 = (dx, dy, dw, dh)
        self.bbox_head = nn.Sequential(*bbox_layers)
        
        # Better weight initialization for faster convergence
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights with Xavier/Kaiming for better convergence.
        
        For bbox head, use smaller initialization to start with boxes near center.
        """
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # Bbox head: initialize smaller to start with boxes near image center
                if 'bbox_head' in name:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        # Initialize bbox biases to predict boxes near center (0.25-0.75 range)
                        # After sigmoid, this gives boxes around center of image
                        nn.init.constant_(m.bias, 0.0)
                else:
                    # Classification head: use Xavier
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning class logits and bbox predictions.
        
        Args:
            features: Backbone features [batch_size, feature_dim]
            
        Returns:
            Tuple of (class_logits, bbox_preds):
            - class_logits: [batch_size, num_anchors, num_classes]
            - bbox_preds: [batch_size, num_anchors, 4] in normalized xyxy format (0..1).
        """
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required for model forward pass")
        
        cls_logits = self.cls_head(features)
        bbox_preds = self.bbox_head(features)
        
        # Reshape to [batch_size, num_anchors, num_classes] and [batch_size, num_anchors, 4]
        batch_size = features.size(0)
        cls_logits = cls_logits.view(batch_size, self.num_anchors, self.num_classes)
        bbox_preds = bbox_preds.view(batch_size, self.num_anchors, 4)

        # Constrain predicted boxes to a valid normalized xyxy range.
        # We first squash to [0,1], then enforce x1<=x2 and y1<=y2.
        boxes = torch.sigmoid(bbox_preds)
        x1 = torch.min(boxes[:, :, 0], boxes[:, :, 2])
        y1 = torch.min(boxes[:, :, 1], boxes[:, :, 3])
        x2 = torch.max(boxes[:, :, 0], boxes[:, :, 2])
        y2 = torch.max(boxes[:, :, 1], boxes[:, :, 3])
        bbox_preds = torch.stack([x1, y1, x2, y2], dim=-1)
        
        return cls_logits, bbox_preds


class MultiHeadModel(nn.Module):  # type: ignore[name-defined]
    """Model that supports multiple heads on a shared backbone.
    
    This allows evaluating the same pretrained backbone with different
    task-specific heads (classification, object detection, etc.).
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        head_type: str = "classification",
        num_classes: Optional[int] = None,
        projection_dim: Optional[int] = None,
        freeze_backbone: bool = True,
        **head_kwargs: Any,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.head_type = head_type
        self.freeze_backbone = freeze_backbone
        
        # Freeze or unfreeze backbone based on parameter
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.backbone.train()
        
        if head_type == "classification":
            if num_classes is None:
                raise ValueError("num_classes required for classification head")
            self.head = nn.Linear(feature_dim, num_classes)
        elif head_type == "detection":
            if num_classes is None:
                raise ValueError("num_classes required for detection head")
            num_anchors = head_kwargs.get("num_anchors", 9)
            hidden_dim = head_kwargs.get("hidden_dim", 256)
            use_bn = head_kwargs.get("use_bn", True)
            dropout = head_kwargs.get("dropout", 0.1)
            self.head = DetectionHead(feature_dim, num_classes, num_anchors, hidden_dim, use_bn=use_bn, dropout=dropout)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass through backbone and head.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            For classification: logits [batch_size, num_classes]
            For detection: (class_logits, bbox_preds) where
                class_logits: [batch_size, num_anchors, num_classes]
                bbox_preds: [batch_size, num_anchors, 4]
        """
        if torch is None:
            raise RuntimeError("PyTorch is required for model forward pass")
        
        # Extract PURE feature vector from backbone (bypass classifier entirely)
        with torch.no_grad():
            # For MobileNet V3, extract features BEFORE classifier
            # MobileNet structure: features -> avgpool -> classifier
            # We want: features -> avgpool -> flatten (skip classifier)
            if hasattr(self.backbone, 'features') and hasattr(self.backbone, 'avgpool'):
                # Pure feature extraction: features + avgpool only
                feat_maps = self.backbone.features(x)  # [B, C, H, W]
                features = self.backbone.avgpool(feat_maps)  # [B, C, 1, 1]
                features = torch.flatten(features, 1)  # [B, C]
            elif hasattr(self.backbone, 'avgpool') and not hasattr(self.backbone, 'features'):
                # For ResNet: conv layers -> avgpool -> fc (fc already replaced with Identity)
                # Extract: conv layers -> avgpool -> flatten
                # ResNet forward goes through all layers, but fc is Identity, so we can use it directly
                # However, to be safe, let's extract manually
                # ResNet structure: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
                # Since fc is Identity, backbone(x) should give us features after avgpool
                features = self.backbone(x)
                if isinstance(features, tuple):
                    features = features[0]
                if features.ndim > 2:
                    features = torch.flatten(features, 1)
            else:
                # Fallback: use backbone as-is (for other architectures)
                features = self.backbone(x)
                if isinstance(features, tuple):
                    features = features[0]
                if features.ndim > 2:
                    features = torch.flatten(features, 1)
        
        # Normalize features for better transfer learning (L2 normalization)
        if F is not None:
            features = F.normalize(features, dim=1, p=2)
        
        # Pass through head
        return self.head(features)


def load_pretrained_backbone(
    checkpoint_path: Union[str, Path],
    model_name: str,
    image_size: int,
    projection_dim: int = 128,
    projection_hidden_dim: Optional[int] = None,
    projection_use_bn: bool = False,
) -> nn.Module:
    """Load a pretrained backbone from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_name: Name of the model architecture
        image_size: Input image size
        projection_dim: Projection dimension (used to build model structure)
        projection_hidden_dim: Projection hidden dimension
        projection_use_bn: Whether to use batch norm in projector
        
    Returns:
        The backbone module (frozen)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for loading checkpoints")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Build model structure (we'll extract just the backbone)
    # num_classes doesn't matter here since we're only extracting backbone
    temp_model = build_model(
        model_name,
        num_classes=10,  # dummy value
        image_size=image_size,
        projection_dim=projection_dim,
        projection_hidden_dim=projection_hidden_dim,
        projection_use_bn=projection_use_bn,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = None
    if isinstance(checkpoint, dict):
        model_state = (
            checkpoint.get("model_state")
            or checkpoint.get("model")
            or checkpoint.get("state_dict")
        )
    elif isinstance(checkpoint, (list, tuple)):
        model_state = None
    else:
        # Some checkpoints are saved as raw state_dict
        model_state = checkpoint
    if model_state is None or not isinstance(model_state, dict):
        raise ValueError(f"No model state dict found in checkpoint: {checkpoint_path}")

    # Strip common prefixes (DataParallel / Lightning) to improve compatibility.
    cleaned_state: Dict[str, Any] = {}
    for k, v in model_state.items():
        key = str(k)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        cleaned_state[key] = v
    model_state = cleaned_state
    
    # Load state into model
    #
    # We often reuse backbones trained on classification tasks (e.g., CIFAR100)
    # for detection with different heads and num_classes. That means some
    # parameters (e.g., classifier layers, small‑image conv1 tweaks) may not
    # match the architecture we're building here (e.g., COCO detection with
    # different image_size / num_classes).
    #
    # `strict=False` ignores missing/unexpected keys, but **shape mismatches
    # still raise errors**, so we proactively filter any tensors whose shapes
    # don't match before loading.
    current_state = temp_model.state_dict()
    filtered_state = {}
    skipped_keys = []
    for key, value in model_state.items():
        if key not in current_state:
            continue
        if current_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        filtered_state[key] = value

    if skipped_keys:
        print(
            "[info] Skipping incompatible checkpoint parameters when loading backbone:\n"
            f"  {', '.join(sorted(skipped_keys))}"
        )

    temp_model.load_state_dict(filtered_state, strict=False)
    
    # Extract backbone
    if not hasattr(temp_model, "backbone"):
        raise ValueError("Model does not have a 'backbone' attribute")
    
    backbone = copy.deepcopy(temp_model.backbone)
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    
    return backbone


def compute_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """Compute Intersection over Union (IoU) between two sets of boxes.
    
    Args:
        boxes1: [N, 4] boxes in format (x1, y1, x2, y2)
        boxes2: [M, 4] boxes in format (x1, y1, x2, y2)
        
    Returns:
        [N, M] IoU matrix
    """
    if torch is None:
        raise RuntimeError("PyTorch is required")
    
    # Compute intersection
    inter_x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0].unsqueeze(0))
    inter_y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1].unsqueeze(0))
    inter_x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2].unsqueeze(0))
    inter_y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3].unsqueeze(0))
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Compute union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    
    # Compute IoU
    iou = inter_area / (union_area + 1e-6)
    return iou


def match_anchors_to_gt(
    pred_boxes: "torch.Tensor",
    gt_boxes: "torch.Tensor",
    *,
    iou_threshold: float,
) -> Tuple["torch.Tensor", "torch.Tensor"]:  # type: ignore[name-defined]
    """Greedy IoU-based matching between predicted boxes and GT boxes.

    This is a simplified alternative to Hungarian matching. It is sufficient for
    training a lightweight evaluation head where we want consistent supervision.

    Args:
        pred_boxes: [A, 4] predicted boxes in xyxy format (same scale as gt_boxes)
        gt_boxes: [G, 4] GT boxes in xyxy format
        iou_threshold: minimum IoU required to accept a match

    Returns:
        (matched_pred_indices, matched_gt_indices)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required")

    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return (
            torch.zeros((0,), dtype=torch.long, device=pred_boxes.device),
            torch.zeros((0,), dtype=torch.long, device=pred_boxes.device),
        )

    iou = compute_iou(pred_boxes, gt_boxes)  # [A, G]
    matched_pred: list[int] = []
    matched_gt: list[int] = []

    # Greedy: repeatedly take the best remaining IoU pair.
    while True:
        max_val, flat_idx = torch.max(iou.view(-1), dim=0)
        if float(max_val.item()) < float(iou_threshold):
            break
        a = int(flat_idx.item()) // int(gt_boxes.size(0))
        g = int(flat_idx.item()) % int(gt_boxes.size(0))
        matched_pred.append(a)
        matched_gt.append(g)
        # Remove this pred and this gt from further consideration.
        iou[a, :] = -1.0
        iou[:, g] = -1.0

    if not matched_pred:
        return (
            torch.zeros((0,), dtype=torch.long, device=pred_boxes.device),
            torch.zeros((0,), dtype=torch.long, device=pred_boxes.device),
        )

    return (
        torch.tensor(matched_pred, dtype=torch.long, device=pred_boxes.device),
        torch.tensor(matched_gt, dtype=torch.long, device=pred_boxes.device),
    )


def apply_nms(
    boxes: "torch.Tensor",
    scores: "torch.Tensor",
    labels: "torch.Tensor",
    *,
    iou_threshold: float,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:  # type: ignore[name-defined]
    """Apply per-class NMS to a set of detections in xyxy format."""
    if torch is None:
        raise RuntimeError("PyTorch is required")

    if boxes.numel() == 0:
        return boxes, scores, labels

    try:
        from torchvision.ops import nms  # type: ignore
    except Exception:
        nms = None  # type: ignore

    def _nms_fallback(b: "torch.Tensor", s: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        # Very small fallback NMS in pure torch.
        keep: list[int] = []
        idxs = torch.argsort(s, descending=True)
        while idxs.numel() > 0:
            i = int(idxs[0].item())
            keep.append(i)
            if idxs.numel() == 1:
                break
            rest = idxs[1:]
            ious = compute_iou(b[i].unsqueeze(0), b[rest]).squeeze(0)
            idxs = rest[ious < iou_threshold]
        return torch.tensor(keep, dtype=torch.long, device=b.device)

    keep_indices: list["torch.Tensor"] = []  # type: ignore[name-defined]
    for cls in torch.unique(labels):
        cls_mask = labels == cls
        cls_idxs = torch.where(cls_mask)[0]
        if cls_idxs.numel() == 0:
            continue
        b = boxes[cls_idxs]
        s = scores[cls_idxs]
        if nms is not None:
            kept = nms(b, s, float(iou_threshold))
        else:
            kept = _nms_fallback(b, s)
        keep_indices.append(cls_idxs[kept])

    if not keep_indices:
        return (
            boxes[:0],
            scores[:0],
            labels[:0],
        )

    keep = torch.cat(keep_indices, dim=0)
    # Sort final keep by score descending.
    keep = keep[torch.argsort(scores[keep], descending=True)]
    return boxes[keep], scores[keep], labels[keep]


def decode_slot_detections(
    cls_logits: "torch.Tensor",
    bbox_preds: "torch.Tensor",
    *,
    score_threshold: float,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:  # type: ignore[name-defined]
    """Convert slot-based outputs into a flat detection list.

    Args:
        cls_logits: [A, C]
        bbox_preds: [A, 4] normalized xyxy
    """
    if torch is None:
        raise RuntimeError("PyTorch is required")

    probs = torch.softmax(cls_logits, dim=-1)
    scores, labels = probs.max(dim=-1)  # [A]
    keep = scores >= float(score_threshold)
    if not keep.any():
        return (
            bbox_preds[:0],
            scores[:0],
            labels[:0],
        )
    return bbox_preds[keep], scores[keep], labels[keep]


def _get_coco_api_from_dataset(dataset: Any) -> Any:
    """Try to get the underlying pycocotools COCO API object from a dataset/subset."""
    try:
        from torch.utils.data import Subset  # type: ignore
    except Exception:
        Subset = None  # type: ignore

    ds = dataset
    # Unwrap common wrappers
    while True:
        if Subset is not None and isinstance(ds, Subset):
            ds = ds.dataset
            continue
        inner = getattr(ds, "dataset", None)
        if inner is not None and inner is not ds:
            ds = inner
            continue
        break

    return getattr(ds, "coco", None)


def _denormalize_boxes_xyxy(
    boxes_norm: "torch.Tensor",
    *,
    orig_h: int,
    orig_w: int,
) -> "torch.Tensor":  # type: ignore[name-defined]
    if torch is None:
        raise RuntimeError("PyTorch is required")
    scale = boxes_norm.new_tensor([orig_w, orig_h, orig_w, orig_h])
    return boxes_norm * scale


def _xyxy_to_xywh(boxes_xyxy: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
    if torch is None:
        raise RuntimeError("PyTorch is required")
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    return torch.stack([x1, y1, (x2 - x1), (y2 - y1)], dim=-1)


def generate_coco_predictions(
    model: MultiHeadModel,
    dataloader: DataLoader,
    *,
    device: "torch.device",
    score_threshold: float,
    nms_iou_threshold: float,
    max_batches: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run model on a detection dataloader and output COCO-format predictions."""
    if torch is None:
        raise RuntimeError("PyTorch is required")

    model.eval()
    preds: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)
            cls_logits, bbox_preds = model(images)

            for i in range(images.size(0)):
                tgt = targets[i] if isinstance(targets, list) else targets
                image_id = int(tgt.get("image_id", -1))
                orig_size = tgt.get("orig_size", None)
                if orig_size is None:
                    # Fallback: assume square resized images
                    orig_h = int(images.size(2))
                    orig_w = int(images.size(3))
                else:
                    try:
                        orig_h = int(orig_size[0])
                        orig_w = int(orig_size[1])
                    except Exception:
                        orig_h = int(images.size(2))
                        orig_w = int(images.size(3))

                boxes, scores, labels = decode_slot_detections(
                    cls_logits[i],
                    bbox_preds[i],
                    score_threshold=score_threshold,
                )
                boxes, scores, labels = apply_nms(
                    boxes,
                    scores,
                    labels,
                    iou_threshold=nms_iou_threshold,
                )
                if boxes.numel() == 0:
                    continue

                boxes_xyxy = _denormalize_boxes_xyxy(boxes, orig_h=orig_h, orig_w=orig_w)
                boxes_xywh = _xyxy_to_xywh(boxes_xyxy)

                for b, s, l in zip(boxes_xywh, scores, labels):
                    contiguous = int(l.item())
                    coco_cat = COCO_CONTIGUOUS_TO_ID.get(contiguous)
                    if coco_cat is None:
                        continue
                    preds.append(
                        {
                            "image_id": image_id,
                            "category_id": int(coco_cat),
                            "bbox": [float(x) for x in b.tolist()],
                            "score": float(s.item()),
                        }
                    )

    return preds


def compute_coco_map(
    coco_gt: Any,
    coco_predictions: List[Dict[str, Any]],
    *,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute COCO mAP@[.5:.95] using pycocotools' COCOeval."""
    try:
        from pycocotools.cocoeval import COCOeval  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pycocotools is required for full COCO mAP evaluation. Install it with `pip install pycocotools`."
        ) from exc

    if coco_gt is None:
        raise RuntimeError("COCO ground-truth API is not available from the dataset.")

    if not coco_predictions:
        return {
            "map": 0.0,
            "map50": 0.0,
            "map75": 0.0,
            "per_class_ap": {},
            "note": "No predictions produced (empty).",
        }

    coco_dt = coco_gt.loadRes(coco_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if iou_thresholds is not None:
        try:
            import numpy as np  # type: ignore

            coco_eval.params.iouThrs = np.array([float(x) for x in iou_thresholds], dtype=np.float32)
        except Exception:
            pass
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    # Per-class AP from precision[T, R, K, A, M]
    per_class_ap: Dict[int, float] = {}
    try:
        precision = coco_eval.eval["precision"]  # type: ignore[index]
        # area=all (0), maxDets=100 (2)
        for k in range(precision.shape[2]):
            p = precision[:, :, k, 0, 2]
            p = p[p > -1]
            if p.numel() == 0:
                ap = float("nan")
            else:
                ap = float(p.mean())
            per_class_ap[int(k)] = ap
    except Exception:
        per_class_ap = {}

    return {
        "map": float(stats[0]),
        "map50": float(stats[1]),
        "map75": float(stats[2]),
        "per_class_ap": per_class_ap,
    }


def evaluate_with_head(
    config: TrainingConfig,
    checkpoint_path: Optional[Union[str, Path]] = None,
    head_type: str = "classification",
    *,
    project_root: Optional[Path] = None,
    report_path: Optional[Path] = None,
    **head_kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate a pretrained backbone with a specific head, or train from scratch.
    
    Args:
        config: Training configuration (used for dataset, model name, etc.)
        checkpoint_path: Path to pretrained checkpoint, or None for training from scratch
        head_type: Type of head ("classification" or "detection")
        project_root: Project root directory
        report_path: Path to save evaluation report
        **head_kwargs: Additional arguments for head construction
        
    Returns:
        Dictionary with evaluation metrics
    """
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for evaluation")
    
    device_str = config.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available. Using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    
    # Load pretrained backbone or build from scratch
    if checkpoint_path is not None:
        # Resolve checkpoint path relative to project_root / cwd when needed.
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            base = Path(project_root) if project_root is not None else Path.cwd()
            checkpoint_path = (base / checkpoint_path).resolve()

        print(f"[info] Loading pretrained backbone from: {checkpoint_path}")
        backbone = load_pretrained_backbone(
            checkpoint_path,
            config.model,
            config.hyperparameters.get("image_size", 224),
            projection_dim=config.hyperparameters.get("projection_dim", 128),
            projection_hidden_dim=config.hyperparameters.get("projection_hidden_dim"),
            projection_use_bn=config.hyperparameters.get("projection_use_bn", False),
        )
    else:
        # Build model from scratch
        print(f"[info] Building model from scratch: {config.model}")
        from .utils import set_seed
        set_seed(config.seed)
        
        # Build the full model (backbone + projection head) but we'll extract just the backbone
        full_model = build_model(
            config.model,
            num_classes=10,  # Dummy, we'll extract backbone
            image_size=config.hyperparameters.get("image_size", 224),
            projection_dim=config.hyperparameters.get("projection_dim", 128),
            projection_hidden_dim=config.hyperparameters.get("projection_hidden_dim"),
            projection_use_bn=config.hyperparameters.get("projection_use_bn", False),
        )
        # Extract backbone (remove classifier/projector)
        backbone = full_model.backbone if hasattr(full_model, "backbone") else full_model
    
    # Get feature dimension from backbone
    # Extract pure features (bypass classifier)
    dummy_input = torch.zeros(1, 3, config.hyperparameters.get("image_size", 224), 
                             config.hyperparameters.get("image_size", 224))
    with torch.no_grad():
        if hasattr(backbone, 'features') and hasattr(backbone, 'avgpool'):
            # MobileNet: Pure feature extraction: features + avgpool only (no classifier)
            feat_maps = backbone.features(dummy_input)
            dummy_features = backbone.avgpool(feat_maps)
            dummy_features = torch.flatten(dummy_features, 1)
        elif hasattr(backbone, 'avgpool') and not hasattr(backbone, 'features'):
            # ResNet: fc is already Identity, so backbone(x) gives features after avgpool
            dummy_features = backbone(dummy_input)
            if isinstance(dummy_features, tuple):
                dummy_features = dummy_features[0]
            if dummy_features.ndim > 2:
                dummy_features = torch.flatten(dummy_features, 1)
        else:
            # Fallback for other architectures
            dummy_features = backbone(dummy_input)
            if isinstance(dummy_features, tuple):
                dummy_features = dummy_features[0]
            if dummy_features.ndim > 2:
                dummy_features = torch.flatten(dummy_features, 1)
        feature_dim = dummy_features.size(1)
    
    # Determine num_classes from dataset
    # For now, we'll use a placeholder and get it from dataloader
    num_classes = head_kwargs.get("num_classes")
    if num_classes is None:
        # Try to infer from dataset
        # This is a simplified approach - in practice you'd want to load the dataset first
        if "cifar10" in config.database.lower():
            num_classes = 10
        elif "cifar100" in config.database.lower():
            num_classes = 100
        else:
            num_classes = 10  # default fallback
    
    # Build model with head
    model = MultiHeadModel(
        backbone,
        feature_dim,
        head_type=head_type,
        num_classes=num_classes,
        **head_kwargs,
    )
    model = model.to(device)
    
    # Get freeze_backbone setting from config (default True for backward compatibility)
    eval_cfg = getattr(config, "evaluation", {}) or {}
    freeze_backbone = eval_cfg.get("freeze_backbone", True)
    # If training from scratch, don't freeze backbone
    if checkpoint_path is None:
        freeze_backbone = False
    
    # Build dataloaders based on head type
    print(f"[info] Building dataloaders for dataset: {config.database}")
    if head_type == "detection":
        # Check if class filtering is enabled
        filter_classes = None
        if eval_cfg.get("filter_coco_to_cifar100", False):
            from .data import get_coco_classes_for_cifar100
            filter_classes = get_coco_classes_for_cifar100()
        
        # Use detection dataloaders for object detection
        train_loader, val_loader, dataset_num_classes = build_detection_dataloaders(
            config.database,
            batch_size=config.hyperparameters.get("batch_size", 32),
            num_workers=config.hyperparameters.get("num_workers", 2),
            val_split=config.hyperparameters.get("val_split", 0.0),
            seed=config.seed,
            project_root=project_root,
            image_size=config.hyperparameters.get("image_size", 224),
            max_train_images=config.hyperparameters.get("max_train_images"),
            max_val_images=config.hyperparameters.get("max_val_images"),
            filter_classes=filter_classes,
        )
    else:
        # Use classification dataloaders for classification
        train_loader, val_loader, dataset_num_classes = build_classification_dataloaders(
            config.database,
            batch_size=config.hyperparameters.get("batch_size", 32),
            num_workers=config.hyperparameters.get("num_workers", 2),
            val_split=config.hyperparameters.get("val_split", 0.1),
            seed=config.seed,
            project_root=project_root,
            image_size=config.hyperparameters.get("image_size", 224),
            contrastive=False,
        )
    
    # Update num_classes if we got it from dataset
    if num_classes != dataset_num_classes:
        print(f"[warn] num_classes mismatch: config={num_classes}, dataset={dataset_num_classes}. Using dataset value.")
        num_classes = dataset_num_classes
    
    # Build model with head (with correct num_classes and freeze_backbone setting)
    model = MultiHeadModel(
        backbone,
        feature_dim,
        head_type=head_type,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        **head_kwargs,
    )
    model = model.to(device)
    
    # Print training mode
    if freeze_backbone:
        print(f"[info] Training {head_type} head (backbone frozen)...")
    else:
        print(f"[info] Training {head_type} head with unfrozen backbone (fine-tuning)...")
    
    if head_type == "classification":
        return _evaluate_classification(model, train_loader, val_loader, device, config)
    elif head_type == "detection":
        return _evaluate_detection(model, train_loader, val_loader, device, config)
    else:
        raise ValueError(f"Unsupported head_type: {head_type}")


def _evaluate_classification(
    model: MultiHeadModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> Dict[str, Any]:
    """Evaluate with classification head."""
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.head.parameters(),
        lr=config.hyperparameters.get("lr", 0.01),
        momentum=config.hyperparameters.get("momentum", 0.9),
        weight_decay=config.weight_decay,
    )
    
    epochs = config.hyperparameters.get("epochs", 10)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                logits = model(inputs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        avg_val_loss = val_loss / len(val_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}")
    
    return {
        "head_type": "classification",
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "final_train_accuracy": train_acc,
        "final_val_accuracy": val_acc,
    }


def _evaluate_detection(
    model: MultiHeadModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> Dict[str, Any]:
    """Evaluate with object detection head using real bounding box annotations.
    
    This implementation uses ground truth boxes from COCO/VOC datasets.
    Note: For production use, implement proper anchor generation, NMS, and mAP computation.
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required")
    
    cls_criterion = nn.CrossEntropyLoss(reduction='none')
    bbox_criterion = nn.SmoothL1Loss(reduction='none')
    
    # Use optimizer builder for flexibility (supports AdamW, SGD, etc.)
    from .optimizers import build_optimizer, build_scheduler
    
    # Check if backbone is frozen
    freeze_backbone = getattr(model, "freeze_backbone", True)
    
    if freeze_backbone:
        # Only optimize head parameters
        optimizer = build_optimizer(
            config.optimizer,
            model.head.parameters(),
            lr=config.hyperparameters.get("lr", 0.01),
            momentum=config.hyperparameters.get("momentum", 0.9),
            weight_decay=config.weight_decay,
        )
    else:
        # Fine-tuning: use different learning rates for backbone and head
        backbone_lr = config.hyperparameters.get("backbone_lr", 0.0001)
        head_lr = config.hyperparameters.get("head_lr", config.hyperparameters.get("lr", 0.001))
        
        # Create parameter groups
        backbone_params = list(model.backbone.parameters())
        head_params = list(model.head.parameters())
        
        # Build optimizer with parameter groups
        if config.optimizer.lower() in ["adam", "adamw"]:
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": backbone_lr, "weight_decay": config.weight_decay},
                    {"params": head_params, "lr": head_lr, "weight_decay": config.weight_decay},
                ]
            )
        elif config.optimizer.lower() == "sgd":
            momentum = config.hyperparameters.get("momentum", 0.9)
            optimizer = torch.optim.SGD(
                [
                    {"params": backbone_params, "lr": backbone_lr, "momentum": momentum, "weight_decay": config.weight_decay},
                    {"params": head_params, "lr": head_lr, "momentum": momentum, "weight_decay": config.weight_decay},
                ]
            )
        else:
            # Fallback: use build_optimizer with all parameters at head_lr
            print(f"[warn] Optimizer {config.optimizer} doesn't support parameter groups. Using single LR for all parameters.")
            all_params = list(model.backbone.parameters()) + list(model.head.parameters())
            optimizer = build_optimizer(
                config.optimizer,
                all_params,
                lr=head_lr,
                momentum=config.hyperparameters.get("momentum", 0.9),
                weight_decay=config.weight_decay,
            )
        
        print(f"[info] Fine-tuning with backbone_lr={backbone_lr}, head_lr={head_lr}")
    
    # Build scheduler if configured
    scheduler = None
    scheduler_config = getattr(config, "scheduler", None)
    if scheduler_config:
        epochs = config.hyperparameters.get("epochs", 10)
        # Adjust scheduler config for detection head training (epoch-based, not step-based)
        if isinstance(scheduler_config, dict):
            sched_type = scheduler_config.get("type", "").lower()
            params = scheduler_config.get("params", {}) or {}
            if sched_type == "cosine_annealing":
                t_max = params.get("t_max")
                if t_max is None:
                    # Auto-set t_max to epochs
                    params = params.copy()
                    params["t_max"] = epochs
                    scheduler_config = scheduler_config.copy()
                    scheduler_config["params"] = params
            elif sched_type == "warmup_cosine":
                # For warmup_cosine, warmup_steps should be in epochs (we step per epoch)
                # Convert warmup_epochs to warmup_steps if needed
                warmup_epochs = params.get("warmup_epochs")
                if warmup_epochs is not None:
                    params = params.copy()
                    params["warmup_steps"] = warmup_epochs
                    params.pop("warmup_epochs", None)
                t_max = params.get("t_max")
                if t_max is None:
                    params = params.copy()
                    params["t_max"] = epochs
                    scheduler_config = scheduler_config.copy()
                    scheduler_config["params"] = params
        scheduler = build_scheduler(scheduler_config, optimizer)
    
    epochs = config.hyperparameters.get("epochs", 10)
    gradient_clip_norm = config.hyperparameters.get("gradient_clip_norm")
    log_every = int(config.hyperparameters.get("log_every", 50) or 50)
    max_train_batches = config.hyperparameters.get("max_train_batches")
    max_val_batches = config.hyperparameters.get("max_val_batches")
    iou_threshold = float(config.hyperparameters.get("iou_threshold", 0.5))
    match_iou_threshold = float(config.hyperparameters.get("match_iou_threshold", 0.3))
    bbox_loss_weight = float(config.hyperparameters.get("bbox_loss_weight", 10.0))  # Increased default
    bbox_loss_warmup_epochs = int(config.hyperparameters.get("bbox_loss_warmup_epochs", 10))  # Warmup epochs
    bbox_loss_max_weight = float(config.hyperparameters.get("bbox_loss_max_weight", 20.0))  # Max weight during warmup
    eval_cfg_local = getattr(config, "evaluation", {}) or {}
    nms_iou_threshold = float(
        eval_cfg_local.get("nms_threshold", config.hyperparameters.get("nms_iou_threshold", 0.5))
    )
    score_threshold = float(config.hyperparameters.get("score_threshold", 0.05))
    try:
        max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    except Exception:
        max_train_batches = None
    try:
        max_val_batches = int(max_val_batches) if max_val_batches is not None else None
    except Exception:
        max_val_batches = None
    
    print("[info] Training detection head with real bounding box annotations")
    print("[warn] This is a simplified implementation. For production, add anchor generation, NMS, and full COCO mAP.")
    
    best_val_loss = float('inf')
    best_epoch = 0

    # Optional checkpoint/CSV logging similar to the main training loop
    save_model = getattr(config, "save_model", False)
    save_model_path = getattr(config, "save_model_path", None) or "output_object_detection/detection_run"
    save_dir = Path(save_model_path if isinstance(save_model_path, str) else str(save_model_path))
    if save_model:
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save a YAML snapshot of the config for later reuse
        try:
            import yaml  # type: ignore
            cfg_path = save_dir / "config.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                # `config` is a dataclass (TrainingConfig); use __dict__ as a fallback
                data = getattr(config, "__dict__", {})
                yaml.safe_dump(data, f, sort_keys=False)
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"[warn] Failed to write detection config snapshot: {exc}")

    report_filename = getattr(config, "report_filename", None)
    if isinstance(report_filename, str) and report_filename:
        report_path = Path(report_filename)
        if not report_path.is_absolute():
            report_path = Path.cwd() / report_path
    else:
        report_path = save_dir / "detection_metrics.csv"

    metrics_rows: List[Dict[str, Any]] = []
    # How often to save checkpoints (default every 50 epochs, matching classification pipeline)
    checkpoint_interval = int(config.hyperparameters.get("checkpoint_interval", 50) or 50)

    # NOTE: IoU-based detection "accuracy" (recall@IoU>=threshold with matching class)
    # is computed per-epoch during validation and logged/printed.

    # CodeCarbon tracking for the full detection-head training run.
    track_emissions = bool(config.hyperparameters.get("track_emissions", True))
    emissions_kg_total: Optional[float] = None
    energy_kwh_total: Optional[float] = None
    emissions_data: Any = None
    duration_seconds: Optional[float] = None
    tracker: Any = None

    start_time = time.perf_counter()
    
    # Use block-based energy tracking (10 epochs per block, matching classification pipeline)
    # so we can log per-block emissions and then aggregate a total at the end.
    block_tracker: Any = None

    def _start_block_tracker() -> Any:
        """Start a new CodeCarbon tracker for a 10-epoch block.
        
        Note: We disable CodeCarbon's own CSV logging (`save_to_file=False`)
        so that we don't create a global `emissions.csv`. Per-block and
        total energy/emissions are instead captured via the tracker object
        and our own `resource_energy_log.csv`.
        """
        if not track_emissions or EmissionsTracker is None:
            return None
        import logging
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
                artifact_dir=save_dir,
                epoch=current_epoch,
                emissions_data=block_data,
                elapsed_seconds=time.perf_counter() - start_time,
                device=device,
            )
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"[warn] Failed to record resource/energy snapshot at epoch {current_epoch}: {exc}")
    
    try:
        for epoch in range(epochs):
            # Start a new 10-epoch energy tracking block at epochs 1, 11, 21, ...
            # (epoch 0 -> start block, epoch 10 -> stop and start new, etc.)
            if track_emissions and ((epoch) % 10 == 0) and block_tracker is None:
                block_tracker = _start_block_tracker()
            
            # Training
            model.train()
            # If backbone is frozen, keep it in eval mode (for proper BatchNorm behavior)
            freeze_backbone = getattr(model, "freeze_backbone", True)
            if freeze_backbone and hasattr(model, "backbone"):
                model.backbone.eval()
            # Ensure head is in train mode
            if hasattr(model, "head"):
                model.head.train()
            
            train_loss = 0.0
            train_cls_loss = 0.0
            train_bbox_loss = 0.0
            train_samples = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                if max_train_batches is not None and batch_idx >= max_train_batches:
                    break
                images = images.to(device)
                
                optimizer.zero_grad()
                cls_logits, bbox_preds = model(images)
                
                batch_size = cls_logits.size(0)
                num_classes = cls_logits.size(2)
                
                # Process each image in the batch
                total_cls_loss = 0.0
                total_bbox_loss = 0.0
                valid_samples = 0
                
                for i in range(batch_size):
                    target = targets[i]
                    gt_boxes = target.get("boxes", torch.zeros((0, 4))).to(device)
                    gt_labels = target.get("labels", torch.zeros((0,), dtype=torch.long)).to(device)
                    
                    if len(gt_boxes) == 0:
                        continue

                    matched_pred, matched_gt = match_anchors_to_gt(
                        bbox_preds[i],
                        gt_boxes,
                        iou_threshold=match_iou_threshold,
                    )
                    # More aggressive fallback: always ensure matches for training
                    if matched_pred.numel() == 0:
                        # Try with very low threshold first
                        matched_pred, matched_gt = match_anchors_to_gt(
                            bbox_preds[i],
                            gt_boxes,
                            iou_threshold=0.01,  # Very low threshold to ensure some matches
                        )
                        if matched_pred.numel() == 0:
                            # Still no matches: match each GT to best anchor (one-to-one)
                            # This ensures every GT box gets a match for training
                            if len(gt_boxes) > 0:
                                iou = compute_iou(bbox_preds[i], gt_boxes)  # [A, G]
                                best_anchors = iou.argmax(dim=0)  # [G] - best anchor per GT
                                matched_pred = best_anchors
                                matched_gt = torch.arange(len(gt_boxes), device=gt_boxes.device)
                            else:
                                continue

                    assigned_cls_logits = cls_logits[i, matched_pred]
                    assigned_labels = gt_labels[matched_gt]
                    valid_mask = (assigned_labels >= 0) & (assigned_labels < num_classes)
                    if valid_mask.any():
                        cls_loss = cls_criterion(assigned_cls_logits[valid_mask], assigned_labels[valid_mask]).mean()
                        total_cls_loss += cls_loss

                    assigned_bbox_preds = bbox_preds[i, matched_pred]
                    assigned_gt_boxes = gt_boxes[matched_gt]
                    # Compute bbox loss: sum over coordinates, then average over boxes
                    # This gives more weight to bbox regression
                    bbox_loss_per_box = bbox_criterion(assigned_bbox_preds, assigned_gt_boxes).sum(dim=1)  # [N] - sum over 4 coords
                    bbox_loss = bbox_loss_per_box.mean()  # Average over matched boxes
                    # Apply adaptive weight (warmup + base weight)
                    if epoch < bbox_loss_warmup_epochs:
                        # Linear warmup: start at base weight, ramp to max weight
                        warmup_factor = epoch / bbox_loss_warmup_epochs
                        current_weight = bbox_loss_weight + (bbox_loss_max_weight - bbox_loss_weight) * warmup_factor
                    else:
                        current_weight = bbox_loss_weight
                    total_bbox_loss += bbox_loss * current_weight

                    valid_samples += 1
                
                if valid_samples > 0:
                    loss = (total_cls_loss + total_bbox_loss) / valid_samples
                    loss.backward()
                    
                    # Gradient clipping for stability
                    if gradient_clip_norm is not None and gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.head.parameters(), gradient_clip_norm)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_cls_loss += (total_cls_loss / valid_samples).item()
                    train_bbox_loss += (total_bbox_loss / valid_samples).item()
                    train_samples += valid_samples

                if log_every > 0 and (batch_idx + 1) % log_every == 0:
                    print(f"[info] epoch {epoch+1}/{epochs} - train batch {batch_idx+1}")
            
            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            avg_train_cls_loss = train_cls_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            avg_train_bbox_loss = train_bbox_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_cls_loss = 0.0
            val_bbox_loss = 0.0

            total_gt_objects = 0
            matched_gt_objects = 0
            
            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(val_loader):
                    if max_val_batches is not None and batch_idx >= max_val_batches:
                        break
                    images = images.to(device)
                    cls_logits, bbox_preds = model(images)
                    
                    batch_size = cls_logits.size(0)
                    total_cls_loss = 0.0
                    total_bbox_loss = 0.0
                    valid_samples = 0
                    
                    for i in range(batch_size):
                        target = targets[i]
                        gt_boxes = target.get("boxes", torch.zeros((0, 4))).to(device)
                        gt_labels = target.get("labels", torch.zeros((0,), dtype=torch.long)).to(device)
                        
                        if len(gt_boxes) == 0:
                            continue

                        matched_pred, matched_gt = match_anchors_to_gt(
                            bbox_preds[i],
                            gt_boxes,
                            iou_threshold=match_iou_threshold,
                        )
                        # Fallback: if no matches, use best IoU pairs anyway (for early training)
                        if matched_pred.numel() == 0:
                            matched_pred, matched_gt = match_anchors_to_gt(
                                bbox_preds[i],
                                gt_boxes,
                                iou_threshold=0.01,  # Very low threshold to ensure some matches
                            )
                            if matched_pred.numel() == 0:
                                # Still no matches: match each GT to best anchor (one-to-one)
                                if len(gt_boxes) > 0:
                                    iou = compute_iou(bbox_preds[i], gt_boxes)  # [A, G]
                                    best_anchors = iou.argmax(dim=0)  # [G] - best anchor per GT
                                    matched_pred = best_anchors
                                    matched_gt = torch.arange(len(gt_boxes), device=gt_boxes.device)
                                else:
                                    continue

                        assigned_cls_logits = cls_logits[i, matched_pred]
                        assigned_labels = gt_labels[matched_gt]
                        valid_mask = (assigned_labels >= 0) & (assigned_labels < num_classes)
                        if valid_mask.any():
                            cls_loss = cls_criterion(assigned_cls_logits[valid_mask], assigned_labels[valid_mask]).mean()
                            total_cls_loss += cls_loss

                        assigned_bbox_preds = bbox_preds[i, matched_pred]
                        assigned_gt_boxes = gt_boxes[matched_gt]
                        # Compute bbox loss: sum over coordinates, then average over boxes
                        bbox_loss_per_box = bbox_criterion(assigned_bbox_preds, assigned_gt_boxes).sum(dim=1)  # [N]
                        bbox_loss = bbox_loss_per_box.mean()
                        # Apply adaptive weight (same as training)
                        if epoch < bbox_loss_warmup_epochs:
                            warmup_factor = epoch / bbox_loss_warmup_epochs
                            current_weight = bbox_loss_weight + (bbox_loss_max_weight - bbox_loss_weight) * warmup_factor
                        else:
                            current_weight = bbox_loss_weight
                        # Scale by number of matched boxes (same as training)
                        num_matched = len(matched_pred)
                        if num_matched > 0:
                            total_bbox_loss += bbox_loss * current_weight * (1.0 + 0.1 * num_matched)
                        else:
                            total_bbox_loss += bbox_loss * current_weight

                        valid_samples += 1

                        pred_boxes, pred_scores, pred_labels = decode_slot_detections(
                            cls_logits[i],
                            bbox_preds[i],
                            score_threshold=score_threshold,
                        )
                        pred_boxes, pred_scores, pred_labels = apply_nms(
                            pred_boxes,
                            pred_scores,
                            pred_labels,
                            iou_threshold=nms_iou_threshold,
                        )

                        total_gt_objects += int(len(gt_boxes))
                        # Debug: track prediction counts and class accuracy
                        if (epoch == 0 or epoch == 9 or epoch == 19) and batch_idx == 0 and i == 0:
                            print(f"[debug] Epoch {epoch+1} - After decode+NMS: {len(pred_boxes)} predictions, {len(gt_boxes)} GT boxes")
                            if len(pred_boxes) > 0:
                                print(f"[debug]   Prediction classes: {pred_labels[:min(5, len(pred_labels))].tolist()}, scores: {pred_scores[:min(5, len(pred_scores))].tolist()}")
                            if len(gt_boxes) > 0:
                                print(f"[debug]   GT classes: {gt_labels[:min(5, len(gt_labels))].tolist()}")
                                # Check if any predictions match GT classes
                                unique_pred_classes = set(pred_labels.tolist())
                                unique_gt_classes = set(gt_labels.tolist())
                                overlap = len(unique_pred_classes & unique_gt_classes)
                                print(f"[debug]   Class overlap: {overlap}/{len(unique_gt_classes)} GT classes predicted")
                        for j in range(len(gt_boxes)):
                            gt_box = gt_boxes[j].unsqueeze(0)
                            gt_label = gt_labels[j]
                            same_cls_mask = pred_labels == gt_label
                            if not same_cls_mask.any():
                                continue
                            candidate_boxes = pred_boxes[same_cls_mask]
                            if candidate_boxes.numel() == 0:
                                continue
                            ious = compute_iou(gt_box, candidate_boxes)[0]
                            if (ious >= iou_threshold).any():
                                matched_gt_objects += 1
                    
                    if valid_samples > 0:
                        loss = (total_cls_loss + total_bbox_loss) / valid_samples
                        val_loss += loss.item()
                        val_cls_loss += (total_cls_loss / valid_samples).item()
                        val_bbox_loss += (total_bbox_loss / valid_samples).item()
            
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            avg_val_cls_loss = val_cls_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            avg_val_bbox_loss = val_bbox_loss / len(val_loader) if len(val_loader) > 0 else 0.0

            iou_recall = float(matched_gt_objects) / float(total_gt_objects) if total_gt_objects > 0 else 0.0
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
            
            # Compute current bbox weight for logging
            if epoch < bbox_loss_warmup_epochs:
                warmup_factor = epoch / bbox_loss_warmup_epochs
                current_bbox_weight = bbox_loss_weight + (bbox_loss_max_weight - bbox_loss_weight) * warmup_factor
            else:
                current_bbox_weight = bbox_loss_weight
            
            # Step scheduler after each epoch
            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"train_loss={avg_train_loss:.4f} (cls={avg_train_cls_loss:.4f}, bbox={avg_train_bbox_loss:.4f}) "
                    f"val_loss={avg_val_loss:.4f} (cls={avg_val_cls_loss:.4f}, bbox={avg_val_bbox_loss:.4f}) "
                    f"iou_recall@{iou_threshold:.2f}={iou_recall:.4f} lr={current_lr:.6f} bbox_w={current_bbox_weight:.1f}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"train_loss={avg_train_loss:.4f} (cls={avg_train_cls_loss:.4f}, bbox={avg_train_bbox_loss:.4f}) "
                    f"val_loss={avg_val_loss:.4f} (cls={avg_val_cls_loss:.4f}, bbox={avg_val_bbox_loss:.4f}) "
                    f"iou_recall@{iou_threshold:.2f}={iou_recall:.4f} bbox_w={current_bbox_weight:.1f}"
                )

            # Energy logging is handled by block tracker (every 10 epochs)
            # No per-epoch logging here to match classification pipeline

            # Optionally save checkpoints every N epochs
            if save_model and checkpoint_interval > 0 and ((epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs):
                try:
                    ckpt_path = save_dir / f"epoch_{epoch+1:04d}.pt"
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state": model.state_dict(),
                            "config": getattr(config, "__dict__", {}),
                        },
                        ckpt_path,
                    )
                except Exception as exc:  # pragma: no cover - best-effort saving
                    print(f"[warn] Failed to save detection checkpoint at epoch {epoch+1}: {exc}")

            metrics_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_cls_loss": avg_train_cls_loss,
                    "train_bbox_loss": avg_train_bbox_loss,
                    "val_loss": avg_val_loss,
                    "val_cls_loss": avg_val_cls_loss,
                    "val_bbox_loss": avg_val_bbox_loss,
                    "iou_recall": iou_recall,
                }
            )

    finally:
        # Stop any remaining block tracker
        if block_tracker is not None:
            _stop_block_tracker(epochs)

        if duration_seconds is None:
            duration_seconds = time.perf_counter() - start_time
    
    # Write CSV with per-epoch metrics if requested
    if save_model and metrics_rows:
        try:
            import csv

            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", newline="") as f:
                fieldnames = [
                    "epoch",
                    "train_loss",
                    "train_cls_loss",
                    "train_bbox_loss",
                    "val_loss",
                    "val_cls_loss",
                    "val_bbox_loss",
                    "iou_recall",
                    "map",
                    "map50",
                    "map75",
                    "predictions_json",
                    "emissions_kg",
                    "energy_kwh",
                    "duration_seconds",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in metrics_rows:
                    # Ensure consistent schema for all rows.
                    row = {
                        **row,
                        "map": "",
                        "map50": "",
                        "map75": "",
                        "predictions_json": "",
                        "emissions_kg": "",
                        "energy_kwh": "",
                        "duration_seconds": "",
                    }
                    writer.writerow(row)
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"[warn] Failed to write detection CSV report to {report_path}: {exc}")

    # Compute full COCO mAP@[.5:.95] if the dataset provides COCO API.
    coco_metrics: Dict[str, Any] = {}
    coco_gt = _get_coco_api_from_dataset(getattr(val_loader, "dataset", None))
    if coco_gt is not None:
        try:
            coco_predictions = generate_coco_predictions(
                model,
                val_loader,
                device=device,
                score_threshold=score_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_batches=max_val_batches,
            )
            # Optional: save COCO-format predictions to JSON for external tooling.
            eval_cfg = getattr(config, "evaluation", {}) or {}
            save_predictions = bool(eval_cfg.get("save_predictions", False))
            predictions_path = eval_cfg.get("predictions_path")
            if save_predictions:
                import json

                if isinstance(predictions_path, str) and predictions_path.strip():
                    out_path = Path(predictions_path.strip())
                    if not out_path.is_absolute():
                        out_path = (Path.cwd() / out_path).resolve()
                else:
                    out_path = save_dir / "coco_predictions.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(coco_predictions, f)
                coco_metrics["predictions_json"] = str(out_path)

            map_iou_thresholds = eval_cfg.get("map_iou_thresholds")
            if isinstance(map_iou_thresholds, (list, tuple)):
                map_iou_thresholds = [float(x) for x in map_iou_thresholds]
            else:
                map_iou_thresholds = None
            coco_metrics = compute_coco_map(coco_gt, coco_predictions, iou_thresholds=map_iou_thresholds)
        except Exception as exc:
            coco_metrics = {"note": f"COCO mAP computation failed: {exc}"}

    results = {
        "head_type": "detection",
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "iou_recall_at_threshold": iou_recall,
        "iou_threshold": iou_threshold,
        "emissions_kg": emissions_kg_total,
        "energy_kwh": energy_kwh_total,
        "duration_seconds": duration_seconds,
        "note": "Slot-based detection head for evaluation (not a production detector).",
    }
    results.update(coco_metrics)

    # Append a summary row with COCO mAP + sustainability metrics if we wrote a CSV.
    if save_model and metrics_rows:
        try:
            import csv

            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("a", newline="") as f:
                fieldnames = [
                    "epoch",
                    "train_loss",
                    "train_cls_loss",
                    "train_bbox_loss",
                    "val_loss",
                    "val_cls_loss",
                    "val_bbox_loss",
                    "iou_recall",
                    "map",
                    "map50",
                    "map75",
                    "predictions_json",
                    "emissions_kg",
                    "energy_kwh",
                    "duration_seconds",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(
                    {
                        "epoch": "summary",
                        "train_loss": "",
                        "train_cls_loss": "",
                        "train_bbox_loss": "",
                        "val_loss": "",
                        "val_cls_loss": "",
                        "val_bbox_loss": "",
                        "iou_recall": results.get("iou_recall_at_threshold", ""),
                        "map": results.get("map", ""),
                        "map50": results.get("map50", ""),
                        "map75": results.get("map75", ""),
                        "predictions_json": results.get("predictions_json", ""),
                        "emissions_kg": results.get("emissions_kg", ""),
                        "energy_kwh": results.get("energy_kwh", ""),
                        "duration_seconds": results.get("duration_seconds", ""),
                    }
                )
        except Exception:
            pass

    return results

