"""Detection model components and factories.

This module intentionally keeps detection-specific model construction out of
`sustainvision.evaluation` so classification evaluation can stay smaller and
focused.
"""

from __future__ import annotations

import copy
import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - runtime safeguard
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from torchvision.models.detection import FasterRCNN  # type: ignore[attr-defined]
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone  # type: ignore[attr-defined]
    from torchvision.models.detection.rpn import AnchorGenerator  # type: ignore[attr-defined]
    from torchvision.ops import MultiScaleRoIAlign  # type: ignore[attr-defined]
    from torchvision.models._utils import IntermediateLayerGetter  # type: ignore[attr-defined]
    from torchvision import models as tv_models  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - runtime safeguard
    FasterRCNN = None  # type: ignore
    resnet_fpn_backbone = None  # type: ignore
    AnchorGenerator = None  # type: ignore
    MultiScaleRoIAlign = None  # type: ignore
    IntermediateLayerGetter = None  # type: ignore
    tv_models = None  # type: ignore

from .models import build_model


class DetectionHead(nn.Module):  # type: ignore[name-defined]
    """Slot-based detection head for lightweight representation evaluation."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_anchors: int = 9,
        hidden_dim: int = 256,
        use_bn: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

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
        bbox_layers.append(nn.Linear(hidden_dim, num_anchors * 4))
        self.bbox_head = nn.Sequential(*bbox_layers)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "bbox_head" in name:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                else:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, features: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        if torch is None:
            raise RuntimeError("PyTorch is required")

        cls_logits = self.cls_head(features)
        bbox_preds = self.bbox_head(features)
        batch_size = features.size(0)
        cls_logits = cls_logits.view(batch_size, self.num_anchors, self.num_classes)
        bbox_preds = bbox_preds.view(batch_size, self.num_anchors, 4)

        boxes = torch.sigmoid(bbox_preds)
        x1 = torch.min(boxes[:, :, 0], boxes[:, :, 2])
        y1 = torch.min(boxes[:, :, 1], boxes[:, :, 3])
        x2 = torch.max(boxes[:, :, 0], boxes[:, :, 2])
        y2 = torch.max(boxes[:, :, 1], boxes[:, :, 3])
        bbox_preds = torch.stack([x1, y1, x2, y2], dim=-1)
        return cls_logits, bbox_preds


def _extract_backbone_features(backbone: "nn.Module", x: "torch.Tensor") -> "torch.Tensor":
    """Extract flattened feature vectors from common torchvision backbones."""
    if torch is None:
        raise RuntimeError("PyTorch is required")

    if hasattr(backbone, "features") and hasattr(backbone, "avgpool"):
        feat_maps = backbone.features(x)  # type: ignore[attr-defined]
        features = backbone.avgpool(feat_maps)  # type: ignore[attr-defined]
        features = torch.flatten(features, 1)
        return features

    features = backbone(x)
    if isinstance(features, tuple):
        features = features[0]
    if features.ndim > 2:
        features = torch.flatten(features, 1)
    return features


class MultiHeadModel(nn.Module):  # type: ignore[name-defined]
    """Attach a classification or slot-detection head to a shared backbone."""

    def __init__(
        self,
        backbone: "nn.Module",
        feature_dim: int,
        head_type: str = "classification",
        num_classes: Optional[int] = None,
        freeze_backbone: bool = True,
        **head_kwargs: Any,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.head_type = str(head_type)
        self.freeze_backbone = bool(freeze_backbone)

        for param in self.backbone.parameters():
            param.requires_grad = not self.freeze_backbone
        if self.freeze_backbone:
            self.backbone.eval()

        if num_classes is None:
            raise ValueError("num_classes is required")

        if self.head_type == "classification":
            self.head = nn.Linear(self.feature_dim, int(num_classes))
        elif self.head_type == "detection":
            self.head = DetectionHead(
                self.feature_dim,
                int(num_classes),
                num_anchors=int(head_kwargs.get("num_anchors", 9)),
                hidden_dim=int(head_kwargs.get("hidden_dim", 256)),
                use_bn=bool(head_kwargs.get("use_bn", True)),
                dropout=float(head_kwargs.get("dropout", 0.1)),
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, x: "torch.Tensor"):
        if torch is None:
            raise RuntimeError("PyTorch is required")

        # Detection fine-tuning must keep gradients through the backbone when it is not frozen.
        # Classification behavior is unchanged because the same `freeze_backbone` switch is used.
        grad_ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
        with grad_ctx:
            features = _extract_backbone_features(self.backbone, x)
            if F is not None:
                features = F.normalize(features, dim=1, p=2)

        return self.head(features)


def infer_feature_dim(backbone: "nn.Module", image_size: int) -> int:
    """Infer the backbone feature dimension from a dummy forward."""
    if torch is None:
        raise RuntimeError("PyTorch is required")
    dummy = torch.zeros(1, 3, int(image_size), int(image_size))
    with torch.no_grad():
        feat = _extract_backbone_features(backbone, dummy)
    return int(feat.size(1))


def load_pretrained_backbone(
    checkpoint_path: Union[str, Path],
    model_name: str,
    image_size: int,
    projection_dim: int = 128,
    projection_hidden_dim: Optional[int] = None,
    projection_use_bn: bool = False,
    adapt_small_models: bool = True,
) -> "nn.Module":
    """Load a SustainVision checkpoint and return the backbone only."""
    if torch is None:
        raise RuntimeError("PyTorch is required")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    temp_model = build_model(
        model_name,
        num_classes=10,
        image_size=int(image_size),
        projection_dim=int(projection_dim),
        projection_hidden_dim=projection_hidden_dim,
        projection_use_bn=bool(projection_use_bn),
        adapt_small_models=adapt_small_models,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = None
    if isinstance(checkpoint, dict):
        model_state = (
            checkpoint.get("model_state")
            or checkpoint.get("model")
            or checkpoint.get("state_dict")
        )
    else:
        model_state = checkpoint
    if model_state is None or not isinstance(model_state, dict):
        raise ValueError(f"No model state dict found in checkpoint: {checkpoint_path}")

    cleaned_state: Dict[str, Any] = {}
    for key, value in model_state.items():
        k = str(key)
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("model."):
            k = k[len("model.") :]
        cleaned_state[k] = value

    current_state = temp_model.state_dict()
    filtered_state = {}
    skipped_keys = []
    for key, value in cleaned_state.items():
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
    if not hasattr(temp_model, "backbone"):
        raise ValueError("Model does not have a 'backbone' attribute")

    backbone = copy.deepcopy(temp_model.backbone)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    return backbone


def build_detection_head_model(
    backbone: "nn.Module",
    *,
    image_size: int,
    head_type: str,
    num_classes: int,
    freeze_backbone: bool,
    **head_kwargs: Any,
) -> MultiHeadModel:
    """Factory for `MultiHeadModel` using an inferred feature dimension."""
    feature_dim = infer_feature_dim(backbone, int(image_size))
    return MultiHeadModel(
        backbone,
        feature_dim,
        head_type=head_type,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        **head_kwargs,
    )


def build_fasterrcnn_fpn(
    *,
    num_classes_with_bg: int,
) -> Tuple["nn.Module", "nn.Module"]:
    """Build torchvision Faster R-CNN with a ResNet-18 + FPN backbone.

    Returns `(model, resnet_body_for_weight_loading)`.
    """
    if (
        FasterRCNN is None
        or resnet_fpn_backbone is None
    ):
        raise RuntimeError("torchvision detection modules are required")

    # Handle torchvision API differences without triggering deprecation warnings.
    kwargs: Dict[str, Any] = {"trainable_layers": 5}
    try:
        sig = inspect.signature(resnet_fpn_backbone)  # type: ignore[arg-type]
        params = sig.parameters
    except Exception:
        params = {}
    if "backbone_name" in params:
        kwargs["backbone_name"] = "resnet18"
    if "weights" in params:
        kwargs["weights"] = None
    elif "pretrained" in params:
        kwargs["pretrained"] = False
    if "backbone_name" not in kwargs:
        # Very old torchvision fallback
        backbone = resnet_fpn_backbone("resnet18", **{k: v for k, v in kwargs.items() if k != "backbone_name"})  # type: ignore[call-arg]
    else:
        backbone = resnet_fpn_backbone(**kwargs)  # type: ignore[call-arg]
    resnet_body_for_loading = backbone.body
    model = FasterRCNN(backbone, num_classes=int(num_classes_with_bg))  # type: ignore[call-arg]
    return model, resnet_body_for_loading


def build_rcnn_c4(
    *,
    num_classes_with_bg: int,
) -> Tuple["nn.Module", "nn.Module"]:
    """Build a Faster R-CNN model using a single C4 feature map (R-CNN-like)."""
    if (
        FasterRCNN is None
        or AnchorGenerator is None
        or MultiScaleRoIAlign is None
        or IntermediateLayerGetter is None
        or tv_models is None
    ):
        raise RuntimeError("torchvision detection modules are required")

    resnet18 = tv_models.resnet18(weights=None)
    resnet18.fc = nn.Identity()  # type: ignore[assignment]
    backbone = IntermediateLayerGetter(resnet18, return_layers={"layer4": "0"})  # type: ignore[call-arg]
    backbone.out_channels = 512  # type: ignore[attr-defined]

    rpn_anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2,
    )
    model = FasterRCNN(  # type: ignore[call-arg]
        backbone,
        num_classes=int(num_classes_with_bg),
        rpn_anchor_generator=rpn_anchor_generator,
        box_roi_pool=box_roi_pool,
    )
    return model, resnet18


def maybe_load_backbone_weights(
    *,
    checkpoint_path: Optional[Union[str, Path]],
    config_model_name: str,
    backbone_image_size: int,
    projection_dim: int,
    projection_hidden_dim: Optional[int],
    projection_use_bn: bool,
    target_resnet_body: "nn.Module",
    adapt_small_models: bool = True,
) -> None:
    """Load SustainVision backbone weights into a torchvision ResNet body when provided."""
    if checkpoint_path is None:
        return

    supcon_backbone = load_pretrained_backbone(
        checkpoint_path,
        config_model_name,
        backbone_image_size,
        projection_dim=projection_dim,
        projection_hidden_dim=projection_hidden_dim,
        projection_use_bn=projection_use_bn,
        adapt_small_models=adapt_small_models,
    )
    missing, unexpected = target_resnet_body.load_state_dict(
        supcon_backbone.state_dict(),
        strict=False,
    )
    if missing:
        print(f"[info] Backbone load missing keys: {len(missing)}")
    if unexpected:
        print(f"[info] Backbone load unexpected keys: {len(unexpected)}")
