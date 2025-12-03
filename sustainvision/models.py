"""Model building utilities."""

from __future__ import annotations

from typing import Optional, Tuple, Sequence, Union

try:
    from torch import nn
    import torch.nn.functional as F
    import torch
    from torchvision import models as tv_models
except Exception:
    nn = None  # type: ignore
    F = None  # type: ignore
    torch = None  # type: ignore
    tv_models = None  # type: ignore


Selector = Union[int, str]


def _safe_get_submodule(root: "nn.Module", path: Sequence[Selector]):  # type: ignore[name-defined]
    """Safely navigate into nested Sequential/attribute structures."""
    module = root
    for key in path:
        try:
            if isinstance(key, int):
                module = module[key]  # type: ignore[index]
            else:
                module = getattr(module, key)
        except Exception:
            return None
    return module


def _adapt_mobilenet_for_small_images(backbone: "nn.Module", image_size: int) -> None:  # type: ignore[name-defined]
    """Reduce initial downsampling so 32x32 inputs don't collapse to 1x1."""
    if image_size > 64 or not hasattr(backbone, "features"):
        return

    features = backbone.features  # type: ignore[attr-defined]
    adjustments = [
        (0, 0),  # first stem conv
        (1, "block", 0, 0),  # first depthwise conv
        (2, "block", 1, 0),  # second block depthwise conv
    ]
    for path in adjustments:
        conv = _safe_get_submodule(features, path)
        if isinstance(conv, nn.Conv2d):
            conv.stride = (1, 1)


class ProjectionModel(nn.Module):  # type: ignore[name-defined]
    """Wrap a backbone with classification and projection heads."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        projection_dim: int,
        *,
        projection_hidden_dim: Optional[int] = None,
        use_batchnorm_projector: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)
        hidden_dim = projection_hidden_dim or max(feature_dim, projection_dim)
        projector_layers = [nn.Linear(feature_dim, hidden_dim)]
        if use_batchnorm_projector:
            projector_layers.append(nn.BatchNorm1d(hidden_dim))
        projector_layers.append(nn.ReLU())
        projector_layers.append(nn.Linear(hidden_dim, projection_dim))
        self.projector = nn.Sequential(*projector_layers)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:  # type: ignore[name-defined]
        """Forward pass returning logits and normalized embeddings."""
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required for model forward pass")
        
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        embeddings = F.normalize(self.projector(features), dim=1) if F is not None else features
        return logits, embeddings


def build_linear_model(
    num_classes: int,
    image_size: int,
    projection_dim: int,
    *,
    projection_hidden_dim: Optional[int] = None,
    projection_use_bn: bool = False,
) -> ProjectionModel:
    """Build a simple MLP model as fallback."""
    if nn is None:
        raise RuntimeError("PyTorch is required for model building")
    
    in_features = image_size * image_size * 3
    feature_dim = 512
    backbone = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, feature_dim),
        nn.ReLU(),
    )
    return ProjectionModel(
        backbone,
        feature_dim,
        num_classes,
        projection_dim,
        projection_hidden_dim=projection_hidden_dim,
        use_batchnorm_projector=projection_use_bn,
    )


def build_model(
    name: str,
    *,
    num_classes: int,
    image_size: int,
    projection_dim: int,
    projection_hidden_dim: Optional[int] = None,
    projection_use_bn: bool = False,
) -> nn.Module:
    """Build a model from name, using torchvision backbones when available."""
    if nn is None:
        raise RuntimeError("PyTorch is required for model building")
    
    key = (name or "resnet18").lower()

    if tv_models is None:
        print("[warn] torchvision models unavailable. Using simple MLP classifier.")
        return build_linear_model(
            num_classes,
            image_size,
            projection_dim,
            projection_hidden_dim=projection_hidden_dim,
            projection_use_bn=projection_use_bn,
        )

    try:
        if key == "resnet18":
            backbone = tv_models.resnet18(weights=None)
            if image_size <= 64:
                backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                backbone.maxpool = nn.Identity()
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            return ProjectionModel(
                backbone,
                feature_dim,
                num_classes,
                projection_dim,
                projection_hidden_dim=projection_hidden_dim,
                use_batchnorm_projector=projection_use_bn,
            )
        if key == "resnet34":
            backbone = tv_models.resnet34(weights=None)
            if image_size <= 64:
                backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                backbone.maxpool = nn.Identity()
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            return ProjectionModel(
                backbone,
                feature_dim,
                num_classes,
                projection_dim,
                projection_hidden_dim=projection_hidden_dim,
                use_batchnorm_projector=projection_use_bn,
            )
        if key == "resnet50":
            backbone = tv_models.resnet50(weights=None)
            if image_size <= 64:
                backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                backbone.maxpool = nn.Identity()
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            return ProjectionModel(
                backbone,
                feature_dim,
                num_classes,
                projection_dim,
                projection_hidden_dim=projection_hidden_dim,
                use_batchnorm_projector=projection_use_bn,
            )
        if key == "mobilenet_v3_small":
            backbone = tv_models.mobilenet_v3_small(weights=None)
            _adapt_mobilenet_for_small_images(backbone, image_size)
            feature_dim = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Identity()
            return ProjectionModel(
                backbone,
                feature_dim,
                num_classes,
                projection_dim,
                projection_hidden_dim=projection_hidden_dim,
                use_batchnorm_projector=projection_use_bn,
            )
        if key == "efficientnet_b0":
            backbone = tv_models.efficientnet_b0(weights=None)
            feature_dim = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Identity()
            return ProjectionModel(
                backbone,
                feature_dim,
                num_classes,
                projection_dim,
                projection_hidden_dim=projection_hidden_dim,
                use_batchnorm_projector=projection_use_bn,
            )
        if key in {"vit_b_16", "vit-b-16", "vit"}:
            backbone = tv_models.vit_b_16(weights=None)
            feature_dim = backbone.heads.head.in_features
            backbone.heads.head = nn.Identity()
            return ProjectionModel(
                backbone,
                feature_dim,
                num_classes,
                projection_dim,
                projection_hidden_dim=projection_hidden_dim,
                use_batchnorm_projector=projection_use_bn,
            )
    except Exception as exc:  # pragma: no cover - model construction safeguard
        print(f"[warn] Failed to build model '{name}': {exc}. Falling back to MLP classifier.")

    print(f"[warn] Unsupported model '{name}'. Using simple MLP classifier instead.")
    return build_linear_model(
        num_classes,
        image_size,
        projection_dim,
        projection_hidden_dim=projection_hidden_dim,
        projection_use_bn=projection_use_bn,
    )

