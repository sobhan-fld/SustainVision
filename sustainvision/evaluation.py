"""Evaluation module for multi-head evaluation of pretrained feature spaces.

This module allows loading a pretrained backbone and evaluating it with
different heads (classification, object detection) without retraining.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
from .data import build_classification_dataloaders, build_detection_dataloaders


class DetectionHead(nn.Module):  # type: ignore[name-defined]
    """Object detection head for bounding box prediction.
    
    Takes backbone features and predicts:
    - Class logits for each anchor/region
    - Bounding box coordinates (x, y, w, h) for each anchor/region
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_anchors: int = 9,  # 3 scales × 3 aspect ratios
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification head: predict class for each anchor
        self.cls_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_anchors * num_classes),
        )
        
        # Regression head: predict bbox offsets (dx, dy, dw, dh) for each anchor
        self.bbox_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_anchors * 4),  # 4 = (dx, dy, dw, dh)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning class logits and bbox predictions.
        
        Args:
            features: Backbone features [batch_size, feature_dim]
            
        Returns:
            Tuple of (class_logits, bbox_preds):
            - class_logits: [batch_size, num_anchors * num_classes]
            - bbox_preds: [batch_size, num_anchors * 4]
        """
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required for model forward pass")
        
        cls_logits = self.cls_head(features)
        bbox_preds = self.bbox_head(features)
        
        # Reshape to [batch_size, num_anchors, num_classes] and [batch_size, num_anchors, 4]
        batch_size = features.size(0)
        cls_logits = cls_logits.view(batch_size, self.num_anchors, self.num_classes)
        bbox_preds = bbox_preds.view(batch_size, self.num_anchors, 4)
        
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
        **head_kwargs: Any,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.head_type = head_type
        
        # Freeze backbone - we only evaluate, not train
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        if head_type == "classification":
            if num_classes is None:
                raise ValueError("num_classes required for classification head")
            self.head = nn.Linear(feature_dim, num_classes)
        elif head_type == "detection":
            if num_classes is None:
                raise ValueError("num_classes required for detection head")
            num_anchors = head_kwargs.get("num_anchors", 9)
            hidden_dim = head_kwargs.get("hidden_dim", 256)
            self.head = DetectionHead(feature_dim, num_classes, num_anchors, hidden_dim)
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
        
        # Extract features from backbone
        with torch.no_grad():
            features = self.backbone(x)
            if isinstance(features, tuple):
                features = features[0]
            if features.ndim > 2:
                features = torch.flatten(features, 1)
        
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
    model_state = checkpoint.get("model_state") or checkpoint.get("model")
    if model_state is None:
        raise ValueError(f"No model state found in checkpoint: {checkpoint_path}")
    
    # Load state into model
    temp_model.load_state_dict(model_state, strict=False)
    
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


def evaluate_with_head(
    config: TrainingConfig,
    checkpoint_path: Union[str, Path],
    head_type: str = "classification",
    *,
    project_root: Optional[Path] = None,
    report_path: Optional[Path] = None,
    **head_kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate a pretrained backbone with a specific head.
    
    Args:
        config: Training configuration (used for dataset, model name, etc.)
        checkpoint_path: Path to pretrained checkpoint
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
    
    # Load pretrained backbone
    print(f"[info] Loading pretrained backbone from: {checkpoint_path}")
    backbone = load_pretrained_backbone(
        checkpoint_path,
        config.model,
        config.hyperparameters.get("image_size", 224),
        projection_dim=config.hyperparameters.get("projection_dim", 128),
        projection_hidden_dim=config.hyperparameters.get("projection_hidden_dim"),
        projection_use_bn=config.hyperparameters.get("projection_use_bn", False),
    )
    
    # Get feature dimension from backbone
    # We'll do a dummy forward pass to get feature dim
    dummy_input = torch.zeros(1, 3, config.hyperparameters.get("image_size", 224), 
                             config.hyperparameters.get("image_size", 224))
    with torch.no_grad():
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
    
    # Build dataloaders based on head type
    print(f"[info] Building dataloaders for dataset: {config.database}")
    if head_type == "detection":
        # Use detection dataloaders for object detection
        train_loader, val_loader, dataset_num_classes = build_detection_dataloaders(
            config.database,
            batch_size=config.hyperparameters.get("batch_size", 32),
            num_workers=config.hyperparameters.get("num_workers", 2),
            val_split=config.hyperparameters.get("val_split", 0.0),
            seed=config.seed,
            project_root=project_root,
            image_size=config.hyperparameters.get("image_size", 224),
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
        # Rebuild model with correct num_classes
        model = MultiHeadModel(
            backbone,
            feature_dim,
            head_type=head_type,
            num_classes=num_classes,
            **head_kwargs,
        )
        model = model.to(device)
    
    # Train the head (backbone stays frozen)
    print(f"[info] Training {head_type} head (backbone frozen)...")
    
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
    
    optimizer = torch.optim.SGD(
        model.head.parameters(),
        lr=config.hyperparameters.get("lr", 0.01),
        momentum=config.hyperparameters.get("momentum", 0.9),
        weight_decay=config.weight_decay,
    )
    
    epochs = config.hyperparameters.get("epochs", 10)
    
    print("[info] Training detection head with real bounding box annotations")
    print("[warn] This is a simplified implementation. For production, add anchor generation, NMS, and mAP.")
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_bbox_loss = 0.0
        train_samples = 0
        
        for images, targets in train_loader:
            images = images.to(device)
            
            optimizer.zero_grad()
            cls_logits, bbox_preds = model(images)
            
            batch_size = cls_logits.size(0)
            num_anchors = cls_logits.size(1)
            num_classes = cls_logits.size(2)
            
            # Process each image in the batch
            total_cls_loss = 0.0
            total_bbox_loss = 0.0
            valid_samples = 0
            
            for i in range(batch_size):
                target = targets[i]
                gt_boxes = target.get("boxes", torch.zeros((0, 4), device=device))
                gt_labels = target.get("labels", torch.zeros((0,), dtype=torch.long, device=device))
                
                if len(gt_boxes) == 0:
                    # Skip images with no objects (background)
                    continue
                
                # Simplified matching: assign each GT box to nearest anchor
                # In production, use proper anchor matching (IoU-based)
                num_objects = len(gt_boxes)
                assigned_anchors = torch.arange(min(num_objects, num_anchors), device=device)
                
                # Classification loss: match GT labels to assigned anchors
                if len(assigned_anchors) > 0:
                    assigned_cls_logits = cls_logits[i, assigned_anchors]  # [num_assigned, num_classes]
                    assigned_labels = gt_labels[:len(assigned_anchors)]
                    cls_loss = cls_criterion(assigned_cls_logits, assigned_labels).mean()
                    total_cls_loss += cls_loss
                
                # Bbox loss: predict box coordinates for assigned anchors
                if len(assigned_anchors) > 0:
                    # Normalize GT boxes to [0, 1] range (assuming image_size normalization)
                    # In production, use proper anchor-based box encoding
                    assigned_bbox_preds = bbox_preds[i, assigned_anchors]  # [num_assigned, 4]
                    assigned_gt_boxes = gt_boxes[:len(assigned_anchors)]
                    
                    # Simple bbox loss (would need proper encoding/decoding in production)
                    bbox_loss = bbox_criterion(assigned_bbox_preds, assigned_gt_boxes).mean()
                    total_bbox_loss += bbox_loss * 0.1  # Weight bbox loss lower
                
                valid_samples += 1
            
            if valid_samples > 0:
                loss = (total_cls_loss + total_bbox_loss) / valid_samples
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_cls_loss += (total_cls_loss / valid_samples).item() if valid_samples > 0 else 0
                train_bbox_loss += (total_bbox_loss / valid_samples).item() if valid_samples > 0 else 0
                train_samples += valid_samples
        
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_train_cls_loss = train_cls_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_train_bbox_loss = train_bbox_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_bbox_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                
                cls_logits, bbox_preds = model(images)
                
                batch_size = cls_logits.size(0)
                total_cls_loss = 0.0
                total_bbox_loss = 0.0
                valid_samples = 0
                
                for i in range(batch_size):
                    target = targets[i]
                    gt_boxes = target.get("boxes", torch.zeros((0, 4), device=device))
                    gt_labels = target.get("labels", torch.zeros((0,), dtype=torch.long, device=device))
                    
                    if len(gt_boxes) == 0:
                        continue
                    
                    num_objects = len(gt_boxes)
                    assigned_anchors = torch.arange(min(num_objects, num_anchors), device=device)
                    
                    if len(assigned_anchors) > 0:
                        assigned_cls_logits = cls_logits[i, assigned_anchors]
                        assigned_labels = gt_labels[:len(assigned_anchors)]
                        cls_loss = cls_criterion(assigned_cls_logits, assigned_labels).mean()
                        total_cls_loss += cls_loss
                        
                        assigned_bbox_preds = bbox_preds[i, assigned_anchors]
                        assigned_gt_boxes = gt_boxes[:len(assigned_anchors)]
                        bbox_loss = bbox_criterion(assigned_bbox_preds, assigned_gt_boxes).mean()
                        total_bbox_loss += bbox_loss * 0.1
                    
                    valid_samples += 1
                
                if valid_samples > 0:
                    loss = (total_cls_loss + total_bbox_loss) / valid_samples
                    val_loss += loss.item()
                    val_cls_loss += (total_cls_loss / valid_samples).item()
                    val_bbox_loss += (total_bbox_loss / valid_samples).item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_val_cls_loss = val_cls_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_val_bbox_loss = val_bbox_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"train_loss={avg_train_loss:.4f} (cls={avg_train_cls_loss:.4f}, bbox={avg_train_bbox_loss:.4f}) "
              f"val_loss={avg_val_loss:.4f} (cls={avg_val_cls_loss:.4f}, bbox={avg_val_bbox_loss:.4f})")
    
    return {
        "head_type": "detection",
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "note": "Simplified detection evaluation. For production, implement proper anchor matching, NMS, and mAP computation.",
    }

