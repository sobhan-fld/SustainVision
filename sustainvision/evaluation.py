"""Evaluation module for classification and detection dispatch.

Detection-specific implementations were moved into dedicated modules:
- `sustainvision.detection_models`
- `sustainvision.detection_metrics`
- `sustainvision.detection_train`
- `sustainvision.rcnn_classic`

This file now keeps:
- classification evaluation
- top-level `evaluate_with_head` dispatcher
- backward-compatible re-exports/wrappers for existing imports
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

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
from .data import build_classification_dataloaders, build_detection_dataloaders
from .models import build_model
from . import detection_models as _dmodels
from . import detection_metrics as _dmetrics
from . import detection_train as _dtrain

# Backward-compatible re-exports (external scripts may import these from sustainvision.evaluation)
DetectionHead = _dmodels.DetectionHead
MultiHeadModel = _dmodels.MultiHeadModel
load_pretrained_backbone = _dmodels.load_pretrained_backbone

compute_iou = _dmetrics.compute_iou
match_anchors_to_gt = _dmetrics.match_anchors_to_gt
apply_nms = _dmetrics.apply_nms
decode_slot_detections = _dmetrics.decode_slot_detections
_get_coco_api_from_dataset = _dmetrics._get_coco_api_from_dataset
_denormalize_boxes_xyxy = _dmetrics._denormalize_boxes_xyxy
_xyxy_to_xywh = _dmetrics._xyxy_to_xywh
generate_coco_predictions_torchvision = _dmetrics.generate_coco_predictions_torchvision
generate_voc_predictions_torchvision = _dmetrics.generate_voc_predictions_torchvision
generate_coco_predictions = _dmetrics.generate_coco_predictions
compute_coco_map = _dmetrics.compute_coco_map
compute_voc_map50 = _dmetrics.compute_voc_map50


def _evaluate_torchvision_frcnn(
    config: TrainingConfig,
    *,
    checkpoint_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Path] = None,
    report_path: Optional[Path] = None,
    detector_variant: str = "torchvision_frcnn",
) -> Dict[str, Any]:
    """Backward-compatible wrapper for the moved torchvision detection loop."""
    return _dtrain._evaluate_torchvision_frcnn(
        config=config,
        checkpoint_path=checkpoint_path,
        project_root=project_root,
        report_path=report_path,
        detector_variant=detector_variant,
    )


def _evaluate_detection(
    model: MultiHeadModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> Dict[str, Any]:
    """Backward-compatible wrapper for the moved slot-detection loop."""
    return _dtrain._evaluate_detection(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
    )


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

    # Import detection helpers lazily to avoid circular imports while refactoring.
    from . import detection_models as det_models
    from . import detection_train as det_train

    # Special-case: use torchvision's Faster R-CNN style detector with ResNet18+FPN.
    if head_type in {"torchvision_frcnn", "torchvision_rcnn"}:
        return det_train.run_torchvision_detection_loop(
            config=config,
            checkpoint_path=checkpoint_path,
            project_root=project_root,
            report_path=report_path,
            detector_variant=head_type,
        )
    if head_type == "rcnn_classic":
        from .rcnn_classic import evaluate_rcnn_classic

        return evaluate_rcnn_classic(
            config=config,
            checkpoint_path=checkpoint_path,
            head_type=head_type,
            project_root=project_root,
            report_path=report_path,
            **head_kwargs,
        )
    
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
        adapt_small_models = bool(config.hyperparameters.get("adapt_small_models", True))
        backbone = det_models.load_pretrained_backbone(
            checkpoint_path,
            config.model,
            config.hyperparameters.get("image_size", 224),
            projection_dim=config.hyperparameters.get("projection_dim", 128),
            projection_hidden_dim=config.hyperparameters.get("projection_hidden_dim"),
            projection_use_bn=config.hyperparameters.get("projection_use_bn", False),
            adapt_small_models=adapt_small_models,
        )
    else:
        # Build model from scratch
        print(f"[info] Building model from scratch: {config.model}")
        from .utils import set_seed
        set_seed(config.seed)
        
        # Build the full model (backbone + projection head) but we'll extract just the backbone
        adapt_small_models = bool(config.hyperparameters.get("adapt_small_models", True))
        full_model = build_model(
            config.model,
            num_classes=10,  # Dummy, we'll extract backbone
            image_size=config.hyperparameters.get("image_size", 224),
            projection_dim=config.hyperparameters.get("projection_dim", 128),
            projection_hidden_dim=config.hyperparameters.get("projection_hidden_dim"),
            projection_use_bn=config.hyperparameters.get("projection_use_bn", False),
            adapt_small_models=adapt_small_models,
        )
        # Extract backbone (remove classifier/projector)
        backbone = full_model.backbone if hasattr(full_model, "backbone") else full_model
    
    # Infer feature dimension using the detection model helper (shared logic for backbones).
    feature_dim = det_models.infer_feature_dim(
        backbone,
        int(config.hyperparameters.get("image_size", 224)),
    )
    
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
            box_coordinate_format="normalized",
            pin_memory=(device.type == "cuda"),
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
            pin_memory=(device.type == "cuda"),
        )
    
    # Update num_classes if we got it from dataset
    if num_classes != dataset_num_classes:
        print(f"[warn] num_classes mismatch: config={num_classes}, dataset={dataset_num_classes}. Using dataset value.")
        num_classes = dataset_num_classes
    
    # Build model with head (with correct num_classes and freeze_backbone setting)
    if head_type == "detection":
        model = det_models.build_detection_head_model(
            backbone,
            image_size=int(config.hyperparameters.get("image_size", 224)),
            head_type=head_type,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            **head_kwargs,
        )
    else:
        model = det_models.MultiHeadModel(
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
        return det_train.run_slot_detection_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
        )
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
