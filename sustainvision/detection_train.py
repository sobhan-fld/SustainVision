"""Detection training and evaluation loops."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - runtime safeguard
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    DataLoader = None  # type: ignore

try:
    from codecarbon import EmissionsTracker  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EmissionsTracker = None  # type: ignore

from .config import TrainingConfig
from .detection_models import (
    AnchorGenerator,
    FasterRCNN,
    IntermediateLayerGetter,
    MultiHeadModel,
    MultiScaleRoIAlign,
    resnet_fpn_backbone,
    tv_models,
)
from .detection_metrics import (
    _denormalize_boxes_xyxy,
    _get_coco_api_from_dataset,
    apply_nms,
    compute_coco_map,
    compute_iou,
    compute_voc_map50,
    decode_slot_detections,
    generate_coco_predictions,
    generate_coco_predictions_torchvision,
    generate_voc_predictions_torchvision,
    match_anchors_to_gt,
)
from .training import _log_resource_and_energy_snapshot


def run_torchvision_detection_loop(
    *,
    config: TrainingConfig,
    checkpoint_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Path] = None,
    report_path: Optional[Path] = None,
    detector_variant: str = "torchvision_frcnn",
) -> Dict[str, Any]:
    """Public entrypoint for torchvision Faster R-CNN / RCNN-C4 loops."""
    return _evaluate_torchvision_frcnn(
        config=config,
        checkpoint_path=checkpoint_path,
        project_root=project_root,
        report_path=report_path,
        detector_variant=detector_variant,
    )


def run_slot_detection_loop(
    *,
    model: Any,
    train_loader: Any,
    val_loader: Any,
    device: Any,
    config: TrainingConfig,
) -> Dict[str, Any]:
    """Public entrypoint for slot-based detection head training/evaluation."""
    return _evaluate_detection(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
    )


def _evaluate_torchvision_frcnn(
    config: TrainingConfig,
    *,
    checkpoint_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Path] = None,
    report_path: Optional[Path] = None,
    detector_variant: str = "torchvision_frcnn",
) -> Dict[str, Any]:
    """Train/evaluate a torchvision detector on COCO/VOC.

    Supported variants:
    - `torchvision_frcnn`: Faster R-CNN with ResNet18+FPN backbone.
    - `torchvision_rcnn`: Faster R-CNN with a single C4 feature map (no FPN).

    This uses torchvision's detection model plus SustainVision's dataloaders,
    reporting, and CodeCarbon integration so we can compare emissions and accuracy
    against our custom detection head under as similar settings as possible.
    """
    if (
        torch is None
        or DataLoader is None
        or FasterRCNN is None
        or resnet_fpn_backbone is None
        or AnchorGenerator is None
        or MultiScaleRoIAlign is None
        or IntermediateLayerGetter is None
        or tv_models is None
    ):
        raise RuntimeError(
            "PyTorch + torchvision detection are required for head_type in {'torchvision_frcnn','torchvision_rcnn'}. "
            "Install torchvision with detection modules."
        )

    from .data import build_detection_dataloaders
    from . import detection_models as det_models
    from .optimizers import build_optimizer, build_scheduler
    from .reporting import write_report_csv
    from .utils import resolve_device

    device = resolve_device(config.device)

    root = Path(project_root or Path.cwd())
    save_dir = root / Path(config.save_model_path or "output_object_detection")
    save_dir.mkdir(parents=True, exist_ok=True)

    if report_path is None:
        report_path = root / Path(
            config.report_filename
            or (save_dir / "torchvision_frcnn_resnet18_metrics.csv")
        )

    # Build detection dataloaders (COCO / VOC) using existing utilities.
    print(f"[info] Building dataloaders for dataset: {config.database}")
    eval_cfg = getattr(config, "evaluation", {}) or {}
    filter_classes = None
    class_remap = None
    if eval_cfg.get("filter_coco_to_cifar100", False) and config.database.lower() == "coco":
        from .data import get_coco_classes_for_cifar100

        filter_classes = get_coco_classes_for_cifar100()

    if "voc" in config.database.lower():
        voc_subset = eval_cfg.get("voc_subset_classes")
        if isinstance(voc_subset, str):
            voc_subset = voc_subset.strip().lower()
            if voc_subset in {"cifar10", "cifar"}:
                from .data import VOC_CIFAR10_SUBSET

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
                    ordered_ids.append(resolved)
            if ordered_ids:
                filter_classes = set(ordered_ids)
                class_remap = {old_id: new_idx for new_idx, old_id in enumerate(ordered_ids)}

    train_loader, val_loader, dataset_num_classes = build_detection_dataloaders(
        config.database,
        batch_size=config.hyperparameters.get("batch_size", 2),
        num_workers=config.hyperparameters.get("num_workers", 2),
        val_split=config.hyperparameters.get("val_split", 0.0),
        seed=config.seed,
        project_root=project_root,
        image_size=config.hyperparameters.get("image_size", 800),
        max_train_images=config.hyperparameters.get("max_train_images"),
        max_val_images=config.hyperparameters.get("max_val_images"),
        filter_classes=filter_classes,
        class_remap=class_remap,
        # Important: torchvision Faster R-CNN normalizes internally.
        resize_images=False,
        normalize_images=False,
        box_coordinate_format="pixel",
        pin_memory=(device.type == "cuda"),
    )

    num_classes = dataset_num_classes  # contiguous [0..C-1]
    num_classes_with_bg = num_classes + 1  # torchvision detectors include background

    if detector_variant == "torchvision_frcnn":
        print("[info] Building torchvision Faster R-CNN (ResNet-18 + FPN) detector")
        model, resnet_body_for_loading = det_models.build_fasterrcnn_fpn(
            num_classes_with_bg=num_classes_with_bg
        )
    elif detector_variant == "torchvision_rcnn":
        print("[info] Building torchvision R-CNN style detector (ResNet-18 C4, no FPN)")
        model, resnet_body_for_loading = det_models.build_rcnn_c4(
            num_classes_with_bg=num_classes_with_bg
        )
    else:
        raise ValueError(f"Unsupported detector_variant: {detector_variant}")

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (root / checkpoint_path).resolve()
        print(f"[info] Loading pretrained backbone weights from: {checkpoint_path}")
        try:
            backbone_image_size = int(
                config.hyperparameters.get(
                    "backbone_image_size",
                    config.hyperparameters.get("image_size", 224),
                )
            )
            adapt_small_models = bool(config.hyperparameters.get("adapt_small_models", True))
            det_models.maybe_load_backbone_weights(
                checkpoint_path=checkpoint_path,
                config_model_name=config.model,
                backbone_image_size=backbone_image_size,
                projection_dim=config.hyperparameters.get("projection_dim", 128),
                projection_hidden_dim=config.hyperparameters.get("projection_hidden_dim"),
                projection_use_bn=config.hyperparameters.get("projection_use_bn", False),
                target_resnet_body=resnet_body_for_loading,
                adapt_small_models=adapt_small_models,
            )
        except Exception as exc:
            print(f"[warn] Failed to load backbone weights: {exc}")

    model.to(device)

    base_lr = float(config.hyperparameters.get("lr", 0.01))
    backbone_lr = float(config.hyperparameters.get("backbone_lr", base_lr))
    head_lr = float(config.hyperparameters.get("head_lr", base_lr))
    freeze_backbone_epochs = int(config.hyperparameters.get("freeze_backbone_epochs", 0))

    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]

    optimizer = build_optimizer(
        config.optimizer,
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        lr=head_lr,
        momentum=config.hyperparameters.get("momentum", 0.9),
        weight_decay=config.weight_decay,
    )

    scheduler_config = getattr(config, "scheduler", None)
    scheduler = build_scheduler(scheduler_config, optimizer) if scheduler_config else None

    epochs = int(config.hyperparameters.get("epochs", 12))
    max_train_batches = config.hyperparameters.get("max_train_batches")
    max_val_batches = config.hyperparameters.get("max_val_batches")
    gradient_clip_norm = config.gradient_clip_norm
    if gradient_clip_norm is None and "gradient_clip_norm" in (config.hyperparameters or {}):
        try:
            gradient_clip_norm = float(config.hyperparameters.get("gradient_clip_norm"))
            print("[warn] `hyperparameters.gradient_clip_norm` is deprecated for detection; use top-level `gradient_clip_norm`.")
        except Exception:
            gradient_clip_norm = None
    score_threshold = float(config.hyperparameters.get("score_threshold", 0.05))

    try:
        max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    except Exception:
        max_train_batches = None
    try:
        max_val_batches = int(max_val_batches) if max_val_batches is not None else None
    except Exception:
        max_val_batches = None

    track_emissions = bool(config.hyperparameters.get("track_emissions", True))
    emissions_kg_total: Optional[float] = None
    energy_kwh_total: Optional[float] = None
    duration_seconds: Optional[float] = None
    start_time = time.perf_counter()
    block_tracker: Any = None

    def _start_block_tracker() -> Any:
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

        if block_kg is not None:
            emissions_kg_total = (emissions_kg_total or 0.0) + float(block_kg)
        if block_data is not None:
            try:
                block_energy = getattr(block_data, "energy_consumed", None)
                if block_energy is not None:
                    energy_kwh_total = (energy_kwh_total or 0.0) + float(block_energy)
            except Exception:
                pass

        try:
            _log_resource_and_energy_snapshot(
                artifact_dir=save_dir,
                epoch=current_epoch,
                emissions_data=block_data,
                elapsed_seconds=time.perf_counter() - start_time,
                device=device,
            )
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(
                f"[warn] Failed to record resource/energy snapshot at epoch {current_epoch}: {exc}"
            )

    metrics_rows: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    save_model = bool(config.save_model)
    checkpoint_interval = int(config.hyperparameters.get("checkpoint_interval", 50))

    def _to_torchvision_targets(images_batch, targets_batch):
        converted: List[Dict[str, Any]] = []
        for i, t in enumerate(targets_batch):
            if not isinstance(t, dict) or "boxes" not in t or "labels" not in t:
                continue
            boxes = t["boxes"]
            labels = t["labels"]
            if not hasattr(boxes, "clone") or not hasattr(labels, "to"):
                continue
            boxes_t = boxes.clone()
            # Backward compatibility: if boxes look normalized, convert to pixels.
            try:
                if boxes_t.numel() > 0 and float(boxes_t.max().item()) <= 1.5:
                    h = images_batch[i].shape[1]
                    w = images_batch[i].shape[2]
                    boxes_t[:, 0] *= w
                    boxes_t[:, 2] *= w
                    boxes_t[:, 1] *= h
                    boxes_t[:, 3] *= h
            except Exception:
                pass
            converted.append(
                {
                    "boxes": boxes_t.to(device),
                    "labels": (labels.to(device) + 1),  # shift for torchvision background class
                }
            )
        return converted

    try:
        for epoch in range(epochs):
            if track_emissions and (epoch % 10 == 0) and block_tracker is None:
                block_tracker = _start_block_tracker()

            # Optional: freeze backbone for early epochs to reduce overfitting.
            if freeze_backbone_epochs > 0 and epoch < freeze_backbone_epochs:
                for param in backbone_params:
                    param.requires_grad = False
                optimizer.param_groups[0]["lr"] = 0.0
                model.backbone.eval()
            else:
                for param in backbone_params:
                    param.requires_grad = True
                optimizer.param_groups[0]["lr"] = backbone_lr
                model.backbone.train()

            # --------------------
            # Training
            # --------------------
            model.train()
            train_loss = 0.0
            train_samples = 0

            try:
                from tqdm.auto import tqdm

                train_iter = tqdm(
                    train_loader,
                    desc=f"Train (FRCNN) {epoch+1}/{epochs}",
                    leave=False,
                    unit="batch",
                )
            except Exception:
                train_iter = train_loader

            for batch_idx, (images, targets) in enumerate(train_iter):
                if max_train_batches is not None and batch_idx >= max_train_batches:
                    break

                images_list = [img.to(device) for img in images]
                converted_targets = _to_torchvision_targets(images, targets)

                if not converted_targets:
                    continue

                optimizer.zero_grad()
                loss_dict = model(images_list, converted_targets)  # type: ignore[call-arg]
                loss = sum(loss for loss in loss_dict.values())
                loss.backward()
                if gradient_clip_norm is not None and gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

                batch_size = len(images_list)
                train_loss += loss.item() * batch_size
                train_samples += batch_size

            avg_train_loss = train_loss / max(train_samples, 1)

            if scheduler is not None:
                scheduler.step()

            # --------------------
            # Validation
            # --------------------
            model.eval()
            val_loss = 0.0
            val_samples = 0

            try:
                from tqdm.auto import tqdm

                val_iter = tqdm(
                    val_loader,
                    desc=f"Val (FRCNN) {epoch+1}/{epochs}",
                    leave=False,
                    unit="batch",
                )
            except Exception:
                val_iter = val_loader

            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(val_iter):
                    if max_val_batches is not None and batch_idx >= max_val_batches:
                        break

                    images_list = [img.to(device) for img in images]
                    converted_targets = _to_torchvision_targets(images, targets)

                    if not converted_targets:
                        continue

                    # Faster R-CNN returns loss dict in train mode, predictions in eval mode
                    # Temporarily switch to train mode to compute validation loss
                    model.train()
                    loss_dict = model(images_list, converted_targets)  # type: ignore[call-arg]
                    model.eval()  # Switch back to eval mode
                    
                    if isinstance(loss_dict, dict):
                        loss = sum(loss for loss in loss_dict.values())
                    else:
                        # Fallback: if somehow we get predictions, skip this batch
                        print(f"[warn] Unexpected loss_dict type in validation: {type(loss_dict)}")
                        continue

                    batch_size = len(images_list)
                    val_loss += loss.item() * batch_size
                    val_samples += batch_size

            avg_val_loss = val_loss / max(val_samples, 1)

            print(
                f"Epoch {epoch+1}/{epochs} (FRCNN) - "
                f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}"
            )

            # Get current learning rate
            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
            
            metrics_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": current_lr,
                }
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1

            if save_model and (
                (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs
            ):
                ckpt_path = save_dir / f"torchvision_frcnn_resnet18_epoch{epoch+1}.pt"
                try:
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"[info] Saved detector checkpoint to {ckpt_path}")
                except Exception as exc:
                    print(f"[warn] Failed to save checkpoint to {ckpt_path}: {exc}")

            if track_emissions and ((epoch + 1) % 10 == 0) and block_tracker is not None:
                _stop_block_tracker(epoch + 1)

        duration_seconds = time.perf_counter() - start_time

        if block_tracker is not None:
            _stop_block_tracker(epochs)

    finally:
        if block_tracker is not None:
            try:
                block_tracker.stop()
            except Exception:
                pass

    # COCO mAP via pycocotools
    eval_cfg_local = getattr(config, "evaluation", {}) or {}
    save_predictions_flag = bool(eval_cfg_local.get("save_predictions", False))
    predictions_path = eval_cfg_local.get("predictions_path")

    coco_metrics: Dict[str, Any] = {}
    voc_metrics: Dict[str, Any] = {}
    coco_gt = _get_coco_api_from_dataset(getattr(val_loader, "dataset", None))
    saved_predictions_path: Optional[str] = None
    if coco_gt is not None:
        try:
            coco_predictions = generate_coco_predictions_torchvision(
                model,
                val_loader,
                device=device,
                score_threshold=score_threshold,
                max_batches=max_val_batches,
            )
            if save_predictions_flag:
                import json

                if isinstance(predictions_path, str) and predictions_path.strip():
                    out_path = Path(predictions_path.strip())
                    if not out_path.is_absolute():
                        out_path = (Path.cwd() / out_path).resolve()
                else:
                    out_path = save_dir / "coco_predictions_torchvision.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(coco_predictions, f)
                saved_predictions_path = str(out_path)

            map_iou_thresholds = eval_cfg_local.get("map_iou_thresholds")
            if isinstance(map_iou_thresholds, (list, tuple)):
                map_iou_thresholds = [float(x) for x in map_iou_thresholds]
            else:
                map_iou_thresholds = None
            coco_metrics = compute_coco_map(
                coco_gt, coco_predictions, iou_thresholds=map_iou_thresholds
            )
            if saved_predictions_path is not None:
                coco_metrics["predictions_json"] = saved_predictions_path
        except Exception as exc:
            coco_metrics = {"note": f"COCO mAP computation failed: {exc}"}
            if saved_predictions_path is not None:
                coco_metrics["predictions_json"] = saved_predictions_path
    elif "voc" in config.database.lower():
        try:
            voc_predictions = generate_voc_predictions_torchvision(
                model,
                val_loader,
                device=device,
                score_threshold=score_threshold,
                max_batches=max_val_batches,
            )

            if save_predictions_flag:
                import json

                if isinstance(predictions_path, str) and predictions_path.strip():
                    out_path = Path(predictions_path.strip())
                    if not out_path.is_absolute():
                        out_path = (Path.cwd() / out_path).resolve()
                else:
                    out_path = save_dir / "voc_predictions_torchvision.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(voc_predictions, f)
                saved_predictions_path = str(out_path)

            voc_ground_truths: List[Dict[str, Any]] = []
            image_index = 0
            for batch_idx, (images, targets) in enumerate(val_loader):
                if max_val_batches is not None and batch_idx >= max_val_batches:
                    break
                for img, tgt in zip(images, targets):
                    if not isinstance(tgt, dict) or "boxes" not in tgt or "labels" not in tgt:
                        image_index += 1
                        continue
                    boxes_norm = tgt["boxes"]
                    labels = tgt["labels"]
                    orig_size = tgt.get("orig_size", None)
                    if orig_size is not None:
                        try:
                            orig_h = int(orig_size[0])
                            orig_w = int(orig_size[1])
                        except Exception:
                            orig_h = int(img.shape[1])
                            orig_w = int(img.shape[2])
                    else:
                        orig_h = int(img.shape[1])
                        orig_w = int(img.shape[2])
                    boxes_abs = _denormalize_boxes_xyxy(
                        boxes_norm,
                        orig_h=orig_h,
                        orig_w=orig_w,
                    )
                    voc_ground_truths.append(
                        {
                            "image_id": image_index,
                            "boxes": boxes_abs.detach().cpu().tolist(),
                            "labels": labels.detach().cpu().tolist(),
                        }
                    )
                    image_index += 1

            voc_metrics = compute_voc_map50(
                voc_ground_truths,
                voc_predictions,
                num_classes=num_classes,
            )
            if saved_predictions_path is not None:
                voc_metrics["predictions_json"] = saved_predictions_path
        except Exception as exc:
            voc_metrics = {"note": f"VOC mAP computation failed: {exc}"}
            if saved_predictions_path is not None:
                voc_metrics["predictions_json"] = saved_predictions_path

    # Write CSV report (using same format as _evaluate_detection)
    if save_model and metrics_rows:
        try:
            import csv
            import datetime

            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", newline="") as f:
                fieldnames = [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "learning_rate",
                    "map",
                    "map50",
                    "map75",
                    "predictions_json",
                    "emissions_kg",
                    "energy_kwh",
                    "duration_seconds",
                    "finished_at",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in metrics_rows:
                    # Ensure consistent schema for all rows
                    row_dict = {
                        "epoch": row.get("epoch", ""),
                        "train_loss": f"{row.get('train_loss', 0.0):.6f}",
                        "val_loss": f"{row.get('val_loss', 0.0):.6f}",
                        "learning_rate": f"{row.get('learning_rate', 0.0):.6f}",
                        "map": "",
                        "map50": "",
                        "map75": "",
                        "predictions_json": "",
                        "emissions_kg": "",
                        "energy_kwh": "",
                        "duration_seconds": "",
                        "finished_at": "",
                    }
                    writer.writerow(row_dict)
                
                # Add summary row with mAP and emissions if available
                if coco_metrics or voc_metrics or emissions_kg_total is not None:
                    map_val = coco_metrics.get("map") if coco_metrics else None
                    map50_val = coco_metrics.get("map50") if coco_metrics else voc_metrics.get("map50")
                    map75_val = coco_metrics.get("map75") if coco_metrics else None
                    predictions_json = coco_metrics.get("predictions_json", "") if coco_metrics else voc_metrics.get("predictions_json", "")
                    finished_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
                    summary_row = {
                        "epoch": "summary",
                        "train_loss": "",
                        "val_loss": "",
                        "learning_rate": "",
                        "map": f"{map_val:.6f}" if map_val is not None else "",
                        "map50": f"{map50_val:.6f}" if map50_val is not None else "",
                        "map75": f"{map75_val:.6f}" if map75_val is not None else "",
                        "predictions_json": predictions_json or "",
                        "emissions_kg": f"{emissions_kg_total:.6f}" if emissions_kg_total is not None else "",
                        "energy_kwh": f"{energy_kwh_total:.6f}" if energy_kwh_total is not None else "",
                        "duration_seconds": f"{duration_seconds:.2f}" if duration_seconds is not None else "",
                        "finished_at": finished_at,
                    }
                    writer.writerow(summary_row)
        except Exception as exc:
            print(f"[warn] Failed to write detection CSV report to {report_path}: {exc}")

    # Save a YAML snapshot of the config next to the weights/metrics.
    if save_model:
        try:
            import yaml  # type: ignore

            cfg_path = save_dir / "config.yaml"
            data = getattr(config, "__dict__", {})
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"[warn] Failed to write config snapshot to {save_dir}: {exc}")

    results = {
        "head_type": "torchvision_frcnn",
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "emissions_kg": emissions_kg_total,
        "energy_kwh": energy_kwh_total,
        "duration_seconds": duration_seconds,
    }
    results.update(coco_metrics)
    results.update(voc_metrics)
    return results

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
    gradient_clip_norm = config.gradient_clip_norm
    if gradient_clip_norm is None and "gradient_clip_norm" in (config.hyperparameters or {}):
        try:
            gradient_clip_norm = float(config.hyperparameters.get("gradient_clip_norm"))
            print("[warn] `hyperparameters.gradient_clip_norm` is deprecated for detection; use top-level `gradient_clip_norm`.")
        except Exception:
            gradient_clip_norm = None
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
            if track_emissions and (epoch % 10 == 0) and block_tracker is None:
                block_tracker = _start_block_tracker()

            # --------------------
            # Training
            # --------------------
            model.train()
            freeze_backbone = getattr(model, "freeze_backbone", True)
            if freeze_backbone and hasattr(model, "backbone"):
                model.backbone.eval()
            if hasattr(model, "head"):
                model.head.train()

            train_loss = 0.0
            train_cls_loss = 0.0
            train_bbox_loss = 0.0
            train_samples = 0

            try:
                from tqdm.auto import tqdm

                train_iterable = tqdm(
                    train_loader,
                    desc=f"Train {epoch+1}/{epochs}",
                    leave=False,
                    unit="batch",
                )
            except ImportError:
                train_iterable = train_loader

            for batch_idx, (images, targets) in enumerate(train_iterable):
                if max_train_batches is not None and batch_idx >= max_train_batches:
                    break
                images = images.to(device)

                optimizer.zero_grad()
                cls_logits, bbox_preds = model(images)

                batch_size = cls_logits.size(0)
                num_classes = cls_logits.size(2)

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
                    if matched_pred.numel() == 0:
                        matched_pred, matched_gt = match_anchors_to_gt(
                            bbox_preds[i],
                            gt_boxes,
                            iou_threshold=0.01,
                        )
                        if matched_pred.numel() == 0:
                            if len(gt_boxes) > 0:
                                iou = compute_iou(bbox_preds[i], gt_boxes)
                                best_anchors = iou.argmax(dim=0)
                                matched_pred = best_anchors
                                matched_gt = torch.arange(len(gt_boxes), device=gt_boxes.device)
                            else:
                                continue

                    assigned_cls_logits = cls_logits[i, matched_pred]
                    assigned_labels = gt_labels[matched_gt]
                    valid_mask = (assigned_labels >= 0) & (assigned_labels < num_classes)
                    if valid_mask.any():
                        cls_loss = cls_criterion(
                            assigned_cls_logits[valid_mask],
                            assigned_labels[valid_mask],
                        ).mean()
                        total_cls_loss += cls_loss

                    assigned_bbox_preds = bbox_preds[i, matched_pred]
                    assigned_gt_boxes = gt_boxes[matched_gt]
                    bbox_loss_per_box = bbox_criterion(assigned_bbox_preds, assigned_gt_boxes).sum(dim=1)
                    bbox_loss = bbox_loss_per_box.mean()
                    if epoch < bbox_loss_warmup_epochs:
                        warmup_factor = epoch / bbox_loss_warmup_epochs
                        current_weight = bbox_loss_weight + (bbox_loss_max_weight - bbox_loss_weight) * warmup_factor
                    else:
                        current_weight = bbox_loss_weight
                    total_bbox_loss += bbox_loss * current_weight

                    valid_samples += 1

                if valid_samples > 0:
                    loss = (total_cls_loss + total_bbox_loss) / valid_samples
                    loss.backward()
                    if gradient_clip_norm is not None and gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.head.parameters(), gradient_clip_norm)
                    optimizer.step()

                    train_loss += loss.item()
                    train_cls_loss += (total_cls_loss / valid_samples).item()
                    train_bbox_loss += (total_bbox_loss / valid_samples).item()
                    train_samples += valid_samples

            avg_train_loss = train_loss / max(train_samples, 1)
            avg_train_cls_loss = train_cls_loss / max(train_samples, 1)
            avg_train_bbox_loss = train_bbox_loss / max(train_samples, 1)

            # --------------------
            # Validation
            # --------------------
            model.eval()
            val_loss = 0.0
            val_cls_loss = 0.0
            val_bbox_loss = 0.0
            val_samples = 0
            total_gt_objects = 0
            matched_gt_objects = 0

            try:
                from tqdm.auto import tqdm

                val_iterable = tqdm(
                    val_loader,
                    desc=f"Val {epoch+1}/{epochs}",
                    leave=False,
                    unit="batch",
                )
            except ImportError:
                val_iterable = val_loader

            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(val_iterable):
                    if max_val_batches is not None and batch_idx >= max_val_batches:
                        break
                    images = images.to(device)
                    cls_logits, bbox_preds = model(images)

                    batch_size = cls_logits.size(0)
                    num_classes = cls_logits.size(2)
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
                        if matched_pred.numel() == 0:
                            matched_pred, matched_gt = match_anchors_to_gt(
                                bbox_preds[i],
                                gt_boxes,
                                iou_threshold=0.01,
                            )
                            if matched_pred.numel() == 0:
                                if len(gt_boxes) > 0:
                                    iou = compute_iou(bbox_preds[i], gt_boxes)
                                    best_anchors = iou.argmax(dim=0)
                                    matched_pred = best_anchors
                                    matched_gt = torch.arange(len(gt_boxes), device=gt_boxes.device)
                                else:
                                    continue

                        assigned_cls_logits = cls_logits[i, matched_pred]
                        assigned_labels = gt_labels[matched_gt]
                        valid_mask = (assigned_labels >= 0) & (assigned_labels < num_classes)
                        if valid_mask.any():
                            cls_loss = cls_criterion(
                                assigned_cls_logits[valid_mask],
                                assigned_labels[valid_mask],
                            ).mean()
                            total_cls_loss += cls_loss

                        assigned_bbox_preds = bbox_preds[i, matched_pred]
                        assigned_gt_boxes = gt_boxes[matched_gt]
                        bbox_loss_per_box = bbox_criterion(assigned_bbox_preds, assigned_gt_boxes).sum(dim=1)
                        bbox_loss = bbox_loss_per_box.mean()
                        if epoch < bbox_loss_warmup_epochs:
                            warmup_factor = epoch / bbox_loss_warmup_epochs
                            current_weight = bbox_loss_weight + (bbox_loss_max_weight - bbox_loss_weight) * warmup_factor
                        else:
                            current_weight = bbox_loss_weight
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
                        if (epoch in {0, 9, 19}) and batch_idx == 0 and i == 0:
                            print(
                                f"[debug] Epoch {epoch+1} - After decode+NMS: "
                                f"{len(pred_boxes)} predictions, {len(gt_boxes)} GT boxes"
                            )
                            if len(pred_boxes) > 0:
                                print(
                                    f"[debug]   Prediction classes: "
                                    f"{pred_labels[:min(5, len(pred_labels))].tolist()}, "
                                    f"scores: {pred_scores[:min(5, len(pred_scores))].tolist()}"
                                )
                            if len(gt_boxes) > 0:
                                print(
                                    f"[debug]   GT classes: {gt_labels[:min(5, len(gt_labels))].tolist()}"
                                )
                                unique_pred_classes = set(pred_labels.tolist())
                                unique_gt_classes = set(gt_labels.tolist())
                                overlap = len(unique_pred_classes & unique_gt_classes)
                                print(
                                    f"[debug]   Class overlap: "
                                    f"{overlap}/{len(unique_gt_classes)} GT classes predicted"
                                )

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
                        val_samples += valid_samples

            avg_val_loss = val_loss / max(val_samples, 1)
            avg_val_cls_loss = val_cls_loss / max(val_samples, 1)
            avg_val_bbox_loss = val_bbox_loss / max(val_samples, 1)

            iou_recall = float(matched_gt_objects) / float(total_gt_objects) if total_gt_objects > 0 else 0.0

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch

            if epoch < bbox_loss_warmup_epochs:
                warmup_factor = epoch / bbox_loss_warmup_epochs
                current_bbox_weight = bbox_loss_weight + (bbox_loss_max_weight - bbox_loss_weight) * warmup_factor
            else:
                current_bbox_weight = bbox_loss_weight

            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"train_loss={avg_train_loss:.4f} (cls={avg_train_cls_loss:.4f}, bbox={avg_train_bbox_loss:.4f}) "
                    f"val_loss={avg_val_loss:.4f} (cls={avg_val_cls_loss:.4f}, bbox={avg_val_bbox_loss:.4f}) "
                    f"iou_recall@{iou_threshold:.2f}={iou_recall:.4f} lr={current_lr:.6f} "
                    f"bbox_w={current_bbox_weight:.1f}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"train_loss={avg_train_loss:.4f} (cls={avg_train_cls_loss:.4f}, bbox={avg_train_bbox_loss:.4f}) "
                    f"val_loss={avg_val_loss:.4f} (cls={avg_val_cls_loss:.4f}, bbox={avg_val_bbox_loss:.4f}) "
                    f"iou_recall@{iou_threshold:.2f}={iou_recall:.4f} bbox_w={current_bbox_weight:.1f}"
                )

            if track_emissions and ((epoch + 1) % 10 == 0) and block_tracker is not None:
                _stop_block_tracker(epoch + 1)

            if save_model and checkpoint_interval > 0 and (
                (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs
            ):
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
