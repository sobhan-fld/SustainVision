from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch.nn as nn

from sustainvision.config import TrainingConfig
from sustainvision.data import build_detection_dataloaders
from sustainvision.detection_models import MultiHeadModel
from sustainvision.detection_train import (
    run_slot_detection_loop,
    run_torchvision_detection_loop,
)
from sustainvision.rcnn_classic import evaluate_rcnn_classic


def _base_detection_config() -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.device = "cpu"
    cfg.database = "synthetic_detection"
    cfg.model = "resnet18"
    cfg.save_model = False
    cfg.gradient_clip_norm = 0.5
    cfg.hyperparameters.update(
        {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "image_size": 64,
            "max_train_batches": 1,
            "max_val_batches": 1,
            "track_emissions": False,
            "score_threshold": 0.01,
            "log_every": 1,
        }
    )
    cfg.evaluation = {
        **cfg.evaluation,
        "enabled": True,
        "head_type": "detection",
        "nms_threshold": 0.5,
        "hidden_dim": 32,
        "num_anchors": 8,
        "rcnn_classic_max_grid_proposals": 8,
        "rcnn_classic_jitter_per_gt": 1,
        "rcnn_classic_roi_size": 32,
    }
    return cfg


def _tiny_backbone(out_dim: int = 16) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 64 * 64, out_dim),
        nn.ReLU(),
    )


def test_slot_detection_loop_smoke_and_gradient_clipping(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_detection_config()
    train_loader, val_loader, _ = build_detection_dataloaders(
        "synthetic_detection",
        batch_size=2,
        num_workers=0,
        image_size=64,
        box_coordinate_format="normalized",
        pin_memory=False,
    )
    model = MultiHeadModel(
        _tiny_backbone(16),
        feature_dim=16,
        head_type="detection",
        num_classes=3,
        freeze_backbone=True,
        num_anchors=8,
        hidden_dim=16,
        use_bn=False,
        dropout=0.0,
    )
    clip_calls = {"count": 0}
    original_clip = torch.nn.utils.clip_grad_norm_

    def _wrapped_clip(*args, **kwargs):
        clip_calls["count"] += 1
        return original_clip(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _wrapped_clip)
    results = run_slot_detection_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        config=cfg,
    )
    assert math.isfinite(float(results["final_train_loss"]))
    assert math.isfinite(float(results["final_val_loss"]))
    assert clip_calls["count"] > 0


def test_torchvision_frcnn_loop_smoke() -> None:
    cfg = _base_detection_config()
    cfg.hyperparameters["batch_size"] = 1
    try:
        results = run_torchvision_detection_loop(
            config=cfg,
            checkpoint_path=None,
            detector_variant="torchvision_frcnn",
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert results["head_type"] == "torchvision_frcnn"
    assert math.isfinite(float(results["best_val_loss"]))


def test_rcnn_classic_loop_smoke() -> None:
    cfg = _base_detection_config()
    cfg.hyperparameters["batch_size"] = 1
    try:
        results = evaluate_rcnn_classic(cfg, checkpoint_path=None)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert results["head_type"] == "rcnn_classic"
    assert math.isfinite(float(results["best_val_loss"]))
