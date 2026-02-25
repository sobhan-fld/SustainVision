from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

import torch.nn as nn

from sustainvision.detection_models import (
    MultiHeadModel,
    build_fasterrcnn_fpn,
    build_rcnn_c4,
)
from sustainvision.rcnn_classic import ClassicRCNNModel


def _tiny_backbone(out_dim: int = 32) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, out_dim),
        nn.ReLU(),
    )


def test_slot_detection_model_forward_shapes() -> None:
    model = MultiHeadModel(
        _tiny_backbone(16),
        feature_dim=16,
        head_type="detection",
        num_classes=5,
        freeze_backbone=True,
        num_anchors=7,
        hidden_dim=16,
        use_bn=False,
        dropout=0.0,
    )
    x = torch.rand(2, 3, 32, 32)
    cls_logits, boxes = model(x)
    assert cls_logits.shape == (2, 7, 5)
    assert boxes.shape == (2, 7, 4)
    assert float(boxes.min().item()) >= 0.0
    assert float(boxes.max().item()) <= 1.0


def test_torchvision_detector_factories_smoke_forward() -> None:
    frcnn, _ = build_fasterrcnn_fpn(num_classes_with_bg=4)
    c4, _ = build_rcnn_c4(num_classes_with_bg=4)
    frcnn.eval()
    c4.eval()
    images = [torch.rand(3, 64, 64)]
    with torch.no_grad():
        out_frcnn = frcnn(images)
        out_c4 = c4(images)
    assert isinstance(out_frcnn, list)
    assert isinstance(out_c4, list)
    assert len(out_frcnn) == 1
    assert len(out_c4) == 1


def test_classic_rcnn_model_forward_smoke() -> None:
    backbone = _tiny_backbone(8)
    model = ClassicRCNNModel(
        backbone,
        image_size=32,
        num_classes=3,
        freeze_backbone=True,
        roi_size=32,
        hidden_dim=16,
    )
    images = torch.rand(1, 3, 32, 32)
    proposals = [torch.tensor([[2.0, 2.0, 20.0, 20.0], [5.0, 5.0, 28.0, 28.0]], dtype=torch.float32)]
    outputs = model(images, proposals)
    assert outputs["logits"].shape[0] == 2
    assert outputs["logits"].shape[1] == 4  # 3 classes + background
    assert outputs["pred_boxes_norm"].shape == (2, 4)

