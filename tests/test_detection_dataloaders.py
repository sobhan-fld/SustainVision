from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sustainvision.data import build_detection_dataloaders


def test_synthetic_detection_dataloader_normalized_boxes() -> None:
    train_loader, val_loader, num_classes = build_detection_dataloaders(
        "synthetic_detection",
        batch_size=2,
        num_workers=0,
        image_size=64,
        box_coordinate_format="normalized",
        pin_memory=False,
    )
    images, targets = next(iter(train_loader))
    assert num_classes == 3
    assert isinstance(images, torch.Tensor)
    assert images.ndim == 4
    assert len(targets) == images.size(0)
    assert targets[0]["boxes"].dtype == torch.float32
    assert targets[0]["labels"].dtype == torch.long
    if targets[0]["boxes"].numel() > 0:
        assert float(targets[0]["boxes"].min().item()) >= 0.0
        assert float(targets[0]["boxes"].max().item()) <= 1.0
    # smoke the val loader too
    _ = next(iter(val_loader))


def test_synthetic_detection_dataloader_pixel_boxes() -> None:
    train_loader, _, _ = build_detection_dataloaders(
        "synthetic_detection",
        batch_size=2,
        num_workers=0,
        image_size=64,
        box_coordinate_format="pixel",
        pin_memory=False,
    )
    images, targets = next(iter(train_loader))
    assert isinstance(images, torch.Tensor)
    if targets[0]["boxes"].numel() > 0:
        assert float(targets[0]["boxes"].max().item()) > 1.0
        assert float(targets[0]["boxes"].max().item()) <= float(images.shape[-1])
