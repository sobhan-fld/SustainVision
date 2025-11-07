"""Dataset management utilities for SustainVision.

Provides a simple menu-driven interface to download or prepare datasets inside
the project `databases/` directory. For now we focus on a handful of common
computer-vision benchmarks and create placeholders if optional dependencies are
missing.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Tuple

import questionary


class DatasetDownloadError(RuntimeError):
    """Raised when a dataset cannot be downloaded or prepared."""


class DatasetPreparationError(RuntimeError):
    """Raised when a dataset cannot be prepared for training."""


def _require_torchvision() -> "module":  # type: ignore[override]
    try:
        import torchvision.datasets as datasets
    except Exception as exc:  # pragma: no cover - runtime safeguard
        raise DatasetDownloadError(
            "torchvision is required to download this dataset. Install it via "
            "`pip install torchvision`."
        ) from exc
    return datasets


def _prepare_root(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base


def download_dataset(dataset: str, *, project_root: Optional[Path] = None) -> Path:
    """Download or prepare a dataset under `databases/`.

    Parameters
    ----------
    dataset:
        Name of the dataset to download (e.g., "cifar10", "mnist", "synthetic").
    project_root:
        Optional project root. Defaults to current working directory.

    Returns
    -------
    Path
        The directory containing the dataset files.
    """

    root = Path(project_root or Path.cwd()) / "databases"
    root.mkdir(parents=True, exist_ok=True)

    name = dataset.lower()

    if name == "synthetic":
        target = root / "synthetic"
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.txt").write_text(
            "Synthetic dataset placeholder. Replace with your custom data.",
            encoding="utf-8",
        )
        return target

    torchvision_datasets = _require_torchvision()

    if name == "cifar10":
        target = root / "cifar10"
        _prepare_root(target)
        torchvision_datasets.CIFAR10(root=str(target), download=True)
        return target

    if name == "mnist":
        target = root / "mnist"
        _prepare_root(target)
        torchvision_datasets.MNIST(root=str(target), download=True)
        return target

    raise DatasetDownloadError(f"Unsupported dataset option: {dataset}")


def prompt_and_download(project_root: Optional[Path] = None) -> Optional[Path]:
    """Prompt the user to choose a dataset and kick off the download."""

    choice = questionary.select(
        "Select a dataset to download:",
        choices=[
            "CIFAR10",
            "MNIST",
            "Synthetic placeholder",
            "Cancel",
        ],
        default="CIFAR10",
    ).ask()

    if choice in ("Cancel", None):
        return None

    mapping = {
        "CIFAR10": "cifar10",
        "MNIST": "mnist",
        "Synthetic placeholder": "synthetic",
    }

    dataset_name = mapping[choice]
    try:
        dataset_path = download_dataset(dataset_name, project_root=project_root)
    except DatasetDownloadError as exc:
        print(f"\n[error] {exc}\n")
        return None
    except Exception as exc:  # pragma: no cover - runtime protection
        print(f"\n[error] Failed to download dataset: {exc}\n")
        return None

    print(f"\nDataset ready at {dataset_path}")
    return dataset_path


def clear_dataset(dataset_path: Path) -> None:
    """Utility to remove a dataset directory (optional helper)."""

    if dataset_path.exists():
        shutil.rmtree(dataset_path)


def _resolve_dataset_source(database: str, root: Path) -> Tuple[str, Path]:
    """Infer dataset type (cifar10, mnist, imagefolder) and root path."""

    database = database.strip()
    lower = database.lower()

    known_roots = {
        "cifar10": root / "databases" / "cifar10",
        "mnist": root / "databases" / "mnist",
    }

    if lower in known_roots:
        return lower, known_roots[lower]

    candidate = Path(database)
    if not candidate.is_absolute():
        candidate = (root / database).resolve()

    if candidate.exists():
        if (candidate / "cifar-10-batches-py").exists():
            return "cifar10", candidate
        if (candidate / "MNIST").exists() or any(p.name.startswith("t10k") for p in candidate.glob("*")):
            return "mnist", candidate
        return "imagefolder", candidate

    if lower in {"cifar10", "mnist"}:
        return lower, known_roots[lower]

    raise DatasetPreparationError(
        f"Unable to resolve dataset '{database}'. Provide a valid path or known dataset name."
    )


def build_classification_dataloaders(
    database: str,
    *,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    project_root: Optional[Path] = None,
    image_size: int = 224,
    contrastive: bool = False,
) -> Tuple["DataLoader", "DataLoader", int]:  # type: ignore[name-defined]
    """Create train/val dataloaders for common classification datasets."""

    try:
        import torch
        from torch.utils.data import DataLoader, Subset
    except Exception as exc:  # pragma: no cover - runtime safeguard
        raise DatasetPreparationError(
            "PyTorch is required for building dataloaders. Install it to proceed."
        ) from exc

    _ = _require_torchvision()
    from torchvision import datasets as tv_datasets, transforms

    root = Path(project_root or Path.cwd())
    dataset_kind, dataset_path = _resolve_dataset_source(database, root)

    val_split = max(0.0, min(float(val_split), 0.5))
    num_workers = max(0, int(num_workers))
    image_size = max(32, int(image_size))

    def _norm(dataset: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        if dataset == "cifar10":
            return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        if dataset == "mnist":
            return (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    mean, std = _norm(dataset_kind)

    def _base_transforms(train: bool) -> transforms.Compose:
        ops = []
        if dataset_kind == "mnist":
            ops.append(transforms.Grayscale(num_output_channels=3))
        resize_target = image_size + 32 if train and image_size >= 64 else image_size
        if train:
            ops.extend(
                [
                    transforms.Resize(resize_target),
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            if contrastive:
                ops.extend(
                    [
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                        transforms.RandomGrayscale(p=0.2),
                    ]
                )
        else:
            ops.extend(
                [
                    transforms.Resize(image_size + 32 if image_size >= 64 else image_size),
                    transforms.CenterCrop(image_size),
                ]
            )
        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(mean, std))
        return transforms.Compose(ops)

    generator = torch.Generator().manual_seed(seed)

    if dataset_kind == "cifar10":
        train_base = tv_datasets.CIFAR10(
            root=str(dataset_path),
            train=True,
            transform=_base_transforms(train=True),
            download=False,
        )
        num_classes = 10

        if val_split > 0:
            val_base = tv_datasets.CIFAR10(
                root=str(dataset_path),
                train=True,
                transform=_base_transforms(train=False),
                download=False,
            )
            total = len(train_base)
            val_count = max(1, int(total * val_split))
            train_count = total - val_count
            indices = torch.randperm(total, generator=generator)
            train_indices = indices[:train_count]
            val_indices = indices[train_count:]
            train_dataset = Subset(train_base, train_indices.tolist())
            val_dataset = Subset(val_base, val_indices.tolist())
        else:
            train_dataset = train_base
            val_dataset = tv_datasets.CIFAR10(
                root=str(dataset_path),
                train=False,
                transform=_base_transforms(train=False),
                download=False,
            )

    elif dataset_kind == "mnist":
        train_base = tv_datasets.MNIST(
            root=str(dataset_path),
            train=True,
            transform=_base_transforms(train=True),
            download=False,
        )
        num_classes = 10

        if val_split > 0:
            val_base = tv_datasets.MNIST(
                root=str(dataset_path),
                train=True,
                transform=_base_transforms(train=False),
                download=False,
            )
            total = len(train_base)
            val_count = max(1, int(total * val_split))
            train_count = total - val_count
            indices = torch.randperm(total, generator=generator)
            train_indices = indices[:train_count]
            val_indices = indices[train_count:]
            train_dataset = Subset(train_base, train_indices.tolist())
            val_dataset = Subset(val_base, val_indices.tolist())
        else:
            train_dataset = train_base
            val_dataset = tv_datasets.MNIST(
                root=str(dataset_path),
                train=False,
                transform=_base_transforms(train=False),
                download=False,
            )

    else:  # imagefolder or custom dataset
        try:
            from torchvision import datasets as img_datasets
        except Exception as exc:  # pragma: no cover
            raise DatasetPreparationError("torchvision is required for ImageFolder datasets.") from exc

        train_dir = dataset_path / "train"
        val_dir = dataset_path / "val"
        test_dir = dataset_path / "test"

        if not train_dir.exists():
            train_dir = dataset_path

        train_base = img_datasets.ImageFolder(train_dir)
        train_base.transform = _base_transforms(train=True)
        num_classes = len(train_base.classes)

        if val_dir.exists():
            val_dataset = img_datasets.ImageFolder(val_dir, transform=_base_transforms(train=False))
            train_dataset = train_base
        elif test_dir.exists():
            val_dataset = img_datasets.ImageFolder(test_dir, transform=_base_transforms(train=False))
            train_dataset = train_base
        elif val_split > 0:
            val_base = img_datasets.ImageFolder(train_dir)
            val_base.transform = _base_transforms(train=False)
            total = len(train_base)
            val_count = max(1, int(total * val_split))
            train_count = total - val_count
            indices = torch.randperm(total, generator=generator)
            train_indices = indices[:train_count]
            val_indices = indices[train_count:]
            train_dataset = Subset(train_base, train_indices.tolist())
            val_dataset = Subset(val_base, val_indices.tolist())
        else:
            raise DatasetPreparationError(
                "Validation data not found. Provide a 'val' or 'test' directory, or set val_split > 0."
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes


