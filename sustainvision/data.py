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

    if name == "cifar100":
        target = root / "cifar100"
        _prepare_root(target)
        torchvision_datasets.CIFAR100(root=str(target), download=True)
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
        choices=["CIFAR10", "CIFAR100", "MNIST", "Synthetic placeholder", "Cancel"],
        default="CIFAR10",
    ).ask()

    if choice in ("Cancel", None):
        return None

    mapping = {
        "CIFAR10": "cifar10",
        "CIFAR100": "cifar100",
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
        "cifar100": root / "databases" / "cifar100",
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
        if (candidate / "cifar-100-python").exists():
            return "cifar100", candidate
        if (candidate / "MNIST").exists() or any(p.name.startswith("t10k") for p in candidate.glob("*")):
            return "mnist", candidate
        return "imagefolder", candidate

    if lower in {"cifar10", "cifar100", "mnist"}:
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
    use_gaussian_blur: bool = False,
    simclr_recipe: bool = False,
    subset_per_class: Optional[int] = None,
    subset_seed: Optional[int] = None,
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
    subset_value: Optional[int] = None
    if subset_per_class is not None:
        try:
            subset_value = int(subset_per_class)
        except Exception:
            subset_value = None
        if subset_value is not None and subset_value <= 0:
            subset_value = None
    subset_seed = subset_seed or seed

    def _norm(dataset: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        if dataset in {"cifar10", "cifar100"}:
            return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        if dataset == "mnist":
            return (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    mean, std = _norm(dataset_kind)

    def _simclr_contrastive_transform():
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )

    def _simclr_linear_transform(train: bool):
        if train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
        return transforms.Compose([transforms.ToTensor()])

    def _maybe_subset(dataset, per_class: Optional[int], seed_value: int, classes: int):
        if per_class is None or per_class <= 0:
            return dataset
        total_needed = per_class * classes
        if total_needed <= 0:
            return dataset
        from torch.utils.data import Subset  # type: ignore

        generator = torch.Generator().manual_seed(seed_value)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        counts = [0 for _ in range(classes)]
        selected: list[int] = []
        for idx in order:
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = int(label.item())
            try:
                label_int = int(label)
            except Exception:
                continue
            if label_int < 0 or label_int >= classes:
                continue
            if counts[label_int] >= per_class:
                continue
            counts[label_int] += 1
            selected.append(idx)
            if len(selected) >= total_needed:
                break
        if len(selected) < total_needed:
            print(
                f"[warn] Requested {per_class} samples per class but only gathered {len(selected)} total samples."
            )
        return Subset(dataset, selected)

    def _base_transforms(train: bool) -> transforms.Compose:
        if dataset_kind in {"cifar10", "cifar100"} and simclr_recipe:
            if contrastive:
                return _simclr_contrastive_transform()
            return _simclr_linear_transform(train)
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
                if use_gaussian_blur:
                    kernel_size = int(0.1 * image_size)
                    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
                    kernel_size = max(kernel_size, 3)
                    ops.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)))
        else:
            ops.extend(
                [
                    transforms.Resize(image_size + 32 if image_size >= 64 else image_size),
                    transforms.CenterCrop(image_size),
                ]
            )
        ops.append(transforms.ToTensor())
        if not (dataset_kind in {"cifar10", "cifar100"} and simclr_recipe):
            ops.append(transforms.Normalize(mean, std))
        return transforms.Compose(ops)

    generator = torch.Generator().manual_seed(seed)

    def _wrap_contrastive(dataset, transform):
        class _ContrastiveDataset(torch.utils.data.Dataset):  # type: ignore[attr-defined]
            def __init__(self, base_ds, aug):
                self.base_ds = base_ds
                self.aug = aug

            def __len__(self):
                return len(self.base_ds)

            def __getitem__(self, idx):
                image, target = self.base_ds[idx]
                view1 = self.aug(image)
                view2 = self.aug(image)
                return (view1, view2), target

        return _ContrastiveDataset(dataset, transform)

    if dataset_kind == "cifar10":
        train_transform = _base_transforms(train=True)
        cifar_train_transform = None if contrastive else train_transform
        train_base = tv_datasets.CIFAR10(
            root=str(dataset_path),
            train=True,
            transform=cifar_train_transform,
            download=False,
        )
        num_classes = 10

        if val_split > 0:
            val_transform = _base_transforms(train=True if contrastive else False)
            val_base = tv_datasets.CIFAR10(
                root=str(dataset_path),
                train=True,
                transform=None if contrastive else _base_transforms(train=False),
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
                transform=None if contrastive else _base_transforms(train=False),
                download=False,
            )

        if contrastive:
            train_dataset = _wrap_contrastive(train_dataset, train_transform)
            val_dataset = _wrap_contrastive(val_dataset, train_transform)

    elif dataset_kind == "cifar100":
        train_transform = _base_transforms(train=True)
        cifar_train_transform = None if contrastive else train_transform
        train_base = tv_datasets.CIFAR100(
            root=str(dataset_path),
            train=True,
            transform=cifar_train_transform,
            download=False,
        )
        num_classes = 100

        if val_split > 0:
            val_transform = _base_transforms(train=True if contrastive else False)
            val_base = tv_datasets.CIFAR100(
                root=str(dataset_path),
                train=True,
                transform=None if contrastive else _base_transforms(train=False),
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
            val_dataset = tv_datasets.CIFAR100(
                root=str(dataset_path),
                train=False,
                transform=None if contrastive else _base_transforms(train=False),
                download=False,
            )

        if contrastive:
            train_dataset = _wrap_contrastive(train_dataset, train_transform)
            val_dataset = _wrap_contrastive(val_dataset, train_transform)

    elif dataset_kind == "mnist":
        train_transform = _base_transforms(train=True)
        mnist_train_transform = None if contrastive else train_transform
        train_base = tv_datasets.MNIST(
            root=str(dataset_path),
            train=True,
            transform=mnist_train_transform,
            download=False,
        )
        num_classes = 10

        if val_split > 0:
            val_base = tv_datasets.MNIST(
                root=str(dataset_path),
                train=True,
                transform=None if contrastive else _base_transforms(train=False),
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
                transform=None if contrastive else _base_transforms(train=False),
                download=False,
            )

        if contrastive:
            train_dataset = _wrap_contrastive(train_dataset, train_transform)
            val_dataset = _wrap_contrastive(val_dataset, train_transform)

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

        train_base = img_datasets.ImageFolder(train_dir, transform=None)
        train_transform = _base_transforms(train=True)
        if not contrastive:
            train_base.transform = train_transform
        num_classes = len(train_base.classes)

        eval_transform = None if contrastive else _base_transforms(train=False)
        if val_dir.exists():
            val_dataset = img_datasets.ImageFolder(val_dir, transform=eval_transform)
            train_dataset = train_base
        elif test_dir.exists():
            val_dataset = img_datasets.ImageFolder(test_dir, transform=eval_transform)
            train_dataset = train_base
        elif val_split > 0:
            val_base = img_datasets.ImageFolder(train_dir)
            val_base.transform = eval_transform
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

        if contrastive:
            train_dataset = _wrap_contrastive(train_dataset, train_transform)
            val_dataset = _wrap_contrastive(val_dataset, train_transform)

    if not contrastive and subset_value:
        train_dataset = _maybe_subset(train_dataset, subset_value, subset_seed, num_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=contrastive,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes


