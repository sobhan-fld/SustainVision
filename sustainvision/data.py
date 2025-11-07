"""Dataset management utilities for SustainVision.

Provides a simple menu-driven interface to download or prepare datasets inside
the project `databases/` directory. For now we focus on a handful of common
computer-vision benchmarks and create placeholders if optional dependencies are
missing.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import questionary


class DatasetDownloadError(RuntimeError):
    """Raised when a dataset cannot be downloaded or prepared."""


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


