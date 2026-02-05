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


# ---------------------------------------------------------------------------
# COCO category mapping (COCO category IDs are non-contiguous and have gaps).
# We map them to contiguous indices [0..79] expected by most heads/losses.
# ---------------------------------------------------------------------------
COCO_CATEGORY_IDS: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 67, 70, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 84, 85,
    86, 87, 88, 89, 90,
]
COCO_ID_TO_CONTIGUOUS: dict[int, int] = {cid: idx for idx, cid in enumerate(COCO_CATEGORY_IDS)}
COCO_CONTIGUOUS_TO_ID: dict[int, int] = {idx: cid for idx, cid in enumerate(COCO_CATEGORY_IDS)}

# COCO class names (in order matching COCO_CATEGORY_IDS indices [0..79])
COCO_CLASS_NAMES: list[str] = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# CIFAR-100 classes that overlap with COCO (semantic matches)
# Maps CIFAR-100 class names to COCO class names
CIFAR100_TO_COCO_MAPPING: dict[str, list[str]] = {
    # Animals
    'bird': ['bird'],
    'cat': ['cat'],
    'dog': ['dog'],
    'horse': ['horse'],
    'sheep': ['sheep'],
    'cow': ['cow'],
    'bear': ['bear'],
    # Vehicles
    'bicycle': ['bicycle'],
    'motorcycle': ['motorcycle'],
    'automobile': ['car'],
    'bus': ['bus'],
    'truck': ['truck'],
    'train': ['train'],
    # Furniture
    'chair': ['chair'],
    'couch': ['couch'],
    'bed': ['bed'],
    'table': ['dining table'],
    # Food
    'apple': ['apple'],
    'orange': ['orange'],
    'banana': ['banana'],
}

def get_coco_classes_for_cifar100() -> set[int]:
    """Return set of COCO contiguous class indices [0..79] that match CIFAR-100 classes.
    
    This filters COCO to only classes that semantically overlap with CIFAR-100,
    reducing the detection task from 80 classes to ~20-30 overlapping classes.
    """
    coco_classes_to_keep = set()
    for coco_idx, coco_name in enumerate(COCO_CLASS_NAMES):
        # Check if this COCO class matches any CIFAR-100 class
        for cifar_class, coco_matches in CIFAR100_TO_COCO_MAPPING.items():
            if coco_name in coco_matches:
                coco_classes_to_keep.add(coco_idx)
                break
    return coco_classes_to_keep


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

    if name == "coco":
        target = root / "coco"
        _prepare_root(target)
        # COCO dataset requires manual download - torchvision doesn't support auto-download
        print("[info] COCO dataset requires manual download.")
        print("[info] Download from: https://cocodataset.org/#download")
        print("[info] Required files:")
        print("  - train2017.zip (~18GB) -> extract to databases/coco/images/train2017/")
        print("  - val2017.zip (~1GB) -> extract to databases/coco/images/val2017/")
        print("  - annotations_trainval2017.zip -> extract to databases/coco/annotations/")
        print("[info] Expected structure:")
        print("  databases/coco/")
        print("    images/")
        print("      train2017/")
        print("      val2017/")
        print("    annotations/")
        print("      instances_train2017.json")
        print("      instances_val2017.json")
        print(f"[info] Directory created at: {target}")
        print("[info] After downloading, you can use the dataset for evaluation.")
        return target

    if name == "voc" or name == "pascal_voc":
        target = root / "voc"
        _prepare_root(target)
        try:
            # Download Pascal VOC 2012
            # Note: torchvision's automatic download may fail due to firewall/network issues
            print("[info] Attempting to download Pascal VOC 2012...")
            print("[info] This may take 10-30 minutes (~2GB download)...")
            print("[warn] If download fails due to firewall, try:")
            print("  - Using a VPN or different network")
            print("  - Downloading from a personal device and transferring")
            print("  - Contacting network admin about firewall policy 10263")
            dataset = torchvision_datasets.VOCDetection(
                root=str(target),
                year="2012",
                image_set="train",
                download=True,
            )
            # Verify download by checking if dataset can be loaded
            if len(dataset) == 0:
                raise ValueError("Downloaded dataset appears empty")
            print(f"[info] Pascal VOC 2012 downloaded successfully! ({len(dataset)} images)")
        except Exception as exc:
            print(f"[warn] Automatic Pascal VOC download failed: {exc}")
            print("[info] The download may be blocked by a firewall (policy 10263).")
            print("[info] Alternative options:")
            print("  1. Use a VPN or different network connection")
            print("  2. Download from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/")
            print("     (on a device not behind the firewall)")
            print(f"  3. Extract to: {target}")
            print("  4. Expected structure: databases/voc/VOCdevkit/VOC2012/")
            print("[info] For now, you can test evaluation with classification on CIFAR-10/100")
        return target

    raise DatasetDownloadError(f"Unsupported dataset option: {dataset}")


def prompt_and_download(project_root: Optional[Path] = None) -> Optional[Path]:
    """Prompt the user to choose a dataset and kick off the download."""

    choice = questionary.select(
        "Select a dataset to download:",
        choices=[
            "CIFAR10",
            "CIFAR100",
            "MNIST",
            "COCO (Object Detection)",
            "Pascal VOC (Object Detection)",
            "Synthetic placeholder",
            "Cancel",
        ],
        default="CIFAR10",
    ).ask()

    if choice in ("Cancel", None):
        return None

    mapping = {
        "CIFAR10": "cifar10",
        "CIFAR100": "cifar100",
        "MNIST": "mnist",
        "COCO (Object Detection)": "coco",
        "Pascal VOC (Object Detection)": "voc",
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
        "coco": root / "databases" / "coco",
        "voc": root / "databases" / "voc",
        "pascal_voc": root / "databases" / "voc",
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
        if (candidate / "images").exists() and (candidate / "annotations").exists():
            # COCO format
            return "coco", candidate
        if (candidate / "VOCdevkit").exists() or (candidate / "VOC2012").exists():
            # Pascal VOC format
            return "voc", candidate
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
    use_m_per_class_sampler: bool = False,
    m_per_class: Optional[int] = None,
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
        # COCO and ImageNet use ImageNet normalization
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

    elif dataset_kind == "coco":
        # COCO as classification dataset: convert detection annotations to image-level labels
        # Use the most common class in each image as the label
        try:
            from pycocotools.coco import COCO  # type: ignore
        except ImportError:
            raise DatasetPreparationError(
                "pycocotools is required for COCO classification. Install with: pip install pycocotools"
            )
        
        # Find COCO dataset - check both possible paths
        coco_root = dataset_path
        train_images = coco_root / "images" / "train2017"
        if not train_images.exists():
            train_images = coco_root / "train2017"
        train_ann = coco_root / "annotations" / "instances_train2017.json"
        
        val_images = coco_root / "images" / "val2017"
        if not val_images.exists():
            val_images = coco_root / "val2017"
        val_ann = coco_root / "annotations" / "instances_val2017.json"
        
        if not train_images.exists() or not train_ann.exists():
            raise DatasetPreparationError(
                f"COCO dataset not found. Expected train images at {train_images} and annotations at {train_ann}"
            )
        
        # Create a classification dataset wrapper for COCO
        class COCOClassificationDataset(torch.utils.data.Dataset):  # type: ignore[attr-defined]
            def __init__(self, coco: COCO, image_dir: Path, transform=None):
                self.coco = coco
                self.image_dir = image_dir
                self.transform = transform
                self.ids = list(coco.imgs.keys())
                # Pre-compute image-level labels (most common class per image)
                self.labels = {}
                print(f"[info] Computing image-level labels for {len(self.ids)} COCO images...")
                for idx, img_id in enumerate(self.ids):
                    if (idx + 1) % 10000 == 0:
                        print(f"[info] Processed {idx + 1}/{len(self.ids)} images...")
                    ann_ids = coco.getAnnIds(imgIds=img_id)
                    anns = coco.loadAnns(ann_ids)
                    if len(anns) > 0:
                        # Count class occurrences
                        class_counts = {}
                        for ann in anns:
                            cat_id = ann.get("category_id")
                            if cat_id is not None:
                                # Map COCO category_id to contiguous [0..79]
                                mapped = COCO_ID_TO_CONTIGUOUS.get(int(cat_id))
                                if mapped is not None:
                                    class_counts[mapped] = class_counts.get(mapped, 0) + 1
                        # Use most common class as label
                        if class_counts:
                            self.labels[img_id] = max(class_counts.items(), key=lambda x: x[1])[0]
                        else:
                            self.labels[img_id] = 0  # Default to class 0 if no valid annotations
                    else:
                        self.labels[img_id] = 0  # Default to class 0 if no annotations
                
                # Expose labels as targets attribute for M-per-class sampler
                self.targets = [self.labels[img_id] for img_id in self.ids]
                print(f"[info] Finished computing labels for {len(self.targets)} images")
            
            def __len__(self):
                return len(self.ids)
            
            def __getitem__(self, idx):
                img_id = self.ids[idx]
                img_info = self.coco.imgs[img_id]
                img_path = self.image_dir / img_info["file_name"]
                
                from PIL import Image
                img = Image.open(img_path).convert("RGB")
                
                if self.transform:
                    img = self.transform(img)
                
                label = self.labels[img_id]
                return img, label
        
        train_transform = _base_transforms(train=True)
        coco_train = COCO(str(train_ann))
        train_dataset = COCOClassificationDataset(coco_train, train_images, transform=None if contrastive else train_transform)
        num_classes = 80  # COCO has 80 classes
        
        if val_split > 0 or (val_images.exists() and val_ann.exists()):
            if val_images.exists() and val_ann.exists():
                coco_val = COCO(str(val_ann))
                val_dataset = COCOClassificationDataset(
                    coco_val, 
                    val_images, 
                    transform=None if contrastive else _base_transforms(train=False)
                )
            else:
                # Split train set
                total = len(train_dataset)
                val_count = max(1, int(total * val_split))
                train_count = total - val_count
                indices = torch.randperm(total, generator=generator)
                train_indices = indices[:train_count]
                val_indices = indices[train_count:]
                from torch.utils.data import Subset
                val_dataset = Subset(train_dataset, val_indices.tolist())
                train_dataset = Subset(train_dataset, train_indices.tolist())
        else:
            # No validation set, use train set
            val_dataset = train_dataset

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

    # Setup sampler for M-per-class sampling (useful for SupCon)
    train_sampler = None
    shuffle = True
    
    if use_m_per_class_sampler and m_per_class is not None and contrastive:
        try:
            from sustainvision.samplers import MPerClassSampler
            
            # Get labels from dataset
            labels = None
            if hasattr(train_dataset, 'targets'):
                labels = train_dataset.targets
            elif hasattr(train_dataset, 'base_ds') and hasattr(train_dataset.base_ds, 'targets'):
                labels = train_dataset.base_ds.targets
            else:
                # Try to extract from dataset (for wrapped contrastive datasets)
                if hasattr(train_dataset, 'base_ds'):
                    base_ds = train_dataset.base_ds
                    if hasattr(base_ds, 'targets'):
                        labels = base_ds.targets
                    elif hasattr(base_ds, 'dataset') and hasattr(base_ds.dataset, 'targets'):
                        labels = base_ds.dataset.targets
            
            if labels is None:
                # Fallback: extract labels by iterating (slow but works)
                print("[warn] Could not find labels attribute, extracting from dataset (this may be slow)...")
                labels = [train_dataset[i][1] if isinstance(train_dataset[i], tuple) else train_dataset[i]['target'] 
                          for i in range(min(len(train_dataset), 1000))]  # Sample first 1000 to get labels
                # If we have a wrapped dataset, try to get full labels
                if hasattr(train_dataset, 'base_ds'):
                    try:
                        labels = [train_dataset.base_ds[i][1] for i in range(len(train_dataset.base_ds))]
                    except:
                        pass
            
            if labels is not None:
                train_sampler = MPerClassSampler(
                    labels=labels,
                    m_per_class=m_per_class,
                    batch_size=batch_size,
                    seed=seed
                )
                shuffle = False  # Must be False when using custom sampler
                print(f"[info] Using MPerClassSampler: {m_per_class} samples per class, {batch_size // m_per_class} classes per batch")
            else:
                print("[warn] Could not extract labels for MPerClassSampler, falling back to random sampling")
        except Exception as e:
            print(f"[warn] Failed to create MPerClassSampler: {e}. Falling back to random sampling.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=contrastive,
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches for faster GPU utilization
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, num_classes


def build_detection_dataloaders(
    database: str,
    *,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.0,
    seed: int = 42,
    project_root: Optional[Path] = None,
    image_size: int = 224,
    max_train_images: Optional[int] = None,
    max_val_images: Optional[int] = None,
    filter_classes: Optional[set[int]] = None,  # Set of COCO class indices [0..79] to keep
    normalize_images: bool = True,
) -> Tuple["DataLoader", "DataLoader", int]:  # type: ignore[name-defined]
    """Build dataloaders for object detection datasets (COCO, Pascal VOC, tiny synthetic).
    
    Returns dataloaders that yield (image, target) where target contains:
    - 'boxes': [N, 4] tensor of bounding boxes in (x1, y1, x2, y2) format
    - 'labels': [N] tensor of class labels
    - 'image_id': image identifier
    
    Parameters
    ----------
    database:
        Dataset name ("coco", "voc") or path to detection dataset
    batch_size:
        Batch size for dataloaders
    num_workers:
        Number of worker processes for data loading
    val_split:
        Validation split ratio (0.0 means use separate val set if available)
    seed:
        Random seed
    project_root:
        Project root directory
    image_size:
        Target image size for resizing
        
    Returns
    -------
    Tuple of (train_loader, val_loader, num_classes)
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise DatasetPreparationError(
            "PyTorch is required for building dataloaders. Install it to proceed."
        ) from exc

    _ = _require_torchvision()
    from torchvision import datasets as tv_datasets, transforms

    root = Path(project_root or Path.cwd())
    database_lower = database.lower().strip()

    # Normalize image size
    image_size = max(32, int(image_size))
    num_workers = max(0, int(num_workers))

    # Define transforms for detection datasets.
    # IMPORTANT: detection targets (boxes) must be consistent with the transformed image.
    # We therefore build a callable that returns:
    # - image tensor resized to (image_size, image_size) and normalized
    # - target dict with normalized boxes in xyxy format (0..1), labels, image_id, orig_size
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def _to_tensor_and_normalize(img):
        from torchvision.transforms import functional as TF  # type: ignore

        img = TF.resize(img, [image_size, image_size])
        tensor = TF.to_tensor(img)
        # NOTE: torchvision detection models (Faster R-CNN / RetinaNet / etc.)
        # apply their own normalization inside GeneralizedRCNNTransform.
        # If we normalize here as well, images get double-normalized, which can
        # lead to degenerate predictions (often empty).
        if normalize_images:
            tensor = TF.normalize(tensor, mean=mean, std=std)
        return tensor

    # Create class remapping if filtering is enabled
    class_remap: Optional[dict[int, int]] = None
    if filter_classes is not None and len(filter_classes) > 0:
        # Create mapping from original COCO indices [0..79] to new contiguous indices [0..len(filter_classes)-1]
        sorted_classes = sorted(filter_classes)
        class_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_classes)}
        print(f"[info] Filtering COCO classes: keeping {len(filter_classes)}/{len(COCO_CATEGORY_IDS)} classes")
        print(f"[info]   Kept classes: {sorted_classes}")
        print(f"[info]   Class names: {[COCO_CLASS_NAMES[i] for i in sorted_classes[:10]]}{'...' if len(sorted_classes) > 10 else ''}")

    def _parse_coco_annotations(annotations: list, orig_w: int, orig_h: int) -> Tuple["torch.Tensor", "torch.Tensor", int]:
        boxes: list[list[float]] = []
        labels: list[int] = []
        image_id = -1
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            if image_id == -1:
                try:
                    image_id = int(ann.get("image_id", -1))
                except Exception:
                    image_id = -1
            bbox = ann.get("bbox")
            if not bbox:
                continue
            try:
                x, y, w, h = bbox
                x1 = float(x)
                y1 = float(y)
                x2 = float(x) + float(w)
                y2 = float(y) + float(h)
            except Exception:
                continue

            cat_id = ann.get("category_id", None)
            mapped = COCO_ID_TO_CONTIGUOUS.get(int(cat_id)) if cat_id is not None else None
            if mapped is None:
                continue
            
            # Filter classes if filter_classes is provided
            if filter_classes is not None and mapped not in filter_classes:
                continue
            
            # Remap class index if filtering is enabled
            if class_remap is not None:
                mapped = class_remap[mapped]

            # Normalize to [0,1] in xyxy format relative to resized image_size
            # Scale boxes from original image coordinates to resized image coordinates
            if orig_w > 0 and orig_h > 0:
                scale_x = image_size / orig_w
                scale_y = image_size / orig_h
                # Scale boxes to resized image coordinates, then normalize to [0,1]
                boxes.append([
                    (x1 * scale_x) / image_size,
                    (y1 * scale_y) / image_size,
                    (x2 * scale_x) / image_size,
                    (y2 * scale_y) / image_size,
                ])
                labels.append(int(mapped))

        if boxes:
            return (
                torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
                image_id,
            )
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            image_id,
        )

    def _parse_voc_target(voc_target: dict, orig_w: int, orig_h: int) -> Tuple["torch.Tensor", "torch.Tensor", int]:
        boxes: list[list[float]] = []
        labels: list[int] = []
        image_id = -1
        annotation = voc_target.get("annotation") if isinstance(voc_target, dict) else None
        if isinstance(annotation, dict):
            objects = annotation.get("object", [])
            if isinstance(objects, dict):
                objects = [objects]
            if isinstance(objects, list):
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue
                    bbox = obj.get("bndbox")
                    if not isinstance(bbox, dict):
                        continue
                    try:
                        x1 = float(bbox.get("xmin", 0))
                        y1 = float(bbox.get("ymin", 0))
                        x2 = float(bbox.get("xmax", 0))
                        y2 = float(bbox.get("ymax", 0))
                    except Exception:
                        continue
                    name = obj.get("name", "unknown")
                    # Stable-ish mapping: keep it in [0..19]
                    label = int(name) if isinstance(name, int) else (hash(str(name)) % 20)
                    if orig_w > 0 and orig_h > 0:
                        # Scale boxes from original image coordinates to resized image coordinates
                        scale_x = image_size / orig_w
                        scale_y = image_size / orig_h
                        # Scale boxes to resized image coordinates, then normalize to [0,1]
                        boxes.append([
                            (x1 * scale_x) / image_size,
                            (y1 * scale_y) / image_size,
                            (x2 * scale_x) / image_size,
                            (y2 * scale_y) / image_size,
                        ])
                        labels.append(label)
        if boxes:
            return (
                torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
                image_id,
            )
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            image_id,
    )

    # ------------------------------------------------------------------
    # Tiny synthetic detection dataset for quickly testing the pipeline
    # ------------------------------------------------------------------
    if database_lower in {"toy_detection", "tiny_detection", "synthetic_detection"}:
        # Very small dataset: a handful of randomly generated images with 1–3 boxes
        # and a small number of classes. This is only meant for sanity‑checking
        # that the detection pipeline (model forward, loss, training loop) works.
        num_classes = 3

        class _TinyDetectionDataset(torch.utils.data.Dataset):  # type: ignore[attr-defined]
            def __init__(self, length: int, image_size: int, seed_value: int) -> None:
                self.length = length
                self.image_size = image_size
                self.generator = torch.Generator().manual_seed(seed_value)

            def __len__(self) -> int:
                return self.length

            def __getitem__(self, idx):  # type: ignore[override]
                # Random image in [0, 1]
                img = torch.rand(3, self.image_size, self.image_size, generator=self.generator)

                # 1–3 random boxes per image
                num_boxes = int(torch.randint(1, 4, (1,), generator=self.generator).item())
                boxes = []
                labels = []
                for _ in range(num_boxes):
                    x1 = torch.randint(0, self.image_size - 4, (1,), generator=self.generator).item()
                    y1 = torch.randint(0, self.image_size - 4, (1,), generator=self.generator).item()
                    x2 = torch.randint(x1 + 2, self.image_size, (1,), generator=self.generator).item()
                    y2 = torch.randint(y1 + 2, self.image_size, (1,), generator=self.generator).item()
                    boxes.append([float(x1), float(y1), float(x2), float(y2)])

                    # Labels in [0, num_classes-1]
                    label = int(torch.randint(0, num_classes, (1,), generator=self.generator).item())
                    labels.append(label)

                target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
                return img, target

        # Use a tiny split for fast tests
        train_dataset = _TinyDetectionDataset(length=32, image_size=image_size, seed_value=seed)
        val_dataset = _TinyDetectionDataset(length=16, image_size=image_size, seed_value=seed + 1)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate_detection_batch,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate_detection_batch,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

        return train_loader, val_loader, num_classes

    # Load COCO dataset
    if database_lower == "coco":
        coco_root = root / "databases" / "coco"
        if not coco_root.exists():
            # Try to resolve as path
            candidate = Path(database)
            if not candidate.is_absolute():
                candidate = (root / database).resolve()
            if candidate.exists():
                coco_root = candidate
            else:
                raise DatasetPreparationError(
                    f"COCO dataset not found at {coco_root}. "
                    "Please download it first using 'Download databases' menu."
                )

        train_images = coco_root / "images" / "train2017"
        train_ann = coco_root / "annotations" / "instances_train2017.json"
        val_images = coco_root / "images" / "val2017"
        val_ann = coco_root / "annotations" / "instances_val2017.json"

        if not train_images.exists() or not train_ann.exists():
            raise DatasetPreparationError(
                f"COCO train set not found. Expected:\n"
                f"  {train_images}\n"
                f"  {train_ann}"
            )

        def _coco_transforms(img, target):
            # img is PIL; target is list[dict]
            orig_w, orig_h = img.size
            image_tensor = _to_tensor_and_normalize(img)
            boxes_t, labels_t, image_id = _parse_coco_annotations(target if isinstance(target, list) else [], orig_w, orig_h)
            return image_tensor, {
                "boxes": boxes_t,
                "labels": labels_t,
                "image_id": int(image_id),
                "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            }

        train_dataset = tv_datasets.CocoDetection(
            root=str(train_images),
            annFile=str(train_ann),
            transforms=_coco_transforms,
        )

        from torch.utils.data import Subset

        if val_images.exists() and val_ann.exists():
            val_dataset = tv_datasets.CocoDetection(
                root=str(val_images),
                annFile=str(val_ann),
                transforms=_coco_transforms,
            )
        else:
            print("[warn] COCO validation set not found, using train set split")
            total = len(train_dataset)
            val_count = max(1, int(total * 0.1))  # 10% for validation
            indices = torch.randperm(total, generator=torch.Generator().manual_seed(seed))
            val_dataset = Subset(train_dataset, indices[:val_count].tolist())
            train_dataset = Subset(train_dataset, indices[val_count:].tolist())

        # Optional: limit dataset size for quick tests
        if max_train_images is not None:
            try:
                max_train_images_int = max(1, int(max_train_images))
                if len(train_dataset) > max_train_images_int:
                    train_dataset = Subset(train_dataset, list(range(max_train_images_int)))
                    print(f"[info] Limiting COCO train to {max_train_images_int} images (quick test).")
            except Exception:
                pass

        if max_val_images is not None:
            try:
                max_val_images_int = max(1, int(max_val_images))
                if len(val_dataset) > max_val_images_int:
                    val_dataset = Subset(val_dataset, list(range(max_val_images_int)))
                    print(f"[info] Limiting COCO val to {max_val_images_int} images (quick test).")
            except Exception:
                pass

        # COCO has 80 classes (1-90, but some IDs are skipped)
        # If filtering is enabled, use filtered class count
        if filter_classes is not None and len(filter_classes) > 0:
            num_classes = len(filter_classes)
        else:
            num_classes = 80

    # Load Pascal VOC dataset
    elif database_lower in {"voc", "pascal_voc"}:
        voc_root = root / "databases" / "voc"
        if not voc_root.exists():
            candidate = Path(database)
            if not candidate.is_absolute():
                candidate = (root / database).resolve()
            if candidate.exists():
                voc_root = candidate
            else:
                raise DatasetPreparationError(
                    f"Pascal VOC dataset not found at {voc_root}. "
                    "Please download it first using 'Download databases' menu."
                )

        def _voc_transforms(img, target):
            orig_w, orig_h = img.size
            image_tensor = _to_tensor_and_normalize(img)
            boxes_t, labels_t, image_id = _parse_voc_target(target if isinstance(target, dict) else {}, orig_w, orig_h)
            return image_tensor, {
                "boxes": boxes_t,
                "labels": labels_t,
                "image_id": int(image_id),
                "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            }

        train_dataset = tv_datasets.VOCDetection(
            root=str(voc_root),
            year="2012",
            image_set="train",
            transforms=_voc_transforms,
            download=False,
        )

        try:
            val_dataset = tv_datasets.VOCDetection(
                root=str(voc_root),
                year="2012",
                image_set="val",
                transforms=_voc_transforms,
                download=False,
            )
        except Exception:
            print("[warn] Pascal VOC validation set not found, using train set split")
            from torch.utils.data import Subset
            total = len(train_dataset)
            val_count = max(1, int(total * 0.1))
            indices = torch.randperm(total, generator=torch.Generator().manual_seed(seed))
            val_dataset = Subset(train_dataset, indices[:val_count].tolist())
            train_dataset = Subset(train_dataset, indices[val_count:].tolist())

        # Pascal VOC has 20 classes
        num_classes = 20

    else:
        raise DatasetPreparationError(
            f"Unsupported detection dataset: {database}. "
            "Supported: 'coco', 'voc', 'pascal_voc'"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_detection_batch,  # Custom collate for detection
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches for faster GPU utilization
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_detection_batch,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, num_classes


def _collate_detection_batch(batch: list) -> Tuple["torch.Tensor", list]:  # type: ignore[name-defined]
    """Custom collate function for detection datasets.
    
    Handles variable number of objects per image by returning targets as a list.
    Torchvision COCO/VOC datasets return targets as dicts with 'annotations' key.
    """
    try:
        import torch
    except Exception:
        raise DatasetPreparationError("PyTorch is required")

    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        
        # If dataset already produced the normalized detection target dict,
        # just forward it. (This is the preferred path for COCO/VOC transforms above.)
        if isinstance(target, dict) and "boxes" in target and "labels" in target:
            targets.append(target)
            continue

        boxes = []
        labels = []

        # Torchvision COCO: target is a **list** of annotation dicts
        if isinstance(target, list):
            annotations = target
            for ann in annotations:
                if "bbox" in ann:
                    # COCO bbox format: [x, y, width, height] -> [x1, y1, x2, y2]
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    # Convert COCO category_id to contiguous [0..79]
                    cat_id = ann.get("category_id", None)
                    mapped = COCO_ID_TO_CONTIGUOUS.get(int(cat_id)) if cat_id is not None else None
                    if mapped is None:
                        # Skip unknown/unsupported category IDs
                        boxes.pop()
                        continue
                    labels.append(mapped)
                elif "segmentation" in ann:
                    # Some COCO annotations might have segmentation but not bbox
                    continue

        # Pascal VOC: target is a dict with 'annotation' key
        elif isinstance(target, dict):
            if "annotation" in target:
                annotation = target["annotation"]
                objects = annotation.get("object", [])
                if isinstance(objects, list):
                    for obj in objects:
                        if "bndbox" in obj:
                            bbox = obj["bndbox"]
                            x1 = float(bbox.get("xmin", 0))
                            y1 = float(bbox.get("ymin", 0))
                            x2 = float(bbox.get("xmax", 0))
                            y2 = float(bbox.get("ymax", 0))
                            boxes.append([x1, y1, x2, y2])
                            # Simple VOC class → ID mapping placeholder
                            name = obj.get("name", "unknown")
                            if isinstance(name, int):
                                labels.append(name)
                            else:
                                labels.append(hash(name) % 20)  # 20 classes for VOC

        # Create target tensor (shared for COCO/VOC)
        if boxes:
            target_tensor = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        else:
            # Empty image - create dummy boxes
            target_tensor = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
            }
        targets.append(target_tensor)

    images = torch.stack(images)
    return images, targets


