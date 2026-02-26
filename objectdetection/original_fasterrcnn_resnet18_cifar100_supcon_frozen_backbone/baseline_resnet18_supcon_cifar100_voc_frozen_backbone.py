import argparse
import csv
import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tv_datasets
from torchvision import transforms as tv_transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

try:
    from codecarbon import EmissionsTracker  # type: ignore
except Exception:
    EmissionsTracker = None

# Reuse SustainVision's checkpoint loader to build the exact CIFAR-style ResNet18
# backbone and load SupCon weights robustly (filters shape-mismatched keys).
# Allow running this script from its own directory (so `import sustainvision` works).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sustainvision.evaluation import load_pretrained_backbone  # noqa: E402


VOC10_CLASSES: List[str] = [
    "aeroplane",
    "bicycle",
    "bird",
    "car",
    "cat",
    "dog",
    "horse",
    "motorbike",
    "person",
    "train",
]
VOC_CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC10_CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Faster R-CNN ResNet-18 (CIFAR100 SupCon) + FPN VOC baseline (frozen backbone by default)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/sfooladi/github/SustainVision/databases/voc",
        help="Path to VOC root (VOCdevkit or Kaggle VOC2012 layout)",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument(
        "--min-size",
        type=int,
        default=800,
        help="Shorter side size used by torchvision detection transform (internal resize).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1333,
        help="Longer side max size used by torchvision detection transform (internal resize).",
    )
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="original_fasterrcnn_resnet18_cifar100_supcon_frozen_backbone/outputs",
    )
    parser.add_argument("--track-emissions", action="store_true", default=True)
    parser.add_argument(
        "--train-backbone",
        action="store_true",
        default=False,
        help="If set, finetune the SupCon backbone. By default the backbone is frozen and only detection heads are trained.",
    )
    parser.add_argument(
        "--supcon-checkpoint",
        type=str,
        default="/home/sfooladi/github/SustainVision/outputs_old2/resnet18_supcon_full_cifar100_v3/resnet18_supcon_full_cifar100_v3_cycle10.pt",
        help="Path to the SupCon CIFAR100 checkpoint (.pt) to initialize the ResNet18 backbone.",
    )
    parser.add_argument(
        "--backbone-image-size",
        type=int,
        default=224,
        help=(
            "Image size used when building the ResNet18 in SustainVision to load SupCon weights. "
            "IMPORTANT: if you set this <= 64, SustainVision adapts ResNet18 to a CIFAR stem "
            "(conv1=3x3,stride=1 and no maxpool), which greatly increases feature-map resolution "
            "and can OOM Faster R-CNN at VOC scales. Use 224+ to keep the standard ImageNet stem."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (VOC mAP@0.5) on a single checkpoint and exit.",
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Evaluate all epoch checkpoints in --output-dir and write baseline_eval_map50.csv.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path for --eval-only. If empty, uses the last epoch checkpoint in --output-dir.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-detections", type=int, default=300)
    return parser.parse_args()


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def _parse_voc_target(target: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    ann = target.get("annotation", {})
    objects = ann.get("object", [])
    if isinstance(objects, dict):
        objects = [objects]

    boxes = []
    labels = []
    for obj in objects:
        name = obj.get("name", "").strip().lower()
        if name not in VOC_CLASS_TO_ID:
            continue
        difficult = obj.get("difficult", "0")
        if isinstance(difficult, dict):
            difficult = difficult.get("#text", "0")
        if str(difficult) == "1":
            continue
        bbox = obj.get("bndbox", {})
        try:
            x1 = float(bbox.get("xmin", 0))
            y1 = float(bbox.get("ymin", 0))
            x2 = float(bbox.get("xmax", 0))
            y2 = float(bbox.get("ymax", 0))
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(VOC_CLASS_TO_ID[name] + 1)  # +1 for background

    if boxes:
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)
    else:
        boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.zeros((0,), dtype=torch.int64)

    return boxes_t, labels_t


class VOC10Detection(tv_datasets.VOCDetection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = tv_transforms.ToTensor()

    def __getitem__(self, idx: int):
        img, target = super().__getitem__(idx)
        img_t = self.transform(img)
        boxes_t, labels_t = _parse_voc_target(target)
        return img_t, {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
        }


class VOC10XmlDetection(torch.utils.data.Dataset):  # type: ignore[attr-defined]
    def __init__(self, base_root: Path, split: str):
        self.base_root = base_root
        self.image_dir = base_root / "JPEGImages"
        self.ann_dir = base_root / "Annotations"
        split_file = base_root / "ImageSets" / "Main" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"VOC split file not found: {split_file}")
        self.image_ids = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.transform = tv_transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        import xml.etree.ElementTree as ET
        from PIL import Image

        image_id = self.image_ids[idx]
        img_path = self.image_dir / f"{image_id}.jpg"
        ann_path = self.ann_dir / f"{image_id}.xml"

        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)

        boxes: List[List[float]] = []
        labels: List[int] = []
        try:
            tree = ET.parse(ann_path)
            root_node = tree.getroot()
        except Exception:
            root_node = None

        if root_node is not None:
            for obj in root_node.findall("object"):
                name_node = obj.find("name")
                name = name_node.text.strip().lower() if name_node is not None else ""
                if name not in VOC_CLASS_TO_ID:
                    continue
                diff_node = obj.find("difficult")
                if diff_node is not None and diff_node.text == "1":
                    continue
                bbox = obj.find("bndbox")
                if bbox is None:
                    continue
                try:
                    x1 = float(bbox.find("xmin").text)
                    y1 = float(bbox.find("ymin").text)
                    x2 = float(bbox.find("xmax").text)
                    y2 = float(bbox.find("ymax").text)
                except Exception:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(VOC_CLASS_TO_ID[name] + 1)

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        return img_t, {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
        }


def build_model(
    *,
    num_classes: int,
    supcon_checkpoint: str,
    backbone_image_size: int,
    train_backbone: bool,
    min_size: int,
    max_size: int,
) -> FasterRCNN:
    # Load CIFAR-style ResNet18 backbone (SupCon) and wrap with FPN.
    body = load_pretrained_backbone(
        supcon_checkpoint,
        model_name="resnet18",
        image_size=int(backbone_image_size),
        projection_dim=128,
        projection_hidden_dim=None,
        projection_use_bn=False,
        adapt_small_models=True,
    )

    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [64, 128, 256, 512]
    out_channels = 256
    backbone = BackboneWithFPN(
        body,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool(),
    )

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=int(min_size),
        max_size=int(max_size),
    )

    # Extra safety: ensure entire backbone (body+fpn) is frozen when requested.
    if not train_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    else:
        for p in model.backbone.parameters():
            p.requires_grad = True

    return model


def _set_bn_eval(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def _box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-9)


def _voc_ap_continuous(rec: torch.Tensor, prec: torch.Tensor) -> float:
    if rec.numel() == 0 or prec.numel() == 0:
        return 0.0

    mrec = torch.cat([torch.tensor([0.0]), rec, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), prec, torch.tensor([0.0])])

    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])

    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()
    return float(ap)


@torch.no_grad()
def evaluate_voc_map50(
    model: FasterRCNN,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.05,
    max_detections: int = 100,
    iou_threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    model.eval()

    gt_by_image: Dict[int, Dict[int, torch.Tensor]] = {}
    gt_used_by_image: Dict[int, Dict[int, torch.Tensor]] = {}
    npos_by_class: Dict[int, int] = {c: 0 for c in range(1, len(VOC10_CLASSES) + 1)}
    preds_by_class: Dict[int, List[Tuple[float, int, torch.Tensor]]] = {c: [] for c in range(1, len(VOC10_CLASSES) + 1)}

    eval_iter = tqdm(loader, desc="Eval (VOC mAP@0.5)", unit="batch", leave=False)
    for images, targets in eval_iter:
        images = [img.to(device) for img in images]
        outputs = model(images)

        outputs_cpu = []
        for out in outputs:
            outputs_cpu.append(
                {
                    "boxes": out["boxes"].detach().cpu(),
                    "labels": out["labels"].detach().cpu(),
                    "scores": out["scores"].detach().cpu(),
                }
            )

        for t, out in zip(targets, outputs_cpu):
            image_id = int(t["image_id"].item()) if isinstance(t["image_id"], torch.Tensor) else int(t["image_id"])
            gt_boxes = t["boxes"].detach().cpu()
            gt_labels = t["labels"].detach().cpu()

            if image_id not in gt_by_image:
                gt_by_image[image_id] = {}
                gt_used_by_image[image_id] = {}

            for c in range(1, len(VOC10_CLASSES) + 1):
                mask = gt_labels == c
                boxes_c = gt_boxes[mask]
                if boxes_c.numel() == 0:
                    continue
                gt_by_image[image_id][c] = boxes_c
                gt_used_by_image[image_id][c] = torch.zeros((boxes_c.shape[0],), dtype=torch.bool)
                npos_by_class[c] += int(boxes_c.shape[0])

            boxes_p = out["boxes"]
            labels_p = out["labels"]
            scores_p = out["scores"]
            if boxes_p.numel() == 0:
                continue

            keep = scores_p >= float(score_threshold)
            boxes_p = boxes_p[keep]
            labels_p = labels_p[keep]
            scores_p = scores_p[keep]
            if boxes_p.numel() == 0:
                continue

            if max_detections is not None and boxes_p.shape[0] > max_detections:
                topk = torch.topk(scores_p, k=int(max_detections))
                idxs = topk.indices
                boxes_p = boxes_p[idxs]
                labels_p = labels_p[idxs]
                scores_p = scores_p[idxs]

            for box, lab, score in zip(boxes_p, labels_p, scores_p):
                c = int(lab.item())
                if c < 1 or c > len(VOC10_CLASSES):
                    continue
                preds_by_class[c].append((float(score.item()), image_id, box))

    ap_by_name: Dict[str, float] = {}
    ap_values: List[float] = []

    for c in range(1, len(VOC10_CLASSES) + 1):
        class_name = VOC10_CLASSES[c - 1]
        preds = preds_by_class[c]
        if npos_by_class[c] == 0:
            ap_by_name[class_name] = float("nan")
            continue
        if len(preds) == 0:
            ap_by_name[class_name] = 0.0
            ap_values.append(0.0)
            continue

        preds.sort(key=lambda x: x[0], reverse=True)
        tp = torch.zeros((len(preds),), dtype=torch.float32)
        fp = torch.zeros((len(preds),), dtype=torch.float32)

        for i, (_score, image_id, box_pred) in enumerate(preds):
            gt_img = gt_by_image.get(image_id, {})
            gt_boxes = gt_img.get(c, None)
            if gt_boxes is None or gt_boxes.numel() == 0:
                fp[i] = 1.0
                continue

            used = gt_used_by_image[image_id][c]
            ious = _box_iou_xyxy(box_pred.view(1, 4), gt_boxes).view(-1)
            max_iou, max_j = torch.max(ious, dim=0)
            j = int(max_j.item())

            if float(max_iou.item()) >= float(iou_threshold) and not bool(used[j].item()):
                tp[i] = 1.0
                used[j] = True
            else:
                fp[i] = 1.0

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        rec = tp_cum / float(npos_by_class[c])
        prec = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-9)
        ap = _voc_ap_continuous(rec, prec)
        ap_by_name[class_name] = ap
        ap_values.append(ap)

    map50 = float(sum(ap_values) / max(len(ap_values), 1))
    return map50, ap_by_name


def _find_last_epoch_checkpoint(output_dir: Path) -> Optional[Path]:
    candidates = list(output_dir.glob("fasterrcnn_resnet18fpn_supcon_cifar100_voc10_frozen_epoch*.pt"))
    if not candidates:
        return None

    def _epoch_num(p: Path) -> int:
        try:
            return int(p.stem.split("epoch")[-1])
        except Exception:
            return -1

    candidates.sort(key=_epoch_num)
    return candidates[-1]


def compute_val_loss(model: FasterRCNN, loader: DataLoader, device: torch.device) -> float:
    model.train()
    model.apply(_set_bn_eval)
    val_loss = 0.0
    samples = 0
    with torch.no_grad():
        val_iter = tqdm(loader, desc="Val", unit="batch", leave=False)
        for images, targets in val_iter:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            batch_size = len(images)
            val_loss += loss.item() * batch_size
            samples += batch_size
            val_iter.set_postfix(loss=loss.item())
    model.eval()
    return val_loss / max(samples, 1)


def _resolve_voc_root(data_root: Path) -> Tuple[str, Path]:
    # Standard VOCdevkit layout.
    if (data_root / "VOCdevkit" / "VOC2012").exists():
        return "torchvision", data_root
    if data_root.name == "VOCdevkit" and (data_root / "VOC2012").exists():
        return "torchvision", data_root.parent

    # Kaggle / flat VOC2012 layout (Annotations/JPEGImages/ImageSets).
    if (data_root / "Annotations").exists() and (data_root / "JPEGImages").exists():
        return "xml", data_root
    kaggle_root = data_root / "VOC2012_train_val" / "VOC2012_train_val"
    if (kaggle_root / "Annotations").exists() and (kaggle_root / "JPEGImages").exists():
        return "xml", kaggle_root

    raise FileNotFoundError(f"VOC2012 layout not found at {data_root}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)
    voc_mode, resolved_root = _resolve_voc_root(data_root)
    if voc_mode == "torchvision":
        train_dataset = VOC10Detection(
            root=str(resolved_root),
            year="2012",
            image_set="train",
            download=False,
        )
        try:
            val_dataset = VOC10Detection(
                root=str(resolved_root),
                year="2012",
                image_set="val",
                download=False,
            )
        except Exception:
            # Some VOC installs don't include the official val split files.
            total = len(train_dataset)
            val_count = max(1, int(total * 0.1))
            g = torch.Generator().manual_seed(42)
            indices = torch.randperm(total, generator=g).tolist()
            val_dataset = Subset(train_dataset, indices[:val_count])
            train_dataset = Subset(train_dataset, indices[val_count:])
            print("[warn] VOC val split not found; using 90/10 split of train as val.")
    else:
        train_dataset = VOC10XmlDetection(resolved_root, "train")
        val_dataset = VOC10XmlDetection(resolved_root, "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = build_model(
        num_classes=len(VOC10_CLASSES) + 1,
        supcon_checkpoint=str(args.supcon_checkpoint),
        backbone_image_size=int(args.backbone_image_size),
        train_backbone=bool(args.train_backbone),
        min_size=int(args.min_size),
        max_size=int(args.max_size),
    )
    model.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Eval-only modes (no training).
    if args.eval_only or args.eval_all:
        eval_csv = output_dir / "baseline_eval_map50.csv"
        if args.eval_all:
            ckpts = sorted(output_dir.glob("fasterrcnn_resnet18fpn_supcon_cifar100_voc10_frozen_epoch*.pt"))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found in {output_dir}")
        else:
            if args.checkpoint:
                ckpts = [Path(args.checkpoint)]
            else:
                last_ckpt = _find_last_epoch_checkpoint(output_dir)
                if last_ckpt is None:
                    raise FileNotFoundError(f"No checkpoints found in {output_dir}")
                ckpts = [last_ckpt]

        rows: List[Dict[str, str]] = []
        for ckpt in ckpts:
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state)
            map50, ap_by_name = evaluate_voc_map50(
                model=model,
                loader=val_loader,
                device=device,
                score_threshold=float(args.score_threshold),
                max_detections=int(args.max_detections),
                iou_threshold=0.5,
            )
            print(f"[eval] checkpoint={ckpt.name} VOC mAP@0.5={map50:.4f}")
            row: Dict[str, str] = {"checkpoint": ckpt.name, "map50": f"{map50:.6f}"}
            for name in VOC10_CLASSES:
                v = ap_by_name.get(name, float("nan"))
                row[f"ap_{name}"] = "" if (v != v) else f"{v:.6f}"  # NaN -> empty
            rows.append(row)

        fieldnames = ["checkpoint", "map50"] + [f"ap_{n}" for n in VOC10_CLASSES]
        with eval_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[eval] Wrote {eval_csv}")
        return

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    tracker = None
    if args.track_emissions and EmissionsTracker is not None:
        tracker = EmissionsTracker(
            project_name="original_fasterrcnn_resnet18_supcon_cifar100_voc12_frozen",
            log_level="warning",
            save_to_file=False,
        )
        tracker.start()

    metrics_rows = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        train_iter = tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{args.epochs}",
            unit="batch",
            leave=False,
        )
        for images, targets in train_iter:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_iter.set_postfix(loss=loss.item())

        lr_scheduler.step()

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        val_loss = compute_val_loss(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
        metrics_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f}"
        )

        ckpt_path = output_dir / f"fasterrcnn_resnet18fpn_supcon_cifar100_voc10_frozen_epoch{epoch + 1}.pt"
        torch.save(model.state_dict(), ckpt_path)

    emissions_kg = None
    energy_kwh = None
    duration_seconds = None
    if tracker is not None:
        emissions_kg = tracker.stop()
        data = getattr(tracker, "final_emissions_data", None)
        if data is not None:
            energy_kwh = getattr(data, "energy_consumed", None)
            duration_seconds = getattr(data, "duration", None)

    csv_path = output_dir / "baseline_metrics.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "learning_rate",
            "emissions_kg",
            "energy_kwh",
            "duration_seconds",
            "finished_at",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_loss": f"{row['train_loss']:.6f}",
                    "val_loss": f"{row['val_loss']:.6f}",
                    "learning_rate": f"{row['learning_rate']:.6f}",
                    "emissions_kg": "",
                    "energy_kwh": "",
                    "duration_seconds": "",
                    "finished_at": "",
                }
            )

        finished_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        writer.writerow(
            {
                "epoch": "summary",
                "train_loss": "",
                "val_loss": "",
                "learning_rate": "",
                "emissions_kg": "" if emissions_kg is None else f"{float(emissions_kg):.6f}",
                "energy_kwh": "" if energy_kwh is None else f"{float(energy_kwh):.6f}",
                "duration_seconds": "" if duration_seconds is None else f"{float(duration_seconds):.2f}",
                "finished_at": finished_at,
            }
        )

    print(f"[done] Wrote {csv_path}")


if __name__ == "__main__":
    main()

