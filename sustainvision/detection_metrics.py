"""Detection metrics and prediction helper utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - runtime safeguard
    torch = None  # type: ignore
    DataLoader = None  # type: ignore

from .data import COCO_CONTIGUOUS_TO_ID


def compute_iou(
    boxes1: "torch.Tensor",
    boxes2: "torch.Tensor",
) -> "torch.Tensor":
    """Compute IoU between two xyxy box sets."""
    if torch is None:
        raise RuntimeError("PyTorch is required")

    inter_x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0].unsqueeze(0))
    inter_y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1].unsqueeze(0))
    inter_x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2].unsqueeze(0))
    inter_y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3].unsqueeze(0))

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    return inter_area / (union_area + 1e-6)


def match_anchors_to_gt(
    pred_boxes: "torch.Tensor",
    gt_boxes: "torch.Tensor",
    *,
    iou_threshold: float,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Greedy IoU matching between predicted slot boxes and GT boxes."""
    if torch is None:
        raise RuntimeError("PyTorch is required")

    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.long, device=pred_boxes.device)
        return empty, empty

    iou = compute_iou(pred_boxes, gt_boxes)
    matched_pred: list[int] = []
    matched_gt: list[int] = []
    while True:
        max_val, flat_idx = torch.max(iou.view(-1), dim=0)
        if float(max_val.item()) < float(iou_threshold):
            break
        a = int(flat_idx.item()) // int(gt_boxes.size(0))
        g = int(flat_idx.item()) % int(gt_boxes.size(0))
        matched_pred.append(a)
        matched_gt.append(g)
        iou[a, :] = -1.0
        iou[:, g] = -1.0

    if not matched_pred:
        empty = torch.zeros((0,), dtype=torch.long, device=pred_boxes.device)
        return empty, empty

    return (
        torch.tensor(matched_pred, dtype=torch.long, device=pred_boxes.device),
        torch.tensor(matched_gt, dtype=torch.long, device=pred_boxes.device),
    )


def apply_nms(
    boxes: "torch.Tensor",
    scores: "torch.Tensor",
    labels: "torch.Tensor",
    *,
    iou_threshold: float,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Apply per-class NMS to detections in xyxy format."""
    if torch is None:
        raise RuntimeError("PyTorch is required")
    if boxes.numel() == 0:
        return boxes, scores, labels

    try:
        from torchvision.ops import nms  # type: ignore
    except Exception:
        nms = None  # type: ignore

    def _nms_fallback(b: "torch.Tensor", s: "torch.Tensor") -> "torch.Tensor":
        keep: list[int] = []
        idxs = torch.argsort(s, descending=True)
        while idxs.numel() > 0:
            i = int(idxs[0].item())
            keep.append(i)
            if idxs.numel() == 1:
                break
            rest = idxs[1:]
            ious = compute_iou(b[i].unsqueeze(0), b[rest]).squeeze(0)
            idxs = rest[ious < iou_threshold]
        return torch.tensor(keep, dtype=torch.long, device=b.device)

    keep_indices: list["torch.Tensor"] = []
    for cls in torch.unique(labels):
        cls_mask = labels == cls
        cls_idxs = torch.where(cls_mask)[0]
        if cls_idxs.numel() == 0:
            continue
        b = boxes[cls_idxs]
        s = scores[cls_idxs]
        kept = nms(b, s, float(iou_threshold)) if nms is not None else _nms_fallback(b, s)
        keep_indices.append(cls_idxs[kept])

    if not keep_indices:
        return boxes[:0], scores[:0], labels[:0]

    keep = torch.cat(keep_indices, dim=0)
    keep = keep[torch.argsort(scores[keep], descending=True)]
    return boxes[keep], scores[keep], labels[keep]


def decode_slot_detections(
    cls_logits: "torch.Tensor",
    bbox_preds: "torch.Tensor",
    *,
    score_threshold: float,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Convert slot-based outputs into a flat detection list."""
    if torch is None:
        raise RuntimeError("PyTorch is required")

    probs = torch.softmax(cls_logits, dim=-1)
    scores, labels = probs.max(dim=-1)
    keep = scores >= float(score_threshold)
    if not keep.any():
        return bbox_preds[:0], scores[:0], labels[:0]
    return bbox_preds[keep], scores[keep], labels[keep]


def _get_coco_api_from_dataset(dataset: Any) -> Any:
    """Try to get the underlying pycocotools COCO API object from a dataset/subset."""
    try:
        from torch.utils.data import Subset  # type: ignore
    except Exception:
        Subset = None  # type: ignore

    ds = dataset
    while True:
        if Subset is not None and isinstance(ds, Subset):
            ds = ds.dataset
            continue
        inner = getattr(ds, "dataset", None)
        if inner is not None and inner is not ds:
            ds = inner
            continue
        break
    return getattr(ds, "coco", None)


def _denormalize_boxes_xyxy(
    boxes_norm: "torch.Tensor",
    *,
    orig_h: int,
    orig_w: int,
) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required")
    scale = boxes_norm.new_tensor([orig_w, orig_h, orig_w, orig_h])
    return boxes_norm * scale


def _xyxy_to_xywh(boxes_xyxy: "torch.Tensor") -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required")
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    return torch.stack([x1, y1, (x2 - x1), (y2 - y1)], dim=-1)


def generate_coco_predictions_torchvision(
    model: "torch.nn.Module",
    dataloader: DataLoader,
    *,
    device: "torch.device",
    score_threshold: float,
    max_batches: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate COCO-format predictions from a torchvision detection model."""
    if torch is None:
        raise RuntimeError("PyTorch is required")

    model.eval()
    preds: List[Dict[str, Any]] = []
    old_model_thresh: Optional[float] = None
    try:
        if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "score_thresh"):
            old_model_thresh = float(model.roi_heads.score_thresh)
            model.roi_heads.score_thresh = float(score_threshold)
    except Exception:
        old_model_thresh = None

    try:
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                images_list: List[torch.Tensor] = [img.to(device) for img in images]
                outputs = model(images_list)  # type: ignore[call-arg]
                if isinstance(outputs, dict):
                    continue

                for out, tgt, img in zip(outputs, targets, images):
                    boxes = out.get("boxes")
                    labels = out.get("labels")
                    scores = out.get("scores")
                    if boxes is None or labels is None or scores is None or boxes.numel() == 0:
                        continue
                    image_id = int(tgt.get("image_id", -1)) if isinstance(tgt, dict) else -1
                    if image_id == -1:
                        continue

                    orig_size = tgt.get("orig_size", None) if isinstance(tgt, dict) else None
                    in_h = int(img.shape[1])
                    in_w = int(img.shape[2])
                    boxes_scaled = boxes
                    if orig_size is not None:
                        try:
                            orig_h = int(orig_size[0])
                            orig_w = int(orig_size[1])
                            if in_h > 0 and in_w > 0:
                                scale = boxes.new_tensor(
                                    [orig_w / in_w, orig_h / in_h, orig_w / in_w, orig_h / in_h]
                                )
                                boxes_scaled = boxes * scale
                        except Exception:
                            boxes_scaled = boxes

                    boxes_xywh = _xyxy_to_xywh(boxes_scaled.detach().cpu())
                    for box_xywh, score, label in zip(boxes_xywh, scores, labels):
                        contiguous = int(label.item()) - 1
                        coco_cat = COCO_CONTIGUOUS_TO_ID.get(contiguous)
                        if coco_cat is None:
                            continue
                        preds.append(
                            {
                                "image_id": image_id,
                                "category_id": int(coco_cat),
                                "bbox": [float(x) for x in box_xywh.tolist()],
                                "score": float(score.item()),
                            }
                        )
    finally:
        try:
            if old_model_thresh is not None and hasattr(model, "roi_heads") and hasattr(model.roi_heads, "score_thresh"):
                model.roi_heads.score_thresh = old_model_thresh
        except Exception:
            pass
    return preds


def generate_voc_predictions_torchvision(
    model: "torch.nn.Module",
    dataloader: DataLoader,
    *,
    device: "torch.device",
    score_threshold: float,
    max_batches: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate VOC-style predictions from a torchvision detection model."""
    if torch is None:
        raise RuntimeError("PyTorch is required")

    model.eval()
    preds: List[Dict[str, Any]] = []

    old_model_thresh: Optional[float] = None
    try:
        if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "score_thresh"):
            old_model_thresh = float(model.roi_heads.score_thresh)
            model.roi_heads.score_thresh = float(score_threshold)
    except Exception:
        old_model_thresh = None

    image_index = 0
    try:
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                images_list: List[torch.Tensor] = [img.to(device) for img in images]
                outputs = model(images_list)  # type: ignore[call-arg]
                if isinstance(outputs, dict):
                    continue

                for out, tgt, img in zip(outputs, targets, images):
                    boxes = out.get("boxes")
                    labels = out.get("labels")
                    scores = out.get("scores")
                    if boxes is None or labels is None or scores is None:
                        image_index += 1
                        continue
                    orig_size = tgt.get("orig_size", None) if isinstance(tgt, dict) else None
                    in_h = int(img.shape[1])
                    in_w = int(img.shape[2])
                    boxes_scaled = boxes
                    if orig_size is not None:
                        try:
                            orig_h = int(orig_size[0])
                            orig_w = int(orig_size[1])
                            if in_h > 0 and in_w > 0:
                                scale = boxes.new_tensor(
                                    [orig_w / in_w, orig_h / in_h, orig_w / in_w, orig_h / in_h]
                                )
                                boxes_scaled = boxes * scale
                        except Exception:
                            boxes_scaled = boxes

                    preds.append(
                        {
                            "image_id": image_index,
                            "boxes": boxes_scaled.detach().cpu().tolist(),
                            "scores": scores.detach().cpu().tolist(),
                            "labels": (labels.detach().cpu() - 1).clamp(min=0).tolist(),
                        }
                    )
                    image_index += 1
    finally:
        try:
            if old_model_thresh is not None and hasattr(model, "roi_heads"):
                model.roi_heads.score_thresh = old_model_thresh
        except Exception:
            pass
    return preds


def generate_coco_predictions(
    model: Any,
    dataloader: DataLoader,
    *,
    device: "torch.device",
    score_threshold: float,
    nms_iou_threshold: float,
    max_batches: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run slot-based detector and output COCO-format predictions."""
    if torch is None:
        raise RuntimeError("PyTorch is required")

    model.eval()
    preds: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)
            cls_logits, bbox_preds = model(images)

            for i in range(images.size(0)):
                tgt = targets[i] if isinstance(targets, list) else targets
                image_id = int(tgt.get("image_id", -1))
                orig_size = tgt.get("orig_size", None)
                if orig_size is None:
                    orig_h = int(images.size(2))
                    orig_w = int(images.size(3))
                else:
                    try:
                        orig_h = int(orig_size[0])
                        orig_w = int(orig_size[1])
                    except Exception:
                        orig_h = int(images.size(2))
                        orig_w = int(images.size(3))

                boxes, scores, labels = decode_slot_detections(
                    cls_logits[i],
                    bbox_preds[i],
                    score_threshold=score_threshold,
                )
                boxes, scores, labels = apply_nms(
                    boxes,
                    scores,
                    labels,
                    iou_threshold=nms_iou_threshold,
                )
                if boxes.numel() == 0:
                    continue

                boxes_xyxy = _denormalize_boxes_xyxy(boxes, orig_h=orig_h, orig_w=orig_w)
                boxes_xywh = _xyxy_to_xywh(boxes_xyxy)
                for box_xywh, score, label in zip(boxes_xywh, scores, labels):
                    contiguous = int(label.item())
                    coco_cat = COCO_CONTIGUOUS_TO_ID.get(contiguous)
                    if coco_cat is None:
                        continue
                    preds.append(
                        {
                            "image_id": image_id,
                            "category_id": int(coco_cat),
                            "bbox": [float(x) for x in box_xywh.tolist()],
                            "score": float(score.item()),
                        }
                    )
    return preds


def compute_coco_map(
    coco_gt: Any,
    coco_predictions: List[Dict[str, Any]],
    *,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute COCO mAP@[.5:.95] using pycocotools."""
    try:
        from pycocotools.cocoeval import COCOeval  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pycocotools is required for full COCO mAP evaluation. Install it with `pip install pycocotools`."
        ) from exc

    if coco_gt is None:
        raise RuntimeError("COCO ground-truth API is not available from the dataset.")
    if not coco_predictions:
        return {"map": 0.0, "map50": 0.0, "map75": 0.0, "per_class_ap": {}, "note": "No predictions produced (empty)."}

    coco_dt = coco_gt.loadRes(coco_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if iou_thresholds is not None:
        try:
            import numpy as np  # type: ignore
            coco_eval.params.iouThrs = np.array([float(x) for x in iou_thresholds], dtype=np.float32)
        except Exception:
            pass
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    per_class_ap: Dict[int, float] = {}
    try:
        precision = coco_eval.eval["precision"]  # type: ignore[index]
        for k in range(precision.shape[2]):
            p = precision[:, :, k, 0, 2]
            p = p[p > -1]
            count = int(getattr(p, "size", 0)) if not hasattr(p, "numel") else int(p.numel())
            ap = float(p.mean()) if count > 0 else float("nan")
            per_class_ap[int(k)] = ap
    except Exception:
        per_class_ap = {}

    return {
        "map": float(stats[0]),
        "map50": float(stats[1]),
        "map75": float(stats[2]),
        "per_class_ap": per_class_ap,
    }


def compute_voc_map50(
    ground_truths: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    *,
    num_classes: int,
) -> Dict[str, Any]:
    """Compute Pascal VOC mAP@0.5 using continuous AP."""

    def _iou(box: list[float], boxes: list[list[float]]) -> Tuple[float, int]:
        best_iou = 0.0
        best_idx = -1
        x1, y1, x2, y2 = box
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        for idx, b in enumerate(boxes):
            xx1 = max(x1, b[0])
            yy1 = max(y1, b[1])
            xx2 = min(x2, b[2])
            yy2 = min(y2, b[3])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            if inter <= 0:
                continue
            b_area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            union = box_area + b_area - inter
            if union <= 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        return best_iou, best_idx

    def _voc_ap(rec: list[float], prec: list[float]) -> float:
        mrec = [0.0] + rec + [1.0]
        mpre = [0.0] + prec + [0.0]
        for i in range(len(mpre) - 1, 0, -1):
            if mpre[i - 1] < mpre[i]:
                mpre[i - 1] = mpre[i]
        ap = 0.0
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap

    gt_by_class: list[Dict[int, Dict[str, Any]]] = []
    gt_count_by_class: list[int] = []
    for cls in range(num_classes):
        gt_by_image: Dict[int, Dict[str, Any]] = {}
        total = 0
        for gt in ground_truths:
            image_id = int(gt["image_id"])
            boxes = gt["boxes"]
            labels = gt["labels"]
            cls_boxes = [box for box, label in zip(boxes, labels) if int(label) == cls]
            if not cls_boxes:
                continue
            gt_by_image[image_id] = {"boxes": cls_boxes, "matched": [False] * len(cls_boxes)}
            total += len(cls_boxes)
        gt_by_class.append(gt_by_image)
        gt_count_by_class.append(total)

    per_class_ap: Dict[int, float] = {}
    ap_values: list[float] = []
    for cls in range(num_classes):
        cls_preds: list[Tuple[int, float, list[float]]] = []
        for pred in predictions:
            image_id = int(pred["image_id"])
            boxes = pred["boxes"]
            scores = pred["scores"]
            labels = pred["labels"]
            for box, score, label in zip(boxes, scores, labels):
                if int(label) == cls:
                    cls_preds.append((image_id, float(score), box))
        cls_preds.sort(key=lambda x: x[1], reverse=True)
        tp = [0.0] * len(cls_preds)
        fp = [0.0] * len(cls_preds)

        gt_by_image = gt_by_class[cls]
        for i, (image_id, _, box) in enumerate(cls_preds):
            gt_record = gt_by_image.get(image_id)
            if gt_record is None:
                fp[i] = 1.0
                continue
            iou, match_idx = _iou(box, gt_record["boxes"])
            if iou >= 0.5 and match_idx >= 0 and not gt_record["matched"][match_idx]:
                tp[i] = 1.0
                gt_record["matched"][match_idx] = True
            else:
                fp[i] = 1.0

        gt_total = gt_count_by_class[cls]
        if gt_total == 0:
            per_class_ap[cls] = float("nan")
            continue

        cum_tp = 0.0
        cum_fp = 0.0
        rec: list[float] = []
        prec: list[float] = []
        for i in range(len(tp)):
            cum_tp += tp[i]
            cum_fp += fp[i]
            rec.append(cum_tp / gt_total)
            prec.append(cum_tp / max(cum_tp + cum_fp, 1e-12))

        ap = _voc_ap(rec, prec)
        per_class_ap[cls] = ap
        ap_values.append(ap)

    map50 = float(sum(ap_values) / len(ap_values)) if ap_values else 0.0
    return {"map50": map50, "per_class_ap": per_class_ap}
