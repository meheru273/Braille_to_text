import torch
import numpy as np
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class Metrics:
    """Metrics container for object detection evaluation"""
    true_positive_count: int
    false_positive_count: int
    mean_average_precision: float
    total_ground_truth_detections: int


class Detection(NamedTuple):
    """Single detection with bounding box, confidence and class"""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: tensor of shape (4,) representing (x1, y1, x2, y2)
        box2: tensor of shape (4,) representing (x1, y1, x2, y2)
    
    Returns:
        IoU value between 0 and 1
    """
    # Get intersection coordinates
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())
    
    # Compute intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def compute_ap_for_class(
    detections: List[Detection], 
    ground_truths: List[torch.Tensor], 
    class_id: int, 
    iou_threshold: float = 0.5
) -> Tuple[float, int, int]:
    """
    Compute Average Precision for a single class
    
    Args:
        detections: List of Detection objects
        ground_truths: List of ground truth bounding boxes per image
        class_id: Class ID to compute AP for
        iou_threshold: IoU threshold for positive detection
    
    Returns:
        Tuple of (AP, true_positives, false_positives)
    """
    # Filter detections for this class and sort by confidence
    class_detections = [d for d in detections if d.class_id == class_id]
    class_detections.sort(key=lambda x: x.confidence, reverse=True)
    
    if not class_detections:
        return 0.0, 0, 0
    
    # Count ground truth boxes for this class
    num_gt = sum(len(gt) for gt in ground_truths)
    
    if num_gt == 0:
        return 0.0, 0, len(class_detections)
    
    # Track which ground truth boxes have been matched
    gt_matched = [torch.zeros(len(gt), dtype=torch.bool) for gt in ground_truths]
    
    tp = []
    fp = []
    
    for detection in class_detections:
        # Find best matching ground truth box
        best_iou = 0
        best_gt_idx = -1
        best_img_idx = -1
        
        for img_idx, gt_boxes in enumerate(ground_truths):
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[img_idx][gt_idx]:
                    continue
                
                iou = compute_iou(torch.tensor(detection.bbox), gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_img_idx = img_idx
        
        if best_iou >= iou_threshold:
            # True positive
            tp.append(1)
            fp.append(0)
            gt_matched[best_img_idx][best_gt_idx] = True
        else:
            # False positive
            tp.append(0)
            fp.append(1)
    
    # Compute precision and recall
    tp = np.array(tp)
    fp = np.array(fp)
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recall = tp_cumsum / num_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # Compute AP using 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap, int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0, int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0


def compute_metrics(
    all_detections: List[List[Detection]], 
    all_class_labels: List[torch.Tensor], 
    all_box_labels: List[torch.Tensor],
    num_classes: int = 8,  # Cityscapes typically has 8 classes
    iou_threshold: float = 0.5
) -> Metrics:
    """
    Compute Pascal VOC style metrics for object detection
    
    Args:
        all_detections: List of detections per image
        all_class_labels: List of class labels per image
        all_box_labels: List of bounding box labels per image
        num_classes: Number of classes
        iou_threshold: IoU threshold for positive detection
    
    Returns:
        Metrics object containing evaluation results
    """
    # Flatten detections
    flat_detections = []
    for img_detections in all_detections:
        flat_detections.extend(img_detections)
    
    # Compute AP for each class
    aps = []
    total_tp = 0
    total_fp = 0
    total_gt = 0
    
    for class_id in range(num_classes):
        ap, tp, fp = compute_ap_for_class(
            flat_detections, 
            all_box_labels, 
            class_id, 
            iou_threshold
        )
        aps.append(ap)
        total_tp += tp
        total_fp += fp
    
    # Count total ground truth detections
    total_gt = sum(len(boxes) for boxes in all_box_labels)
    
    # Compute mean AP
    mean_ap = np.mean(aps) if aps else 0.0
    
    return Metrics(
        true_positive_count=total_tp,
        false_positive_count=total_fp,
        mean_average_precision=mean_ap,
        total_ground_truth_detections=total_gt
    )