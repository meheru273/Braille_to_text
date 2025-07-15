import pathlib
import os
import logging
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from PIL import Image
from pycocotools.coco import COCO
import cv2

from Train import COCOData, collate_fn
from FPN import FPN
from inference import compute_detections, render_detections_to_image, tensor_to_image


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU score
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_metrics(
    detections: List[Dict[str, np.ndarray]], 
    gt_per_image: List[Dict[str, np.ndarray]],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score.
    
    Args:
        detections: List of detection dicts (per image) with keys:
            'boxes', 'labels', 'scores'
        gt_per_image: List of ground truth dicts (per image) with keys:
            'boxes', 'labels'
        num_classes: Number of classes
        iou_threshold: IoU threshold for true positive
    
    Returns:
        Dictionary with micro-averaged precision, recall, and F1 score
    """
    # Initialize counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Process each image
    for img_idx, (det, gt) in enumerate(zip(detections, gt_per_image)):
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']
        det_boxes = det['boxes']
        det_labels = det['labels']
        det_scores = det['scores']
        
        # Sort detections by confidence (descending)
        sorted_indices = np.argsort(det_scores)[::-1]
        det_boxes = det_boxes[sorted_indices]
        det_labels = det_labels[sorted_indices]
        
        # Track matched ground truths
        matched_gt = [False] * len(gt_boxes)
        
        # Match detections to ground truth
        for det_box, det_label in zip(det_boxes, det_labels):
            matched = False
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if matched_gt[gt_idx]:
                    continue
                if det_label != gt_label:
                    continue
                iou = compute_iou(det_box, gt_box)
                if iou >= iou_threshold:
                    matched_gt[gt_idx] = True
                    matched = True
                    break
            
            if matched:
                total_tp += 1
            else:
                total_fp += 1
        
        # Count unmatched ground truths as false negatives
        total_fn += sum(not matched for matched in matched_gt)
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def metric(test_dir: pathlib.Path, model_path: pathlib.Path):
    IMAGE_SIZE = (800, 1200)
    
    test_dir = pathlib.Path(test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory {test_dir} does not exist.")
    
    test_dataset = COCOData(test_dir, image_size=IMAGE_SIZE, min_area=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    num_classes = test_dataset.get_num_classes()
    class_names = test_dataset.get_class_names()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = FPN(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    # Compute detections
    detections = compute_detections(
        model, test_loader, device=device, 
        class_names=class_names, image_size=IMAGE_SIZE
    )
    
    # Collect ground truth data
    gt_per_image = []
    for i in range(len(test_dataset)):
        _, targets = test_dataset[i]
        # Convert tensors to numpy arrays
        gt_data = {
            "boxes": targets["boxes"].numpy(),
            "labels": targets["labels"].numpy()
        }
        gt_per_image.append(gt_data)
    
    # Convert detection tensors to numpy if needed
    for i, det in enumerate(detections):
        if isinstance(det["boxes"], torch.Tensor):
            detections[i]["boxes"] = det["boxes"].cpu().numpy()
        if isinstance(det["labels"], torch.Tensor):
            detections[i]["labels"] = det["labels"].cpu().numpy()
        if isinstance(det["scores"], torch.Tensor):
            detections[i]["scores"] = det["scores"].cpu().numpy()
    
    # Calculate metrics
    metrics = calculate_metrics(detections, gt_per_image, num_classes)
    
    # Print results
    print("\nEvaluation Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")


if __name__ == "__main__":
    test_dir = pathlib.Path(r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\dataset.coco\test") 
    model_path = pathlib.Path(r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\runs\fcos_custom\fcos_epoch30.pth")
    metric(test_dir, model_path)