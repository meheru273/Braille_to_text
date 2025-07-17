import pathlib
import os
import logging
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from PIL import Image
from pycocotools.coco import COCO
import cv2

from Train import COCOData, collate_fn
from FPN import FPN, normalize_batch
from inference import detections_from_network_output, render_detections_to_image, tensor_to_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate intersection area
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def match_detections_to_ground_truth(detections, ground_truth, iou_threshold=0.01):
    """
    Match detections to ground truth boxes using IoU threshold.
    
    Args:
        detections: List of detection dictionaries with 'bbox', 'class', 'score'
        ground_truth: List of ground truth dictionaries with 'bbox', 'class'
        iou_threshold: IoU threshold for matching
    
    Returns:
        matched_detections: List of matched detection indices
        matched_gt: List of matched ground truth indices
        unmatched_detections: List of unmatched detection indices
        unmatched_gt: List of unmatched ground truth indices
    """
    if len(detections) == 0 or len(ground_truth) == 0:
        return [], [], list(range(len(detections))), list(range(len(ground_truth)))
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(detections), len(ground_truth)))
    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truth):
            if det['class'] == gt['class']:  # Only match same class
                iou_matrix[i, j] = calculate_iou(det['bbox'], gt['bbox'])
    
    # Greedy matching (highest IoU first)
    matched_detections = []
    matched_gt = []
    used_gt = set()
    used_det = set()
    
    # Sort detections by confidence score (highest first)
    sorted_det_indices = sorted(range(len(detections)), 
                               key=lambda i: detections[i]['score'], reverse=True)
    
    for det_idx in sorted_det_indices:
        if det_idx in used_det:
            continue
            
        best_gt_idx = -1
        best_iou = 0
        
        for gt_idx in range(len(ground_truth)):
            if gt_idx in used_gt:
                continue
            if iou_matrix[det_idx, gt_idx] > best_iou and iou_matrix[det_idx, gt_idx] >= iou_threshold:
                best_iou = iou_matrix[det_idx, gt_idx]
                best_gt_idx = gt_idx
        
        if best_gt_idx != -1:
            matched_detections.append(det_idx)
            matched_gt.append(best_gt_idx)
            used_det.add(det_idx)
            used_gt.add(best_gt_idx)
    
    unmatched_detections = [i for i in range(len(detections)) if i not in used_det]
    unmatched_gt = [i for i in range(len(ground_truth)) if i not in used_gt]
    
    return matched_detections, matched_gt, unmatched_detections, unmatched_gt

def calculate_precision_recall(all_detections, all_ground_truth, iou_threshold=0.01):
    """
    Calculate precision and recall for all test images.
    
    Args:
        all_detections: List of detection lists (one per image)
        all_ground_truth: List of ground truth lists (one per image)
        iou_threshold: IoU threshold for matching
    
    Returns:
        precision: Precision value
        recall: Recall value
        f1_score: F1 score value
    """
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives
    
    for detections, ground_truth in zip(all_detections, all_ground_truth):
        matched_det, matched_gt, unmatched_det, unmatched_gt = match_detections_to_ground_truth(
            detections, ground_truth, iou_threshold
        )
        
        total_tp += len(matched_det)
        total_fp += len(unmatched_det)
        total_fn += len(unmatched_gt)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score

def compute_detections(model, test_loader, device, class_names, image_size, confidence_threshold=0.01):
    model.eval()
    all_detections = []
    all_ground_truth = []
    
    with torch.no_grad():
        for batch_idx, (images, class_labels, box_labels) in enumerate(test_loader):
            images = images.to(device)
            batch_norm = normalize_batch(images)
            
            # Forward pass
            cls_pred, cen_pred, box_pred = model(batch_norm)
            
            # Get image dimensions
            H, W = images.shape[2], images.shape[3]
            
            # Convert predictions to detections
            detections = detections_from_network_output(
                H, W, cls_pred, cen_pred, box_pred, 
                model.scales, model.strides
            )
            
            # Process detections for each image in batch
            for i in range(len(images)):
                image_detections = []
                image_ground_truth = []
                
                # Process detections with proper attribute access
                for det in detections[i]:
                    # Check if det has attributes or is a dict
                    if hasattr(det, 'score'):
                        # Detection object with attributes
                        score = det.score
                        bbox = det.bbox if hasattr(det, 'bbox') else det.box
                        if hasattr(det, 'class_id'):
                            class_id = det.class_id
                        else:
                            # Use getattr to safely access 'class' attribute
                            class_id = getattr(det, 'class', None)
                    else:
                        # Dictionary format
                        score = det['score']
                        bbox = det['bbox']
                        class_id = det['class']
                    
                    if score >= confidence_threshold:
                        image_detections.append({
                            'bbox': bbox,
                            'class': class_id,
                            'score': score
                        })
                
                # Process ground truth
                for j in range(len(class_labels[i])):
                    image_ground_truth.append({
                        'bbox': box_labels[i][j].tolist(),
                        'class': class_labels[i][j].item()
                    })
                
                all_detections.append(image_detections)
                all_ground_truth.append(image_ground_truth)
            
            # Optional: Log progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}")
    
    return all_detections, all_ground_truth


def calculate_map(all_detections, all_ground_truth, iou_thresholds=[0.01, 0.05, 0.08, 0.65, 0.7]):
    """
    Calculate mean Average Precision (mAP) across multiple IoU thresholds.
    
    Args:
        all_detections: List of detections for each image
        all_ground_truth: List of ground truth for each image
        iou_thresholds: List of IoU thresholds
    
    Returns:
        mAP: Mean Average Precision
        mAP_50: mAP at IoU threshold 0.5
        mAP_75: mAP at IoU threshold 0.75
    """
    ap_scores = []
    
    for iou_threshold in iou_thresholds:
        precision, recall, _ = calculate_precision_recall(
            all_detections, all_ground_truth, iou_threshold
        )
        ap_scores.append(precision)  # Simplified AP calculation
    
    mAP = np.mean(ap_scores)
    mAP_50 = ap_scores[0] if len(ap_scores) > 0 else 0.0
    mAP_75 = ap_scores[5] if len(ap_scores) > 5 else 0.0
    
    return mAP, mAP_50, mAP_75

def evaluate_model(test_dir: pathlib.Path, model_path: pathlib.Path, confidence_threshold: float = 0.5):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        test_dir: Path to test dataset directory
        model_path: Path to trained model checkpoint
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    IMAGE_SIZE = (800, 1200)
    
    # Validate paths
    test_dir = pathlib.Path(test_dir)
    model_path = pathlib.Path(model_path)
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory {test_dir} does not exist.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    # Load dataset
    logger.info("Loading test dataset...")
    test_dataset = COCOData(test_dir, image_size=IMAGE_SIZE, min_area=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                           collate_fn=collate_fn, num_workers=2)
    
    num_classes = test_dataset.get_num_classes()
    class_names = test_dataset.get_class_names()
    
    logger.info(f"Test dataset loaded: {len(test_dataset)} images, {num_classes} classes")
    
    # Load model
    logger.info("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = FPN(num_classes=num_classes)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Compute detections
    logger.info("Computing detections...")
    all_detections, all_ground_truth = compute_detections(
        model, test_loader, device, class_names, IMAGE_SIZE, confidence_threshold
    )
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    precision, recall, f1_score = calculate_precision_recall(all_detections, all_ground_truth)
    mAP, mAP_50, mAP_75 = calculate_map(all_detections, all_ground_truth)
    
    # Count total detections and ground truth
    total_detections = sum(len(dets) for dets in all_detections)
    total_ground_truth = sum(len(gt) for gt in all_ground_truth)
    
    # Create results dictionary
    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mAP': mAP,
        'mAP_50': mAP_50,
        'mAP_75': mAP_75,
        'total_detections': total_detections,
        'total_ground_truth': total_ground_truth,
        'num_test_images': len(test_dataset),
        'confidence_threshold': confidence_threshold
    }
    
    return results

def print_evaluation_results(results: Dict):
    """
    Print formatted evaluation results.
    
    Args:
        results: Dictionary containing evaluation metrics
    """
    print("\n" + "="*60)
    print("OBJECT DETECTION EVALUATION RESULTS")
    print("="*60)
    
    print(f"Test Images: {results['num_test_images']}")
    print(f"Confidence Threshold: {results['confidence_threshold']:.2f}")
    print(f"Total Detections: {results['total_detections']}")
    print(f"Total Ground Truth: {results['total_ground_truth']}")
    print("-" * 60)
    
    print(f"Precision:    {results['precision']:.4f}")
    print(f"Recall:       {results['recall']:.4f}")
    print(f"F1 Score:     {results['f1_score']:.4f}")
    print("-" * 60)
    
    print(f"mAP:          {results['mAP']:.4f}")
    print(f"mAP@0.5:      {results['mAP_50']:.4f}")
    print(f"mAP@0.75:     {results['mAP_75']:.4f}")
    print("="*60)

if __name__ == "__main__":
    # Set up paths
    test_dir = pathlib.Path(r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\dataset.coco\test") 
    model_path = pathlib.Path(r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\runs\fcos_custom\fcos_epoch10.pth")
    
    try:
        # Run evaluation
        results = evaluate_model(test_dir, model_path, confidence_threshold=0.01)
        
        # Print results
        print_evaluation_results(results)
        
        # Save results to file
        results_file = model_path.parent / "evaluation_results.txt"
        with open(results_file, 'w') as f:
            f.write("OBJECT DETECTION EVALUATION RESULTS\n")
            f.write("="*60 + "\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
