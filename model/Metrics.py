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

from FPNAttention import ImprovedFPN
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

def compute_detections(model, test_loader, device, class_names, image_size, confidence_threshold=0.5):
    """
    FIXED: Now properly handles 1-26 class IDs without conversion
    """
    model.eval()
    all_detections = []
    all_ground_truth = []
    
    print(f"Computing detections with confidence threshold: {confidence_threshold}")
    
    with torch.no_grad():
        for batch_idx, (images, class_labels, box_labels) in enumerate(test_loader):
            images = images.to(device)
            batch_norm = normalize_batch(images)
            
            # Forward pass - handle both standard FPN and ImprovedFPN
            try:
                # Try ImprovedFPN first (returns 3 values)
                cls_pred, box_pred, att_map = model(batch_norm)
            except ValueError:
                # Fallback to standard FPN (returns 2 values)
                cls_pred, box_pred = model(batch_norm)
            
            H, W = images.shape[2], images.shape[3]
            
            detections = detections_from_network_output(
                H, W, cls_pred, box_pred, 
                model.scales, model.strides
            )
            
            for i in range(len(images)):
                image_detections = []
                image_ground_truth = []
                
                # Process detections - KEEP ORIGINAL CLASS IDs (1-26)
                for det in detections[i]:
                    if hasattr(det, 'score'):
                        score = det.score
                        bbox = det.bbox if hasattr(det, 'bbox') else det.box
                        # Get class ID from detection
                        if hasattr(det, 'object_class'):
                            class_id = det.object_class
                        elif hasattr(det, 'class_id'):
                            class_id = det.class_id
                        else:
                            class_id = getattr(det, 'class', None)
                    else:
                        score = det['score']
                        bbox = det['bbox']
                        class_id = det['class']
                    
                    if score >= confidence_threshold and class_id is not None:
                        # CRITICAL FIX: Keep original class IDs (1-26)
                        # NO CONVERSION - Use class_id as-is
                        adjusted_class_id = class_id - 1
                        if 0 <= adjusted_class_id < 26:
                            image_detections.append({
                            'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                            'class': adjusted_class_id,  # Use corrected ID
                            'score': score
                        })
                
                # Process ground truth - KEEP ORIGINAL CLASS IDs (1-26)
                for j in range(len(class_labels[i])):
                    gt_class = class_labels[i][j].item()
                    # CRITICAL FIX: Keep original class IDs (1-26)
                    # NO CONVERSION - Use gt_class as-is
                    image_ground_truth.append({
                        'bbox': box_labels[i][j].tolist(),
                        'class': gt_class  # Keep original 1-26 range
                    })
                
                all_detections.append(image_detections)
                all_ground_truth.append(image_ground_truth)
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}")
                # Debug: Print some detection info
                # In the same function, where you print samples:
               # In compute_detections function:
                if batch_idx == 0 and len(all_detections) > 0 and len(all_detections[0]) > 0:
                    det = all_detections[0][0]
                    print(f"Sample detection: class={det['class']} ({class_names[det['class']]}), score={det['score']:.3f}")  # Fixed line
                if batch_idx == 0 and len(all_ground_truth) > 0 and len(all_ground_truth[0]) > 0:
                    gt = all_ground_truth[0][0]
                    print(f"Sample ground truth: class={gt['class']} ({class_names[gt['class']]})")  # Also fixed for consistency
    
    return all_detections, all_ground_truth


def calculate_map(all_detections, all_ground_truth, iou_thresholds=None):
    """
    Calculate mAP with proper Average Precision computation.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                         0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    ap_scores = []
    
    for iou_threshold in iou_thresholds:
        precision, recall, _ = calculate_precision_recall(
            all_detections, all_ground_truth, iou_threshold
        )
        # Use precision as simplified AP (could be improved with proper PR curve)
        ap_scores.append(precision)
    
    mAP = np.mean(ap_scores) if ap_scores else 0.0
    
    # Find closest thresholds for standard metrics
    mAP_50_idx = min(range(len(iou_thresholds)), 
                     key=lambda i: abs(iou_thresholds[i] - 0.5))
    mAP_75_idx = min(range(len(iou_thresholds)), 
                     key=lambda i: abs(iou_thresholds[i] - 0.75))
    
    mAP_50 = ap_scores[mAP_50_idx] if ap_scores else 0.0
    mAP_75 = ap_scores[mAP_75_idx] if ap_scores else 0.0
    
    return mAP, mAP_50, mAP_75

def calculate_per_class_metrics(all_detections, all_ground_truth, num_classes=26, iou_threshold=0.5):
    """
    Calculate precision, recall, and F1 score for each class.
    """
    per_class_metrics = {}
    
    # Initialize counters for each class (1-26, skip class 0)
    for class_id in range(1, num_classes):
        per_class_metrics[class_id] = {'tp': 0, 'fp': 0, 'fn': 0}
    
    # Count TP, FP, FN for each class
    for detections, ground_truth in zip(all_detections, all_ground_truth):
        matched_det, matched_gt, unmatched_det, unmatched_gt = match_detections_to_ground_truth(
            detections, ground_truth, iou_threshold
        )
        
        # True positives
        for det_idx in matched_det:
            class_id = detections[det_idx]['class']
            if class_id in per_class_metrics:
                per_class_metrics[class_id]['tp'] += 1
        
        # False positives
        for det_idx in unmatched_det:
            class_id = detections[det_idx]['class']
            if class_id in per_class_metrics:
                per_class_metrics[class_id]['fp'] += 1
        
        # False negatives
        for gt_idx in unmatched_gt:
            class_id = ground_truth[gt_idx]['class']
            if class_id in per_class_metrics:
                per_class_metrics[class_id]['fn'] += 1
    
    # Calculate precision, recall, F1 for each class
    results = {}
    for class_id, counts in per_class_metrics.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return results

def evaluate_model(test_dir: pathlib.Path, model_path: pathlib.Path, confidence_threshold: float = 0.5):
    """
    Comprehensive model evaluation with automatic class count detection from checkpoint
    """
    IMAGE_SIZE = (1600,2000)
    
    # Validate paths
    test_dir = pathlib.Path(test_dir)
    model_path = pathlib.Path(model_path)
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory {test_dir} does not exist.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    # STEP 1: Load checkpoint to determine original model configuration
    logger.info("Loading checkpoint to determine model configuration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        
        # CRITICAL: Extract number of classes from the saved model weights
        # Look at the classification layer size to determine the original class count
        if 'classification_to_class.weight' in ckpt['model_state']:
            model_num_classes = ckpt['model_state']['classification_to_class.weight'].shape[0]
            print(f" Model was trained with {model_num_classes} classes")
        else:
            # Fallback - check other possible layer names
            for key in ckpt['model_state'].keys():
                if 'to_class' in key and 'weight' in key:
                    model_num_classes = ckpt['model_state'][key].shape[0]
                    print(f" Model was trained with {model_num_classes} classes (from {key})")
                    break
            else:
                raise ValueError("Could not determine number of classes from model")
        
        # Check if it's an improved model
        use_improved_model = ckpt.get('use_improved_model', True)
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise
    
    # STEP 2: Create model with the ORIGINAL class count from checkpoint
    logger.info(f"Creating model with {model_num_classes} classes...")
    try:
        if use_improved_model:
            model = ImprovedFPN(num_classes=model_num_classes, use_coord=False, use_cbam=True, use_deform=False)
            print(" Created ImprovedFPN model")
        else:
            model = FPN(num_classes=model_num_classes)
            print(" Created standard FPN model")
            
        # Load the state dict
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        logger.info(f" Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Error creating/loading model: {e}")
        raise
    
    # STEP 3: Load dataset after we know the model's class count
    logger.info("Loading test dataset...")
    test_dataset = COCOData(test_dir, image_size=IMAGE_SIZE, min_area=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                           collate_fn=collate_fn, num_workers=2)
    
    dataset_num_classes = test_dataset.get_num_classes()
    class_names = test_dataset.get_class_names()
    
    print(f" Dataset reports {dataset_num_classes} classes")
    print(f" Model expects {model_num_classes} classes")
    
    # STEP 4: Handle class name alignment
    # Ensure class names list matches model expectations
    if len(class_names) != model_num_classes:
        print(f"  Adjusting class names from {len(class_names)} to {model_num_classes}")
        # If dataset has 27 classes but model expects 26, remove the first (background) class
        if len(class_names) == model_num_classes + 1:
            class_names = class_names[1:]  # Remove first class (background)
            print(f" Removed background class, now have {len(class_names)} classes")
        elif len(class_names) < model_num_classes:
            # Pad with dummy names if needed
            while len(class_names) < model_num_classes:
                class_names.append(f"class_{len(class_names)}")
            print(f" Padded class names to {len(class_names)} classes")
    
    # Print class mapping for verification
    print("\n Class mapping verification:")
    for i, name in enumerate(class_names[:min(30, len(class_names))]):  # Show first 10
        print(f"  Model Class {i}: {name}")
    
    
    logger.info(f" Test dataset loaded: {len(test_dataset)} images")
    
    # STEP 5: Compute detections
    logger.info("Computing detections...")
    all_detections, all_ground_truth = compute_detections(
        model, test_loader, device, class_names, IMAGE_SIZE, confidence_threshold, model_num_classes
    )
    
    # STEP 6: Debug class mapping
    print("\n" + "="*80)
    print("DEBUGGING CLASS MAPPING MISMATCH")
    print("="*80)
    
    # Create sample predictions and ground truth for debugging
    sample_predictions = []
    sample_ground_truth = []
    
    for i, (dets, gts) in enumerate(zip(all_detections, all_ground_truth)):
        if i >= 3:  # Only check first 3 images
            break
        
        for det in dets:
            sample_predictions.append({
                'category_id': det['class'],
                'bbox': det['bbox'],
                'score': det['score']
            })
        
        for gt in gts:
            sample_ground_truth.append({
                'category_id': gt['class'],
                'bbox': gt['bbox']
            })
    
    
    # Print detection statistics
    total_detections = sum(len(dets) for dets in all_detections)
    total_ground_truth = sum(len(gt) for gt in all_ground_truth)
    print(f"\nDetection Statistics:")
    print(f"  Total detections: {total_detections}")
    print(f"  Total ground truth: {total_ground_truth}")
    
    # Show class distribution in detections
    det_class_counts = {}
    for detections in all_detections:
        for det in detections:
            class_id = det['class']
            det_class_counts[class_id] = det_class_counts.get(class_id, 0) + 1
    
    print(f"  Detection class distribution: {sorted(det_class_counts.items())}")
    
    # Show class distribution in ground truth
    gt_class_counts = {}
    for ground_truth in all_ground_truth:
        for gt in ground_truth:
            class_id = gt['class']
            gt_class_counts[class_id] = gt_class_counts.get(class_id, 0) + 1
    
    print(f"  Ground truth class distribution: {sorted(gt_class_counts.items())}")
    
    # Calculate overall metrics
    logger.info("Calculating overall metrics...")
    precision, recall, f1_score = calculate_precision_recall(all_detections, all_ground_truth)
    mAP, mAP_50, mAP_75 = calculate_map(all_detections, all_ground_truth)
    
    # Calculate per-class metrics (use model's class count)
    logger.info("Calculating per-class metrics...")
    per_class_metrics = calculate_per_class_metrics(all_detections, all_ground_truth, model_num_classes)
    
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
        'confidence_threshold': confidence_threshold,
        'per_class_metrics': per_class_metrics,
        'detection_class_counts': det_class_counts,
        'ground_truth_class_counts': gt_class_counts,
        'model_num_classes': model_num_classes,
        'dataset_num_classes': dataset_num_classes
    }
    
    return results


def compute_detections(model, test_loader, device, class_names, image_size, confidence_threshold=0.5, model_num_classes=26):
    """
    FIXED: Handle class ID conversion between dataset (1-26) and model (0-25)
    """
    model.eval()
    all_detections = []
    all_ground_truth = []
    
    print(f" Computing detections with confidence threshold: {confidence_threshold}")
    print(f" Model expects {model_num_classes} classes (0-{model_num_classes-1})")
    
    with torch.no_grad():
        for batch_idx, (images, class_labels, box_labels) in enumerate(test_loader):
            images = images.to(device)
            batch_norm = normalize_batch(images)
            
            # Forward pass - handle both standard FPN and ImprovedFPN
            try:
                # Try ImprovedFPN first (returns 3 values)
                cls_pred, box_pred, att_map = model(batch_norm)
            except ValueError:
                # Fallback to standard FPN (returns 2 values)
                cls_pred, box_pred = model(batch_norm)
            
            H, W = images.shape[2], images.shape[3]
            
            detections = detections_from_network_output(
                H, W, cls_pred, box_pred, 
                model.scales, model.strides
            )
            
            for i in range(len(images)):
                image_detections = []
                image_ground_truth = []
                
                # Process detections - model outputs 0-(model_num_classes-1), keep as is
                for det in detections[i]:
                    if hasattr(det, 'score'):
                        score = det.score
                        bbox = det.bbox if hasattr(det, 'bbox') else det.box
                        # Get class ID from detection
                        if hasattr(det, 'object_class'):
                            class_id = det.object_class
                        elif hasattr(det, 'class_id'):
                            class_id = det.class_id
                        else:
                            class_id = getattr(det, 'class', None)
                    else:
                        score = det['score']
                        bbox = det['bbox']
                        class_id = det['class']
                    
                    if score >= confidence_threshold and class_id is not None:
                        # Model outputs 0-(model_num_classes-1), keep as is
                        if 0 <= class_id < model_num_classes:
                            image_detections.append({
                                'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                                'class': class_id,  # Keep model's range
                                'score': score
                            })
                
                # Process ground truth - dataset gives 1-26, convert to 0-25 to match model
                for j in range(len(class_labels[i])):
                    gt_class = class_labels[i][j].item()
                    # Convert from dataset's 1-26 to model's 0-(model_num_classes-1)
                    converted_class = gt_class - 1  # Convert 1-26 to 0-25
                    if 0 <= converted_class < model_num_classes:
                        image_ground_truth.append({
                            'bbox': box_labels[i][j].tolist(),
                            'class': converted_class  # Convert to model's range
                        })
                
                all_detections.append(image_detections)
                all_ground_truth.append(image_ground_truth)
            
            if batch_idx % 10 == 0:
                print(f" Processed batch {batch_idx}")
                # Debug: Print some detection info
                if batch_idx == 0 and len(all_detections) > 0 and len(all_detections[0]) > 0:
                    det = all_detections[0][0]
                    if det['class'] < len(class_names):
                        print(f"  Sample detection: class={det['class']} ({class_names[det['class']]}), score={det['score']:.3f}")
                if batch_idx == 0 and len(all_ground_truth) > 0 and len(all_ground_truth[0]) > 0:
                    gt = all_ground_truth[0][0]
                    if gt['class'] < len(class_names):
                        print(f"  Sample ground truth: class={gt['class']} ({class_names[gt['class']]})")
    
    return all_detections, all_ground_truth

def print_evaluation_results(results: Dict, class_names=None):
    """
    Print formatted evaluation results with per-class breakdown.
    """
    print("\n" + "="*60)
    print("OBJECT DETECTION EVALUATION RESULTS")
    print("="*60)
    
    print(f"Test Images: {results['num_test_images']}")
    print(f"Confidence Threshold: {results['confidence_threshold']:.2f}")
    print(f"Total Detections: {results['total_detections']}")
    print(f"Total Ground Truth: {results['total_ground_truth']}")
    print("-" * 60)
    
    print("OVERALL METRICS:")
    print(f"Precision:    {results['precision']:.4f}")
    print(f"Recall:       {results['recall']:.4f}")
    print(f"F1 Score:     {results['f1_score']:.4f}")
    print("-" * 60)
    
    print(f"mAP:          {results['mAP']:.4f}")
    print(f"mAP@0.5:      {results['mAP_50']:.4f}")
    print(f"mAP@0.75:     {results['mAP_75']:.4f}")
    print("-" * 60)
    
    # Per-class metrics
    print("PER-CLASS METRICS:")
    per_class = results.get('per_class_metrics', {})
    for class_id in sorted(per_class.keys()):
        metrics = per_class[class_id]
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class_{class_id}"
        print(f"Class {class_id} ({class_name}):")
        print(f"  Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
        print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    print("="*60)

if __name__ == "__main__":
    # Set up paths
    test_dir = pathlib.Path(r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\dataset.coco\test") 
    model_path = pathlib.Path(r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\runs\fcos\nocord_cus.pth")
    
    try:
        # Run evaluation with different confidence thresholds
        confidence_thresholds = [0.01, 0.1, 0.2, 0.3, 0.5]
        
        print("Testing different confidence thresholds...")
        
        for conf_thresh in confidence_thresholds:
            print(f"\n{'='*80}")
            print(f"TESTING WITH CONFIDENCE THRESHOLD: {conf_thresh}")
            print(f"{'='*80}")
            
            results = evaluate_model(test_dir, model_path, confidence_threshold=conf_thresh)
            
            # Get class names for better display
            test_dataset = COCOData(test_dir, image_size=(1600,2000), min_area=2)
            class_names = test_dataset.get_class_names()
            
            # Print results
            print_evaluation_results(results, class_names)
            
            # Save results to file
            results_file = model_path.parent / f"evaluation_results_conf_{conf_thresh}.txt"
            with open(results_file, 'w') as f:
                f.write(f"OBJECT DETECTION EVALUATION RESULTS (Confidence: {conf_thresh})\n")
                f.write("="*60 + "\n")
                for key, value in results.items():
                    if key != 'per_class_metrics':
                        f.write(f"{key}: {value}\n")
                    else:
                        f.write("per_class_metrics:\n")
                        for class_id, metrics in value.items():
                            f.write(f"  Class {class_id}: {metrics}\n")
            
            logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise