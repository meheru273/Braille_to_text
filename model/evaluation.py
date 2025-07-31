import pathlib
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Set UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    import io
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Import your existing modules
from Dataset import COCOData, collate_fn
from FPN import FPN, normalize_batch
from FPNAttention import FPN
from inference import detections_from_network_output


class EnhancedMetricsCalculator:
    """Enhanced metrics calculator with detailed analysis capabilities"""
    
    def __init__(self, iou_thresholds: List[float] = None, conf_threshold: float = 0.5):
        if iou_thresholds is None:
            self.iou_thresholds = np.arange(0.5, 1.0, 0.7).tolist()
        else:
            self.iou_thresholds = iou_thresholds
        self.conf_threshold = conf_threshold
        
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_predictions_to_ground_truth(self, 
                                        pred_boxes: np.ndarray, 
                                        pred_scores: np.ndarray,
                                        pred_classes: np.ndarray,
                                        gt_boxes: np.ndarray,
                                        gt_classes: np.ndarray,
                                        iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, int]:
        """Match predictions to ground truth boxes"""
        if len(pred_boxes) == 0:
            return np.array([]), np.array([]), len(gt_boxes)
        
        if len(gt_boxes) == 0:
            return np.zeros(len(pred_boxes)), np.ones(len(pred_boxes)), 0
        
        # Sort predictions by confidence score (descending)
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sorted_indices]
        pred_classes = pred_classes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        
        for pred_idx in range(len(pred_boxes)):
            pred_class = pred_classes[pred_idx]
            pred_box = pred_boxes[pred_idx]
            
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth box
            for gt_idx in range(len(gt_boxes)):
                if gt_matched[gt_idx] or gt_classes[gt_idx] != pred_class:
                    continue
                
                iou = self.calculate_iou(pred_box, gt_boxes[gt_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is good enough
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[pred_idx] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[pred_idx] = 1
        
        return tp, fp, len(gt_boxes)
    
    def calculate_precision_recall_curve(self, 
                                       tp: np.ndarray, 
                                       fp: np.ndarray, 
                                       num_gt: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate precision-recall curve"""
        if len(tp) == 0:
            return np.array([1.0, 0.0]), np.array([0.0, 0.0])
        
        # Cumulative true positives and false positives
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Calculate precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / (num_gt + 1e-8)
        
        # Add point (0, 1) at the beginning
        precision = np.concatenate(([1.0], precision))
        recall = np.concatenate(([0.0], recall))
        
        return precision, recall
    
    def calculate_ap(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """Calculate Average Precision using 101-point interpolation"""
        recall_levels = np.linspace(0, 1, 101)
        ap = 0.0
        
        for r in recall_levels:
            valid_precisions = precision[recall >= r]
            if len(valid_precisions) > 0:
                ap += np.max(valid_precisions)
        
        return ap / len(recall_levels)
    
    def evaluate_single_class(self, 
                             predictions: List[Dict], 
                             ground_truths: List[Dict],
                             class_id: int,
                             iou_threshold: float = 0.5) -> Dict:
        """Evaluate metrics for a single class with detailed statistics"""
        
        # Collect all predictions and ground truths for this class
        pred_by_image = defaultdict(list)
        gt_by_image = defaultdict(list)
        
        for pred in predictions:
            if pred['class_id'] == class_id:
                pred_by_image[pred['image_id']].append(pred)
        
        for gt in ground_truths:
            if gt['class_id'] == class_id:
                gt_by_image[gt['image_id']].append(gt)
        
        # Get all unique image IDs
        all_image_ids = set(list(pred_by_image.keys()) + list(gt_by_image.keys()))
        
        tp_list = []
        fp_list = []
        scores_list = []
        total_gt = 0
        
        for img_id in all_image_ids:
            img_preds = pred_by_image[img_id]
            img_gts = gt_by_image[img_id]
            
            if len(img_preds) == 0:
                total_gt += len(img_gts)
                continue
            
            pred_boxes = np.array([p['bbox'] for p in img_preds])
            pred_scores = np.array([p['score'] for p in img_preds])
            pred_classes = np.array([p['class_id'] for p in img_preds])
            
            if len(img_gts) > 0:
                gt_boxes = np.array([g['bbox'] for g in img_gts])
                gt_classes = np.array([g['class_id'] for g in img_gts])
            else:
                gt_boxes = np.empty((0, 4))
                gt_classes = np.array([])
            
            tp, fp, num_gt = self.match_predictions_to_ground_truth(
                pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold
            )
            
            tp_list.extend(tp)
            fp_list.extend(fp)
            scores_list.extend(pred_scores)
            total_gt += num_gt
        
        if len(tp_list) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'ap': 0.0,
                'num_predictions': 0,
                'num_ground_truth': total_gt,
                'tp': 0,
                'fp': 0,
                'fn': total_gt,
                'score_stats': {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            }
        
        tp_array = np.array(tp_list)
        fp_array = np.array(fp_list)
        scores_array = np.array(scores_list)
        
        # Sort by scores
        sorted_indices = np.argsort(scores_array)[::-1]
        tp_sorted = tp_array[sorted_indices]
        fp_sorted = fp_array[sorted_indices]
        
        # Calculate precision-recall curve
        precision, recall = self.calculate_precision_recall_curve(tp_sorted, fp_sorted, total_gt)
        
        # Calculate AP
        ap = self.calculate_ap(precision, recall)
        
        # Calculate precision, recall, F1 at confidence threshold
        total_tp = np.sum(tp_array)
        total_fp = np.sum(fp_array)
        total_fn = total_gt - total_tp
        
        precision_at_threshold = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall_at_threshold = total_tp / total_gt if total_gt > 0 else 0.0
        f1_at_threshold = (2 * precision_at_threshold * recall_at_threshold) / \
                         (precision_at_threshold + recall_at_threshold) if \
                         (precision_at_threshold + recall_at_threshold) > 0 else 0.0
        
        # Score statistics
        score_stats = {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array))
        }
        
        return {
            'precision': precision_at_threshold,
            'recall': recall_at_threshold,
            'f1': f1_at_threshold,
            'ap': ap,
            'num_predictions': len(tp_list),
            'num_ground_truth': total_gt,
            'tp': int(total_tp),
            'fp': int(total_fp),
            'fn': int(total_fn),
            'score_stats': score_stats
        }

    def analyze_class_distribution(self, predictions: List[Dict], ground_truths: List[Dict], 
                                 class_names: List[str]) -> Dict:
        """Analyze class distribution and imbalance"""
        pred_counts = defaultdict(int)
        gt_counts = defaultdict(int)
        
        for pred in predictions:
            pred_counts[pred['class_id']] += 1
            
        for gt in ground_truths:
            gt_counts[gt['class_id']] += 1
        
        # Calculate ratios and statistics
        class_analysis = {}
        for class_id in range(len(class_names)):
            pred_count = pred_counts[class_id]
            gt_count = gt_counts[class_id]
            
            ratio = pred_count / gt_count if gt_count > 0 else float('inf')
            
            class_analysis[class_id] = {
                'name': class_names[class_id],
                'predictions': pred_count,
                'ground_truth': gt_count,
                'prediction_ratio': ratio,
                'is_overdetected': ratio > 2.0,
                'is_underdetected': ratio < 0.5 and gt_count > 0
            }
        
        return class_analysis

    def analyze_confidence_distribution(self, predictions: List[Dict]) -> Dict:
        """Analyze confidence score distribution"""
        scores = [pred['score'] for pred in predictions]
        
        if not scores:
            return {'count': 0}
        
        scores = np.array(scores)
        
        # Calculate percentiles and statistics
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(scores, percentiles)
        
        # Count predictions by confidence ranges
        ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        range_counts = {}
        for low, high in ranges:
            count = np.sum((scores >= low) & (scores < high))
            range_counts[f'{low}-{high}'] = int(count)
        
        return {
            'count': len(scores),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'percentiles': dict(zip(percentiles, percentile_values)),
            'range_counts': range_counts
        }


def load_model(model_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """Load a trained model from checkpoint with proper configuration handling"""
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract state dict and config
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"Loading model from {model_path}")
    print(f"State dict has {len(state_dict)} parameters")
    
    # Try to get config from checkpoint first
    config = checkpoint.get('config', None) if isinstance(checkpoint, dict) else None
    
    if config:
        print(f"Found saved configuration: {config}")
        # Use saved configuration
        model = FPN(
            num_classes=num_classes,
            use_coord=config.get('use_coord', False),
            use_cbam=config.get('use_cbam', False),
            use_deform=config.get('use_deform', False)
        )
        print(f"Created FPN with saved config: coord={config.get('use_coord')}, cbam={config.get('use_cbam')}, deform={config.get('use_deform')}")
    else:
        # Fallback to auto-detection with improved logic
        state_dict_keys = list(state_dict.keys())
        
        # More robust feature detection
        has_cbam = any('channel_attention.fc' in key and 'weight' in key for key in state_dict_keys)
        has_coord = any(('conv_h' in key or 'conv_w' in key) and 'weight' in key for key in state_dict_keys)
        has_feature_fusion = any('feature_fusion' in key and 'conv' in key for key in state_dict_keys)
        
        print(f"Auto-detected features: CBAM={has_cbam}, CoordAtt={has_coord}, FeatureFusion={has_feature_fusion}")
        
        if has_cbam or has_coord or has_feature_fusion:
            model = FPN(
                num_classes=num_classes,
                use_coord=has_coord,
                use_cbam=has_cbam,
                use_deform=False
            )
            print(f"Created FPN with auto-detection: coord={has_coord}, cbam={has_cbam}")
        else:
            # Fallback to baseline FPN
            from FPN import FPN
            model = FPN(num_classes=num_classes)
            print("Created baseline FPN")
    
    # Load state dict with proper error handling
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ Model loaded successfully with strict=True")
    except RuntimeError as e:
        print(f"⚠ Strict loading failed: {str(e)}")
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
            print("✓ Model loaded with strict=False")
        except Exception as e2:
            print(f"✗ Failed to load model: {str(e2)}")
            raise e2
    
    model.to(device)
    model.eval()
    
    return model


def get_predictions_from_model_with_analysis(model: torch.nn.Module, 
                                           dataloader: DataLoader, 
                                           device: torch.device,
                                           conf_thresholds: List[float] = [0.9, 0.5, 0.7],
                                           nms_iou_threshold: float = 0.8) -> Dict:
    """Get predictions and ground truths from model with multiple confidence thresholds"""
    
    all_predictions = []  # Store all predictions without confidence filtering
    ground_truths = []
    
    print(f"Testing with confidence thresholds: {conf_thresholds}, NMS IoU threshold: {nms_iou_threshold}")
    
    with torch.no_grad():
        for batch_idx, (images, class_labels, box_labels) in enumerate(dataloader):
            try:
                images = images.to(device)
                normalized_images = normalize_batch(images)
                model_outputs = model(normalized_images)
                
                # Handle different output formats
                if len(model_outputs) == 2:
                    cls_pred, box_pred = model_outputs
                elif len(model_outputs) == 3:
                    cls_pred, box_pred, attention_maps = model_outputs
                else:
                    raise ValueError(f"Unexpected number of model outputs: {len(model_outputs)}")
                
                # Process each image in batch
                for img_idx in range(len(images)):
                    image_id = batch_idx * len(images) + img_idx
                    H, W = images[img_idx].shape[1], images[img_idx].shape[2]
                    
                    try:
                        detections = detections_from_network_output(
                            H, W, cls_pred, box_pred, 
                            model.scales, model.strides
                        )
                        
                        # Add ground truths
                        if img_idx < len(class_labels) and img_idx < len(box_labels):
                            img_class_labels = class_labels[img_idx]
                            img_box_labels = box_labels[img_idx]
                            for class_id, bbox in zip(img_class_labels, img_box_labels):
                                if class_id >= 0:
                                    if torch.is_tensor(bbox):
                                        bbox = bbox.cpu().detach().numpy()
                                    elif hasattr(bbox, 'cpu'):
                                        bbox = bbox.cpu().detach().numpy()
                                    ground_truths.append({
                                        'image_id': image_id,
                                        'class_id': int(class_id),
                                        'bbox': bbox.astype(float)
                                    })

                        # Store all predictions (no confidence filtering yet)
                        if img_idx < len(detections):
                            img_detections = detections[img_idx]
                            for detection in img_detections:
                                bbox = detection.bbox
                                if torch.is_tensor(bbox):
                                    bbox = bbox.cpu().detach().numpy()
                                elif hasattr(bbox, 'cpu'):
                                    bbox = bbox.cpu().detach().numpy()
                                all_predictions.append({
                                    'image_id': image_id,
                                    'class_id': int(detection.object_class),
                                    'bbox': bbox.astype(float),
                                    'score': float(detection.score)
                                })
                                    
                    except Exception as e:
                        print(f"Error processing image {img_idx} in batch {batch_idx}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Now filter and apply NMS for each confidence threshold
    results = {}
    for conf_threshold in conf_thresholds:
        # Filter by confidence
        filtered_predictions = [p for p in all_predictions if p['score'] >= conf_threshold]
        
        # Apply NMS per image
        predictions_with_nms = []
        images_with_predictions = defaultdict(list)
        
        # Group predictions by image
        for pred in filtered_predictions:
            images_with_predictions[pred['image_id']].append(pred)
        
        # Apply NMS per image
        for image_id, img_predictions in images_with_predictions.items():
            if not img_predictions:
                continue
                
            # Sort by confidence
            img_predictions.sort(key=lambda x: x['score'], reverse=True)
            
            # Apply NMS
            keep = []
            while img_predictions:
                current = img_predictions.pop(0)
                keep.append(current)
                
                if not img_predictions:
                    break
                
                # Calculate IoU with remaining predictions
                remaining = []
                for pred in img_predictions:
                    iou = calculate_iou(current['bbox'], pred['bbox'])
                    if iou < nms_iou_threshold:
                        remaining.append(pred)
                
                img_predictions = remaining
            
            predictions_with_nms.extend(keep)
        
        results[conf_threshold] = {
            'predictions': predictions_with_nms,
            'ground_truths': ground_truths
        }
    
    return results


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes for NMS."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def create_analysis_plots(results: Dict, output_dir: str, model_name: str):
    """Create analysis plots for the evaluation results"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Evaluation Analysis: {model_name}', fontsize=16)
    
    # Plot 1: Confidence threshold vs metrics
    conf_thresholds = sorted(results.keys())
    metrics = ['precision', 'recall', 'f1', 'mAP@50']
    
    for metric in metrics:
        values = [results[conf]['summary'][metric] for conf in conf_thresholds]
        axes[0, 0].plot(conf_thresholds, values, marker='o', label=metric)
    
    axes[0, 0].set_xlabel('Confidence Threshold')
    axes[0, 0].set_ylabel('Metric Value')
    axes[0, 0].set_title('Metrics vs Confidence Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Number of predictions vs confidence threshold
    pred_counts = [results[conf]['summary']['num_predictions'] for conf in conf_thresholds]
    gt_count = results[conf_thresholds[0]]['summary']['num_ground_truth']
    
    axes[0, 1].bar(range(len(conf_thresholds)), pred_counts, alpha=0.7, label='Predictions')
    axes[0, 1].axhline(y=gt_count, color='red', linestyle='--', label=f'Ground Truth ({gt_count})')
    axes[0, 1].set_xlabel('Confidence Threshold')
    axes[0, 1].set_ylabel('Number of Detections')
    axes[0, 1].set_title('Predictions vs Ground Truth')
    axes[0, 1].set_xticks(range(len(conf_thresholds)))
    axes[0, 1].set_xticklabels([f'{conf:.1f}' for conf in conf_thresholds])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Class distribution (for best confidence threshold)
    best_conf = conf_thresholds[len(conf_thresholds)//2]  # Use middle threshold
    class_analysis = results[best_conf]['class_analysis']
    
    class_ids = list(class_analysis.keys())[:20]  # Show top 20 classes
    pred_counts = [class_analysis[cid]['predictions'] for cid in class_ids]
    gt_counts = [class_analysis[cid]['ground_truth'] for cid in class_ids]
    
    x = np.arange(len(class_ids))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, pred_counts, width, label='Predictions', alpha=0.7)
    axes[1, 0].bar(x + width/2, gt_counts, width, label='Ground Truth', alpha=0.7)
    axes[1, 0].set_xlabel('Class ID')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Class Distribution (Top 20, conf={best_conf})')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_ids, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Confidence score distribution
    conf_analysis = results[best_conf]['confidence_analysis']
    if 'range_counts' in conf_analysis:
        ranges = list(conf_analysis['range_counts'].keys())
        counts = list(conf_analysis['range_counts'].values())
        
        axes[1, 1].bar(ranges, counts, alpha=0.7)
        axes[1, 1].set_xlabel('Confidence Range')
        axes[1, 1].set_ylabel('Number of Predictions')
        axes[1, 1].set_title('Confidence Score Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{model_name}_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Analysis plots saved to {plot_path}")


def evaluate_model_comprehensive(model_path: str, 
                               dataset_path: str, 
                               device: torch.device,
                               image_size: Tuple[int, int] = (1600, 2000),
                               conf_thresholds: List[float] = [0.3, 0.5, 0.7]) -> Dict:
    """Comprehensive model evaluation with multiple confidence thresholds"""
    
    print(f"\nComprehensive evaluation of {os.path.basename(model_path)} on {os.path.basename(os.path.dirname(dataset_path))}")
    
    try:
        # Load dataset
        dataset = COCOData(pathlib.Path(dataset_path), image_size=image_size, min_area=2)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        num_classes = dataset.get_num_classes()
        class_names = dataset.get_class_names()
        
        print(f"Dataset loaded: {len(dataset)} images, {num_classes} classes")
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise e
    
    # Load model
    model = load_model(model_path, num_classes, device)
    
    # Get predictions for all confidence thresholds
    all_results = get_predictions_from_model_with_analysis(
        model, dataloader, device, conf_thresholds
    )
    
    # Initialize enhanced metrics calculator
    metrics_calc = EnhancedMetricsCalculator()
    
    # Evaluate for each confidence threshold
    comprehensive_results = {}
    
    for conf_threshold in conf_thresholds:
        print(f"\n--- Evaluating at confidence threshold {conf_threshold} ---")
        
        predictions = all_results[conf_threshold]['predictions']
        ground_truths = all_results[conf_threshold]['ground_truths']
        
        print(f"Predictions: {len(predictions)}, Ground truths: {len(ground_truths)}")
        
        # Calculate metrics for each IoU threshold
        iou_results = {}
        for iou_name, iou_thresh in [('mAP@50', 0.5), ('mAP@70', 0.7)]:
            class_aps = []
            class_precisions = []
            class_recalls = []
            class_f1s = []
            detailed_class_metrics = {}
            
            for class_id in range(num_classes):
                class_metrics = metrics_calc.evaluate_single_class(
                    predictions, ground_truths, class_id, iou_thresh
                )
                
                class_aps.append(class_metrics['ap'])
                class_precisions.append(class_metrics['precision'])
                class_recalls.append(class_metrics['recall'])
                class_f1s.append(class_metrics['f1'])
                detailed_class_metrics[class_id] = class_metrics
            
            iou_results[iou_name] = {
                'map': np.mean(class_aps) if class_aps else 0.0,
                'precision': np.mean(class_precisions) if class_precisions else 0.0,
                'recall': np.mean(class_recalls) if class_recalls else 0.0,
                'f1': np.mean(class_f1s) if class_f1s else 0.0,
                'class_metrics': detailed_class_metrics
            }
        
        # Analyze class distribution and confidence scores
        class_analysis = metrics_calc.analyze_class_distribution(predictions, ground_truths, class_names)
        confidence_analysis = metrics_calc.analyze_confidence_distribution(predictions)
        
        # Overall summary using IoU@0.5
        summary = {
            'precision': iou_results['mAP@50']['precision'],
            'recall': iou_results['mAP@50']['recall'],
            'f1': iou_results['mAP@50']['f1'],
            'mAP@50': iou_results['mAP@50']['map'],
            'mAP@70': iou_results['mAP@70']['map'],
            'num_predictions': len(predictions),
            'num_ground_truth': len(ground_truths),
            'prediction_ratio': len(predictions) / len(ground_truths) if len(ground_truths) > 0 else float('inf')
        }
        
        comprehensive_results[conf_threshold] = {
            'summary': summary,
            'iou_results': iou_results,
            'class_analysis': class_analysis,
            'confidence_analysis': confidence_analysis,
            'predictions': predictions,
            'ground_truths': ground_truths
        }
        
        print(f"✓ Results at threshold {conf_threshold}:")
        print(f"  - Precision: {summary['precision']:.4f}")
        print(f"  - Recall: {summary['recall']:.4f}")
        print(f"  - F1-Score: {summary['f1']:.4f}")
        print(f"  - mAP@50: {summary['mAP@50']:.4f}")
        print(f"  - mAP@70: {summary['mAP@70']:.4f}")
        print(f"  - Prediction ratio: {summary['prediction_ratio']:.2f}")
    
    return comprehensive_results


def print_detailed_analysis(results: Dict, class_names: List[str], model_name: str):
    """Print detailed analysis of the evaluation results"""
    
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: {model_name}")
    print(f"{'='*80}")
    
    # Find best performing confidence threshold
    best_conf = None
    best_f1 = 0
    for conf, result in results.items():
        if result['summary']['f1'] > best_f1:
            best_f1 = result['summary']['f1']
            best_conf = conf
    
    print(f"\nBest performing confidence threshold: {best_conf} (F1: {best_f1:.4f})")
    
    # Analyze prediction ratios across thresholds
    print(f"\n--- Prediction Ratios by Confidence Threshold ---")
    for conf in sorted(results.keys()):
        ratio = results[conf]['summary']['prediction_ratio']
        status = "BALANCED" if 0.8 <= ratio <= 1.5 else "IMBALANCED"
        print(f"Conf {conf:.1f}: {results[conf]['summary']['num_predictions']:6d} preds / "
              f"{results[conf]['summary']['num_ground_truth']:6d} GT = {ratio:6.2f} ({status})")
    
    # Analyze class-level performance
    best_result = results[best_conf]
    class_analysis = best_result['class_analysis']
    
    print(f"\n--- Class-Level Analysis (at best threshold {best_conf}) ---")
    
    # Find problematic classes
    overdetected = []
    underdetected = []
    well_balanced = []
    
    for class_id, analysis in class_analysis.items():
        if analysis['ground_truth'] == 0:
            continue
            
        if analysis['is_overdetected']:
            overdetected.append((class_id, analysis))
        elif analysis['is_underdetected']:
            underdetected.append((class_id, analysis))
        else:
            well_balanced.append((class_id, analysis))
    
    print(f"\nOverdetected classes ({len(overdetected)}):")
    for class_id, analysis in sorted(overdetected, key=lambda x: x[1]['prediction_ratio'], reverse=True)[:10]:
        print(f"  Class {class_id:2d} ({analysis['name'][:20]:20s}): "
              f"{analysis['predictions']:4d} preds / {analysis['ground_truth']:4d} GT = "
              f"{analysis['prediction_ratio']:6.2f}x")
    
    print(f"\nUnderdetected classes ({len(underdetected)}):")
    for class_id, analysis in sorted(underdetected, key=lambda x: x[1]['prediction_ratio'])[:10]:
        print(f"  Class {class_id:2d} ({analysis['name'][:20]:20s}): "
              f"{analysis['predictions']:4d} preds / {analysis['ground_truth']:4d} GT = "
              f"{analysis['prediction_ratio']:6.2f}x")
    
    # Analyze confidence score distribution
    conf_analysis = best_result['confidence_analysis']
    print(f"\n--- Confidence Score Analysis ---")
    print(f"Total predictions: {conf_analysis['count']}")
    print(f"Mean confidence: {conf_analysis['mean']:.4f}")
    print(f"Std confidence: {conf_analysis['std']:.4f}")
    print(f"Min confidence: {conf_analysis['min']:.4f}")
    print(f"Max confidence: {conf_analysis['max']:.4f}")
    
    if 'percentiles' in conf_analysis:
        print(f"Confidence percentiles:")
        for p, v in conf_analysis['percentiles'].items():
            print(f"  {p:2d}th: {v:.4f}")
    
    # Performance recommendations
    print(f"\n--- RECOMMENDATIONS ---")
    
    best_ratio = best_result['summary']['prediction_ratio']
    if best_ratio > 2.0:
        print("⚠ HIGH FALSE POSITIVE RATE detected!")
        print("  - Consider increasing confidence threshold")
        print("  - Review NMS IoU threshold (current: 0.5)")
        print("  - Check for model overfitting")
        
    elif best_ratio < 0.5:
        print("⚠ HIGH FALSE NEGATIVE RATE detected!")
        print("  - Consider decreasing confidence threshold")
        print("  - Review detection head sensitivity")
        print("  - Check for class imbalance in training data")
        
    else:
        print("✓ Prediction ratio is reasonably balanced")
    
    if best_result['summary']['mAP@50'] < 0.3:
        print("⚠ LOW mAP@50 detected!")
        print("  - Model may need more training")
        print("  - Check data quality and annotations")
        print("  - Consider data augmentation")
    
    if len(overdetected) > len(class_names) * 0.3:
        print("⚠ Many classes are overdetected!")
        print("  - Review class balance in training data")
        print("  - Consider class-specific confidence thresholds")
    
    print(f"\n{'='*80}")


def save_comprehensive_results(results: Dict, model_name: str, dataset_name: str, output_dir: str):
    """Save comprehensive results to files"""
    
    # Create summary DataFrame
    summary_data = []
    for conf_threshold, result in results.items():
        summary = result['summary']
        summary_data.append({
            'Model': model_name,
            'Dataset': dataset_name,
            'Confidence_Threshold': conf_threshold,
            'Precision': summary['precision'],
            'Recall': summary['recall'],
            'F1_Score': summary['f1'],
            'mAP@50': summary['mAP@50'],
            'mAP@70': summary['mAP@70'],
            'Num_Predictions': summary['num_predictions'],
            'Num_Ground_Truth': summary['num_ground_truth'],
            'Prediction_Ratio': summary['prediction_ratio']
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save summary CSV
    summary_path = os.path.join(output_dir, f'{model_name}_{dataset_name}_comprehensive_results.csv')
    df_summary.to_csv(summary_path, index=False)
    
    # Create class-level analysis for best threshold
    best_conf = df_summary.loc[df_summary['F1_Score'].idxmax(), 'Confidence_Threshold']
    class_analysis = results[best_conf]['class_analysis']
    
    class_data = []
    for class_id, analysis in class_analysis.items():
        class_data.append({
            'Class_ID': class_id,
            'Class_Name': analysis['name'],
            'Predictions': analysis['predictions'],
            'Ground_Truth': analysis['ground_truth'],
            'Prediction_Ratio': analysis['prediction_ratio'],
            'Is_Overdetected': analysis['is_overdetected'],
            'Is_Underdetected': analysis['is_underdetected']
        })
    
    df_classes = pd.DataFrame(class_data)
    class_path = os.path.join(output_dir, f'{model_name}_{dataset_name}_class_analysis.csv')
    df_classes.to_csv(class_path, index=False)
    
    # Save detailed JSON results
    json_path = os.path.join(output_dir, f'{model_name}_{dataset_name}_detailed_results.json')
    
    # Prepare JSON-serializable results
    json_results = {}
    for conf_threshold, result in results.items():
        json_results[str(conf_threshold)] = {
            'summary': result['summary'],
            'class_analysis': {str(k): v for k, v in result['class_analysis'].items()},
            'confidence_analysis': result['confidence_analysis']
        }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"✓ Results saved:")
    print(f"  - Summary: {summary_path}")
    print(f"  - Class analysis: {class_path}")
    print(f"  - Detailed JSON: {json_path}")
    
    return summary_path, class_path, json_path


def main():
    """Enhanced main evaluation function"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_path = r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch"
    runs_path = os.path.join(base_path, "runs")
    
    # Model paths
    models = {
        'baseline_dsbi': os.path.join(runs_path, 'baseline_all_false_dsbi.pth'),
        'cbam_only_dsbi': os.path.join(runs_path, 'cbam_only_dsbi.pth'),
        'coord_cbam_dsbi': os.path.join(runs_path, 'coord_cbam_dsbi.pth'),
        'baseline_custom' : os.path.join(runs_path,'baseline_all_false_custom.pth')
    }
    
    # Dataset paths
    datasets = {
        'dsbi': os.path.join(base_path, 'dsbi.coco', 'valid'),
        'custom': os.path.join(base_path, 'custom.coco', 'valid')
    }
    
    # Check available models and datasets
    available_models = {k: v for k, v in models.items() if os.path.exists(v)}
    available_datasets = {k: v for k, v in datasets.items() if os.path.exists(v)}
    
    print(f"Found {len(available_models)} models, {len(available_datasets)} datasets")
    
    if not available_models or not available_datasets:
        print("No models or datasets found!")
        return
    
    # Create output directory
    output_dir = os.path.join(base_path, 'comprehensive_evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Confidence thresholds to test
    conf_thresholds =  [0.8, 0.9, 0.95]
    
    # Results storage
    all_results = {}
    
    # Evaluate models
    for model_name, model_path in available_models.items():
        model_results = {}
        
        # Determine which datasets to test
        if 'dsbi' in model_name and 'dsbi' in available_datasets:
            datasets_to_test = {'dsbi': available_datasets['dsbi']}
        elif 'custom' in model_name and 'custom' in available_datasets:
            datasets_to_test = {'custom': available_datasets['custom']}
        else:
            datasets_to_test = available_datasets
        
        for dataset_name, dataset_path in datasets_to_test.items():
            print(f"\n{'='*100}")
            print(f"COMPREHENSIVE EVALUATION: {model_name} on {dataset_name}")
            print(f"{'='*100}")
            
            try:
                # Comprehensive evaluation
                results = evaluate_model_comprehensive(
                    model_path, dataset_path, device, conf_thresholds=conf_thresholds
                )
                
                model_results[dataset_name] = results
                
                # Print detailed analysis
                class_names = list(results[conf_thresholds[0]]['class_analysis'].values())[0]['name'] if results else []
                if results:
                    # Get class names properly
                    try:
                        dataset = COCOData(pathlib.Path(dataset_path), image_size=(1600, 2000), min_area=2)
                        class_names = dataset.get_class_names()
                    except:
                        class_names = [f"Class_{i}" for i in range(64)]  # Fallback
                    
                    print_detailed_analysis(results, class_names, f"{model_name}_{dataset_name}")
                
                # Save results and create plots
                save_comprehensive_results(results, model_name, dataset_name, output_dir)
                create_analysis_plots(results, output_dir, f"{model_name}_{dataset_name}")
                
            except Exception as e:
                print(f"✗ Error evaluating {model_name} on {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if model_results:
            all_results[model_name] = model_results
    
    # Create overall comparison
    if all_results:
        print(f"\n{'='*120}")
        print("OVERALL COMPARISON - BEST PERFORMANCE PER MODEL")
        print("="*120)
        
        comparison_data = []
        for model_name, model_results in all_results.items():
            for dataset_name, results in model_results.items():
                # Find best F1 score configuration
                best_conf = None
                best_f1 = 0
                for conf, result in results.items():
                    if result['summary']['f1'] > best_f1:
                        best_f1 = result['summary']['f1']
                        best_conf = conf
                
                if best_conf is not None:
                    best_result = results[best_conf]['summary']
                    comparison_data.append({
                        'Model': model_name,
                        'Dataset': dataset_name,
                        'Best_Conf_Threshold': best_conf,
                        'Precision': best_result['precision'],
                        'Recall': best_result['recall'],
                        'F1_Score': best_result['f1'],
                        'mAP@50': best_result['mAP@50'],
                        'mAP@70': best_result['mAP@70'],
                        'Prediction_Ratio': best_result['prediction_ratio']
                    })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            print(df_comparison.to_string(index=False, float_format='%.4f'))
            
            # Save comparison
            comparison_path = os.path.join(output_dir, 'model_comparison_best_performance.csv')
            df_comparison.to_csv(comparison_path, index=False)
            print(f"\n✓ Comparison saved to {comparison_path}")
        
        print(f"\n✓ All results saved in: {output_dir}")


if __name__ == "__main__":
    main()