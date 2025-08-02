import pathlib
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import pandas as pd
from collections import defaultdict

from Dataset import COCOData, collate_fn
from FPNAttention import FPN, normalize_batch
from inference import detections_from_network_output


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def load_model(model_path: str, num_classes: int, device: torch.device):
    """Load model with auto-detection of attention features"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
    
    # Auto-detect features from model name and state dict
    model_name = os.path.basename(model_path).lower()
    state_keys = list(state_dict.keys())
    
    use_cbam = 'cbam' in model_name or any('channel_attention' in k for k in state_keys)
    use_coord = 'coord' in model_name or any('conv_h' in k or 'conv_w' in k for k in state_keys)
    use_pos = 'pos' in model_name or any('pos_attention' in k for k in state_keys)
    
    if 'baseline' in model_name and 'all_false' in model_name:
        use_cbam = use_coord = use_pos = False
    
    print(f"Loading {model_name}: coord={use_coord}, cbam={use_cbam}, pos={use_pos}")
    
    model = FPN(num_classes=num_classes, use_coord=use_coord, use_cbam=use_cbam, use_pos=use_pos)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    
    return model


def get_predictions(model, dataloader, device, conf_threshold=0.5):
    """Get predictions and ground truths from model"""
    predictions, ground_truths = [], []
    
    with torch.no_grad():
        for batch_idx, (images, class_labels, box_labels) in enumerate(dataloader):
            images = normalize_batch(images.to(device))
            outputs = model(images)
            cls_pred, box_pred = outputs[:2]  # Handle 2 or 3 outputs
            
            for img_idx in range(len(images)):
                H, W = images[img_idx].shape[1:3]
                image_id = batch_idx * len(images) + img_idx
                
                # Get detections
                detections = detections_from_network_output(H, W, cls_pred, box_pred, model.scales, model.strides)
                
                # Add predictions
                if img_idx < len(detections):
                    for det in detections[img_idx]:
                        if det.score >= conf_threshold:
                            bbox = det.bbox.cpu().numpy() if torch.is_tensor(det.bbox) else det.bbox
                            predictions.append({
                                'image_id': image_id,
                                'class_id': int(det.object_class),
                                'bbox': bbox.astype(float),
                                'score': float(det.score)
                            })
                
                # Add ground truths
                if img_idx < len(class_labels):
                    for cls_id, bbox in zip(class_labels[img_idx], box_labels[img_idx]):
                        if cls_id >= 0:
                            bbox = bbox.cpu().numpy() if torch.is_tensor(bbox) else bbox
                            ground_truths.append({
                                'image_id': image_id,
                                'class_id': int(cls_id),
                                'bbox': bbox.astype(float)
                            })
                            
            # In get_predictions_from_model_with_analysis, after getting predictions
            sample_pred = predictions[0] if predictions else None
            if sample_pred:
                print("\nDEBUG: Sample prediction analysis")
                print(f"Image ID: {sample_pred['image_id']}")
                print(f"Class ID: {sample_pred['class_id']}")
                print(f"Confidence: {sample_pred['score']:.4f}")
                print(f"Bounding box: {sample_pred['bbox']}")
                print(f"Box width: {sample_pred['bbox'][2]-sample_pred['bbox'][0]:.2f} pixels")
                print(f"Box height: {sample_pred['bbox'][3]-sample_pred['bbox'][1]:.2f} pixels")
                
                return predictions, ground_truths


def calculate_metrics(predictions, ground_truths, num_classes, iou_threshold=0.5):
    """Calculate precision, recall, F1, and mAP"""
    class_metrics = []
    
    for class_id in range(num_classes):
        # Filter by class
        cls_preds = [p for p in predictions if p['class_id'] == class_id]
        cls_gts = [g for g in ground_truths if g['class_id'] == class_id]
        
        if not cls_gts:
            class_metrics.append({'precision': 0, 'recall': 0, 'f1': 0, 'ap': 0})
            continue
        
        if not cls_preds:
            class_metrics.append({'precision': 0, 'recall': 0, 'f1': 0, 'ap': 0})
            continue
        
        # Group by image
        pred_by_img = defaultdict(list)
        gt_by_img = defaultdict(list)
        for p in cls_preds:
            pred_by_img[p['image_id']].append(p)
        for g in cls_gts:
            gt_by_img[g['image_id']].append(g)
        
        tp, fp = [], []
        all_images = set(list(pred_by_img.keys()) + list(gt_by_img.keys()))
        
        for img_id in all_images:
            img_preds = pred_by_img[img_id]
            img_gts = gt_by_img[img_id]
            
            if not img_preds:
                continue
            
            # Sort by confidence
            img_preds.sort(key=lambda x: x['score'], reverse=True)
            gt_matched = [False] * len(img_gts)
            
            for pred in img_preds:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(img_gts):
                    if gt_matched[gt_idx]:
                        continue
                    iou = calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    tp.append(1)
                    fp.append(0)
                    gt_matched[best_gt_idx] = True
                else:
                    tp.append(0)
                    fp.append(1)
        
        # Calculate metrics
        if not tp:
            class_metrics.append({'precision': 0, 'recall': 0, 'f1': 0, 'ap': 0})
            continue
        
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precision = tp_cum / (tp_cum + fp_cum + 1e-8)
        recall = tp_cum / len(cls_gts)
        
        # AP calculation (simplified)
        ap = np.mean(precision) if len(precision) > 0 else 0
        
        # Final metrics
        final_precision = precision[-1] if len(precision) > 0 else 0
        final_recall = recall[-1] if len(recall) > 0 else 0
        f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-8)
        
        class_metrics.append({
            'precision': final_precision,
            'recall': final_recall,
            'f1': f1,
            'ap': ap
        })
    
    # Overall metrics
    overall = {
        'precision': np.mean([m['precision'] for m in class_metrics]),
        'recall': np.mean([m['recall'] for m in class_metrics]),
        'f1': np.mean([m['f1'] for m in class_metrics]),
        'mAP': np.mean([m['ap'] for m in class_metrics])
    }
    
    return overall, class_metrics


def evaluate_model(model_path: str, dataset_path: str, device: torch.device, 
                  conf_thresholds: List[float] = [0.3, 0.5, 0.7]):
    """Evaluate model on dataset with multiple confidence thresholds"""
    
    # Load dataset
    dataset = COCOData(pathlib.Path(dataset_path), image_size=(1600, 2000), min_area=2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    num_classes = dataset.get_num_classes()
    model = load_model(model_path, num_classes, device)
    
    results = {}
    
    for conf_threshold in conf_thresholds:
        print(f"Evaluating at confidence threshold {conf_threshold}")
        
        predictions, ground_truths = get_predictions(model, dataloader, device, conf_threshold)
        overall, class_metrics = calculate_metrics(predictions, ground_truths, num_classes)
        
        results[conf_threshold] = {
            'overall': overall,
            'num_predictions': len(predictions),
            'num_ground_truth': len(ground_truths),
            'prediction_ratio': len(predictions) / len(ground_truths) if ground_truths else 0
        }
        
        print(f"Results: P={overall['precision']:.3f}, R={overall['recall']:.3f}, "
              f"F1={overall['f1']:.3f}, mAP={overall['mAP']:.3f}")
    
    return results


def main():
    """Main evaluation function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_path = r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch"
    runs_path = os.path.join(base_path, "runs")
    
    models = {
        'baseline_dsbi': os.path.join(runs_path, 'baseline_all_false_dsbi.pth'),
        'cbam_only_dsbi': os.path.join(runs_path, 'cbam_only_dsbi.pth'),
        'coord_cbam_dsbi': os.path.join(runs_path, 'coord_cbam_dsbi.pth'),
        'coord_cbam_pos_dsbi': os.path.join(runs_path, 'coord_cbam_pos_dsbi.pth'),
        'baseline_custom': os.path.join(runs_path, 'baseline_all_false_custom.pth')
    }
    
    datasets = {
        'dsbi': os.path.join(base_path, 'dsbi.coco', 'valid'),
        'custom': os.path.join(base_path, 'custom.coco', 'valid')
    }
    
    # Filter existing files
    available_models = {k: v for k, v in models.items() if os.path.exists(v)}
    available_datasets = {k: v for k, v in datasets.items() if os.path.exists(v)}
    
    print(f"Found {len(available_models)} models, {len(available_datasets)} datasets")
    
    # Results storage
    all_results = []
    
    for model_name, model_path in available_models.items():
        # Match model to dataset
        if 'dsbi' in model_name and 'dsbi' in available_datasets:
            test_datasets = {'dsbi': available_datasets['dsbi']}
        elif 'custom' in model_name and 'custom' in available_datasets:
            test_datasets = {'custom': available_datasets['custom']}
        else:
            test_datasets = available_datasets
        
        for dataset_name, dataset_path in test_datasets.items():
            print(f"\nEvaluating {model_name} on {dataset_name}")
            
            try:
                results = evaluate_model(model_path, dataset_path, device)
                
                # Find best result
                best_conf = max(results.keys(), key=lambda k: results[k]['overall']['f1'])
                best_result = results[best_conf]
                
                all_results.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Best_Conf': best_conf,
                    'Precision': best_result['overall']['precision'],
                    'Recall': best_result['overall']['recall'],
                    'F1': best_result['overall']['f1'],
                    'mAP': best_result['overall']['mAP'],
                    'Predictions': best_result['num_predictions'],
                    'Ground_Truth': best_result['num_ground_truth'],
                    'Ratio': best_result['prediction_ratio']
                })
                
            except Exception as e:
                print(f"Error evaluating {model_name} on {dataset_name}: {e}")
    
    # Print comparison table
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS")
        print("="*80)
        print(df.to_string(index=False, float_format='%.3f'))
        
        # Save results
        output_dir = os.path.join(base_path, 'evaluation_results')
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
        print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()