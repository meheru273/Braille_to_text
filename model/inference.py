import math
import cv2
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torchvision

from FPNAttention import FPN, normalize_batch

MIN_SCORE = 0.001
DEFAULT_MAX_DETECTIONS = 3000

@dataclass
class Detection:
    score: float
    object_class: int
    bbox: np.ndarray  # (min_x, min_y, max_x, max_y)

def render_detections_to_image(img: np.ndarray, detections: List[Detection]):
    # Ensure img is a contiguous uint8 NumPy array
    img = np.ascontiguousarray(img, dtype=np.uint8)
    for detection in detections:
        if detection.score > 0.0:
            x1, y1, x2, y2 = detection.bbox
            start_point = (int(round(x1)), int(round(y1)))
            end_point   = (int(round(x2)), int(round(y2)))
            # cv2.rectangle works in-place; no need to reassign
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
    return img

def compute_detections(model: FPN, img: np.ndarray, device) -> List[Detection]:
    model.eval()
    if img.dtype == np.uint8:
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
    else:
        # assume float [0,1]
        tensor = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0)
    tensor = tensor.to(device)

    # 2. Normalize batch
    batch = normalize_batch(tensor)

    # 3. Forward pass
    with torch.no_grad():
        classes_by_feature, boxes_by_feature, _ = model(batch)  # Note: added _ for attention maps

    # 4. Decode detections
    H, W = tensor.shape[2], tensor.shape[3]
    detections = detections_from_network_output(
        H, W,
        classes_by_feature, boxes_by_feature,
        model.scales, model.strides
    )
    # returns List[List[Detection]]; since batch size=1, you can return detections[0]
    return detections[0]

# In inference.py, modify detections_from_network_output:
def detections_from_network_output(img_height, img_width, classes, boxes, scales, strides):
    """Fixed version with consistent sigmoid handling"""
    all_classes = []
    all_boxes = []
    n_classes = classes[0].shape[-1]
    batch_size = classes[0].shape[0]

    for feat_classes, feat_boxes, scale, stride in zip(classes, boxes, scales, strides):
        # Apply sigmoid to classification (convert logits to probabilities)
        feat_classes_probs = torch.sigmoid(feat_classes)
        
        # Decode boxes with proper scaling
        boxes_decoded = _boxes_from_regression(
            feat_boxes, img_height, img_width, stride
        )

        all_classes.append(feat_classes_probs.view(batch_size, -1, n_classes))
        all_boxes.append(boxes_decoded.view(batch_size, -1, 4))

    classes_ = torch.cat(all_classes, dim=1)
    boxes_ = torch.cat(all_boxes, dim=1)
    gathered_boxes, gathered_classes, gathered_scores = _gather_detections(classes_, boxes_)
    return detections_from_net(gathered_boxes, gathered_classes, gathered_scores)

# In inference.py, replace _boxes_from_regression function:
def _boxes_from_regression(reg, img_height, img_width, stride):
    """
    CORRECTED: Proper FCOS box decoding with tiny box prevention
    """
    batch, rows, cols, _ = reg.shape

    # Create anchor points (centers of grid cells)
    y = torch.linspace(stride/2, img_height - stride/2, rows).to(reg.device)
    x = torch.linspace(stride/2, img_width - stride/2, cols).to(reg.device)
    center_y, center_x = torch.meshgrid(y, x, indexing='ij')
    center_y = center_y.unsqueeze(0).expand(batch, -1, -1)
    center_x = center_x.unsqueeze(0).expand(batch, -1, -1)

    # FIXED: Apply bounds to prevent extreme values
    reg_clamped = torch.clamp(reg, min=0.01, max=8.0)  # Reasonable bounds
    
    left_dist = reg_clamped[..., 0] * stride
    top_dist = reg_clamped[..., 1] * stride
    right_dist = reg_clamped[..., 2] * stride
    bottom_dist = reg_clamped[..., 3] * stride

    # Convert to absolute coordinates
    x_min = center_x - left_dist
    y_min = center_y - top_dist
    x_max = center_x + right_dist
    y_max = center_y + bottom_dist

    # Clamp to image boundaries
    x_min = torch.clamp(x_min, 0, img_width)
    y_min = torch.clamp(y_min, 0, img_height)
    x_max = torch.clamp(x_max, 0, img_width)
    y_max = torch.clamp(y_max, 0, img_height)

    # CRITICAL FIX: Filter out tiny boxes
    min_box_size = stride / 2  # Minimum reasonable box size
    
    widths = x_max - x_min
    heights = y_max - y_min
    
    # Create mask for valid boxes
    valid_mask = (widths >= min_box_size) & (heights >= min_box_size)
    
    # Zero out invalid boxes
    x_min = torch.where(valid_mask, x_min, torch.zeros_like(x_min))
    y_min = torch.where(valid_mask, y_min, torch.zeros_like(y_min))
    x_max = torch.where(valid_mask, x_max, torch.zeros_like(x_max))
    y_max = torch.where(valid_mask, y_max, torch.zeros_like(y_max))

    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)



def detections_from_net(boxes_by_batch, classes_by_batch, scores_by_batch=None) -> List[List[Detection]]:
    """Convert network outputs to Detection objects"""
    result = []

    for batch in range(len(classes_by_batch)):
        scores = scores_by_batch[batch] if scores_by_batch is not None else None
        classes = classes_by_batch[batch]
        boxes = boxes_by_batch[batch]

        batch_detections = []
        for i in range(boxes.shape[0]):
            # FIXED: Ensure positive box dimensions
            bbox = boxes[i].cpu().numpy()
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid box
                batch_detections.append(
                    Detection(
                        score=scores[i].item() if scores is not None else 1.0,
                        object_class=classes[i].item(),
                        bbox=bbox.astype(int),
                    )
                )
        
        result.append(batch_detections)

    return result

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to image array"""
    arr = tensor.detach().cpu().numpy()

    if arr.max() <= 1.0:
        arr = arr * 255.0
    
    if len(arr.shape) == 3:  
        img = arr.transpose(1, 2, 0)  # [H, W, C]
    elif len(arr.shape) == 2:  
        img = arr
    else:
        raise ValueError(f"Unexpected tensor shape: {arr.shape}")
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def _gather_detections(classes, boxes, max_detections=3000, min_score=0.05):
    """
    IMPROVED: Better detection gathering with size filtering
    """
    class_scores, class_indices = torch.max(classes, dim=2)
    boxes_by_batch = []
    classes_by_batch = []
    scores_by_batch = []

    n_batches = boxes.shape[0]
    for i in range(n_batches):
        class_scores_i = class_scores[i]
        boxes_i = boxes[i]
        class_indices_i = class_indices[i]

        # FIXED: Higher minimum score threshold
        valid_mask = class_scores_i > min_score
        
        # ADDED: Size filtering for boxes
        box_widths = boxes_i[:, 2] - boxes_i[:, 0]
        box_heights = boxes_i[:, 3] - boxes_i[:, 1]
        size_mask = (box_widths > 2) & (box_heights > 2)  # Filter tiny boxes
        
        final_mask = valid_mask & size_mask
        
        if not final_mask.any():
            # No valid detections
            boxes_by_batch.append(torch.empty(0, 4).to(boxes.device))
            classes_by_batch.append(torch.empty(0).long().to(boxes.device))
            scores_by_batch.append(torch.empty(0).to(boxes.device))
            continue

        class_scores_i = class_scores_i[final_mask]
        boxes_i = boxes_i[final_mask]
        class_indices_i = class_indices_i[final_mask]

        # Select top detections
        num_detections = min(class_scores_i.shape[0], max_detections)
        if num_detections > 0:
            _, top_detection_indices = torch.topk(class_scores_i, num_detections, dim=0)

            top_boxes_i = torch.index_select(boxes_i, 0, top_detection_indices)
            top_classes_i = torch.index_select(class_indices_i, 0, top_detection_indices)
            top_scores_i = torch.index_select(class_scores_i, 0, top_detection_indices)

            # Apply NMS with appropriate threshold for small objects
            if len(top_boxes_i) > 0:
                boxes_to_keep = torchvision.ops.nms(top_boxes_i, top_scores_i, 0.5)  # Higher threshold for small objects
                top_boxes_i = top_boxes_i[boxes_to_keep]
                top_classes_i = top_classes_i[boxes_to_keep] 
                top_scores_i = top_scores_i[boxes_to_keep]
        else:
            top_boxes_i = torch.empty(0, 4).to(boxes.device)
            top_classes_i = torch.empty(0).long().to(boxes.device)
            top_scores_i = torch.empty(0).to(boxes.device)
        
        boxes_by_batch.append(top_boxes_i)
        classes_by_batch.append(top_classes_i)
        scores_by_batch.append(top_scores_i)

    return boxes_by_batch, classes_by_batch, scores_by_batch


# Debug function to monitor improvements
def debug_regression_outputs(reg_outputs, stride, feature_level):
    """Debug function to monitor regression output quality"""
    reg_flat = reg_outputs.view(-1, 4)
    
    stats = {
        'min_vals': reg_flat.min(dim=0)[0].cpu().numpy(),
        'max_vals': reg_flat.max(dim=0)[0].cpu().numpy(), 
        'mean_vals': reg_flat.mean(dim=0).cpu().numpy(),
        'std_vals': reg_flat.std(dim=0).cpu().numpy()
    }
    
    # Check for problematic values
    tiny_boxes = (reg_flat < 0.01).any(dim=1).float().mean().item()
    huge_boxes = (reg_flat > 8.0).any(dim=1).float().mean().item()
    
    print(f"\nFeature level {feature_level} (stride={stride}):")
    print(f"  Regression stats: min={stats['min_vals']}, max={stats['max_vals']}")
    print(f"  Mean: {stats['mean_vals']}, Std: {stats['std_vals']}")
    print(f"  Tiny boxes ratio: {tiny_boxes:.4f}")
    print(f"  Huge boxes ratio: {huge_boxes:.4f}")
    
    return stats