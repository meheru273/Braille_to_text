from typing import Tuple, List
import torch
import math


import torch
import math

def generate_targets_adaptive(img_shape, class_labels_by_batch, box_labels_by_batch, strides, 
                                  size_ranges=None, center_sampling=True, center_radius=1.5):
    """
    FIXED target generation that prevents tiny box predictions
    """
    batch_size, _, img_h, img_w = img_shape
    
    # Define proper size ranges for Braille characters
    if size_ranges is None:
        size_ranges = [
            (0, 24),      # stride=2: very small Braille dots
            (12, 48),     # stride=4: small characters
            (24, 96),     # stride=8: medium characters
            (48, 192),    # stride=16: large characters
            (96, 512)     # stride=32: very large characters
        ]
    
    class_targets = []
    box_targets = []
    
    for level_idx, (stride, size_range) in enumerate(zip(strides, size_ranges)):
        min_size, max_size = size_range
        feat_h = img_h // stride
        feat_w = img_w // stride
        
        cls_target = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.long)
        box_target = torch.zeros((batch_size, feat_h, feat_w, 4), dtype=torch.float32)
        
        for batch_idx, (labels, boxes) in enumerate(zip(class_labels_by_batch, box_labels_by_batch)):
            if len(boxes) == 0:
                continue
            
            for obj_idx, (label, box) in enumerate(zip(labels, boxes)):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # Use geometric mean for size assignment (better for small objects)
                obj_size = math.sqrt(w * h)
                
                # Strict size filtering - only assign to appropriate levels
                if not (min_size <= obj_size < max_size):
                    continue
                
                # Calculate object center
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                if center_sampling:
                    # Adaptive center sampling based on object size
                    cx_feat = cx / stride
                    cy_feat = cy / stride
                    
                    # CRITICAL FIX: Limit sampling radius to prevent tiny boxes
                    radius = min(center_radius, max(w, h) / stride * 0.3)  # Max 30% of object size
                    radius = max(0.5, radius)  # Ensure minimum radius
                    
                    x_min = int(max(0, cx_feat - radius))
                    x_max = int(min(feat_w - 1, cx_feat + radius)) + 1
                    y_min = int(max(0, cy_feat - radius))
                    y_max = int(min(feat_h - 1, cy_feat + radius)) + 1
                else:
                    # Conservative bbox assignment
                    x_min = max(0, int((x1 + w * 0.25) / stride))  # Start 25% into box
                    y_min = max(0, int((y1 + h * 0.25) / stride))
                    x_max = min(feat_w, int((x2 - w * 0.25) / stride) + 1)  # End 25% before edge
                    y_max = min(feat_h, int((y2 - h * 0.25) / stride) + 1)
                
                # Ensure at least one pixel assignment
                if x_max <= x_min:
                    x_max = x_min + 1
                if y_max <= y_min:
                    y_max = y_min + 1
                
                # Assign targets with quality control
                for feat_y in range(y_min, y_max):
                    for feat_x in range(x_min, x_max):
                        px = feat_x * stride + stride / 2
                        py = feat_y * stride + stride / 2
                        
                        # Strict bounds checking
                        if not (x1 <= px <= x2 and y1 <= py <= y2):
                            continue
                        
                        # Calculate regression targets
                        left = px - x1
                        top = py - y1
                        right = x2 - px
                        bottom = y2 - py
                        
                        # CRITICAL: Quality control for regression targets
                        min_distance = stride * 0.1  # Minimum 10% of stride
                        max_distance = stride * 8.0  # Maximum 8x stride
                        
                        if (left >= min_distance and top >= min_distance and 
                            right >= min_distance and bottom >= min_distance and
                            left <= max_distance and top <= max_distance and
                            right <= max_distance and bottom <= max_distance):
                            
                            # Only assign if current location is empty or has larger object
                            current_label = cls_target[batch_idx, feat_y, feat_x].item()
                            current_size = float('inf')
                            
                            if current_label > 0:
                                # Calculate current object size from existing targets
                                curr_box = box_target[batch_idx, feat_y, feat_x]
                                current_size = (curr_box[0] + curr_box[2]) * (curr_box[1] + curr_box[3])
                            
                            obj_reg_size = (left + right) * (top + bottom)
                            
                            # Prefer smaller objects (they're harder to detect)
                            if current_label == 0 or obj_reg_size < current_size:
                                cls_target[batch_idx, feat_y, feat_x] = label
                                
                                # FIXED: Proper normalization with bounds
                                box_target[batch_idx, feat_y, feat_x] = torch.tensor([
                                    left / stride,
                                    top / stride,
                                    right / stride,
                                    bottom / stride
                                ], dtype=torch.float32)
        
        class_targets.append(cls_target)
        box_targets.append(box_target)
    
    return class_targets, box_targets


def generate_targets_multiscale(img_shape, class_labels_by_batch, box_labels_by_batch, strides):
    """
    Multi-scale assignment: assign each object to multiple FPN levels
    This can help with small object detection
    """
    batch_size, _, img_h, img_w = img_shape
    
    class_targets = []
    box_targets = []
    
    # Pre-calculate feature map dimensions
    feature_sizes = []
    for stride in strides:
        feat_h = img_h // stride
        feat_w = img_w // stride
        feature_sizes.append((feat_h, feat_w))
    
    for level_idx, (stride, (feat_h, feat_w)) in enumerate(zip(strides, feature_sizes)):
        # Initialize targets
        cls_target = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.long)
        box_target = torch.zeros((batch_size, feat_h, feat_w, 4), dtype=torch.float32)
        
        for batch_idx, (labels, boxes) in enumerate(zip(class_labels_by_batch, box_labels_by_batch)):
            if len(boxes) == 0:
                continue
            
            for obj_idx, (label, box) in enumerate(zip(labels, boxes)):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # Calculate object size
                obj_size = math.sqrt(w * h)
                
                # Multi-scale assignment weight
                # Smaller objects get higher weight on finer levels
                if level_idx == 0:  # Finest level
                    weight = 1.0 if obj_size < 32 else 0.5
                elif level_idx == len(strides) - 1:  # Coarsest level
                    weight = 1.0 if obj_size > 128 else 0.5
                else:  # Middle levels
                    weight = 0.8
                
                # Random sampling based on weight
                if torch.rand(1).item() > weight:
                    continue
                
                # Calculate center
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Center sampling region
                cx_feat = cx / stride
                cy_feat = cy / stride
                
                # Adaptive radius based on object size
                radius = max(1.0, min(2.5, obj_size / stride / 4))
                
                x_min = int(max(0, cx_feat - radius))
                x_max = int(min(feat_w - 1, cx_feat + radius)) + 1
                y_min = int(max(0, cy_feat - radius))
                y_max = int(min(feat_h - 1, cy_feat + radius)) + 1
                
                # Assign targets
                for feat_y in range(y_min, y_max):
                    for feat_x in range(x_min, x_max):
                        px = feat_x * stride + stride / 2
                        py = feat_y * stride + stride / 2
                        
                        if x1 <= px <= x2 and y1 <= py <= y2:
                            left = px - x1
                            top = py - y1
                            right = x2 - px
                            bottom = y2 - py
                            
                            if left > 0 and top > 0 and right > 0 and bottom > 0:
                                # Distance from center for weighting
                                dist = math.sqrt((px - cx)**2 + (py - cy)**2)
                                center_weight = math.exp(-dist / (obj_size / 2))
                                
                                # Only assign if weight is high enough
                                if center_weight > 0.3:
                                    cls_target[batch_idx, feat_y, feat_x] = label
                                    box_target[batch_idx, feat_y, feat_x] = torch.tensor([
                                        left / stride,
                                        top / stride,
                                        right / stride,
                                        bottom / stride
                                    ], dtype=torch.float32)
        
        class_targets.append(cls_target)
        box_targets.append(box_target)
    
    return class_targets, box_targets


# Wrapper function to maintain compatibility
def generate_targets(img_shape, class_labels_by_batch, box_labels_by_batch, strides, 
                    method='adaptive', **kwargs):
    """
    Generate targets using specified method
    
    Args:
        method: 'adaptive', 'multiscale', or 'original'
    """
    if method == 'adaptive':
        return generate_targets_adaptive(
            img_shape, class_labels_by_batch, box_labels_by_batch, strides, **kwargs
        )
    elif method == 'multiscale':
        return generate_targets_multiscale(
            img_shape, class_labels_by_batch, box_labels_by_batch, strides
        )
    else:
        # Fall back to your original implementation
        # You can import and use your original function here
        raise NotImplementedError(f"Method {method} not implemented")