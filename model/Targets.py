from typing import Tuple, List
import torch
import math


def generate_targets_adaptive(img_shape, class_labels_by_batch, box_labels_by_batch, strides, 
                             size_ranges=None, center_sampling=True, center_radius=1.5):
    """
    Improved target generation for small object detection (like Braille characters)
    
    Args:
        img_shape: Shape of input images [B, C, H, W]
        class_labels_by_batch: List of class labels per image
        box_labels_by_batch: List of bounding boxes per image
        strides: List of stride values for each FPN level
        size_ranges: Custom size ranges for each level (auto-calculated if None)
        center_sampling: Use center sampling strategy
        center_radius: Radius for center sampling (in stride units)
    """
    batch_size, _, img_h, img_w = img_shape
    
    # Auto-calculate size ranges based on strides if not provided
    if size_ranges is None:
        # For small objects like Braille, use stride-based ranges
        size_ranges = []
        for i, stride in enumerate(strides):
            if i == 0:
                min_size = 0
                max_size = stride * 4
            elif i == len(strides) - 1:
                min_size = strides[i-1] * 4
                max_size = float('inf')
            else:
                min_size = strides[i-1] * 4
                max_size = stride * 4
            size_ranges.append((min_size, max_size))
    
    
    class_targets = []
    box_targets = []
    
    # Pre-calculate feature map dimensions
    feature_sizes = []
    for stride in strides:
        feat_h = img_h // stride
        feat_w = img_w // stride
        feature_sizes.append((feat_h, feat_w))
    
    # Statistics for debugging
    level_assignments = [0] * len(strides)
    
    for level_idx, (stride, size_range, (feat_h, feat_w)) in enumerate(zip(strides, size_ranges, feature_sizes)):
        min_size, max_size = size_range
        
        # Initialize targets
        cls_target = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.long)
        box_target = torch.zeros((batch_size, feat_h, feat_w, 4), dtype=torch.float32)
        
        for batch_idx, (labels, boxes) in enumerate(zip(class_labels_by_batch, box_labels_by_batch)):
            if len(boxes) == 0:
                continue
            
            for obj_idx, (label, box) in enumerate(zip(labels, boxes)):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # Use geometric mean for size (better for small objects)
                obj_size = math.sqrt(w * h)
                
                # Check size range
                if not (min_size <= obj_size < max_size):
                    continue
                
                level_assignments[level_idx] += 1
                
                # Calculate object center
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                if center_sampling:
                    # Center sampling: only assign pixels near object center
                    # This is especially important for small objects
                    cx_feat = cx / stride
                    cy_feat = cy / stride
                    
                    # Define sampling region
                    radius = center_radius
                    x_min = int(max(0, cx_feat - radius))
                    x_max = int(min(feat_w - 1, cx_feat + radius)) + 1
                    y_min = int(max(0, cy_feat - radius))
                    y_max = int(min(feat_h - 1, cy_feat + radius)) + 1
                else:
                    # Original method: assign all pixels inside bbox
                    x_min = max(0, int(x1 / stride))
                    y_min = max(0, int(y1 / stride))
                    x_max = min(feat_w, int(x2 / stride) + 1)
                    y_max = min(feat_h, int(y2 / stride) + 1)
                
                # Ensure we have at least one pixel
                if x_max <= x_min:
                    x_max = x_min + 1
                if y_max <= y_min:
                    y_max = y_min + 1
                
                # Assign targets to feature map locations
                for feat_y in range(y_min, y_max):
                    for feat_x in range(x_min, x_max):
                        # Feature map location center in image space
                        px = feat_x * stride + stride / 2
                        py = feat_y * stride + stride / 2
                        
                        # Additional check: ensure pixel center is inside bbox
                        if x1 <= px <= x2 and y1 <= py <= y2:
                            # Calculate regression targets (distances to box edges)
                            left = px - x1
                            top = py - y1
                            right = x2 - px
                            bottom = y2 - py
                            
                            # All distances must be positive
                            if left > 0 and top > 0 and right > 0 and bottom > 0:
                                # Only update if this location isn't already assigned
                                # or if current object is smaller (prioritize small objects)
                                current_label = cls_target[batch_idx, feat_y, feat_x].item()
                                if current_label == 0:
                                    cls_target[batch_idx, feat_y, feat_x] = label
                                    
                                    # Normalize by stride for scale invariance
                                    box_target[batch_idx, feat_y, feat_x] = torch.tensor([
                                        left / stride,
                                        top / stride,
                                        right / stride,
                                        bottom / stride
                                    ], dtype=torch.float32)
        
        class_targets.append(cls_target)
        box_targets.append(box_target)
    
    # Print assignment statistics
    total_objects = sum(len(labels) for labels in class_labels_by_batch)
    # print(f"Target assignment statistics:")
    # print(f"  Total objects: {total_objects}")
    # for i, count in enumerate(level_assignments):
    #     print(f"  Level {i} (stride {strides[i]}): {count} objects")
    
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