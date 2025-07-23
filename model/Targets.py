from typing import Tuple, List
import torch
import math

def generate_targets(img_shape, class_labels_by_batch, box_labels_by_batch, strides):
    """
    Fixed FCOS target generation with proper size-based assignment
    """
    batch_size, _, img_h, img_w = img_shape
    
    size_ranges = [(0,32),(32,  64),(64,  128),(128, 256),(256, float('inf')),]
    class_targets = []
    box_targets = []
    
    # Pre-calculate feature map dimensions to avoid mismatches
    feature_sizes = []
    for stride in strides:
        feat_h = img_h // stride
        feat_w = img_w // stride
        feature_sizes.append((feat_h, feat_w))
    
    print(f"Image size: ({img_h}, {img_w})")
    print(f"Feature sizes: {feature_sizes}")
    
    for level_idx, (stride, size_range, (feat_h, feat_w)) in enumerate(zip(strides, size_ranges, feature_sizes)):
        min_size, max_size = size_range
        
        # Initialize targets with correct dimensions
        cls_target = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.long)
        box_target = torch.zeros((batch_size, feat_h, feat_w, 4), dtype=torch.float32)
        
        print(f"Level {level_idx}: stride={stride}, size_range=({min_size}, {max_size}), feat_size=({feat_h}, {feat_w})")
        
        for batch_idx, (labels, boxes) in enumerate(zip(class_labels_by_batch, box_labels_by_batch)):
            if len(boxes) == 0:
                continue
            
            objects_assigned = 0
            
            for obj_idx, (label, box) in enumerate(zip(labels, boxes)):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # CRITICAL FIX: Use max side length for size-based assignment
                max_side = max(w, h)
                
                # CRITICAL FIX: Only assign objects to appropriate pyramid level
                if not (min_size <= max_side < max_size):
                    continue
                
                objects_assigned += 1
                
                # Convert box coordinates to feature map coordinates
                feat_x1 = max(0, int(x1 / stride))
                feat_y1 = max(0, int(y1 / stride))
                feat_x2 = min(feat_w, int((x2 / stride) + 0.5))  # Better rounding
                feat_y2 = min(feat_h, int((y2 / stride) + 0.5))
                
                # Ensure we have at least one pixel
                if feat_x2 <= feat_x1:
                    feat_x2 = feat_x1 + 1
                if feat_y2 <= feat_y1:
                    feat_y2 = feat_y1 + 1
                
                # OPTIMIZED: Vectorized assignment instead of nested loops
                for feat_y in range(feat_y1, feat_y2):
                    for feat_x in range(feat_x1, feat_x2):
                        # Calculate center point of this feature map location
                        center_x = feat_x * stride + stride // 2
                        center_y = feat_y * stride + stride // 2
                        
                        # CRITICAL FIX: Check if center point is inside the box
                        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                            # Calculate regression targets (distances to box edges)
                            left = center_x - x1
                            top = center_y - y1
                            right = x2 - center_x
                            bottom = y2 - center_y
                            
                            # CRITICAL FIX: Ensure all distances are positive
                            if left > 0 and top > 0 and right > 0 and bottom > 0:
                                # Assign positive class label
                                cls_target[batch_idx, feat_y, feat_x] = label
                                
                                # FIXED: Proper regression targets (normalized by stride)
                                box_target[batch_idx, feat_y, feat_x] = torch.tensor([
                                    left / stride,   # l*
                                    top / stride,    # t*  
                                    right / stride,  # r*
                                    bottom / stride  # b*
                                ], dtype=torch.float32)
            
            if objects_assigned > 0:
                print(f"  Batch {batch_idx}: {objects_assigned} objects assigned to level {level_idx}")
        
        # Count positive targets for debugging
        positive_targets = (cls_target > 0).sum().item()
        print(f"Level {level_idx}: {positive_targets} positive targets generated")
        
        class_targets.append(cls_target)
        box_targets.append(box_target)
    
    return class_targets, box_targets
