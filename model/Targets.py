from typing import Tuple, List
import torch
import math

def generate_targets(img_shape, class_labels_by_batch, box_labels_by_batch, strides):
    """Optimized target generation with progress logging"""
    
    print(f"🎯 Starting target generation for {len(class_labels_by_batch)} images")
    
    batch_size = img_shape[0]
    img_h, img_w = img_shape[2], img_shape[3]
    
    print(f"Image shape: {img_h}×{img_w}, Strides: {strides}")
    
    # Fixed size ranges for your strides
    if len(strides) == 5:
        m = (0, 20, 40, 60, 120, 320, math.inf)
    else:
        m = (0, 20, 40, 80, math.inf)
    
    class_targets_by_feature = []
    centerness_target_by_feature = []
    box_targets_by_feature = []
    
    for i, stride in enumerate(strides):
        print(f"Processing level {i}, stride {stride}")
        
        feat_h = img_h // stride
        feat_w = img_w // stride
        
        # Initialize targets
        class_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, dtype=torch.long)
        centerness_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, dtype=torch.float32)
        box_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, 4, dtype=torch.float32)
        
        min_box_side = m[i]
        max_box_side = m[i + 1]
        
        positive_count = 0
        
        for batch_idx, (class_labels, box_labels) in enumerate(zip(class_labels_by_batch, box_labels_by_batch)):
            if len(box_labels) == 0:
                continue
            
            # OPTIMIZED: Vectorized computation
            heights = box_labels[:, 3] - box_labels[:, 1]
            widths = box_labels[:, 2] - box_labels[:, 0]
            max_sides = torch.maximum(heights, widths)
            
            # Filter valid boxes
            valid_mask = (max_sides >= min_box_side) & (max_sides < max_box_side)
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                continue
            
            # Process only valid boxes (much faster)
            for j in valid_indices:
                box = box_labels[j]
                class_id = class_labels[j]
                
                # Convert to grid coordinates
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2 / stride
                center_y = (y1 + y2) / 2 / stride
                
                if 0 <= center_x < feat_w and 0 <= center_y < feat_h:
                    grid_x, grid_y = int(center_x), int(center_y)
                    
                    # Quick assignment
                    class_target_for_feature[batch_idx, grid_y, grid_x] = class_id
                    positive_count += 1
                    
                    # Basic centerness (simplified)
                    left = center_x - x1 / stride
                    top = center_y - y1 / stride
                    right = x2 / stride - center_x
                    bottom = y2 / stride - center_y
                    
                    if left > 0 and top > 0 and right > 0 and bottom > 0:
                        centerness = 0.8  # Simplified centerness
                        centerness_target_for_feature[batch_idx, grid_y, grid_x] = centerness
                        box_target_for_feature[batch_idx, grid_y, grid_x] = torch.tensor([
                            left, top, right, bottom
                        ])
        
        
        class_targets_by_feature.append(class_target_for_feature)
        centerness_target_by_feature.append(centerness_target_for_feature)
        box_targets_by_feature.append(box_target_for_feature)
    
    print("🎯 Target generation complete")
    return class_targets_by_feature, centerness_target_by_feature, box_targets_by_feature
