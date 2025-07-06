from typing import Tuple, List
import torch
import math

def generate_targets(img_shape, class_labels_by_batch, box_labels_by_batch, strides):
    """Fixed target generation with proper coordinate handling"""
    
    batch_size = img_shape[0]
    img_h, img_w = img_shape[2], img_shape[3]
    
    
    # Size ranges for target assignment
    m = (0, 20, 40, 80, 160, 320, math.inf)
    
    class_targets_by_feature = []
    centerness_target_by_feature = []
    box_targets_by_feature = []
    
    for i, stride in enumerate(strides):
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
            
            for j in range(len(box_labels)):
                box = box_labels[j]
                class_id = class_labels[j]
                
                x1, y1, x2, y2 = box
                
                # FIXED: Proper center calculation
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                max_side = max(width, height)
                
                # Size filtering
                if max_side < min_box_side or max_side >= max_box_side:
                    continue
                
                # FIXED: Convert to feature map coordinates
                feat_center_x = center_x / stride
                feat_center_y = center_y / stride
                
                # Check bounds
                if 0 <= feat_center_x < feat_w and 0 <= feat_center_y < feat_h:
                    grid_x = int(feat_center_x)
                    grid_y = int(feat_center_y)
                    
                    # Assign target
                    class_target_for_feature[batch_idx, grid_y, grid_x] = class_id
                    positive_count += 1
                    
                    # Calculate regression targets (distances from center to edges)
                    left = feat_center_x - x1 / stride
                    top = feat_center_y - y1 / stride
                    right = x2 / stride - feat_center_x
                    bottom = y2 / stride - feat_center_y
                    
                    # Calculate centerness
                    if left > 0 and top > 0 and right > 0 and bottom > 0:
                        centerness = math.sqrt(
                            (min(left, right) / max(left, right)) *
                            (min(top, bottom) / max(top, bottom))
                        )
                        centerness_target_for_feature[batch_idx, grid_y, grid_x] = centerness
                        box_target_for_feature[batch_idx, grid_y, grid_x] = torch.tensor([
                            left, top, right, bottom
                        ])
        
        
        class_targets_by_feature.append(class_target_for_feature)
        centerness_target_by_feature.append(centerness_target_for_feature)
        box_targets_by_feature.append(box_target_for_feature)
    
    return class_targets_by_feature, centerness_target_by_feature, box_targets_by_feature
