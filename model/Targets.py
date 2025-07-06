from typing import Tuple, List
import torch
import math

def debug_generated_targets(class_targets_by_feature, centerness_target_by_feature, box_targets_by_feature, strides):
    """Debug the generated targets"""
    print("\n=== GENERATED TARGETS ANALYSIS ===")
    
    for level_idx in range(len(class_targets_by_feature)):
        cls_target = class_targets_by_feature[level_idx][0]  # First batch
        cen_target = centerness_target_by_feature[level_idx][0]
        box_target = box_targets_by_feature[level_idx][0]
        
        stride = strides[level_idx]
        
        print(f"\nLevel {level_idx} (stride {stride}):")
        print(f"  Feature map shape: {cls_target.shape}")
        print(f"  Unique classes: {torch.unique(cls_target)}")
        
        # Count positive samples
        positive_mask = cls_target > 0
        positive_count = positive_mask.sum().item()
        total_locations = cls_target.numel()
        positive_ratio = positive_count / total_locations
        
        print(f"  Positive samples: {positive_count}/{total_locations} ({positive_ratio:.4f})")
        
        if positive_count > 0:
            # Show where positive samples are
            pos_y, pos_x = torch.where(positive_mask)
            print(f"  Positive locations Y: {pos_y.min().item()}-{pos_y.max().item()}")
            print(f"  Positive locations X: {pos_x.min().item()}-{pos_x.max().item()}")
            
            # Show centerness stats for positive samples
            pos_centerness = cen_target[positive_mask]
            print(f"  Centerness range: {pos_centerness.min():.3f} to {pos_centerness.max():.3f}")
            print(f"  Mean centerness: {pos_centerness.mean():.3f}")
            
            # Show some example positive samples
            print(f"  Example positive samples:")
            for i in range(min(3, len(pos_y))):
                y, x = pos_y[i].item(), pos_x[i].item()
                cls = cls_target[y, x].item()
                cent = cen_target[y, x].item()
                box = box_target[y, x]
                print(f"    ({x}, {y}): class={cls}, cent={cent:.3f}, box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        
        # Check for potential issues - FIXED: Remove Unicode characters
        if positive_ratio > 0.1:  # More than 10%
            print(f"  WARNING: Too many positive samples ({positive_ratio:.1%})")
        elif positive_ratio < 0.001:  # Less than 0.1%
            print(f"  WARNING: Very few positive samples ({positive_ratio:.1%})")
        else:
            print(f"  [OK] Positive ratio looks good ({positive_ratio:.1%})")  # Changed from ✓


def debug_generated_targets(class_targets_by_feature, centerness_target_by_feature, box_targets_by_feature, strides):
    """Debug the generated targets"""
    print("\n=== GENERATED TARGETS ANALYSIS ===")
    
    for level_idx in range(len(class_targets_by_feature)):
        cls_target = class_targets_by_feature[level_idx][0]  # First batch
        cen_target = centerness_target_by_feature[level_idx][0]
        box_target = box_targets_by_feature[level_idx][0]
        
        stride = strides[level_idx]
        
        print(f"\nLevel {level_idx} (stride {stride}):")
        print(f"  Feature map shape: {cls_target.shape}")
        print(f"  Unique classes: {torch.unique(cls_target)}")
        
        # Count positive samples
        positive_mask = cls_target > 0
        positive_count = positive_mask.sum().item()
        total_locations = cls_target.numel()
        positive_ratio = positive_count / total_locations
        
        print(f"  Positive samples: {positive_count}/{total_locations} ({positive_ratio:.4f})")
        
        if positive_count > 0:
            # Show where positive samples are
            pos_y, pos_x = torch.where(positive_mask)
            print(f"  Positive locations Y: {pos_y.min().item()}-{pos_y.max().item()}")
            print(f"  Positive locations X: {pos_x.min().item()}-{pos_x.max().item()}")
            
            # Show centerness stats for positive samples
            pos_centerness = cen_target[positive_mask]
            print(f"  Centerness range: {pos_centerness.min():.3f} to {pos_centerness.max():.3f}")
            print(f"  Mean centerness: {pos_centerness.mean():.3f}")
            
            # Show some example positive samples
            print(f"  Example positive samples:")
            for i in range(min(3, len(pos_y))):
                y, x = pos_y[i].item(), pos_x[i].item()
                cls = cls_target[y, x].item()
                cent = cen_target[y, x].item()
                box = box_target[y, x]
                print(f"    ({x}, {y}): class={cls}, cent={cent:.3f}, box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        
        # Check for potential issues
        if positive_ratio > 0.1:  # More than 10%
            print(f"  WARNING: Too many positive samples ({positive_ratio:.1%})")
        elif positive_ratio < 0.001:  # Less than 0.1%
            print(f"  WARNING: Very few positive samples ({positive_ratio:.1%})")
        else:
            print(f"  ✓ Positive ratio looks good ({positive_ratio:.1%})")

def generate_targets(img_shape, class_labels_by_batch, box_labels_by_batch, strides, debug=False):
    """Fixed target generation with proper coordinate handling"""
    
    if debug:
        debug_target_generation(class_labels_by_batch, box_labels_by_batch, img_shape, strides)
    
    batch_size = img_shape[0]
    img_h, img_w = img_shape[2], img_shape[3]
    
    # FIXED: Size ranges for Braille characters
    m = (0, 64, 128, 256, 512, 1024, math.inf)
    
    class_targets_by_feature = []
    centerness_target_by_feature = []
    box_targets_by_feature = []
    
    total_positive_assigned = 0
    
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
        
        if debug:
            print(f"\nProcessing Level {i} (stride {stride}):")
            print(f"  Feature map: {feat_w}x{feat_h}")
            print(f"  Size range: {min_box_side} to {max_box_side}")
        
        for batch_idx, (class_labels, box_labels) in enumerate(zip(class_labels_by_batch, box_labels_by_batch)):
            if len(box_labels) == 0:
                continue
            
            for j in range(len(box_labels)):
                box = box_labels[j]
                class_id = class_labels[j]
                
                x1, y1, x2, y2 = box
                
                # Calculate box properties
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                max_side = max(width, height)
                
                # Size filtering
                if max_side < min_box_side or max_side >= max_box_side:
                    if debug and j < 3:
                        print(f"    Box {j}: size {max_side:.1f} outside range [{min_box_side}, {max_box_side})")
                    continue
                
                # Convert to feature map coordinates
                feat_center_x = center_x / stride
                feat_center_y = center_y / stride
                
                # Check bounds
                if 0 <= feat_center_x < feat_w and 0 <= feat_center_y < feat_h:
                    grid_x = int(feat_center_x)
                    grid_y = int(feat_center_y)
                    
                    # Only assign if background
                    if class_target_for_feature[batch_idx, grid_y, grid_x] == 0:
                        # Assign class
                        class_target_for_feature[batch_idx, grid_y, grid_x] = class_id
                        positive_count += 1
                        
                        # FIXED: Calculate regression targets properly
                        feat_x1 = x1 / stride
                        feat_y1 = y1 / stride
                        feat_x2 = x2 / stride
                        feat_y2 = y2 / stride
                        
                        # Grid cell center
                        grid_center_x = grid_x + 0.5
                        grid_center_y = grid_y + 0.5
                        
                        # Distances from grid center to box edges
                        left = grid_center_x - feat_x1
                        top = grid_center_y - feat_y1
                        right = feat_x2 - grid_center_x
                        bottom = feat_y2 - grid_center_y
                        
                        # Ensure all distances are positive
                        if left > 0 and top > 0 and right > 0 and bottom > 0:
                            # Calculate centerness
                            centerness = math.sqrt(
                                (min(left, right) / max(left, right)) *
                                (min(top, bottom) / max(top, bottom))
                            )
                            centerness_target_for_feature[batch_idx, grid_y, grid_x] = centerness
                            box_target_for_feature[batch_idx, grid_y, grid_x] = torch.tensor([
                                left, top, right, bottom
                            ])
                            
                            if debug and positive_count <= 3:
                                print(f"    Assigned box {j} to grid ({grid_x}, {grid_y}): "
                                      f"class={class_id}, cent={centerness:.3f}, "
                                      f"box=[{left:.1f}, {top:.1f}, {right:.1f}, {bottom:.1f}]")
                        else:
                            # Reset if distances are invalid
                            class_target_for_feature[batch_idx, grid_y, grid_x] = 0
                            positive_count -= 1
                            if debug:
                                print(f"    Invalid distances for box {j}: l={left:.2f}, t={top:.2f}, r={right:.2f}, b={bottom:.2f}")
        
        if debug:
            print(f"  Level {i} assigned {positive_count} positive samples")
        
        total_positive_assigned += positive_count
        
        class_targets_by_feature.append(class_target_for_feature)
        centerness_target_by_feature.append(centerness_target_for_feature)
        box_targets_by_feature.append(box_target_for_feature)
    
    if debug:
        print(f"\nTotal positive samples assigned across all levels: {total_positive_assigned}")
        debug_generated_targets(class_targets_by_feature, centerness_target_by_feature, box_targets_by_feature, strides)
    
    return class_targets_by_feature, centerness_target_by_feature, box_targets_by_feature
