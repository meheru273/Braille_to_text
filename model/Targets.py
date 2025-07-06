from typing import Tuple, List
import torch
import math

def debug_target_generation(class_labels_by_batch, box_labels_by_batch, img_shape, strides):
    """Debug what targets are actually being generated"""
    print("=== TARGET GENERATION DEBUG ===")
    print(f"Image shape: {img_shape}")
    print(f"Strides: {strides}")
    
    for batch_idx, (cls_batch, box_batch) in enumerate(zip(class_labels_by_batch, box_labels_by_batch)):
        print(f"\nBatch {batch_idx}:")
        print(f"  Classes: {cls_batch}")
        print(f"  Boxes shape: {box_batch.shape}")
        print(f"  Number of boxes: {len(box_batch)}")
        
        if len(box_batch) == 0:
            print("  WARNING: No ground truth boxes!")
            continue
            
        # Check if boxes are reasonable
        for i, box in enumerate(box_batch):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            print(f"    Box {i}: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) "
                  f"size=({w:.1f}x{h:.1f}) class={cls_batch[i]}")
            
            if w <= 0 or h <= 0:
                print(f"    ERROR: Invalid box dimensions!")
            if x1 < 0 or y1 < 0 or x2 >= img_shape[3] or y2 >= img_shape[2]:
                print(f"    ERROR: Box outside image bounds!")
                print(f"    Image bounds: width={img_shape[3]}, height={img_shape[2]}")

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
    
    # ADD DEBUG HERE
    if debug:
        debug_target_generation(class_labels_by_batch, box_labels_by_batch, img_shape, strides)
    
    batch_size = img_shape[0]
    img_h, img_w = img_shape[2], img_shape[3]
    
    # Size ranges for target assignment
    m = (0, 20, 40, 80, 160, 320, math.inf)
    
    class_targets_by_feature = []
    centerness_target_by_feature = []
    box_targets_by_feature = []
    
    total_positive_assigned = 0  # Track total positive assignments
    
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
                if debug:
                    print(f"  Batch {batch_idx}: No boxes")
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
                    if debug and j < 3:  # Only show first few for debugging
                        print(f"    Box {j}: size {max_side:.1f} outside range [{min_box_side}, {max_box_side})")
                    continue
                
                # FIXED: Convert to feature map coordinates
                feat_center_x = center_x / stride
                feat_center_y = center_y / stride
                
                # Check bounds
                if 0 <= feat_center_x < feat_w and 0 <= feat_center_y < feat_h:
                    grid_x = int(feat_center_x)
                    grid_y = int(feat_center_y)
                    
                    # POTENTIAL FIX: Check if this location is already assigned
                    if class_target_for_feature[batch_idx, grid_y, grid_x] == 0:  # Only assign if background
                        # Assign target
                        class_target_for_feature[batch_idx, grid_y, grid_x] = class_id
                        positive_count += 1
                        
                        # Calculate regression targets (distances from center to edges)
                        left = feat_center_x - x1 / stride
                        top = feat_center_y - y1 / stride
                        right = x2 / stride - feat_center_x
                        bottom = y2 / stride - feat_center_y
                        
                        # FIXED: Ensure all distances are positive
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
                            
                            if debug and positive_count <= 3:  # Show first few assignments
                                print(f"    Assigned box {j} to grid ({grid_x}, {grid_y}): "
                                      f"class={class_id}, cent={centerness:.3f}")
                        else:
                            # Reset if distances are invalid
                            class_target_for_feature[batch_idx, grid_y, grid_x] = 0
                            positive_count -= 1
                            if debug:
                                print(f"    Invalid distances for box {j}: l={left:.2f}, t={top:.2f}, r={right:.2f}, b={bottom:.2f}")
                else:
                    if debug and j < 3:
                        print(f"    Box {j} center ({feat_center_x:.1f}, {feat_center_y:.1f}) outside feature map bounds")
        
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
