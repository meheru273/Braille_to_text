from typing import Tuple, List
import torch  # ← ADD THIS LINE
import math


from typing import Tuple, List
import torch
import math


def generate_targets(img_shape, class_labels_by_batch, box_labels_by_batch, strides):
    """Memory-efficient target generation with comprehensive debugging"""
    
    batch_size, _, img_h, img_w = img_shape
    
    # ✅ CRITICAL: Determine device early
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # === INPUT VALIDATION AND DEBUGGING ===
    print(f"\n=== TARGET GENERATION DEBUG ===")
    print(f"Input image shape: {img_shape}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Image dimensions: {img_h} x {img_w}")
    print(f"Strides: {strides}")
    print(f"Number of batches: {len(class_labels_by_batch)}")
    
    # Debug input annotations
    total_annotations = 0
    for batch_idx, (labels, boxes) in enumerate(zip(class_labels_by_batch, box_labels_by_batch)):
        print(f"Batch {batch_idx}: {len(boxes)} annotations, classes: {labels}")
        total_annotations += len(boxes)
        
        # Show sample boxes
        if len(boxes) > 0:
            for i, box in enumerate(boxes[:3]):  # Show first 3 boxes
                width = box[2] - box[0]
                height = box[3] - box[1]
                print(f"  Box {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}] "
                      f"size: {width:.1f}x{height:.1f}, area: {width*height:.1f}")
    
    print(f"Total annotations across all batches: {total_annotations}")
    
    if total_annotations == 0:
        print("⚠️  WARNING: No annotations found - model will only learn background!")
    
    class_targets = []
    box_targets = []
    
    # Size ranges for FPN levels (like working FCOS)
    size_ranges = [(0, 32), (24, 64), (32, 84), (64, 128), (128, float('inf'))]
    print(f"Size ranges for pyramid levels: {size_ranges}")
    
    total_positive_locations = 0
    
    for level_idx, stride in enumerate(strides):
        print(f"\n--- PROCESSING PYRAMID LEVEL {level_idx} (stride={stride}) ---")
        
        feat_h, feat_w = img_h // stride, img_w // stride
        print(f"Feature map size: {feat_h} x {feat_w}")
        
        # ✅ FIXED: Create tensors on correct device from start
        cls_target = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.long, device=device)
        box_target = torch.zeros((batch_size, feat_h, feat_w, 4), dtype=torch.float32, device=device)
        
        positive_locations = 0
        level_assignments = 0
        filtered_boxes = 0
        
        min_size, max_size = size_ranges[level_idx]
        print(f"Size range for this level: {min_size} - {max_size}")
        
        for batch_idx, (labels, boxes) in enumerate(zip(class_labels_by_batch, box_labels_by_batch)):
            print(f"  Processing batch {batch_idx} with {len(boxes)} boxes")
            
            if len(boxes) == 0:
                print(f"    No boxes in batch {batch_idx}")
                continue
            
            # ✅ FIXED: Convert to tensors and sort by area
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
            
            heights = boxes_tensor[:, 3] - boxes_tensor[:, 1]
            widths = boxes_tensor[:, 2] - boxes_tensor[:, 0]
            areas = heights * widths
            sorted_indices = torch.argsort(areas, descending=True)
            
            print(f"    Box areas: {areas.tolist()}")
            print(f"    Processing boxes in order of decreasing area: {sorted_indices.tolist()}")
            
            for box_num, idx in enumerate(sorted_indices):
                label = labels_tensor[idx].item()
                x1, y1, x2, y2 = boxes_tensor[idx].tolist()
                
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                print(f"    Box {box_num} (class {label}): "
                      f"coords=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), "
                      f"size={box_width:.1f}x{box_height:.1f}")
                
                # ✅ ADDED: Size filtering
                max_side = max(x2 - x1, y2 - y1)
                print(f"      Max side: {max_side:.1f}, range check: {min_size} <= {max_side:.1f} < {max_size}")
                
                if max_side < min_size or max_side >= max_size:
                    print(f"      ❌ Box filtered out - size {max_side:.1f} not in range [{min_size}, {max_size})")
                    filtered_boxes += 1
                    continue
                
                # ✅ FIXED: Efficient coordinate mapping
                min_x = max(int(x1 / stride), 0)
                min_y = max(int(y1 / stride), 0)
                max_x = min(int(x2 / stride) + 1, feat_w)
                max_y = min(int(y2 / stride) + 1, feat_h)
                
                print(f"      Mapped coordinates: feature_map[{min_y}:{max_y}, {min_x}:{max_x}]")
                print(f"      Interior region: feature_map[{min_y+1}:{max_y-1}, {min_x+1}:{max_x-1}]")
                
                # Assign targets to interior points only
                if max_x > min_x + 1 and max_y > min_y + 1:
                    interior_width = (max_x - 1) - (min_x + 1)
                    interior_height = (max_y - 1) - (min_y + 1)
                    interior_pixels = interior_width * interior_height
                    
                    print(f"      ✅ Assigning {interior_pixels} interior pixels to class {label}")
                    
                    cls_target[batch_idx, min_y + 1:max_y - 1, min_x + 1:max_x - 1] = label
                    level_assignments += 1
                    
                    # Generate regression targets
                    pixels_assigned = 0
                    for x in range(min_x + 1, max_x - 1):
                        for y in range(min_y + 1, max_y - 1):
                            # Calculate distances to box edges in feature map coordinates
                            left = x * stride - x1
                            top = y * stride - y1
                            right = x2 - x * stride
                            bottom = y2 - y * stride
                            
                            # Verify all distances are positive
                            if left > 0 and top > 0 and right > 0 and bottom > 0:
                                box_target[batch_idx, y, x] = torch.tensor([left, top, right, bottom], device=device)
                                positive_locations += 1
                                pixels_assigned += 1
                            else:
                                print(f"        ⚠️  Invalid distances at ({x},{y}): "
                                      f"left={left:.2f}, top={top:.2f}, right={right:.2f}, bottom={bottom:.2f}")
                    
                    print(f"      Assigned regression targets to {pixels_assigned} pixels")
                else:
                    print(f"      ❌ Box too small after mapping - no interior pixels")
                    print(f"      Need: max_x({max_x}) > min_x+1({min_x+1}) and max_y({max_y}) > min_y+1({min_y+1})")
        
        print(f"  Level {level_idx} summary:")
        print(f"    Boxes assigned: {level_assignments}")
        print(f"    Boxes filtered: {filtered_boxes}")
        print(f"    Positive pixel locations: {positive_locations}")
        total_positive_locations += positive_locations
        
        # Sample the target tensors to verify content
        pos_cls_count = (cls_target > 0).sum().item()
        pos_box_count = (box_target.sum(dim=-1) != 0).sum().item()
        
        print(f"    Classification target stats: {pos_cls_count} positive pixels")
        print(f"    Box target stats: {pos_box_count} non-zero regression targets")
        
        if pos_cls_count != pos_box_count:
            print(f"    ⚠️  Mismatch between classification and regression positive counts!")
        
        class_targets.append(cls_target)
        box_targets.append(box_target)
    
    # === FINAL SUMMARY ===
    print(f"\n=== TARGET GENERATION SUMMARY ===")
    print(f"Total positive locations across all levels: {total_positive_locations}")
    print(f"Number of target levels generated: {len(class_targets)}")
    
    for i, (cls_target, box_target) in enumerate(zip(class_targets, box_targets)):
        pos_cls = (cls_target > 0).sum().item()
        pos_box = (box_target.sum(dim=-1) != 0).sum().item()
        total_pixels = cls_target.numel()
        
        print(f"Level {i}: {pos_cls} positive classification targets ({pos_cls/total_pixels*100:.3f}%)")
        print(f"         {pos_box} positive regression targets")
    
    if total_positive_locations == 0:
        print("🚨 CRITICAL: No positive targets generated - model will not learn to detect objects!")
        print("   Possible causes:")
        print("   - All boxes filtered out by size ranges")
        print("   - Boxes too small after stride mapping")
        print("   - Coordinate system mismatch")
        print("   - Invalid annotation format")
    else:
        print("✅ Target generation completed successfully")
    
    return class_targets, box_targets
