from typing import Tuple, List
import torch  # ← ADD THIS LINE
import math


def generate_targets(img_shape, class_labels_by_batch, box_labels_by_batch, strides):
    """Fixed target generation with proper debugging"""
    
    batch_size, _, img_h, img_w = img_shape
    print(f" Input image shape: {img_shape}")
    
    class_targets = []
    box_targets = []
    
    for level_idx, stride in enumerate(strides):
        feat_h, feat_w = img_h // stride, img_w // stride
        
        # ✅ Initialize targets
        cls_target = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.long)
        box_target = torch.zeros((batch_size, feat_h, feat_w, 4), dtype=torch.float32)
        
        print(f" Level {level_idx} (stride={stride}): feature map size ({feat_h}, {feat_w})")
        print(f" Target shapes: cls_target={cls_target.shape}, box_target={box_target.shape}")
        
        positive_locations = 0  # Debug counter
        
        for batch_idx, (labels, boxes) in enumerate(zip(class_labels_by_batch, box_labels_by_batch)):
            if len(boxes) == 0:
                continue
                
            for obj_idx, (label, box) in enumerate(zip(labels, boxes)):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # ✅ Add size filtering (optional - uncomment if needed)
                # max_side = max(w, h)
                # size_range = get_size_range_for_level(level_idx)  # You can implement this
                # if not (size_range[0] <= max_side < size_range[1]):
                #     continue
                
                # ✅ Improved feature map coordinate calculation
                # Use floating point for more accurate mapping
                feat_x1 = max(0, int(x1 / stride))
                feat_y1 = max(0, int(y1 / stride))
                feat_x2 = min(feat_w, int((x2 - 1) / stride) + 1)  # ✅ Fixed boundary
                feat_y2 = min(feat_h, int((y2 - 1) / stride) + 1)  # ✅ Fixed boundary
                
                for feat_y in range(feat_y1, feat_y2):
                    for feat_x in range(feat_x1, feat_x2):
                        # ✅ More precise coordinate mapping
                        img_x = feat_x * stride + stride // 2
                        img_y = feat_y * stride + stride // 2
                        
                        # ✅ Improved bounds checking
                        if (x1 <= img_x <= x2 and y1 <= img_y <= y2):
                            # Calculate distances to box edges
                            left = img_x - x1
                            top = img_y - y1
                            right = x2 - img_x
                            bottom = y2 - img_y
                            
                            # ✅ Ensure all distances are positive
                            if left > 0 and top > 0 and right > 0 and bottom > 0:
                                # Assign positive label
                                cls_target[batch_idx, feat_y, feat_x] = label
                                
                                # ✅ Direct tensor assignment (more efficient)
                                box_target[batch_idx, feat_y, feat_x, 0] = left / stride
                                box_target[batch_idx, feat_y, feat_x, 1] = top / stride
                                box_target[batch_idx, feat_y, feat_x, 2] = right / stride
                                box_target[batch_idx, feat_y, feat_x, 3] = bottom / stride
                                
                                positive_locations += 1
        
        print(f" Level {level_idx}: {positive_locations} positive locations assigned")
        
        class_targets.append(cls_target)
        box_targets.append(box_target)
    
    # ✅ Debug: Print final target list information
    print(f" Generated targets for {len(class_targets)} levels")
    for i, (cls_t, box_t) in enumerate(zip(class_targets, box_targets)):
        print(f"Level {i}: cls_target {cls_t.shape}, box_target {box_t.shape}")
        print(f"Level {i}: positive samples = {(cls_t > 0).sum().item()}")
    
    return class_targets, box_targets
