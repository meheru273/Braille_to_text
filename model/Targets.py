from typing import Tuple, List
import torch  # ← ADD THIS LINE
import math


def generate_targets(img_shape, class_labels_by_batch, box_labels_by_batch, strides):
    """Memory-efficient target generation based on working FCOS implementation"""
    
    batch_size, _, img_h, img_w = img_shape
    
    # ✅ CRITICAL: Determine device early
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    class_targets = []
    box_targets = []
    
    # Size ranges for FPN levels (like working FCOS)
    size_ranges = [(0, 32), (24, 64), (32, 84), (64, 128), (128, float('inf'))]
    
    for level_idx, stride in enumerate(strides):
        feat_h, feat_w = img_h // stride, img_w // stride
        
        # ✅ FIXED: Create tensors on correct device from start
        cls_target = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.long, device=device)
        box_target = torch.zeros((batch_size, feat_h, feat_w, 4), dtype=torch.float32, device=device)
        
        
        positive_locations = 0
        min_size, max_size = size_ranges[level_idx]
        
        for batch_idx, (labels, boxes) in enumerate(zip(class_labels_by_batch, box_labels_by_batch)):
            if len(boxes) == 0:
                continue
            
            # ✅ FIXED: Convert to tensors and sort by area
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
            
            heights = boxes_tensor[:, 3] - boxes_tensor[:, 1]
            widths = boxes_tensor[:, 2] - boxes_tensor[:, 0]
            areas = heights * widths
            sorted_indices = torch.argsort(areas, descending=True)
            
            for idx in sorted_indices:
                label = labels_tensor[idx].item()
                x1, y1, x2, y2 = boxes_tensor[idx].tolist()
                
                # ✅ ADDED: Size filtering
                max_side = max(x2 - x1, y2 - y1)
                if max_side < min_size or max_side >= max_size:
                    continue
                
                # ✅ FIXED: Efficient coordinate mapping
                min_x = max(int(x1 / stride), 0)
                min_y = max(int(y1 / stride), 0)
                max_x = min(int(x2 / stride) + 1, feat_w)
                max_y = min(int(y2 / stride) + 1, feat_h)
                
                # Assign targets to interior points only
                if max_x > min_x + 1 and max_y > min_y + 1:
                    cls_target[batch_idx, min_y + 1:max_y - 1, min_x + 1:max_x - 1] = label
                    
                    for x in range(min_x + 1, max_x - 1):
                        for y in range(min_y + 1, max_y - 1):
                            left = x - x1 / stride
                            top = y - y1 / stride
                            right = x2 / stride - x
                            bottom = y2 / stride - y
                            
                            box_target[batch_idx, y, x] = torch.tensor([left, top, right, bottom], device=device)
                            positive_locations += 1
            
        
        class_targets.append(cls_target)
        box_targets.append(box_target)
    
    return class_targets, box_targets
