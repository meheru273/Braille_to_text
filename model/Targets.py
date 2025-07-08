from typing import Tuple, List
import torch  # ← ADD THIS LINE
import math
def generate_targets(img_shape, class_labels_by_batch, box_labels_by_batch, strides):

    batch_size, _, img_h, img_w = img_shape
    
    class_targets = []
    centerness_targets = []
    box_targets = []
    
    # Use original FCOS size ranges
    size_ranges = [ (0, 64), (32, 128), (64, 256), (128, 512), (256, float('inf'))]
    
    for level_idx, stride in enumerate(strides):
        feat_h, feat_w = img_h // stride, img_w // stride
        min_size, max_size = size_ranges[level_idx]
    
        cls_target = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.long)
        cen_target = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.float32)
        box_target = torch.zeros((batch_size, feat_h, feat_w, 4), dtype=torch.float32)
        
        for batch_idx, (labels, boxes) in enumerate(zip(class_labels_by_batch, box_labels_by_batch)):
            if len(boxes) == 0:
                continue
                
            for obj_idx, (label, box) in enumerate(zip(labels, boxes)):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                max_side = max(w, h)
                
                if not (min_size <= max_side < max_size):
                    continue
                
                # Find all feature map locations inside this box
                feat_x1 = max(0, int(x1 / stride))
                feat_y1 = max(0, int(y1 / stride))
                feat_x2 = min(feat_w, int(x2 / stride) + 1)
                feat_y2 = min(feat_h, int(y2 / stride) + 1)
                
                for feat_y in range(feat_y1, feat_y2):
                    for feat_x in range(feat_x1, feat_x2):
                        # Convert back to image coordinates
                        img_x = feat_x * stride + stride // 2
                        img_y = feat_y * stride + stride // 2
                        
                        # Check if this location is inside the box
                        if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                            # Calculate distances to box edges
                            left = img_x - x1
                            top = img_y - y1
                            right = x2 - img_x
                            bottom = y2 - img_y
                            
                            if left > 0 and top > 0 and right > 0 and bottom > 0:
                                # Assign positive label
                                cls_target[batch_idx, feat_y, feat_x] = label
                                
                                # Calculate centerness
                                centerness = math.sqrt(
                                    (min(left, right) / max(left, right)) *
                                    (min(top, bottom) / max(top, bottom))
                                )
                                cen_target[batch_idx, feat_y, feat_x] = centerness
                                
                                # Box regression targets (normalized by stride)
                                box_target[batch_idx, feat_y, feat_x] = torch.tensor([
                                    left / stride, top / stride, right / stride, bottom / stride
                                ])
        
        class_targets.append(cls_target)
        centerness_targets.append(cen_target)
        box_targets.append(box_target)
    
    return class_targets, centerness_targets, box_targets
