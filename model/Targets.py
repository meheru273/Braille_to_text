from typing import Tuple, List
import torch
import math

def generate_targets(
    img_shape: torch.LongTensor,
    class_labels_by_batch: List[torch.LongTensor],
    box_labels_by_batch: List[torch.FloatTensor],
    strides: List[int],
) -> Tuple[List[torch.LongTensor], List[torch.FloatTensor], List[torch.FloatTensor]]:
    """
    Generate targets for FCOS with proper error handling and return statement
    """
    print(f"generate_targets called with img_shape: {img_shape}")
    print(f"Number of strides: {len(strides)}")
    print(f"Strides: {strides}")
    
    if not len(box_labels_by_batch) == len(class_labels_by_batch) == img_shape[0]:
        raise ValueError("labels and batch size must match")

    batch_size = img_shape[0]
    img_h, img_w = img_shape[2], img_shape[3]

    class_targets_by_feature = []
    centerness_target_by_feature = []
    box_targets_by_feature = []

    # Size ranges for 5 FPN levels
    m = ( 40,60,80, 160, math.inf)

    try:
        for i, stride in enumerate(strides):
            print(f"Processing level {i}, stride {stride}")
            
            feat_h = int(img_h / stride)
            feat_w = int(img_w / stride)
            

            # Initialize targets with correct dtypes
            class_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, dtype=torch.long)
            centerness_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, dtype=torch.float32)
            box_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, 4, dtype=torch.float32)

            min_box_side = m[i]
            max_box_side = m[i + 1]

            for batch_idx, (class_labels, box_labels) in enumerate(
                zip(class_labels_by_batch, box_labels_by_batch)
            ):
                if len(box_labels) == 0:
                    continue
                    
                # Calculate box dimensions
                heights = box_labels[:, 3] - box_labels[:, 1]
                widths = box_labels[:, 2] - box_labels[:, 0]
                areas = torch.mul(widths, heights)

                # Process boxes from largest to smallest
                for j in torch.argsort(areas, dim=0, descending=True):
                    box = box_labels[j]  # [x1, y1, x2, y2]
                    class_id = class_labels[j]
                    
                    # Check if box size is appropriate for this FPN level
                    max_side = max(heights[j], widths[j])
                    if max_side < min_box_side or max_side >= max_box_side:
                        continue

                    # Convert box coordinates to feature map coordinates
                    x1, y1, x2, y2 = box
                    fx1, fy1 = x1 / stride, y1 / stride
                    fx2, fy2 = x2 / stride, y2 / stride
                    
                    # Get the range of feature map cells this box covers
                    min_x = max(int(fx1), 0)
                    min_y = max(int(fy1), 0)
                    max_x = min(int(fx2) + 1, feat_w)
                    max_y = min(int(fy2) + 1, feat_h)
                    
                    # Skip if box is outside feature map
                    if min_x >= max_x or min_y >= max_y:
                        continue

                    # Assign targets to feature map cells
                    for x in range(min_x, max_x):
                        for y in range(min_y, max_y):
                            # Calculate distances from cell center to box edges
                            cell_center_x = x + 0.5
                            cell_center_y = y + 0.5
                            
                            left = cell_center_x - fx1
                            top = cell_center_y - fy1
                            right = fx2 - cell_center_x
                            bottom = fy2 - cell_center_y
                            
                            # Only assign if cell center is inside the box
                            if left > 0 and top > 0 and right > 0 and bottom > 0:
                                # Set class target
                                class_target_for_feature[batch_idx, y, x] = class_id
                                
                                # Calculate centerness
                                min_lr = min(left, right)
                                max_lr = max(left, right)
                                min_tb = min(top, bottom)
                                max_tb = max(top, bottom)
                                
                                if max_lr > 0 and max_tb > 0:
                                    centerness = math.sqrt((min_lr / max_lr) * (min_tb / max_tb))
                                else:
                                    centerness = 0.0
                                    
                                centerness_target_for_feature[batch_idx, y, x] = centerness
                                
                                # Set box regression targets (distances to edges)
                                box_target_for_feature[batch_idx, y, x] = torch.tensor([
                                    left, top, right, bottom
                                ], dtype=torch.float32)

            class_targets_by_feature.append(class_target_for_feature)
            centerness_target_by_feature.append(centerness_target_for_feature)
            box_targets_by_feature.append(box_target_for_feature)

        print(f"Successfully generated targets for {len(class_targets_by_feature)} levels")
        # Debug: Check if targets are actually being created
        for i, (class_targets, centerness_targets, box_targets) in enumerate(zip(class_targets_by_feature, centerness_target_by_feature, box_targets_by_feature)):
            positive_samples = (class_targets > 0).sum()
            print(f"Level {i}: Positive samples = {positive_samples}")
            if positive_samples > 0:
                print(f"  Class range: {class_targets.min()}-{class_targets.max()}")
                print(f"  Centerness range: {centerness_targets.min():.4f}-{centerness_targets.max():.4f}")
        # IMPORTANT: Make sure we return the tuple
        return class_targets_by_feature, centerness_target_by_feature, box_targets_by_feature
        
    except Exception as e:
        print(f"Error in generate_targets: {e}")
        import traceback
        traceback.print_exc()
        # Return empty targets to avoid None
        return [], [], []
