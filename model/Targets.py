from typing import Tuple, List
import torch
import math


def generate_targets(
    img_shape: torch.LongTensor,
    class_labels_by_batch: List[torch.LongTensor],
    box_labels_by_batch: List[torch.FloatTensor],
    strides: List[int],
) -> Tuple[List[torch.LongTensor], List[torch.FloatTensor]]:
    """
    Generate FCOS targets for classification and regression without centerness.
    Includes proper size-based filtering for pyramid levels.
    """
    if not len(box_labels_by_batch) == len(class_labels_by_batch) == img_shape[0]:
        raise ValueError("labels and batch size must match")

    batch_size = img_shape[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class_targets_by_feature = []
    box_targets_by_feature = []

    # Size ranges for each pyramid level (proper FCOS ranges)
    size_ranges = [64, 128, 256, 512, math.inf]

    for level_idx, stride in enumerate(strides):
        feat_h = int(img_shape[2] / stride)
        feat_w = int(img_shape[3] / stride)

        # Initialize targets for this pyramid level
        class_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, dtype=torch.long, device=device)
        box_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, 4, dtype=torch.float32, device=device)

        # Size filtering bounds for this level
        min_box_side = 0 if level_idx == 0 else size_ranges[level_idx - 1]
        max_box_side = size_ranges[level_idx]

        for batch_idx, (class_labels, box_labels) in enumerate(
            zip(class_labels_by_batch, box_labels_by_batch)
        ):
            if len(box_labels) == 0:
                continue
                
            # Convert to tensors if needed
            if not isinstance(box_labels, torch.Tensor):
                box_labels = torch.tensor(box_labels, dtype=torch.float32, device=device)
            if not isinstance(class_labels, torch.Tensor):
                class_labels = torch.tensor(class_labels, dtype=torch.long, device=device)

            # Calculate box dimensions
            heights = box_labels[:, 3] - box_labels[:, 1]
            widths = box_labels[:, 2] - box_labels[:, 0]
            areas = torch.mul(widths, heights)

            # Process boxes in descending order by area (larger boxes first)
            for j in torch.argsort(areas, dim=0, descending=True):
                box_height = heights[j].item()
                box_width = widths[j].item()
                max_side = max(box_height, box_width)

                # Size-based filtering - only assign to appropriate pyramid level
                if max_side < min_box_side or max_side >= max_box_side:
                    continue

                # Map box coordinates to feature map coordinates
                x1, y1, x2, y2 = box_labels[j].tolist()
                
                min_x = max(int(x1 / stride), 0)
                min_y = max(int(y1 / stride), 0)
                max_x = min(int(x2 / stride) + 1, feat_w)
                max_y = min(int(y2 / stride) + 1, feat_h)

                # Only assign to interior points (avoid boundary conflicts)
                if max_x > min_x + 1 and max_y > min_y + 1:
                    # Assign classification targets to interior region
                    class_target_for_feature[
                        batch_idx, min_y + 1:max_y - 1, min_x + 1:max_x - 1
                    ] = class_labels[j]

                    # Assign regression targets to interior points
                    for x in range(min_x + 1, max_x - 1):
                        for y in range(min_y + 1, max_y - 1):
                            # Calculate distances to box edges in feature map coordinates
                            left = x - x1 / stride
                            top = y - y1 / stride
                            right = x2 / stride - x
                            bottom = y2 / stride - y
                            
                            # Only assign if all distances are positive (inside the box)
                            if left > 0 and top > 0 and right > 0 and bottom > 0:
                                box_target_for_feature[batch_idx, y, x] = torch.tensor(
                                    [left, top, right, bottom], 
                                    dtype=torch.float32, 
                                    device=device
                                )

        class_targets_by_feature.append(class_target_for_feature)
        box_targets_by_feature.append(box_target_for_feature)

    return class_targets_by_feature, box_targets_by_feature
