from typing import Tuple, List
import torch
import math


def generate_targets(
    img_shape: torch.LongTensor,
    class_labels_by_batch: List[torch.LongTensor],
    box_labels_by_batch: List[torch.FloatTensor],
    strides: List[int],
) -> List[Tuple[torch.FloatTensor, torch.LongTensor]]:
    """
    Given the shape of the input image, the box and class labels, and the stride of each
    feature map, construct the model targets for FCOS at each feature map scale.
    """
    if not len(box_labels_by_batch) == len(class_labels_by_batch) == img_shape[0]:
        raise ValueError("labels and batch size must match")

    batch_size = img_shape[0]

    class_targets_by_feature = []
    centerness_target_by_feature = []
    box_targets_by_feature = []

    # Using overlapping size ranges for better detection coverage
    # Format: [(min_size, max_size), ...]
    # This allows objects at boundary sizes to be detected by multiple levels
    size_ranges = [
        (0, 64),      # Level 0 (stride 8): very small objects
        (32, 128),     # Level 1 (stride 16): small objects  
        (128, 256),    # Level 2 (stride 32): medium objects
        (256, 512),    # Level 3 (stride 64): large objects
        (512, math.inf) # Level 4 (stride 128): very large objects
    ]

    for i, stride in enumerate(strides):
        feat_h = int((img_shape[2] + stride - 1) // stride)  # Ceiling division
        feat_w = int((img_shape[3] + stride - 1) // stride)

        class_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, dtype=int)
        centerness_target_for_feature = torch.zeros(batch_size, feat_h, feat_w)
        box_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, 4)

        min_box_side, max_box_side = size_ranges[i]

        for batch_idx, (class_labels, box_labels) in enumerate(
            zip(class_labels_by_batch, box_labels_by_batch)
        ):
            heights = box_labels[:, 3] - box_labels[:, 1]
            widths = box_labels[:, 2] - box_labels[:, 0]
            areas = torch.mul(widths, heights)

            for j in torch.argsort(areas, dim=0, descending=True):
                if max(heights[j], widths[j]) < min_box_side or max(heights[j], widths[j]) > max_box_side:
                    continue

                min_x = max(int(box_labels[j][0] / stride), 0)
                min_y = max(int(box_labels[j][1] / stride), 0)
                max_x = min(int(box_labels[j][2] / stride) + 1, feat_w)
                max_y = min(int(box_labels[j][3] / stride) + 1, feat_h)

                mid_x = (max_x + min_x) / 2.0
                mid_y = (max_y + min_y) / 2.0
                b_w = max_x - min_x
                b_h = max_y - min_y
                for x in range(min_x, max_x):
                    for y in range(min_y, max_y):
                        if x == 0 or y == 0 or x == max_x - 1 or y == max_y - 1:
                            dist = 0
                        else:
                            l = x - min_x
                            r = max_x - 1 - x
                            t = y - min_y
                            b = max_y - 1 - y
                            dist = math.sqrt(min(l, r) / float(max(l, r)) * min(t, b) / float(max(t, b)))

                        centerness = dist
                        centerness_target_for_feature[batch_idx, y, x] = centerness

                class_target_for_feature[
                    batch_idx, min_y + 1 : max_y - 1, min_x + 1 : max_x - 1
                ] = class_labels[j]

                for x in range(min_x + 1, max_x - 1):
                    for y in range(min_y + 1, max_y - 1):
                        target = torch.Tensor(
                            [
                                x - float(box_labels[j][0]) / stride,
                                y - float(box_labels[j][1]) / stride,
                                float(box_labels[j][2]) / stride - x,
                                float(box_labels[j][3]) / stride - y,
                            ]
                        )
                        box_target_for_feature[batch_idx, y, x] = target

        class_targets_by_feature.append(class_target_for_feature)
        centerness_target_by_feature.append(centerness_target_for_feature)
        box_targets_by_feature.append(box_target_for_feature)

    # Return [BHWC, BHW[min_x, min_y, max_x, max_y]]
    return class_targets_by_feature, centerness_target_by_feature, box_targets_by_feature