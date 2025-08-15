import math

import cv2
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torchvision

from AttentionFPN import FPN, normalize_batch

MIN_SCORE = 0.05
DEFAULT_MAX_DETECTIONS = 3000


@dataclass
class Detection:
    score: float
    object_class: int
    bbox: np.ndarray  # (min_x, min_y, max_x, max_y)


def render_detections_to_image(img: np.ndarray, detections: List[Detection]):
    for detection in detections:
        if detection.score > 0.3:
            start_point = (detection.bbox[0], detection.bbox[1])
            end_point = (detection.bbox[2], detection.bbox[3])
            img = cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)

    return img


def compute_detections(model: FPN, img: np.ndarray, device) -> List[Detection]:
    """
    Take an image using opencv conventions and return a list of detections.
    """
    tensor = img
    return compute_detections_for_tensor(model, tensor, device)


def compute_detections_for_tensor(model: FPN, x, device) -> List[Detection]:
    """
    Updated for single detection layer output from the modified FPN
    """
    with torch.no_grad():
        x = x.to(device)
        batch = normalize_batch(x)
        
        # Updated: Model now returns single tensors instead of lists
        classes, centernesses, boxes, attention_maps = model(batch)
        
        # Convert single tensors to lists for compatibility with existing code
        classes_list = [classes]
        centernesses_list = [centernesses]  
        boxes_list = [boxes]
        
        # Single scale and stride (now just one value)
        scales = model.scales  # This is a tensor with one element
        strides = model.strides  # This is a list with one element
        
        return detections_from_network_output(
            x.shape[2], x.shape[3], 
            classes_list, centernesses_list, boxes_list, 
            scales, strides
        )


def detections_from_network_output(
    img_height, img_width, classes, centernesses, boxes, scales, strides
) -> List[List[Detection]]:
    """
    Updated to handle single detection layer
    """
    all_classes = []
    all_centernesses = []
    all_boxes = []

    n_classes = classes[0].shape[-1]
    batch_size = classes[0].shape[0]

    # Now we only have one feature level
    for feat_classes, feat_centernesses, feat_boxes, scale, stride in zip(
        classes, centernesses, boxes, scales, strides
    ):
        # Convert scale from tensor to scalar if needed
        if isinstance(scale, torch.Tensor):
            scale = scale.item()
            
        boxes = _boxes_from_regression(feat_boxes, img_height, img_width, scale, stride)

        all_classes.append(feat_classes.view(batch_size, -1, n_classes))
        all_centernesses.append(feat_centernesses.view(batch_size, -1))
        all_boxes.append(boxes.view(batch_size, -1, 4))

    classes_ = torch.cat(all_classes, dim=1)
    centernesses_ = torch.cat(all_centernesses, dim=1)
    boxes_ = torch.cat(all_boxes, dim=1)

    gathered_boxes, gathered_classes, gathered_scores = _gather_detections(classes_, centernesses_, boxes_)
    return detections_from_net(gathered_boxes, gathered_classes, gathered_scores)


def _boxes_from_regression(reg, img_height, img_width, scale, stride):
    """
    Returns B[x_min, y_min, x_max, y_max], in image space, given regression
    values, which represent offsets (left, top, right, bottom).
    """
    half_stride = stride // 2
    _, rows, cols, _ = reg.shape

    # Create coordinate grids based on actual feature map size and stride
    y = torch.arange(0, rows, dtype=torch.float32, device=reg.device) * stride + half_stride
    x = torch.arange(0, cols, dtype=torch.float32, device=reg.device) * stride + half_stride

    center_y, center_x = torch.meshgrid(y, x, indexing='ij')

    # Expand dimensions to match batch size
    center_x = center_x.unsqueeze(0).expand(reg.shape[0], -1, -1)
    center_y = center_y.unsqueeze(0).expand(reg.shape[0], -1, -1)

    # Calculate bounding box coordinates
    x_min = center_x - reg[:, :, :, 0]
    y_min = center_y - reg[:, :, :, 1] 
    x_max = center_x + reg[:, :, :, 2]
    y_max = center_y + reg[:, :, :, 3]

    # Clamp to image boundaries
    x_min = torch.clamp(x_min, 0, img_width)
    y_min = torch.clamp(y_min, 0, img_height)
    x_max = torch.clamp(x_max, 0, img_width)
    y_max = torch.clamp(y_max, 0, img_height)

    return torch.stack([x_min, y_min, x_max, y_max], dim=3)


def detections_from_net(boxes_by_batch, classes_by_batch, scores_by_batch=None) -> List[List[Detection]]:
    """
    - BHW[c] class index of each box (int)
    - BHW[p] class probability of each box (float)
    - BHW[min_x, y_min, x_min, y_max, x_max] (box dimensions, floats)
    """
    result = []

    for batch in range(len(classes_by_batch)):
        scores = scores_by_batch[batch] if scores_by_batch is not None else None
        classes = classes_by_batch[batch]
        boxes = boxes_by_batch[batch]

        result.append(
            [
                Detection(
                    score=scores[i].item() if scores is not None else 1.0,
                    object_class=classes[i].item(),
                    bbox=boxes[i].cpu().numpy().astype(int),
                )
                for i in range(boxes.shape[0])
                if classes[i] != 0
            ]
        )

    return result


def _gather_detections(classes, centernesses, boxes, max_detections=DEFAULT_MAX_DETECTIONS):
    """
    Gather and filter detections from network outputs
    """
    # classes: BHW[c] class probabilities
    class_scores, class_indices = torch.max(classes, dim=2)

    boxes_by_batch = []
    classes_by_batch = []
    scores_by_batch = []

    n_batches = boxes.shape[0]
    for i in range(n_batches):
        # Filter out background predictions (class 0)
        non_background_points = class_indices[i] > 0

        class_scores_i = class_scores[i][non_background_points]
        boxes_i = boxes[i][non_background_points]
        centerness_i = torch.sigmoid(centernesses[i][non_background_points])  # Apply sigmoid to centerness
        class_indices_i = class_indices[i][non_background_points]

        # Combine classification and centerness scores
        final_scores_i = class_scores_i * centerness_i

        # Filter by minimum score threshold
        score_filter = final_scores_i > MIN_SCORE
        final_scores_i = final_scores_i[score_filter]
        boxes_i = boxes_i[score_filter]
        class_indices_i = class_indices_i[score_filter]

        if final_scores_i.shape[0] == 0:
            # No detections for this batch
            boxes_by_batch.append(torch.empty((0, 4), device=boxes.device))
            classes_by_batch.append(torch.empty((0,), dtype=torch.long, device=boxes.device))
            scores_by_batch.append(torch.empty((0,), device=boxes.device))
            continue

        # Get top K detections
        num_detections = min(final_scores_i.shape[0], max_detections)
        if num_detections > 0:
            _, top_detection_indices = torch.topk(final_scores_i, num_detections, dim=0)

            top_boxes_i = torch.index_select(boxes_i, 0, top_detection_indices)
            top_classes_i = torch.index_select(class_indices_i, 0, top_detection_indices)
            top_scores_i = torch.index_select(final_scores_i, 0, top_detection_indices)

            # Apply Non-Maximum Suppression
            if top_boxes_i.shape[0] > 0:
                boxes_to_keep = torchvision.ops.nms(top_boxes_i, top_scores_i, 0.5)  # Lower NMS threshold for denser objects

                top_boxes_i = top_boxes_i[boxes_to_keep]
                top_classes_i = top_classes_i[boxes_to_keep]
                top_scores_i = top_scores_i[boxes_to_keep]
        else:
            top_boxes_i = torch.empty((0, 4), device=boxes.device)
            top_classes_i = torch.empty((0,), dtype=torch.long, device=boxes.device)
            top_scores_i = torch.empty((0,), device=boxes.device)

        boxes_by_batch.append(top_boxes_i)
        classes_by_batch.append(top_classes_i)
        scores_by_batch.append(top_scores_i)

    return boxes_by_batch, classes_by_batch, scores_by_batch