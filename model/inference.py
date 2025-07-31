import math

import cv2
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torchvision

from FPNAttention import FPN, normalize_batch

MIN_SCORE = 0.001
DEFAULT_MAX_DETECTIONS = 3000


@dataclass
class Detection:
    score: float
    object_class: int
    bbox: np.ndarray  # (min_x, min_y, max_x, max_y)


def render_detections_to_image(img: np.ndarray, detections: List[Detection]):
    # Ensure img is a contiguous uint8 NumPy array
    img = np.ascontiguousarray(img, dtype=np.uint8)
    for detection in detections:
        if detection.score > 0.0:
            x1, y1, x2, y2 = detection.bbox
            start_point = (int(round(x1)), int(round(y1)))
            end_point   = (int(round(x2)), int(round(y2)))
            # cv2.rectangle works in-place; no need to reassign
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
    return img

def compute_detections(model: FPN, img: np.ndarray, device) -> List[Detection]:
    
    model.eval()
    if img.dtype == np.uint8:
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
    else:
        # assume float [0,1]
        tensor = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0)
    tensor = tensor.to(device)

    # 2. Normalize batch
    batch = normalize_batch(tensor)

    # 3. Forward pass
    with torch.no_grad():
        classes_by_feature, boxes_by_feature = model(batch)

    # 4. Decode detections
    H, W = tensor.shape[2], tensor.shape[3]
    detections = detections_from_network_output(
        H, W,
        classes_by_feature, boxes_by_feature,
        model.scales, model.strides
    )
    # returns List[List[Detection]]; since batch size=1, you can return detections[0]
    return detections[0]


def detections_from_network_output(
    img_height, img_width, classes, boxes, scales, strides
) -> List[List[Detection]]:
    all_classes = []
    all_boxes = []
    n_classes = classes[0].shape[-1]
    batch_size = classes[0].shape[0]

    for feat_classes, feat_boxes, scale, stride in zip(
        classes,  boxes, scales, strides
    ):
        boxes = _boxes_from_regression(feat_boxes, img_height, img_width, scale, stride)

        all_classes.append(feat_classes.view(batch_size, -1, n_classes))
        all_boxes.append(boxes.view(batch_size, -1, 4))

    classes_ = torch.cat(all_classes, dim=1)
    boxes_ = torch.cat(all_boxes, dim=1)

    gathered_boxes, gathered_classes, gathered_scores = _gather_detections(classes_, boxes_)
    return detections_from_net(gathered_boxes, gathered_classes, gathered_scores)


def _boxes_from_regression(reg, img_height, img_width, scale, stride):
    """Returns boxes in image space without background consideration"""
    half_stride = stride // 2
    batch, rows, cols, _ = reg.shape

    y = torch.linspace(0, img_height - stride, rows).to(reg.device)
    x = torch.linspace(0, img_width - stride, cols).to(reg.device)

    center_y, center_x = torch.meshgrid(y, x, indexing='ij')
    center_y = center_y.unsqueeze(0).expand(batch, -1, -1)
    center_x = center_x.unsqueeze(0).expand(batch, -1, -1)

    # Direct box calculation without background handling
    x_min = center_x - reg[..., 0] * stride
    y_min = center_y - reg[..., 1] * stride
    x_max = center_x + reg[..., 2] * stride
    y_max = center_y + reg[..., 3] * stride

    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def detections_from_net(boxes_by_batch, classes_by_batch, scores_by_batch=None) -> List[List[Detection]]:
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
                # REMOVED BACKGROUND FILTER: if classes[i] != 0
            ]
        )

    return result

# Add this to inference.py after the existing detections_from_network_output function

def debug_detections_from_network_output(
    img_height, img_width, classes, boxes, scales, strides
):
    print("\n=== DETAILED DETECTION DEBUGGING ===")
    
    all_classes = []
    all_boxes = []
    n_classes = classes[0].shape[-1]
    batch_size = classes[0].shape[0]
    
    for i, (feat_classes, feat_boxes, scale, stride) in enumerate(zip(
        classes, boxes, scales, strides)):
        
        # Check class predictions
        class_scores, class_indices = torch.max(feat_classes, dim=-1)
        print(f"  Class scores: max={class_scores.max():.4f}, mean={class_scores.mean():.4f}")
        print(f"  Non-background pixels: {(class_indices > 0).sum()}")
        
        
        # Convert boxes
        boxes_converted = _boxes_from_regression(feat_boxes, img_height, img_width, scale, stride)
        
        all_classes.append(feat_classes.view(batch_size, -1, n_classes))
        all_boxes.append(boxes_converted.view(batch_size, -1, 4))
    
    # Continue with normal processing...
    classes_ = torch.cat(all_classes, dim=1)
    boxes_ = torch.cat(all_boxes, dim=1)
    
    print(f"\nCombined stats:")
    print(f"  Total predictions: {classes_.shape[1]}")
    
    # Debug the gathering process
    class_scores, class_indices = torch.max(classes_, dim=2)
    print(f"  Non-background: {(class_indices[0] > 0).sum()}")
    
    
    return _gather_detections(classes_, boxes_)



def _gather_detections(classes, boxes, max_detections=DEFAULT_MAX_DETECTIONS):
    # Get class scores and indices
    class_scores, class_indices = torch.max(classes, dim=2)
    boxes_by_batch = []
    classes_by_batch = []
    scores_by_batch = []

    n_batches = boxes.shape[0]
    for i in range(n_batches):
        # REMOVED: non_background_points = class_indices[i] > 0
        # Process ALL points
        
        class_scores_i = class_scores[i]  # Use all scores
        boxes_i = boxes[i]
        class_indices_i = class_indices[i]

        non_minimal_points = class_scores_i > MIN_SCORE

        class_scores_i = class_scores_i[non_minimal_points]
        boxes_i = boxes_i[non_minimal_points]
        class_indices_i = class_indices_i[non_minimal_points]

        num_detections = min(class_scores_i.shape[0], max_detections)
        _, top_detection_indices = torch.topk(class_scores_i, num_detections, dim=0)

        top_boxes_i = torch.index_select(boxes_i, 0, top_detection_indices)
        top_classes_i = torch.index_select(class_indices_i, 0, top_detection_indices)
        top_scores_i = torch.index_select(class_scores_i, 0, top_detection_indices)

        boxes_to_keep = torchvision.ops.nms(top_boxes_i, top_scores_i, 0.1)

        top_boxes_i = top_boxes_i[boxes_to_keep]
        top_classes_i = top_classes_i[boxes_to_keep]
        top_scores_i = top_scores_i[boxes_to_keep]
        
        boxes_by_batch.append(top_boxes_i)
        classes_by_batch.append(top_classes_i)
        scores_by_batch.append(top_scores_i)

    return boxes_by_batch, classes_by_batch, scores_by_batch

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    
    arr = tensor.detach().cpu().numpy()

    if arr.max() <= 1.0:
        arr = arr * 255.0
    
    if len(arr.shape) == 3:  
        img = arr.transpose(1, 2, 0)  # [H, W, C]
    elif len(arr.shape) == 2:  
        img = arr
    else:
        raise ValueError(f"Unexpected tensor shape: {arr.shape}")
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img

