#postprocessing.py 
import cv2
import numpy as np
import torch
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    
    return inter_area / union if union > 0 else 0.0

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []
    
    while len(idxs):
        current = idxs[0]
        keep.append(current)
        rest = idxs[1:]
        
        if len(rest) == 0:
            break
            
        ious = np.array([compute_iou(boxes[current], boxes[i]) for i in rest])
        idxs = rest[ious < iou_threshold]
    
    return keep

def generate_colors(num_classes):
    """Generate distinct colors for each class"""
    colors = []
    for i in range(num_classes):
        # Generate colors in HSV space for better distribution
        hue = int(180 * i / num_classes)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(color[0]), int(color[1]), int(color[2])))
    return colors

def refine_large_boxes(detections, shrink_factor=.6):
    """Shrink oversized bounding boxes while maintaining center"""
    refined_boxes = []
    
    for detection in detections:
        # FIX: Use dot notation instead of dictionary access
        x1, y1, x2, y2 = detection.bbox
        
        # Calculate center and current size
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Shrink box if too large
        if width * height > 2000:  # Threshold for "too large"
            new_width = width * shrink_factor
            new_height = height * shrink_factor
            
            # Recalculate coordinates
            new_x1 = center_x - new_width / 2
            new_y1 = center_y - new_height / 2
            new_x2 = center_x + new_width / 2
            new_y2 = center_y + new_height / 2
            
            # FIX: Create new Detection object with refined bbox
            from inference import Detection
            refined_detection = Detection(
                score=detection.score,
                object_class=detection.object_class,
                bbox=np.array([new_x1, new_y1, new_x2, new_y2])
            )
            refined_boxes.append(refined_detection)
        else:
            # Keep original detection if box size is acceptable
            refined_boxes.append(detection)
    
    return refined_boxes


import numpy as np
import cv2

def letterbox_image(img_rgb, target_size=(720, 1024)):
    
    orig_h, orig_w = img_rgb.shape[:2]
    target_w, target_h = target_size

    # Compute uniform scale for aspect ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize image with OpenCV
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create black background canvas
    letterboxed = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Center the resized image
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    letterboxed[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = img_resized

    return letterboxed, scale, paste_x, paste_y


def map_detections(detections, letterbox_shape=(720, 1024)):
    """Keep detections in letterboxed coordinates and clip to letterbox bounds"""
    letterbox_h, letterbox_w = letterbox_shape
    
    mapped_detections = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        
        # Clip to letterboxed image bounds (720x1024)
        x1_letterbox = max(0, min(letterbox_w, x1))
        y1_letterbox = max(0, min(letterbox_h, y1))
        x2_letterbox = max(0, min(letterbox_w, x2))
        y2_letterbox = max(0, min(letterbox_h, y2))
        
        # Ensure valid bounding box
        if x2_letterbox > x1_letterbox and y2_letterbox > y1_letterbox:
            from inference import Detection
            mapped_det = Detection(
                score=det.score,
                object_class=det.object_class,
                bbox=np.array([x1_letterbox, y1_letterbox, x2_letterbox, y2_letterbox])
            )
            mapped_detections.append(mapped_det)
    
    return mapped_detections
