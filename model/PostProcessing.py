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

def refine_large_boxes(detections, shrink_factor=0.5):
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


def letterbox_image(img_rgb, target_size=(800, 1200)):
    """Apply letterboxing to match training preprocessing"""
    orig_h, orig_w = img_rgb.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale to maintain aspect ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    # Resize image
    img_resized = cv2.resize(img_rgb, (new_w, new_h))
    
    # Create letterboxed image with gray padding
    letterboxed = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    
    # Calculate paste position (center)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    
    # Paste resized image onto letterboxed background
    letterboxed[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = img_resized
    
    return letterboxed, scale, paste_x, paste_y


def map_detections_to_original(detections, scale, paste_x, paste_y, orig_shape):
    """Map detections from letterboxed coordinates back to original image"""
    orig_h, orig_w = orig_shape[:2]
    
    mapped_detections = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        
        # Remove letterbox padding
        x1_orig = (x1 - paste_x) / scale
        y1_orig = (y1 - paste_y) / scale
        x2_orig = (x2 - paste_x) / scale
        y2_orig = (y2 - paste_y) / scale
        
        # Clip to original image bounds
        x1_orig = max(0, min(orig_w, x1_orig))
        y1_orig = max(0, min(orig_h, y1_orig))
        x2_orig = max(0, min(orig_w, x2_orig))
        y2_orig = max(0, min(orig_h, y2_orig))
        
        # Create new detection with mapped coordinates
        if x2_orig > x1_orig and y2_orig > y1_orig:
            from inference import Detection
            mapped_det = Detection(
                score=det.score,
                object_class=det.object_class,
                bbox=np.array([x1_orig, y1_orig, x2_orig, y2_orig])
            )
            mapped_detections.append(mapped_det)
    
    return mapped_detections