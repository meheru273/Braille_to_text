import cv2
import numpy as np
import torch
from FPN import FPN
from inference import compute_detections, render_detections_to_image, tensor_to_image
from PostProcessing import (generate_colors, non_max_suppression, 
                           refine_large_boxes, letterbox_image,
                           map_detections,compute_iou)

from torchviz import make_dot
import torch


def apply_postprocessing_pipeline(detections, confidence_threshold=0.8, nms_threshold=0.2):
    """Apply complete postprocessing pipeline to raw detections"""
    
    if len(detections) == 0:
        return detections
    
    # Step 1: Filter by confidence
    filtered_detections = []
    for det in detections:
        if det.score >= confidence_threshold:
            filtered_detections.append(det)
    
    print(f"[INFO] After confidence filtering: {len(filtered_detections)} detections")
    
    if len(filtered_detections) == 0:
        return filtered_detections
    
    # Step 2: Prepare data for NMS
    boxes = np.array([det.bbox for det in filtered_detections])
    scores = np.array([det.score for det in filtered_detections])
    classes = np.array([det.object_class for det in filtered_detections])
    
    # Step 3: Apply Non-Maximum Suppression
    keep_indices = non_max_suppression(boxes, scores, iou_threshold=nms_threshold)
    
    # Filter detections based on NMS results
    nms_detections = [filtered_detections[i] for i in keep_indices]
    print(f"[INFO] After NMS: {len(nms_detections)} detections")
    
    shrink_detections = refine_large_boxes(nms_detections)
    
    
    return shrink_detections


def analyze_detection_distribution(detections, img_shape):
    """Check if detections are properly distributed across image quadrants"""
    if len(detections) == 0:
        print("No detections to analyze")
        return
        
    orig_h, orig_w = img_shape[:2]
    quadrants = {'top_left': 0, 'top_right': 0, 'bottom_left': 0, 'bottom_right': 0}
    
    for det in detections:
        x_center = (det.bbox[0] + det.bbox[2]) / 2
        y_center = (det.bbox[1] + det.bbox[3]) / 2
        
        # Complete the quadrant classification
        if x_center < orig_w / 2 and y_center < orig_h / 2:
            quadrants['top_left'] += 1
        elif x_center >= orig_w / 2 and y_center < orig_h / 2:
            quadrants['top_right'] += 1
        elif x_center < orig_w / 2 and y_center >= orig_h / 2:
            quadrants['bottom_left'] += 1
        else:
            quadrants['bottom_right'] += 1
    
    print("Detection distribution:")
    for quadrant, count in quadrants.items():
        percentage = count / len(detections) * 100 if detections else 0
        print(f"  {quadrant}: {count} ({percentage:.1f}%)")


def main():
    print("=== BRAILLE DETECTION WITH POSTPROCESSING ===")
    
    # 1. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FPN(num_classes=27)
    
    try:
        ckpt = torch.load("runs/basic_fcos/fcos_epoch50dsbi.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        print(f"[OK] Model loaded on {device}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # 2. Load and preprocess image
    img_path = "testImage/before2.jpg"
    
    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[ERROR] Could not load image: {img_path}")
            return
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_shape = img_rgb.shape
        print(f"[OK] Loaded image: {orig_shape}")
        
        
        # FIXED: Properly handle letterbox_image return value
        letterbox_result = letterbox_image(img_rgb, (700, 1024))
        img_resized = letterbox_result[0]
        scale_info = letterbox_result[1] if len(letterbox_result) > 1 else None
    
            
        print(f"[OK] Image resized to: {img_resized.shape}")
         
        # 3. Run detection
        print("\n=== RUNNING DETECTION ===")
        raw_detections = compute_detections(model, img_resized, device)
        print(f"[OK] Raw detections: {len(raw_detections)}")
        
        if len(raw_detections) == 0:
            print("[WARNING] No detections found. Check your model and image.")
            return
        
        # 4. Apply postprocessing pipeline
        print("\n=== APPLYING POSTPROCESSING ===")
        processed_detections = apply_postprocessing_pipeline(
            raw_detections, 
            confidence_threshold=0.01, 
            nms_threshold=0.5
        )
        
        # Show sample results
        print("\n=== DETECTION RESULTS ===")
        for i, det in enumerate(processed_detections[:10]):
            width = det.bbox[2] - det.bbox[0]
            height = det.bbox[3] - det.bbox[1]
            print(f"  {i}: class={det.object_class}, score={det.score:.3f}, "
                  f"size={width:.0f}x{height:.0f}, area={width*height:.0f}")
        
        # 5. Map to original coordinates
        print("\n=== MAPPING TO ORIGINAL COORDINATES ===")
        original_detections = map_detections(processed_detections)
        
        # 6. Visualize results with colors
        print("\n=== CREATING VISUALIZATIONS ===")
        
        # Generate colors for classes
        colors = generate_colors(27)
        
        # Create enhanced visualization
        img_for_viz = img_resized.copy() 
        
        img_with_detections = create_enhanced_visualization(
            img_for_viz, original_detections, colors
        )
        
        # Save results
        cv2.imwrite("detections_postprocessed.jpg", 
                   cv2.cvtColor(img_with_detections, cv2.COLOR_RGB2BGR))
        print("[OK] Saved 'detections_postprocessed.jpg'")
        
        # Generate model architecture diagram
        try:
            # Create a dummy input for visualization
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
            dot.render("fpn_architecture", format="png")
            print("[OK] Saved model architecture diagram")
        except Exception as viz_error:
            print(f"[WARNING] Could not save architecture diagram: {viz_error}")
        
        # Final summary
        print("\n=== DETECTION DISTRIBUTION ===")
        analyze_detection_distribution(original_detections, orig_shape)
        
        print(f"\n=== SUMMARY ===")
        print(f"Raw detections: {len(raw_detections)}")
        print(f"Final detections: {len(original_detections)}")
        print(f"Improvement: {len(raw_detections) - len(original_detections)} duplicates/false positives removed")
        
    except Exception as e:
        print(f"[ERROR] Detection/postprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return


def create_enhanced_visualization(img, detections, colors):
    """Create enhanced visualization with class-specific colors"""
    
    # Make a copy to avoid modifying the original
    img_vis = img.copy()
    
    for det in detections:
        try:
            x1, y1, x2, y2 = det.bbox.astype(int)
            class_id = det.object_class
            score = det.score
            
            # Ensure coordinates are within image bounds
            h, w = img_vis.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Convert color to tuple of ints for OpenCV
            color = tuple(int(c) for c in color)
            
            # Draw bounding box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw class label and score
            label = f"Class {class_id}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Ensure label fits within image
            label_y = max(y1, label_size[1] + 5)
            
            # Background rectangle for text
            cv2.rectangle(img_vis, (x1, label_y - label_size[1] - 5), 
                         (x1 + label_size[0], label_y), color, -1)
            
            # Text
            cv2.putText(img_vis, label, (x1, label_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        except Exception as viz_error:
            print(f"[WARNING] Failed to visualize detection: {viz_error}")
            continue
    
    return img_vis


if __name__ == "__main__":
    main()
