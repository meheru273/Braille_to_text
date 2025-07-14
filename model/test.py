import cv2
import numpy as np
import torch
from FPN import FPN
from inference import compute_detections, render_detections_to_image, tensor_to_image
from PostProcessing import generate_colors,compute_iou, non_max_suppression,refine_large_boxes,letterbox_image, map_detections_to_original


def apply_postprocessing_pipeline(detections, confidence_threshold=0.3, nms_threshold=0.3):
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
    
    # Step 4: Refine large boxes
    refined_detections = refine_large_boxes(nms_detections, shrink_factor=0.6)
    print(f"[INFO] After box refinement: {len(refined_detections)} detections")
    
    return refined_detections

def main():
    print("=== BRAILLE DETECTION WITH POSTPROCESSING ===")
    
    # 1. Load model (same as before)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FPN(num_classes=28)
    
    try:
        ckpt = torch.load("runs/fcos_custom/fcos_epoch30.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()  # Important: set to evaluation mode
        print(f"[OK] Model loaded on {device}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # 2. Load and preprocess image
    img_path = "before2.jpg"
    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[ERROR] Could not load image: {img_path}")
            return
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_shape = img_rgb.shape
        print(f"[OK] Loaded image: {orig_shape}")
        
        # Apply letterboxing
        img_letterboxed, scale, paste_x, paste_y = letterbox_image(img_rgb)
        print(f"[OK] Letterboxed image: {img_letterboxed.shape}")
        
    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {e}")
        return
    
    # 3. Run detection
    try:
        print("\n=== RUNNING DETECTION ===")
        raw_detections = compute_detections(model, img_letterboxed, device)
        print(f"[OK] Raw detections: {len(raw_detections)}")
        
        # 4. Apply postprocessing pipeline
        print("\n=== APPLYING POSTPROCESSING ===")
        processed_detections = apply_postprocessing_pipeline(
            raw_detections, 
            confidence_threshold=0.3, 
            nms_threshold=0.3
        )
        
        # Show sample results
        print("\n=== DETECTION RESULTS ===")
        for i, det in enumerate(processed_detections[:10]):
            width = det.bbox[2] - det.bbox[0]
            height = det.bbox[3] - det.bbox[1]
            print(f"  {i}: class={det.object_class}, score={det.score:.3f}, "
                  f"size={width:.0f}x{height:.0f}, area={width*height:.0f}")
        
    except Exception as e:
        print(f"[ERROR] Detection/postprocessing failed: {e}")
        return
    
    # 5. Map to original coordinates
    try:
        print("\n=== MAPPING TO ORIGINAL COORDINATES ===")
        original_detections = map_detections_to_original(
            processed_detections, scale, paste_x, paste_y, orig_shape
        )
        print(f"[OK] Mapped {len(original_detections)} detections")
        
    except Exception as e:
        print(f"[ERROR] Coordinate mapping failed: {e}")
        return
    
    # 6. Visualize results with colors
    try:
        print("\n=== CREATING VISUALIZATIONS ===")
        
        # Generate colors for classes
        colors = generate_colors(28)
        
        # Create enhanced visualization
        img_with_detections = create_enhanced_visualization(
            img_rgb.copy(), original_detections, colors
        )
        
        # Save results
        cv2.imwrite("detections_postprocessed.jpg", 
                   cv2.cvtColor(img_with_detections, cv2.COLOR_RGB2BGR))
        print("[OK] Saved 'detections_postprocessed.jpg'")
        
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
        return
    
    print(f"\n=== SUMMARY ===")
    print(f"Raw detections: {len(raw_detections)}")
    print(f"Final detections: {len(original_detections)}")
    print(f"Improvement: {len(raw_detections) - len(original_detections)} duplicates/false positives removed")

def create_enhanced_visualization(img, detections, colors):
    """Create enhanced visualization with class-specific colors"""
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox.astype(int)
        class_id = det.object_class
        score = det.score
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw class label and score
        label = f"Class {class_id}: {score:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Background rectangle for text
        cv2.rectangle(img, (x1, y1 - label_size[1] - 5), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Text
        cv2.putText(img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

if __name__ == "__main__":
    main()
