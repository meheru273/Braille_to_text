import cv2
import numpy as np
import torch
from FPN import FPN
from inference import compute_detections, render_detections_to_image, tensor_to_image

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

def main():
    print("=== SIMPLE FCOS DETECTION TEST ===")
    
    # 1. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FPN(num_classes=28)
    
    try:
        ckpt = torch.load("runs/fcos_custom/fcos_epoch20.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        print(f"[OK] Model loaded on {device}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # 2. Load and preprocess image
    img_path = "B38.jpg"
    try:
        # Load original image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[ERROR] Could not load image: {img_path}")
            return
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_shape = img_rgb.shape
        print(f"[OK] Loaded image: {orig_shape}")
        
        # Apply letterboxing (same as training)
        img_letterboxed, scale, paste_x, paste_y = letterbox_image(img_rgb)
        print(f"[OK] Letterboxed image: {img_letterboxed.shape}")
        print(f"     Scale: {scale:.3f}, Paste: ({paste_x}, {paste_y})")
        
    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {e}")
        return
    
    # 3. Run detection using inference.py functions
    try:
        print("\n=== RUNNING DETECTION ===")
        detections = compute_detections(model, img_letterboxed, device)
        print(f"[OK] Found {len(detections)} detections")
        
        # Show detection details
        for i, det in enumerate(detections[:10]):  # Show first 10
            print(f"  {i}: class={det.object_class}, score={det.score:.3f}, "
                  f"bbox=[{det.bbox[0]:.0f}, {det.bbox[1]:.0f}, {det.bbox[2]:.0f}, {det.bbox[3]:.0f}]")
        
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        return
    
    # 4. Map detections back to original coordinates
    try:
        print("\n=== MAPPING TO ORIGINAL COORDINATES ===")
        original_detections = map_detections_to_original(
            detections, scale, paste_x, paste_y, orig_shape
        )
        print(f"[OK] Mapped {len(original_detections)} detections to original image")
        
    except Exception as e:
        print(f"[ERROR] Coordinate mapping failed: {e}")
        return
    
    # 5. Visualize results
    try:
        print("\n=== SAVING RESULTS ===")
        
        
        # Visualize on original image
        img_original_with_detections = render_detections_to_image(img_rgb.copy(), original_detections)
        cv2.imwrite("detections_original.jpg", cv2.cvtColor(img_original_with_detections, cv2.COLOR_RGB2BGR))
        print("[OK] Saved 'detections_original.jpg'")
        
       
        
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
        return
    
    print("\n=== TEST COMPLETED ===")
    print("Check these files:")
    print("1. 'detections_original.jpg' - Final results on original image")
    print("2. 'detections_letterboxed.jpg' - Results on letterboxed image")
    print("3. 'input_letterboxed.jpg' - The letterboxed input to the model")

if __name__ == "__main__":
    main()
