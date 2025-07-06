import torch
from FPN import FPN, normalize_batch
from inference import debug_detections_from_network_output, tensor_to_image, render_detections_to_image
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def simple_centerness_detection_fixed(model, image_tensor, min_centerness=0.6, max_detections=30):
    """Fixed simple detection with proper coordinate handling"""
    
    with torch.no_grad():
        batch_norm = normalize_batch(image_tensor)
        cls_pred, cen_pred, box_pred = model(batch_norm)
        
        detections = []
        H, W = image_tensor.shape[2], image_tensor.shape[3]
        
        print(f"Image size: {W}x{H}")
        
        # Check ALL levels, not just Level 1
        strides = [4, 8, 16, 32, 64]
        
        for level_idx, (cent, stride) in enumerate(zip(cen_pred, strides)):
            cent = cent[0]  # Remove batch dimension
            feat_h, feat_w = cent.shape
            
            print(f"Level {level_idx}: feature map {feat_w}x{feat_h}, stride={stride}")
            
            # Find high centerness locations across ENTIRE feature map
            high_mask = cent > min_centerness
            y_coords, x_coords = torch.where(high_mask)
            
            print(f"  High centerness locations: {len(y_coords)}")
            if len(y_coords) > 0:
                # Show distribution of x-coordinates
                x_vals = x_coords.cpu().numpy()
                print(f"  X-coordinate range: {x_vals.min()}-{x_vals.max()} (feature map width: {feat_w})")
                
                # FIXED: Sample from entire feature map, not just edges
                for i in range(min(10, len(y_coords))):
                    feat_y, feat_x = y_coords[i].item(), x_coords[i].item()
                    score = cent[feat_y, feat_x].item()
                    
                    # PROPER coordinate transformation
                    img_x = feat_x * stride + stride // 2  # Center of the grid cell
                    img_y = feat_y * stride + stride // 2
                    
                    # Ensure coordinates are within image bounds
                    if img_x >= W or img_y >= H:
                        continue
                        
                    print(f"    Feat({feat_x}, {feat_y}) -> Img({img_x}, {img_y}), score={score:.3f}")
                    
                    # Fixed box size for Braille characters
                    box_size = min(40, stride * 2)  # Reasonable size
                    
                    x1 = max(0, img_x - box_size//2)
                    y1 = max(0, img_y - box_size//2)
                    x2 = min(W, img_x + box_size//2)
                    y2 = min(H, img_y + box_size//2)
                    
                    detections.append({
                        'class': f'L{level_idx}',
                        'score': score,
                        'bbox': [x1, y1, x2, y2],
                        'level': level_idx
                    })
        
        # Sort by score and return top detections
        detections.sort(key=lambda x: x['score'], reverse=True)
        return detections[:max_detections]

def debug_centerness_distribution(model, image_tensor):
    """Debug where the model thinks objects are"""
    
    with torch.no_grad():
        batch_norm = normalize_batch(image_tensor)
        cls_pred, cen_pred, box_pred = model(batch_norm)
        
        strides = [4, 8, 16, 32, 64]
        
        print("=== CENTERNESS DISTRIBUTION ANALYSIS ===")
        
        for level_idx, (cent, stride) in enumerate(zip(cen_pred, strides)):
            cent = cent[0]  # Remove batch dimension
            feat_h, feat_w = cent.shape
            
            # Analyze centerness distribution
            cent_np = cent.cpu().numpy()
            
            print(f"\nLevel {level_idx} (stride {stride}):")
            print(f"  Feature map: {feat_w}x{feat_h}")
            print(f"  Centerness range: {cent_np.min():.3f} to {cent_np.max():.3f}")
            print(f"  Mean centerness: {cent_np.mean():.3f}")
            
            # Check distribution across x-axis
            x_means = cent_np.mean(axis=0)  # Average across height
            print(f"  X-axis distribution (first 10): {x_means[:10]}")
            print(f"  X-axis distribution (last 10): {x_means[-10:]}")
            
            # Find peak locations
            high_locs = np.where(cent_np > 0.7)
            if len(high_locs[0]) > 0:
                x_peaks = high_locs[1]
                y_peaks = high_locs[0]
                print(f"  High centerness X-coordinates: {np.unique(x_peaks)}")
                print(f"  High centerness Y-coordinates: {np.unique(y_peaks)}")

def render_centerness_detections(image, detections):
    """Render detections with centerness info"""
    import cv2
    import numpy as np
    
    if isinstance(image, torch.Tensor):
        # Convert tensor to numpy
        img = image.cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        img = image.copy()
    
    # Color for each level
    level_colors = [
        (255, 0, 0),    # Level 0: Red
        (0, 255, 0),    # Level 1: Green  
        (0, 0, 255),    # Level 2: Blue
        (255, 255, 0),  # Level 3: Yellow
        (255, 0, 255),  # Level 4: Magenta
    ]
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        score = det['score']
        level = det.get('level', 0)
        
        # Get color for this level
        color = level_colors[level % len(level_colors)]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw level and score
        label = f"L{level}: {score:.3f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Background for text
        cv2.rectangle(img, (x1, y1 - label_size[1] - 5), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Text
        cv2.putText(img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

def tensor_to_image(tensor):
    """Convert tensor to numpy array for visualization"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

# 1. Load model
num_classes = 28
model = FPN(num_classes=num_classes)
ckpt = torch.load("runs/fcos_custom/fcos_epoch10.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 2. Load and preprocess image
img_path = "before2.jpg"
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f"Original image shape: {img_rgb.shape}")

# CHANGED: Use width=800, height=1200 (width, height format for cv2.resize)
img_resized = cv2.resize(img_rgb, (800, 1200))
img_norm = img_resized.astype(np.float32) / 255.0
input_tensor = torch.from_numpy(img_norm.transpose(2,0,1)).unsqueeze(0)

print(f"Input tensor shape: {input_tensor.shape}")  # Should be [1, 3, 1200, 800]

# 3. Forward pass with debugging
with torch.no_grad():
    batch = normalize_batch(input_tensor)
    print(f"Normalized batch stats: min={batch.min():.3f}, max={batch.max():.3f}, mean={batch.mean():.3f}")
    
    classes, centerness, boxes = model(batch)
    
    # Add raw output analysis
    print("\n=== RAW OUTPUT ANALYSIS ===")
    for i, (cls, cent, box) in enumerate(zip(classes, centerness, boxes)):
        print(f"Level {i}: cls_shape={cls.shape}, cent_shape={cent.shape}, box_shape={box.shape}")
        
        # Get the actual class probabilities
        cls_raw = cls.view(-1, cls.shape[-1])
        cent_raw = cent.view(-1)
        
        # Find highest scoring locations
        class_scores, class_indices = torch.max(cls_raw, dim=1)
        combined = class_scores * cent_raw
        
        # Get top 5 predictions (reduced from 10 for cleaner output)
        top_k = min(5, len(combined))
        top_values, top_indices = torch.topk(combined, top_k)
        
        print(f"Level {i} top predictions:")
        for j in range(top_k):
            idx = top_indices[j]
            print(f"  {j}: class={class_indices[idx]}, cls_score={class_scores[idx]:.4f}, "
                  f"cent={cent_raw[idx]:.4f}, combined={top_values[j]:.4f}")

# 4. CHANGED: Use actual tensor dimensions for detection
H, W = input_tensor.shape[2], input_tensor.shape[3]  # H=1200, W=800
print(f"\nUsing image dimensions: H={H}, W={W}")

# Add this to your test.py after loading the model

print("=== CENTERNESS-BASED DETECTION ===")
print("=" * 50)

# First, debug the distribution
debug_centerness_distribution(model, input_tensor)

# Then try fixed detection
print("\n=== FIXED DETECTION ===")
fixed_detections = simple_centerness_detection_fixed(
    model, input_tensor, 
    min_centerness=0.7,
    max_detections=30
)

print(f"Fixed detections: {len(fixed_detections)}")

# Show detection distribution
if len(fixed_detections) > 0:
    x_coords = [det['bbox'][0] for det in fixed_detections]
    y_coords = [det['bbox'][1] for det in fixed_detections]
    print(f"Detection X-coordinates: {min(x_coords):.0f} to {max(x_coords):.0f}")
    print(f"Detection Y-coordinates: {min(y_coords):.0f} to {max(y_coords):.0f}")
    
    # Visualize
    vis_fixed = render_centerness_detections(input_tensor[0], fixed_detections)
    cv2.imwrite("fixed_detections.jpg", cv2.cvtColor(vis_fixed, cv2.COLOR_RGB2BGR))
    print("Saved fixed detections as 'fixed_detections.jpg'")
else:
    print("No detections found - trying lower threshold...")
    low_thresh = simple_centerness_detection_fixed(model, input_tensor, min_centerness=0.5)
    print(f"With threshold 0.5: {len(low_thresh)} detections")
