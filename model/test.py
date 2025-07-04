import torch
from FPN import FPN, normalize_batch
from inference import debug_detections_from_network_output, tensor_to_image, render_detections_to_image
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. Load model
num_classes = 28
model = FPN(num_classes=num_classes)
ckpt = torch.load("runs/fcos_custom/fcos_epoch20.pth", map_location="cpu", weights_only=False)
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

print("\n" + "="*50)
print("CALLING DEBUG DETECTION FUNCTION")
print("="*50)

# Use correct dimensions for detection
gathered_boxes, gathered_classes, gathered_scores = debug_detections_from_network_output(
    H, W, classes, centerness, boxes, model.scales, model.strides
)

# Convert to Detection objects
from inference import detections_from_net
detections_list = detections_from_net(gathered_boxes, gathered_classes, gathered_scores)
detections = detections_list[0]  # batch size = 1

print(f"\nFinal detections: {len(detections)}")
if len(detections) > 0:
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        print(f"Detection {i}: class={det.object_class}, score={det.score:.4f}")
        print(f"  bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}], size={width:.0f}x{height:.0f}, aspect={aspect_ratio:.2f}")

# 5. Visualize
vis = tensor_to_image(input_tensor[0])
vis = render_detections_to_image(vis, detections)

# CHANGED: Adjust figure size for 800x1200 aspect ratio
plt.figure(figsize=(8, 12))  # Maintain aspect ratio
plt.imshow(vis)
plt.title(f"Detections: {len(detections)} (800x1200)")
plt.axis('off')
plt.show()

# 6. Save debug image
cv2.imwrite("debug_output.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print("Debug output saved as 'debug_output.jpg'")

# Check if model was actually trained
print("\n=== MODEL WEIGHT ANALYSIS ===")
for name, param in model.named_parameters():
    if 'classification_to_class' in name or 'classification_to_centerness' in name:
        print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
