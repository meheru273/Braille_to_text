
import pathlib
import os
import logging
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gc
from torch.cuda.amp import autocast, GradScaler

# Import existing functions from your modules
from inference import detections_from_network_output
from FPN import FPN, normalize_batch, FocalLoss
from Targets import generate_targets
from loss import _compute_loss
from Dataset import DSBIData, COCOData, collate_fn
def calculate_model_memory(batch_size=1, image_size=(800, 1200), fpn_channels=256, num_classes=10):
    """Calculate expected memory usage for FCOS model"""
    
    print("=== Model Memory Calculation ===")
    H, W = image_size
    
    # 1. INPUT TENSORS
    input_memory = batch_size * 3 * H * W * 4 / (1024**2)  # 4 bytes per float32
    print(f"Input tensor ({batch_size}, 3, {H}, {W}): {input_memory:.1f} MB")
    
    # 2. BACKBONE ACTIVATIONS (EfficientNet or ResNet)
    backbone_activations = 0
    
    # Typical backbone feature map sizes and channels
    backbone_stages = [
        (H//2, W//2, 64),    # Stage 1: /2 stride
        (H//4, W//4, 128),   # Stage 2: /4 stride  
        (H//8, W//8, 256),   # Stage 3: /8 stride
        (H//16, W//16, 512), # Stage 4: /16 stride
        (H//32, W//32, 1024) # Stage 5: /32 stride
    ]
    
    for i, (h, w, c) in enumerate(backbone_stages):
        stage_memory = batch_size * c * h * w * 4 / (1024**2)
        backbone_activations += stage_memory
        print(f"Backbone Stage {i+1} ({h}, {w}, {c}): {stage_memory:.1f} MB")
    
    print(f"Total Backbone Activations: {backbone_activations:.1f} MB")
    
    # 3. FPN FEATURE MAPS
    fpn_levels = [
        (H//4, W//4, fpn_channels),   # P2
        (H//8, W//8, fpn_channels),   # P3  
        (H//16, W//16, fpn_channels), # P4
        (H//32, W//32, fpn_channels), # P5
        (H//64, W//64, fpn_channels)  # P6
    ]
    
    fpn_memory = 0
    for i, (h, w, c) in enumerate(fpn_levels):
        level_memory = batch_size * c * h * w * 4 / (1024**2)
        fpn_memory += level_memory
        print(f"FPN P{i+2} ({h}, {w}, {c}): {level_memory:.1f} MB")
    
    print(f"Total FPN Activations: {fpn_memory:.1f} MB")
    
    # 4. DETECTION HEAD OUTPUTS
    detection_memory = 0
    for i, (h, w, c) in enumerate(fpn_levels):
        # Classification head output
        cls_memory = batch_size * num_classes * h * w * 4 / (1024**2)
        # Regression head output  
        reg_memory = batch_size * 4 * h * w * 4 / (1024**2)
        
        level_det_memory = cls_memory + reg_memory
        detection_memory += level_det_memory
        print(f"Detection outputs P{i+2}: {level_det_memory:.1f} MB")
    
    print(f"Total Detection Outputs: {detection_memory:.1f} MB")
    
    # 5. MODEL PARAMETERS
    # Typical FCOS parameter counts
    backbone_params = 20_000_000  # ~20M for EfficientNet-B2
    fpn_params = 2_000_000        # ~2M for FPN lateral convs
    detection_head_params = 1_000_000  # ~1M for classification + regression heads
    
    total_params = backbone_params + fpn_params + detection_head_params
    param_memory = total_params * 4 / (1024**2)  # 4 bytes per parameter
    print(f"Model Parameters ({total_params:,}): {param_memory:.1f} MB")
    
    # 6. GRADIENTS (same size as parameters)
    gradient_memory = param_memory
    print(f"Gradients: {gradient_memory:.1f} MB")
    
    # 7. OPTIMIZER STATE (Adam = 2x parameters)
    optimizer_memory = param_memory * 2  # Adam keeps momentum + variance
    print(f"Optimizer State (Adam): {optimizer_memory:.1f} MB")
    
    # 8. MIXED PRECISION OVERHEAD
    mixed_precision_overhead = (backbone_activations + fpn_memory + detection_memory) * 0.3
    print(f"Mixed Precision Overhead: {mixed_precision_overhead:.1f} MB")
    
    # TOTAL CALCULATION
    total_memory = (input_memory + backbone_activations + fpn_memory + 
                   detection_memory + param_memory + gradient_memory + 
                   optimizer_memory + mixed_precision_overhead)
    
    print(f"\n=== TOTAL MEMORY BREAKDOWN ===")
    print(f"Activations: {input_memory + backbone_activations + fpn_memory + detection_memory:.1f} MB")
    print(f"Parameters: {param_memory:.1f} MB") 
    print(f"Gradients: {gradient_memory:.1f} MB")
    print(f"Optimizer: {optimizer_memory:.1f} MB")
    print(f"Mixed Precision: {mixed_precision_overhead:.1f} MB")
    print(f"TOTAL EXPECTED: {total_memory:.1f} MB ({total_memory/1024:.2f} GB)")
    
    return total_memory

# Calculate for different configurations
print("Configuration 1: Full Size")
calculate_model_memory(batch_size=1, image_size=(800, 1200), fpn_channels=256, num_classes=10)

print("\nConfiguration 2: Reduced Size")  
calculate_model_memory(batch_size=1, image_size=(512, 768), fpn_channels=128, num_classes=10)

print("\nConfiguration 3: Minimal Size")
calculate_model_memory(batch_size=1, image_size=(320, 480), fpn_channels=64, num_classes=10)
