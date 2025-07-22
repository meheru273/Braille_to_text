
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

import torch
import torch.jit
from FPN import FPN  # Your current model
import os

def export_to_torchscript(checkpoint_path, export_path, num_classes=27):
    """Export PyTorch model to TorchScript format (.pt file)"""
    
    try:
    # Load your current model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FPN(num_classes=num_classes)
        
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        model.eval()
        # Create example input
        example_input = torch.randn(1, 3, 800, 1200).to(device)
        
        # Trace and save
        print("[PROGRESS] Tracing model for export...")
        traced_model = torch.jit.trace(model, example_input)
        
        print(f"[SAVE] Saving to: {export_path}")
        traced_model.save(export_path)
        print("[SUCCESS] Export completed successfully!")
        
        # FIXED: Safe verification with proper shape checking
        print("[VERIFY] Testing exported model...")
        loaded_model = torch.jit.load(export_path, map_location=device)
        
        with torch.no_grad():
            test_output = loaded_model(example_input)
        
        # Safe shape extraction
        def get_safe_shapes(output):
            """Safely extract shapes from model output"""
            shapes = []
            
            if isinstance(output, (list, tuple)):
                for i, item in enumerate(output):
                    if hasattr(item, 'shape'):
                        shapes.append(f"Output[{i}]: {item.shape}")
                    elif isinstance(item, (list, tuple)):
                        # Handle nested structures
                        for j, subitem in enumerate(item):
                            if hasattr(subitem, 'shape'):
                                shapes.append(f"Output[{i}][{j}]: {subitem.shape}")
                            else:
                                shapes.append(f"Output[{i}][{j}]: {type(subitem)}")
                    else:
                        shapes.append(f"Output[{i}]: {type(item)}")
            elif hasattr(output, 'shape'):
                shapes.append(f"Single output: {output.shape}")
            else:
                shapes.append(f"Output type: {type(output)}")
            
            return shapes
        
        output_shapes = get_safe_shapes(test_output)
        print("[SUCCESS] Test verification completed!")
        for shape_info in output_shapes:
            print(f"  {shape_info}")
        
        return export_path
        
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    

# Export your current model
export_path = export_to_torchscript(
    checkpoint_path="runs/fcos_Cord/fcos_epoch50.pth",
    export_path="runs/fcos_Cord/standalone.pt",
    num_classes=27
)
