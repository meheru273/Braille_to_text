import pathlib
import os
import logging
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gc
from torch.amp import autocast, GradScaler 

# Import existing functions from your modules
from inference import detections_from_network_output
from FPN import FPN, normalize_batch, FocalLoss
from Targets import generate_targets
from loss import _compute_loss
from Dataset import DSBIData, collate_fn,COCOData 
import re
logger = logging.getLogger(__name__)

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'


def tensor_to_image(tensor):
    """Convert tensor to numpy array for visualization"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

def create_optimizer(model, base_lr=1e-4):
    """Create optimizer with different learning rates for backbone and new layers"""
    backbone_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            new_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': base_lr * 0.1},
        {'params': new_params, 'lr': base_lr}
    ])
    
    return optimizer

def train(train_dir: pathlib.Path, val_dir: pathlib.Path, writer, resume_ckpt_path=None):
    """
    Memory-optimized training function with proper autocast and GradScaler usage
    """
    # Training hyperparameters
    BATCH_SIZE = 2
    IMAGE_SIZE = (700, 1024)
    BASE_LR = 1e-4
    NUM_EPOCHS = 50
    
    # Ensure paths exist
    train_dir = pathlib.Path(train_dir)
    val_dir = pathlib.Path(val_dir)
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir.absolute()}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir.absolute()}")
    
    print("Creating datasets...")
    train_dataset = DSBIData(train_dir, image_size=IMAGE_SIZE, min_area=2)
    val_dataset = DSBIData(val_dir, image_size=IMAGE_SIZE, min_area=2)
    
    
    num_classes = train_dataset.get_num_classes()
    class_names = train_dataset.get_class_names()
    print(f"Number of classes: {num_classes}")
    print(f"CPU count: {os.cpu_count()}")
    
    # Data loaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=min(4, os.cpu_count()),  # Reduced for memory efficiency
        collate_fn=collate_fn,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Avoid worker respawn overhead
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=min(4, os.cpu_count()),
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Creating model...")
    model = FPN(num_classes=num_classes)
    model.to(device)
    
    # Enable mixed precision training optimizations
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Create optimizer and scaler
    optimizer = create_optimizer(model, BASE_LR)
    scaler = torch.amp.GradScaler(
        init_scale=2.**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    start_epoch = 1
    
    # Resume from checkpoint if provided
    if resume_ckpt_path is not None and os.path.isfile(resume_ckpt_path):
        print(f"Loading checkpoint from {resume_ckpt_path}")
        checkpoint = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        if 'scaler_state' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Training will resume from epoch {start_epoch}")
        logger.info(f"Resuming training from epoch {start_epoch}")
    else:
        print("Starting training from scratch")
    
    # Initialize focal loss
    focal_loss = FocalLoss(alpha="auto", gamma=2.0, num_classes=num_classes, reduction='mean')
    focal_loss.calculate_auto_alpha(train_dataset)
    
    print(f"Starting training from epoch {start_epoch} to {NUM_EPOCHS}...")
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{NUM_EPOCHS} start")
        
        # =================== TRAINING PHASE ===================
        model.train()
        epoch_losses = []
        epoch_cls_losses = []
        epoch_reg_losses = []
        
        for batch_idx, (x, class_labels, box_labels) in enumerate(train_loader):
            # Memory management - clear cache before each batch
            if batch_idx > 0:  # Less frequent clearing to reduce overhead
                torch.cuda.empty_cache()
                gc.collect()
            
            # Move data to device with non_blocking for efficiency
            x = x.to(device, non_blocking=True)

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
            
            # =================== FORWARD PASS WITH AUTOCAST ===================
            with torch.amp.autocast(device_type='cuda'):
                batch_norm = normalize_batch(x)
                cls_pred, box_pred = model(batch_norm)
                
                class_t, box_t = generate_targets(
                    x.shape, class_labels, box_labels, model.strides
                )
                
                total_loss, cls_loss, reg_loss = _compute_loss(
                    cls_pred, box_pred, class_t, box_t, focal_loss,
                    box_labels_by_batch=box_labels,
                    img_shape=x.shape,
                    strides=model.strides,
                )
            
            # Check for NaN losses
            if not torch.isfinite(total_loss):
                print(f"Warning: Non-finite loss detected at epoch {epoch}, batch {batch_idx}")
                continue
            
            # =================== BACKWARD PASS WITH GRADSCALER ===================
            scaler.scale(total_loss).backward()
            
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            # Track losses
            epoch_losses.append(total_loss.item())
            epoch_cls_losses.append(cls_loss.item())
            epoch_reg_losses.append(reg_loss.item())
            
            # Clear references for memory efficiency
            del total_loss, cls_loss, reg_loss, cls_pred, box_pred, batch_norm
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average losses
        avg_loss = np.mean(epoch_losses)
        avg_cls_loss = np.mean(epoch_cls_losses)
        avg_reg_loss = np.mean(epoch_reg_losses)
        
        # Log losses
        print(f"Epoch {epoch} completed:")
        print(f"  Avg Total Loss: {avg_loss:.4f}")
        print(f"  Avg Cls Loss: {avg_cls_loss:.4f}")
        print(f"  Avg Reg Loss: {avg_reg_loss:.4f}")
        print(f"  Current LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # =================== VALIDATION PHASE ===================
        if epoch % 5 == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for i, (x, class_labels, box_labels) in enumerate(val_loader):
                    if i >= 5:  # Limit validation samples
                        break
                    
                    x = x.to(device, non_blocking=True)
                    
                    # Use autocast for validation too (memory efficient)
                    with autocast(device_type='cuda'):
                        batch_norm = normalize_batch(x)
                        cls_pred, box_pred = model(batch_norm)
                        
                        # Optional: compute validation loss
                        class_t, box_t = generate_targets(
                            x.shape, class_labels, box_labels, model.strides
                        )
                        val_loss, _, _ = _compute_loss(
                            cls_pred, box_pred, class_t, box_t, focal_loss,
                            box_labels_by_batch=box_labels,
                            img_shape=x.shape,
                            strides=model.strides,
                        )
                        val_losses.append(val_loss.item())
                    
                    # Generate detections for evaluation
                    H, W = x.shape[2], x.shape[3]
                    detections = detections_from_network_output(
                        H, W, cls_pred, box_pred, model.scales, model.strides
                    )
                    
                    # Clear validation tensors
                    del cls_pred, box_pred, batch_norm, val_loss
            
            if val_losses:
                avg_val_loss = np.mean(val_losses)
                print(f"  Avg Validation Loss: {avg_val_loss:.4f}")
        
        # =================== CHECKPOINT SAVING ===================
        if epoch % 20 == 0 or epoch == NUM_EPOCHS:
            checkpoint_dir = os.path.join(writer.log_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            ckpt_path = os.path.join(checkpoint_dir, f"fcos_epoch{epoch}.pth")
            
            # Save checkpoint with scaler state
            checkpoint = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),  # Important for resuming mixed precision
                'epoch': epoch,
                'class_names': class_names,
                'num_classes': num_classes,
                'losses': {
                    'total': avg_loss,
                    'classification': avg_cls_loss,
                    'regression': avg_reg_loss,
                },
                'hyperparameters': {
                    'batch_size': BATCH_SIZE,
                    'image_size': IMAGE_SIZE,
                    'base_lr': BASE_LR,
                    'num_epochs': NUM_EPOCHS
                }
            }
            
            torch.save(checkpoint, ckpt_path)
            logger.info(f"Saved checkpoint {ckpt_path}")
            print(f"Checkpoint saved: {ckpt_path}")
        
        # Final memory cleanup at end of epoch
        torch.cuda.empty_cache()
        gc.collect()

# Usage remains the same
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from torch.utils.tensorboard import SummaryWriter
    import pathlib
    
    train_dir = pathlib.Path(r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\dataset.coco\train")
    val_dir = pathlib.Path(r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\dataset.coco\valid")
    writer = SummaryWriter("runs/fcos_custom")
    
    resume_ckpt_path = None  # or path to checkpoint
    
    train(train_dir, val_dir, writer, resume_ckpt_path)
