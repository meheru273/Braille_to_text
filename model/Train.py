import pathlib
import os
import logging
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gc
from torch.amp import autocast, GradScaler

# Import your existing modules
from Targets import generate_targets
from loss import FocalLoss, _compute_loss  
from Dataset import collate_fn, COCOData
from AttentionFPN import FPN, normalize_batch

logger = logging.getLogger(__name__)

# Memory optimization settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'


def train(train_dir: pathlib.Path, 
    val_dir: pathlib.Path, 
    writer, 
    resume_ckpt_path=None,
    use_improved_model=True
):
    """Enhanced training function with attention mechanisms"""
    
    # Training hyperparameters
    BATCH_SIZE = 1
    IMAGE_SIZE = (1600, 2000)
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    
    # Ensure paths exist
    train_dir = pathlib.Path(train_dir)
    val_dir = pathlib.Path(val_dir)
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir.absolute()}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir.absolute()}")
    
    print("Creating datasets...")
    train_dataset = COCOData(train_dir, image_size=IMAGE_SIZE, min_area=2)
    val_dataset = COCOData(val_dir, image_size=IMAGE_SIZE, min_area=2)
    
    num_classes = train_dataset.get_num_classes()
    class_names = train_dataset.get_class_names()
    print(f"Number of classes: {num_classes}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=min(4, os.cpu_count()),
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True
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
    
    # Create model with the correct parameters
    if use_improved_model:
        print("Creating improved model with FPN cross-attention...")
        model = FPN(
            num_classes=num_classes,
            use_fpn_at=True  # Only this parameter is valid
        )
    else:
        print("Creating baseline model...")
        model = FPN(
            num_classes=num_classes,
            use_fpn_at=False
        )

    model.to(device)
    
    # Enable mixed precision optimizations
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(init_scale=2.**16)
    
    start_epoch = 1
    best_val_loss = float('inf')
    
    # Resume from checkpoint
    if resume_ckpt_path is not None and os.path.isfile(resume_ckpt_path):
        print(f"Loading checkpoint from {resume_ckpt_path}")
        checkpoint = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if 'scaler_state' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state'])
        
        start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        
        print(f"✅ Training will resume from epoch {start_epoch}")
    
    # Initialize focal loss
    focal_loss = FocalLoss(alpha="auto", gamma=2.0, num_classes=num_classes, reduction='mean')
    focal_loss.calculate_auto_alpha(train_dataset)
    
    print(f"Starting training from epoch {start_epoch} to {NUM_EPOCHS}...")
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{NUM_EPOCHS} start")
        
        # =================== TRAINING PHASE ===================
        model.train()
        epoch_losses = {
            'total': [], 'cls': [], 'centerness': [], 'reg': []
        }
        
        for batch_idx, (x, class_labels, box_labels) in enumerate(train_loader):
            # Memory management
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with autocast
            with autocast(device_type='cuda'):
                batch_norm = normalize_batch(x)
                
                # Model returns (classes, centerness, regression, attention_maps)
                cls_pred, centerness_pred, box_pred, attention_maps = model(batch_norm)
                
                # Generate targets (returns class_targets, centerness_targets, box_targets)
                class_t, centerness_t, box_t = generate_targets(
                    x.shape, class_labels, box_labels, model.strides
                )
                
                # Compute FCOS losses using your _compute_loss function
                cls_loss, centerness_loss, reg_loss = _compute_loss(
                    cls_pred, centerness_pred, box_pred,
                    class_t, centerness_t, box_t,
                    focal_loss,
                    classification_weight=1.0,
                    centerness_weight=1.0,
                    regression_weight=1.0
                )
                
                # Total loss is sum of all components
                total_loss = cls_loss + centerness_loss + reg_loss
            
            # Skip if NaN
            if not torch.isfinite(total_loss):
                print(f"Warning: Non-finite loss at epoch {epoch}, batch {batch_idx}")
                continue
            
            # Backward pass
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Track losses
            epoch_losses['total'].append(total_loss.item())
            epoch_losses['cls'].append(cls_loss.item())
            epoch_losses['centerness'].append(centerness_loss.item())
            epoch_losses['reg'].append(reg_loss.item())
            
            # Clean up
            del total_loss, cls_loss, centerness_loss, reg_loss, attention_maps
        
        # Calculate average losses
        avg_losses = {k: np.mean(v) if v else 0 for k, v in epoch_losses.items()}
        
        # Log losses
        print(f"\nEpoch {epoch} Training Metrics:")
        print(f"  Total Loss: {avg_losses['total']:.4f}")
        print(f"  Classification Loss: {avg_losses['cls']:.4f}")
        print(f"  Centerness Loss: {avg_losses['centerness']:.4f}")
        print(f"  Regression Loss: {avg_losses['reg']:.4f}")
        
        # Write to tensorboard
        writer.add_scalar('Loss/Total', avg_losses['total'], epoch)
        writer.add_scalar('Loss/Classification', avg_losses['cls'], epoch)
        writer.add_scalar('Loss/Centerness', avg_losses['centerness'], epoch)
        writer.add_scalar('Loss/Regression', avg_losses['reg'], epoch)
        writer.add_scalar('LearningRate', LEARNING_RATE, epoch)
        
        # =================== VALIDATION PHASE ===================
        current_val_loss = None
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:  # Validate every 5 epochs and at the end
            model.eval()
            val_losses = {'total': [], 'cls': [], 'centerness': [], 'reg': []}
            
            print("Running validation...")
            with torch.no_grad():
                for i, (x, class_labels, box_labels) in enumerate(val_loader):
                    if i >= 20:  # Evaluate on 20 validation samples
                        break
                    
                    x = x.to(device, non_blocking=True)
                    
                    with autocast(device_type='cuda'):
                        batch_norm = normalize_batch(x)
                        
                        # Model returns (classes, centerness, regression, attention_maps)
                        cls_pred, centerness_pred, box_pred, attention_maps = model(batch_norm)
                        
                        # Generate targets for validation
                        class_t, centerness_t, box_t = generate_targets(
                            x.shape, class_labels, box_labels, model.strides
                        )
                        
                        # Compute validation loss
                        cls_loss, centerness_loss, reg_loss = _compute_loss(
                            cls_pred, centerness_pred, box_pred,
                            class_t, centerness_t, box_t,
                            focal_loss,
                            classification_weight=1.0,
                            centerness_weight=1.0,
                            regression_weight=1.0
                        )
                        
                        val_total_loss = cls_loss + centerness_loss + reg_loss
                        
                        # Only add finite losses
                        if torch.isfinite(val_total_loss):
                            val_losses['total'].append(val_total_loss.item())
                            val_losses['cls'].append(cls_loss.item())
                            val_losses['centerness'].append(centerness_loss.item())
                            val_losses['reg'].append(reg_loss.item())
            
            # Calculate average validation losses
            avg_val_losses = {k: np.mean(v) if v else float('inf') for k, v in val_losses.items()}
            current_val_loss = avg_val_losses['total']
            
            print(f"  Validation Metrics:")
            print(f"    Total Loss: {avg_val_losses['total']:.4f}")
            print(f"    Classification Loss: {avg_val_losses['cls']:.4f}")
            print(f"    Centerness Loss: {avg_val_losses['centerness']:.4f}")
            print(f"    Regression Loss: {avg_val_losses['reg']:.4f}")
            
            # Log to tensorboard
            writer.add_scalar('Loss/Validation_Total', avg_val_losses['total'], epoch)
            writer.add_scalar('Loss/Validation_Classification', avg_val_losses['cls'], epoch)
            writer.add_scalar('Loss/Validation_Centerness', avg_val_losses['centerness'], epoch)
            writer.add_scalar('Loss/Validation_Regression', avg_val_losses['reg'], epoch)
            
            # Check if this is the best model
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                print(f"  🎉 New best validation loss: {best_val_loss:.4f}")
                
                # Save best model checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': {
                        'num_classes': num_classes,
                        'use_fpn_at': model.use_fpn_at,
                    }
                }
                
                torch.save(checkpoint, f'best_model_epoch_{epoch}.pth')
                print(f"  💾 Best model saved as best_model_epoch_{epoch}.pth")
        
        # Save regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {
                    'num_classes': num_classes,
                    'use_fpn_at': model.use_fpn_at,
                }
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
            print(f"  💾 Checkpoint saved as checkpoint_epoch_{epoch}.pth")
    
    print(f"\n🎊 Training completed! Best validation loss: {best_val_loss:.4f}")
    return best_val_loss