import pathlib
import os
import logging
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gc
from torch.amp import autocast, GradScaler

# Import your existing modules
from inference import detections_from_network_output
from Targets import generate_targets
from loss import _compute_loss
from Dataset import collate_fn, COCOData
from FPNAttention import (
    ImprovedFPN, compute_loss_with_attention, 
    SpatialAttentionLoss, CenterAttentionLoss
)
from FPNAttention import normalize_batch
from loss import FocalLoss

logger = logging.getLogger(__name__)

# Memory optimization settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'


class AdaptiveLearningRateScheduler:
    """Custom scheduler that adapts based on loss plateaus"""
    def __init__(self, optimizer, patience=10, factor=0.5, min_lr=1e-7):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait = 0
        
    def step(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.wait = 0
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * self.factor, self.min_lr)
            return True
        return False
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def create_optimizer_with_layer_wise_lr(model, base_lr=1e-4):
    """Create optimizer with layer-wise learning rates"""
    param_groups = [
        # Backbone - lowest learning rate
        {'params': [], 'lr': base_lr * 0.1, 'name': 'backbone'},
        # FPN lateral/extra convs - medium learning rate
        {'params': [], 'lr': base_lr * 0.5, 'name': 'fpn'},
        # Detection heads - highest learning rate
        {'params': [], 'lr': base_lr, 'name': 'heads'},
        # Attention modules - medium learning rate
        {'params': [], 'lr': base_lr * 0.5, 'name': 'attention'}
    ]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'backbone' in name:
            param_groups[0]['params'].append(param)
        elif 'lateral' in name or 'extra_conv' in name:
            param_groups[1]['params'].append(param)
        elif 'classification_head' in name or 'regression_head' in name or 'to_class' in name or 'to_bbox' in name:
            param_groups[2]['params'].append(param)
        elif 'attention' in name:
            param_groups[3]['params'].append(param)
        else:
            param_groups[1]['params'].append(param)  # Default to FPN group
    
    # Remove empty groups
    param_groups = [g for g in param_groups if g['params']]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    return optimizer


def train(train_dir: pathlib.Path, 
    val_dir: pathlib.Path, 
    writer, 
    resume_ckpt_path=None,
    use_improved_model=True
):
    """Enhanced training function with attention mechanisms"""
    
    # Training hyperparameters
    BATCH_SIZE = 1
    IMAGE_SIZE = (1600,2000)
    BASE_LR = 1e-4
    NUM_EPOCHS = 50
    WARMUP_EPOCHS = 5
    
    # Attention loss weights (can be tuned)
    SPATIAL_ATT_WEIGHT = 0.1
    CENTER_ATT_WEIGHT = 0.05
    
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
    
    # Create model
    if use_improved_model:
        print("Creating improved model with attention...")
        model = ImprovedFPN(num_classes=num_classes,use_coord=False,use_cbam=True,use_deform=False)   
    
    
    model.to(device)
    
    # Enable mixed precision optimizations
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Create optimizer and scalers
    optimizer = create_optimizer_with_layer_wise_lr(model, BASE_LR)
    scaler = GradScaler(init_scale=2.**16)
    
    # Learning rate schedulers
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    adaptive_scheduler = AdaptiveLearningRateScheduler(optimizer)
    
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
            'total': [], 'cls': [], 'reg': [], 
            'spatial_att': [], 'center_att': []
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
                
                if use_improved_model:
                    cls_pred, box_pred, attention_maps = model(batch_norm)
                else:
                    cls_pred, box_pred = model(batch_norm)
                    attention_maps = None
                
                class_t, box_t = generate_targets(
                    x.shape, class_labels, box_labels, model.strides
                )
                
                # Compute loss with attention
                if use_improved_model:
                    total_loss, cls_loss, reg_loss, spatial_loss, center_loss = \
                        compute_loss_with_attention(
                            cls_pred, box_pred, class_t, box_t,
                            focal_loss, attention_maps,
                            classification_weight=2.0,
                            regression_weight=2.0,
                            spatial_attention_weight=SPATIAL_ATT_WEIGHT,
                            center_attention_weight=CENTER_ATT_WEIGHT,
                            box_labels_by_batch=box_labels,
                            img_shape=x.shape,
                            strides=model.strides
                        )
                else:
                    total_loss, cls_loss, reg_loss = _compute_loss(
                        cls_pred, box_pred, class_t, box_t, focal_loss,
                        box_labels_by_batch=box_labels,
                        img_shape=x.shape,
                        strides=model.strides
                    )
                    spatial_loss = center_loss = torch.tensor(0.0)
            
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
            epoch_losses['reg'].append(reg_loss.item())
            if use_improved_model:
                epoch_losses['spatial_att'].append(spatial_loss.item())
                epoch_losses['center_att'].append(center_loss.item())
            
            # Clean up
            del total_loss, cls_loss, reg_loss
            if use_improved_model:
                del spatial_loss, center_loss, attention_maps
        
        # Update schedulers
        if epoch <= WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        # Calculate average losses
        avg_losses = {k: np.mean(v) if v else 0 for k, v in epoch_losses.items()}
        
        # Adaptive learning rate adjustment
        if epoch > WARMUP_EPOCHS:
            plateau_detected = adaptive_scheduler.step(avg_losses['total'])
            if plateau_detected:
                print("Learning rate reduced due to plateau")
        
        # Log losses
        print(f"\nEpoch {epoch} Training Metrics:")
        print(f"  Total Loss: {avg_losses['total']:.4f}")
        print(f"  Classification Loss: {avg_losses['cls']:.4f}")
        print(f"  Regression Loss: {avg_losses['reg']:.4f}")
        if use_improved_model:
            print(f"  Spatial Attention Loss: {avg_losses['spatial_att']:.4f}")
            print(f"  Center Attention Loss: {avg_losses['center_att']:.4f}")
        
        
        # Write to tensorboard
        writer.add_scalar('Loss/Total', avg_losses['total'], epoch)
        writer.add_scalar('Loss/Classification', avg_losses['cls'], epoch)
        writer.add_scalar('Loss/Regression', avg_losses['reg'], epoch)
        if use_improved_model:
            writer.add_scalar('Loss/SpatialAttention', avg_losses['spatial_att'], epoch)
            writer.add_scalar('Loss/CenterAttention', avg_losses['center_att'], epoch)
        
        # =================== VALIDATION PHASE ===================
        current_val_loss = None
        if epoch % 20 == 0 or epoch == NUM_EPOCHS:  # Validate every 20 epochs and at the end
            model.eval()
            val_losses = []
            
            print("Running validation...")
            with torch.no_grad():
                for i, (x, class_labels, box_labels) in enumerate(val_loader):
                    if i >= 20:  # Evaluate on 20 validation samples
                        break
                    
                    x = x.to(device, non_blocking=True)
                    
                    with autocast(device_type='cuda'):
                        batch_norm = normalize_batch(x)
                        
                        if use_improved_model:
                            cls_pred, box_pred, attention_maps = model(batch_norm)
                        else:
                            cls_pred, box_pred = model(batch_norm)
                            attention_maps = None
                        
                        # Generate targets for validation
                        class_t, box_t = generate_targets(
                            x.shape, class_labels, box_labels, model.strides
                        )
                        
                        # Compute validation loss
                        if use_improved_model:
                            val_loss, _, _, _, _ = compute_loss_with_attention(
                                cls_pred, box_pred, class_t, box_t,
                                focal_loss, attention_maps,
                                classification_weight=2.0,
                                regression_weight=2.0,
                                spatial_attention_weight=SPATIAL_ATT_WEIGHT,
                                center_attention_weight=CENTER_ATT_WEIGHT,
                                box_labels_by_batch=box_labels,
                                img_shape=x.shape,
                                strides=model.strides
                            )
                        else:
                            val_loss, _, _ = _compute_loss(
                                cls_pred, box_pred, class_t, box_t, focal_loss,
                                box_labels_by_batch=box_labels,
                                img_shape=x.shape,
                                strides=model.strides
                            )
                        
                        # Only add finite losses
                        if torch.isfinite(val_loss):
                            val_losses.append(val_loss.item())
            
            # Calculate average validation loss
            current_val_loss = np.mean(val_losses) if val_losses else float('inf')
            print(f"  Validation Loss: {current_val_loss:.4f}")
            
            # Log to tensorboard
            writer.add_scalar('Loss/Validation', current_val_loss, epoch)
            
            # Check if this is the best model
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                print(f"  🎉 New best validation loss: {best_val_loss:.4f}")
        
        