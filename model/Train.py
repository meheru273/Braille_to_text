import pathlib
import os
import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from PIL import Image
from pycocotools.coco import COCO
import cv2

# Import existing functions from your modules
from inference import detections_from_network_output
from FPN import FPN, normalize_batch, FocalLoss
from Targets import generate_targets
logger = logging.getLogger(__name__)

from loss import _compute_loss
from Dataset import DSBIData ,COCOData ,collate_fn
import gc
from torch.cuda.amp import autocast, GradScaler

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.set_per_process_memory_fraction(0.80)

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
        {'params': backbone_params, 'lr': base_lr * 0.1},  # Lower LR for pretrained backbone
        {'params': new_params, 'lr': base_lr}              # Higher LR for new layers
    ])
    
    return optimizer


def train(train_dir: pathlib.Path, val_dir: pathlib.Path, writer, resume_ckpt_path=None):
    """
    Enhanced training function with resume capability 
    """
    # Training hyperparameters optimized for small Braille characters
    BATCH_SIZE = 2
    IMAGE_SIZE = (800,1200)
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
    train_dataset = COCOData(train_dir, image_size=IMAGE_SIZE, min_area=2)
    val_dataset = COCOData(val_dir, image_size=IMAGE_SIZE, min_area=2)
    

    num_classes = train_dataset.get_num_classes()
    class_names = train_dataset.get_class_names()
    print("number of classes:", num_classes)
    print("cpu count:", os.cpu_count())
    # Data loaders
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=min(8, os.cpu_count()),  # Reduce workers to avoid multiprocessing issues
                             collate_fn=collate_fn )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            num_workers=min(8, os.cpu_count()), collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Creating model...")
    model = FPN(num_classes=num_classes)
    model.to(device)
    
    # Create optimizer with different learning rates
    optimizer = create_optimizer(model, BASE_LR)
    scaler = torch.amp.GradScaler()
    
    # Learning rate scheduler
    if resume_ckpt_path is not None and os.path.isfile(resume_ckpt_path):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    start_epoch = 1
    
    
    if resume_ckpt_path is not None and os.path.isfile(resume_ckpt_path):
        print(f"Loading checkpoint from {resume_ckpt_path}")
        checkpoint = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Load scheduler state
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        if 'scaler_state' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Training will resume from epoch {start_epoch}")
        
        logger.info(f"Resuming training from epoch {start_epoch}")
    else:
        print("Starting training from scratch")
        
    focal_loss = FocalLoss(alpha="auto", gamma=2.0, num_classes=num_classes, reduction='mean')
    focal_loss.calculate_auto_alpha(train_dataset) 

    print(f"Starting training from epoch {start_epoch} to {NUM_EPOCHS}...")
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{NUM_EPOCHS} start")
        
        # Training phase
        model.train()
        epoch_losses = []
        epoch_cls_losses = []
        epoch_reg_losses = []
        epoch_att_losses = []
        
        for batch_idx, (x, class_labels, box_labels) in enumerate(train_loader):
            
            torch.cuda.empty_cache()
            gc.collect()

            optimizer.zero_grad()
            x = x.to(device)
            print("batch no :", batch_idx, "of epoch", epoch)
            
            # ✅ NEW CODE with gradient scaler
            with torch.amp.autocast(device_type='cuda'):
                batch_norm = normalize_batch(x)
                cls_pred, box_pred = model(batch_norm)
                class_t, box_t = generate_targets(x.shape, class_labels, box_labels, model.strides)
                total_loss, cls_loss, reg_loss = _compute_loss(
                    cls_pred, box_pred, class_t, box_t, focal_loss,
                    box_labels_by_batch=box_labels,
                    img_shape=x.shape,
                    strides=model.strides,
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

                        
            # ✅ FIXED: Track losses only once
            epoch_losses.append(total_loss.item())
            epoch_cls_losses.append(cls_loss.item())
            epoch_reg_losses.append(reg_loss.item())
        
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

        
        # Validation phase
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                for i, (x, class_labels, box_labels) in enumerate(val_loader):
                    if i >= 5:
                        break
                        
                    x = x.to(device)
                    batch_norm = normalize_batch(x)
                    cls_pred, box_pred = model(batch_norm)
                    
                    H, W = x.shape[2], x.shape[3]
                    detections = detections_from_network_output(
                        H, W, cls_pred, box_pred, model.scales, model.strides
                    )
                         
                    print(f"Validation image 0: {len(detections[0])} detections, "
                              f"{len(box_labels[0])} ground truth boxes")

        
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            # Create checkpoint directory if it doesn't exist
            checkpoint_dir = os.path.join(writer.log_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            ckpt_path = os.path.join(checkpoint_dir, f"fcos_epoch{epoch}.pth")
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
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
            }, ckpt_path)
            logger.info(f"Saved checkpoint {ckpt_path}")
            print(f"Checkpoint saved: {ckpt_path}")


# Usage:
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from torch.utils.tensorboard import SummaryWriter
    import pathlib
    
    train_dir = pathlib.Path(r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\dataset.coco\train")
    val_dir = pathlib.Path(r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\dataset.coco\valid")
    writer = SummaryWriter("runs/fcos_custom")
    
    # Set resume checkpoint path - change this to the checkpoint you want to resume from
    resume_ckpt_path = None  # or None to start fresh
    
    train(train_dir, val_dir, writer, resume_ckpt_path)
