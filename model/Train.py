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
from inference import detections_from_network_output, render_detections_to_image
from FPN import FPN, normalize_batch, FocalLoss
from Targets import generate_targets
logger = logging.getLogger(__name__)

from loss import _compute_loss, compute_attention_loss
import time
class COCOData(Dataset):
    """
    Enhanced COCO dataset with proper preprocessing for Braille detection
    """
    def __init__(self,
                 split_dir: pathlib.Path,
                 image_size=(800, 1200),
                 min_area=2,
                 max_detections=None):
        self.split_dir = pathlib.Path(split_dir)
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Folder not found: {self.split_dir}")
        # Find annotation JSON
        jsons = list(self.split_dir.glob("*.json"))
        if not jsons:
            raise FileNotFoundError(f"No JSON annotation file in {self.split_dir}")
        ann_file = next((jf for jf in jsons if "annotation" in jf.name.lower()), jsons[0])
        print(f"Loading COCO annotations from: {ann_file}")
        self.coco = COCO(str(ann_file))
        self.images_dir = self.split_dir
        self.image_ids = list(self.coco.imgs.keys())
        # Configurable parameters
        self.image_size = image_size  # (width, height)
        self.min_area = min_area
        self.max_detections = max_detections

        # Map original COCO category IDs to contiguous 1..N
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = sorted(cats, key=lambda x: x['id'])
        self.cat_id_to_contiguous = {cat['id']: i+1 for i, cat in enumerate(cats)}
        self.num_classes = len(cats) + 1  # +1 for background=0

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.coco.imgs[img_id]
        img_path = self.images_dir / info['file_name']
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        # Load and letterbox image
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = info['width'], info['height']
        target_w, target_h = self.image_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        letterboxed = Image.new('RGB', (target_w, target_h), (114, 114, 114))
        paste_x, paste_y = (target_w - new_w) // 2, (target_h - new_h) // 2
        letterboxed.paste(img_resized, (paste_x, paste_y))
        img_np = np.array(letterboxed).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2,0,1)).float()
        # Load and transform annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w * h < self.min_area:
                continue
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1n = x1 * scale + paste_x
            y1n = y1 * scale + paste_y
            x2n = x2 * scale + paste_x
            y2n = y2 * scale + paste_y
            x1n, y1n = max(0, x1n), max(0, y1n)
            x2n, y2n = min(target_w, x2n), min(target_h, y2n)
            
            if x2n > x1n and y2n > y1n:
                if (x2n - x1n) * (y2n - y1n) >= self.min_area:
                    boxes.append([x1n, y1n, x2n, y2n])
                    labels.append(self.cat_id_to_contiguous[ann['category_id']])
                    if self.max_detections and len(boxes) >= self.max_detections:
                        break
        if boxes:
            return img_tensor, torch.tensor(labels, dtype=torch.long), torch.tensor(boxes, dtype=torch.float32)
        return img_tensor, torch.zeros((0,), dtype=torch.long), torch.zeros((0,4), dtype=torch.float32)
        
    def get_num_classes(self):
        return self.num_classes

    def get_class_names(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = sorted(cats, key=lambda x: x['id'])
        return ['__background__'] + [c['name'] for c in cats]

def collate_fn(batch):
    """
    Custom collate function for object detection with variable-sized annotations
    """
    images = []
    class_labels = []
    box_labels = []
    
    for sample in batch:
        if len(sample) == 3:  # (image, class_labels, box_labels)
            img, cls_lbl, box_lbl = sample
            
            # Ensure image is a proper tensor, not a view
            if isinstance(img, torch.Tensor):
                img = img.clone()
            
            images.append(img)
            class_labels.append(cls_lbl)
            box_labels.append(box_lbl)
        else:
            print(f"Warning: Unexpected sample structure: {len(sample)} elements")
            continue
    
    # Stack images if they all have the same shape
    try:
        if len(images) > 0 and all(img.shape == images[0].shape for img in images):
            images = torch.stack(images, dim=0)
        else:
            # Keep as list if shapes differ
            pass
    except Exception as e:
        print(f"Warning: Could not stack images: {e}")
        # Keep as list
    
    return images, class_labels, box_labels



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
    Enhanced training function with resume capability and fixed attention loss tracking
    """
    # Training hyperparameters optimized for small Braille characters
    BATCH_SIZE = 4
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

    # Data loaders
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,  # Reduce workers to avoid multiprocessing issues
                            pin_memory=True,persistent_workers=True, collate_fn=collate_fn )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=4, collate_fn=collate_fn,prefetch_factor=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Creating model with pretrained backbone...")
    model = FPN(num_classes=num_classes)
    model.to(device)
    
    # Create optimizer with different learning rates
    optimizer = create_optimizer(model, BASE_LR)
    
    # Learning rate scheduler
    if resume_ckpt_path is not None and os.path.isfile(resume_ckpt_path):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    start_epoch = 1
    
    # ✅ FIXED: Complete checkpoint loading logic
    if resume_ckpt_path is not None and os.path.isfile(resume_ckpt_path):
        print(f"Loading checkpoint from {resume_ckpt_path}")
        checkpoint = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state'])
        print(f"✅ Model state loaded from epoch {checkpoint['epoch']}")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"✅ Optimizer state loaded")
        
        # Load scheduler state
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        print(f"✅ Scheduler state loaded")
        
        # ✅ CRITICAL: Set start_epoch to resume from next epoch
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
        epoch_cen_losses = []
        epoch_reg_losses = []
        epoch_att_losses = []
        
        for batch_idx, (x, class_labels, box_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            print("batch no :", batch_idx, "of epoch", epoch)
            
            # Normalize batch
            batch_norm = normalize_batch(x)
            
            # Forward pass
            cls_pred, cen_pred, box_pred = model(batch_norm)

            # Generate targets
            class_t, cen_t, box_t = generate_targets(
                x.shape, class_labels, box_labels, model.strides)

            total_loss, cls_loss, cen_loss, reg_loss, att_loss = _compute_loss(
                cls_pred, cen_pred, box_pred,
                class_t, cen_t, box_t,
                focal_loss,
                attention_maps=model.attention_maps,
                box_labels_by_batch=box_labels,
                img_shape=x.shape,
                strides=model.strides,
                attention_weight=0.1)

            # ✅ FIXED: Proper gradient handling
            total_loss.backward()
            
            # ✅ FIXED: Gradient clipping BEFORE optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # ✅ FIXED: Single optimizer step
            optimizer.step()

            # ✅ FIXED: Track losses only once
            epoch_losses.append(total_loss.item())
            epoch_cls_losses.append(cls_loss.item())
            epoch_cen_losses.append(cen_loss.item())
            epoch_reg_losses.append(reg_loss.item())
            epoch_att_losses.append(att_loss.item())
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average losses
        avg_loss = np.mean(epoch_losses)
        avg_cls_loss = np.mean(epoch_cls_losses)
        avg_cen_loss = np.mean(epoch_cen_losses)
        avg_reg_loss = np.mean(epoch_reg_losses)
        avg_att_loss = np.mean(epoch_att_losses)
        
        # Log losses
        print(f"Epoch {epoch} completed:")
        print(f"  Avg Total Loss: {avg_loss:.4f}")
        print(f"  Avg Cls Loss: {avg_cls_loss:.4f}")
        print(f"  Avg Cen Loss: {avg_cen_loss:.4f}")
        print(f"  Avg Reg Loss: {avg_reg_loss:.4f}")
        print(f"  Avg Att Loss: {avg_att_loss:.4f}")
        
        # Log to tensorboard
        if writer:
            writer.add_scalar('Train/Total_Loss', avg_loss, epoch)
            writer.add_scalar('Train/Classification_Loss', avg_cls_loss, epoch)
            writer.add_scalar('Train/Centerness_Loss', avg_cen_loss, epoch)
            writer.add_scalar('Train/Regression_Loss', avg_reg_loss, epoch)
            writer.add_scalar('Train/Attention_Loss', avg_att_loss, epoch)
            writer.add_scalar('Train/Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        # Validation phase
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                for i, (x, class_labels, box_labels) in enumerate(val_loader):
                    if i >= 5:
                        break
                        
                    x = x.to(device)
                    batch_norm = normalize_batch(x)
                    cls_pred, cen_pred, box_pred = model(batch_norm)
                    
                    H, W = x.shape[2], x.shape[3]
                    detections = detections_from_network_output(
                        H, W, cls_pred, cen_pred, box_pred, model.scales, model.strides
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
                'epoch': epoch,
                'class_names': class_names,
                'num_classes': num_classes,
                'losses': {
                    'total': avg_loss,
                    'classification': avg_cls_loss,
                    'centerness': avg_cen_loss,
                    'regression': avg_reg_loss,
                    'attention': avg_att_loss
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

        persistent_path = "/kaggle/working/latest_checkpoint.pth"
        torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'epoch': epoch,
        'timestamp': time.time()
    }, persistent_path)
    logger.info("Training completed!")


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
