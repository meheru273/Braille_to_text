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


import pathlib
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

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
    imgs, lbls, bxs = zip(*batch)
    return torch.stack(imgs), list(lbls), list(bxs)


def tensor_to_image(tensor):
    """Convert tensor to numpy array for visualization"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def _compute_loss(
    classes, centernesses, boxes, class_targets, centerness_targets, box_targets,
    focal_loss_fn, classification_weight=2.0, centerness_weight=1.0, regression_weight=2.0
):
    """Fixed loss computation that handles shape mismatches"""

    device = classes[0].device
    num_classes = classes[0].shape[-1]

    box_loss_fn = torch.nn.SmoothL1Loss(reduction='none')
    cen_loss_fn = torch.nn.BCELoss(reduction='none')

    classification_losses = []
    centerness_losses = []
    regression_losses = []

    for idx in range(len(classes)):
        cls_p = classes[idx]  # [B, H, W, num_classes]
        cen_p = centernesses[idx]  # [B, H, W]
        box_p = boxes[idx]    # [B, H, W, 4]

        cls_t = class_targets[idx].to(device)  # [B, H_target, W_target]
        cen_t = centerness_targets[idx].to(device)  # [B, H_target, W_target]
        box_t = box_targets[idx].to(device)  # [B, H_target, W_target, 4]

        # Resize targets to match model output if shapes don't match
        B, H_pred, W_pred = cls_p.shape[:3]
        B_t, H_target, W_target = cls_t.shape

        if (H_pred, W_pred) != (H_target, W_target):
            # Resize class targets using nearest neighbor
            cls_t = F.interpolate(cls_t.float().unsqueeze(1),size=(H_pred, W_pred),
                                  mode='nearest').squeeze(1).long()
            # Resize centerness targets using bilinear
            cen_t = F.interpolate(cen_t.unsqueeze(1),size=(H_pred, W_pred),
                mode='bilinear',align_corners=False).squeeze(1)

            # Resize box targets
            box_t = F.interpolate( box_t.permute(0, 3, 1, 2),size=(H_pred, W_pred),
                mode='bilinear',align_corners=False).permute(0, 2, 3, 1)

        # Flatten for loss computation
        cls_p_flat = cls_p.view(-1, num_classes)
        cls_t_flat = cls_t.view(-1)
        cen_p_flat = cen_p.view(-1)
        cen_t_flat = cen_t.view(-1)
        box_p_flat = box_p.view(-1, 4)
        box_t_flat = box_t.view(-1, 4)

        pos_mask = cls_t_flat > 0

        # Classification loss
        cls_loss = focal_loss_fn(cls_p_flat, cls_t_flat)
        classification_losses.append(cls_loss)

        # Centerness loss (only on positive samples)
        if pos_mask.sum() > 0:
            cen_loss = cen_loss_fn(cen_p_flat[pos_mask], cen_t_flat[pos_mask]).mean()
            centerness_losses.append(cen_loss)

            # Regression loss (only on positive samples)
            reg_loss = box_loss_fn(box_p_flat[pos_mask], box_t_flat[pos_mask]).mean()
            regression_losses.append(reg_loss)

    # Compute total loss
    total_cls_loss = torch.stack(classification_losses).mean() if classification_losses else torch.tensor(0.0, device=device)
    total_cen_loss = torch.stack(centerness_losses).mean() if centerness_losses else torch.tensor(0.0, device=device)
    total_reg_loss = torch.stack(regression_losses).mean() if regression_losses else torch.tensor(0.0, device=device)

    total_loss = (classification_weight * total_cls_loss +
                  centerness_weight * total_cen_loss +
                  regression_weight * total_reg_loss)

    return total_loss, total_cls_loss, total_cen_loss, total_reg_loss




def _render_targets_to_image(img: np.ndarray, box_labels: torch.Tensor):
    """Render ground truth boxes on image"""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    for i in range(box_labels.shape[0]):
        x1, y1, x2, y2 = box_labels[i].tolist()
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)
    return img


def unfreeze_backbone_gradually(model, epoch):
    """Gradually unfreeze backbone layers during training"""
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'):
        if epoch == 2:
            # Unfreeze layer4
            for param in model.backbone.layer4.parameters():
                param.requires_grad = True
        elif epoch == 4:
            # Unfreeze layer3
            for param in model.backbone.layer3.parameters():
                param.requires_grad = True
        elif epoch == 6:
            # Unfreeze layer2
            for param in model.backbone.layer2.parameters():
                param.requires_grad = True
        elif epoch == 8:
            # Unfreeze layer2
            for param in model.backbone.layer1.parameters():
                param.requires_grad = True


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
    BATCH_SIZE = 2  # Reduced for ResNet-50
    IMAGE_SIZE = (800,1200)  # Better aspect ratio for documents
    BASE_LR = 1e-4
    NUM_EPOCHS = 50
    
    # Loss weights
    CLASSIFICATION_WEIGHT = 2.0
    CENTERNESS_WEIGHT = 1.0
    REGRESSION_WEIGHT = 2.0
    
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
    logger.info(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    print("Creating model with pretrained backbone...")
    model = FPN(num_classes=num_classes)
    model.to(device)
    
    # Create optimizer with different learning rates
    optimizer = create_optimizer(model, BASE_LR)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    # Initialize start epoch
    start_epoch = 1
        
    # Focal loss for class imbalance
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0, num_classes=num_classes)  # Initialize with default
    focal_loss.update_alpha_from_dataloader(train_loader)
    

    print(f"Starting training from epoch {start_epoch} to {NUM_EPOCHS}...")
    
    # FIXED: Use start_epoch instead of starting from 1
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{NUM_EPOCHS} start")
        
        # Gradually unfreeze backbone
        unfreeze_backbone_gradually(model, epoch)
        
        # Training phase
        model.train()
        epoch_losses = []
        epoch_cls_losses = []
        epoch_cen_losses = []
        epoch_reg_losses = []
        
        for batch_idx, (x, class_labels, box_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            
            # Normalize batch
            batch_norm = normalize_batch(x)
            
            # Forward pass
            cls_pred, cen_pred, box_pred = model(batch_norm)
            
            # Generate targets
            debug_enabled = (epoch == 1 and batch_idx < 3)  # Debug first 3 batches of first epoch
            class_targets, centerness_targets, box_targets = generate_targets(
                x.shape, class_labels, box_labels, model.strides
            )
            
            total_loss, cls_loss, cen_loss, reg_loss = _compute_loss(
                cls_pred, cen_pred, box_pred,
                class_targets, centerness_targets, box_targets,
                focal_loss, CLASSIFICATION_WEIGHT, CENTERNESS_WEIGHT, REGRESSION_WEIGHT
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            # Track losses
            epoch_losses.append(total_loss.item())
            epoch_cls_losses.append(cls_loss.item())
            epoch_cen_losses.append(cen_loss.item())
            epoch_reg_losses.append(reg_loss.item())
            
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_cls_loss = np.mean(epoch_cls_losses)
        avg_cen_loss = np.mean(epoch_cen_losses)
        avg_reg_loss = np.mean(epoch_reg_losses)
        
        print(f"Epoch {epoch} completed:")
        print(f"  Avg Total Loss: {avg_loss:.4f}")
        print(f"  Avg Cls Loss: {avg_cls_loss:.4f}")
        print(f"  Avg Cen Loss: {avg_cen_loss:.4f}")
        print(f"  Avg Reg Loss: {avg_reg_loss:.4f}")
        
        # Validation phase
        if epoch % 5 == 0:  # Validate every 5 epochs
            model.eval()
            with torch.no_grad():
                for i, (x, class_labels, box_labels) in enumerate(val_loader):
                    if i >= 5:  # Only validate on first 5 images
                        break
                        
                    x = x.to(device)
                    batch_norm = normalize_batch(x)
                    cls_pred, cen_pred, box_pred = model(batch_norm)
                    
                    H, W = x.shape[2], x.shape[3]
                    detections = detections_from_network_output(
                        H, W, cls_pred, cen_pred, box_pred, model.scales, model.strides
                    )
                    
                    if i == 0:  # Visualize first validation image
                        img_vis = tensor_to_image(x[0])
                        img_with_dets = render_detections_to_image(img_vis.copy(), detections[0])
                        img_with_targets = _render_targets_to_image(img_vis.copy(), box_labels[0])
                        
                        print(f"Validation image 0: {len(detections[0])} detections, "
                                   f"{len(box_labels[0])} ground truth boxes")

        # Save checkpoint every 3 epochs
        if epoch % 10 == 0 or epoch == NUM_EPOCHS:
            ckpt_path = os.path.join(writer.log_dir, f"fcos_epoch{epoch}.pth")
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': epoch,
                'class_names': class_names,
                'losses': {
                    'total': avg_loss,
                    'classification': avg_cls_loss,
                    'centerness': avg_cen_loss,
                    'regression': avg_reg_loss
                }
            }, ckpt_path)
            logger.info(f"Saved checkpoint {ckpt_path}")

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
