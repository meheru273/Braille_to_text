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

logger = logging.getLogger(__name__)

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'



class DSBIData(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', file_list=None, transforms=None,min_area=2, image_size=(700, 1024)):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.image_size = image_size
        
        # Set number of classes (background + 26 Braille characters)
        self.num_classes = 27
        
        if file_list is None:
            self.img_files = self._discover_images_by_split()
        else:
            with open(file_list, 'r') as f:
                self.img_files = [line.strip() for line in f.readlines()]
        
        print(f"Found {len(self.img_files)} images for {split} split")

    def _discover_images_by_split(self):
        """Discover images based on train/test split logic"""
        img_files = []
        data_dir = os.path.join(self.root_dir, "data")
        
        if not os.path.exists(data_dir):
            print(f"Warning: data directory not found at {data_dir}")
            return []
        
        for category_folder in os.listdir(data_dir):
            category_path = os.path.join(data_dir, category_folder)
            
            if os.path.isdir(category_path):
                jpg_files = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
                
                for file in jpg_files:
                    if self._should_include_file(category_folder, file):
                        rel_path = os.path.join("data", category_folder, file)
                        img_files.append(rel_path)
        
        img_files.sort()
        return img_files
    
    def _should_include_file(self, category, filename):
        """Determine if file should be included based on split"""
        number_match = re.search(r'\+(\d+)\.jpg$', filename)
        
        if not number_match:
            return False
        
        file_number = int(number_match.group(1))
        
        existing_categories = ['Massage', 'Math', 'Ordinary Printed Document', 'Shaver Yang Fengting']
        test_only_categories = ['Fundamentals of Massage', 
                               'The Second Volume of Ninth Grade Chinese Book 1',
                               'The Second Volume of Ninth Grade Chinese Book 2']
        
        if self.split == 'train':
            # Special handling for each category in train
            if category == 'Massage' and 1 <= file_number <= 10:
                return True
            elif category == 'Math' and 1 <= file_number <= 10:
                return True
            elif category == 'Ordinary Printed Document' and 1 <= file_number <= 3:
                return True
            elif category == 'Shaver Yang Fengting' and 3 <= file_number <= 5:
                return True
                
        elif self.split == 'test':
            # Existing categories with higher numbers
            if category == 'Massage' and file_number >= 11:
                return True
            elif category == 'Math' and file_number >= 11:
                return True
            elif category == 'Ordinary Printed Document' and 4 <= file_number <= 6:
                return True
            elif category == 'Shaver Yang Fengting' and 6 <= file_number <= 8:
                return True
            # All test-only categories
            elif category in test_only_categories:
                return True
        
        return False
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root_dir, self.img_files[idx])
        img = Image.open(img_path).convert("RGB")
        
        # Apply letterboxing to match training preprocessing
        if self.image_size:
            img = self._letterbox_image(img)
        
        # Load annotations
        ann_path = img_path.replace('.jpg', '+recto.txt')
        boxes = []
        labels = []

        if os.path.exists(ann_path):
            boxes, labels = self._parse_annotation(ann_path)
        
        # Convert to tensors (handle empty annotations)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        # Apply transforms if specified
        if self.transforms:
            img = self.transforms(img)
        
        # Return format expected by your training pipeline
        return img, labels, boxes
    
    def _letterbox_image(self, img):
        """Apply letterboxing to match training preprocessing"""
        target_w, target_h = self.image_size
        orig_w, orig_h = img.size
        
        # Calculate scale to maintain aspect ratio
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        
        # Resize image
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Create letterboxed image with black padding
        letterboxed = Image.new('RGB', (target_w, target_h), (0, 0, 0))
        
        # Calculate paste position (center)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        
        # Paste resized image onto letterboxed background
        letterboxed.paste(img_resized, (paste_x, paste_y))
        
        return letterboxed
    
    def _parse_annotation(self, ann_path):
        """Parse DSBI annotation format"""
        boxes = []
        labels = []
        
        try:
            with open(ann_path, 'r') as f:
                lines = f.readlines()
                
                if len(lines) >= 3:
                    # Parse grid structure
                    vertical_lines = list(map(int, lines[1].strip().split()))
                    horizontal_lines = list(map(int, lines[2].strip().split()))
                    
                    # Parse cell data
                    cell_lines = lines[3:]
                    for line in cell_lines:
                        parts = line.strip().split()
                        if len(parts) == 8:
                            row_num = int(parts[0])
                            col_num = int(parts[1])
                            dots = list(map(int, parts[2:]))
                            
                            # Calculate bounding box coordinates
                            if (col_num <= len(vertical_lines) and 
                                row_num <= len(horizontal_lines) and
                                col_num > 0 and row_num > 0):
                                
                                x_min = vertical_lines[col_num-1]
                                x_max = vertical_lines[col_num] if col_num < len(vertical_lines) else vertical_lines[-1]
                                y_min = horizontal_lines[row_num-1]
                                y_max = horizontal_lines[row_num] if row_num < len(horizontal_lines) else horizontal_lines[-1]

                                # Only add if valid bounding box
                                if x_max > x_min and y_max > y_min:
                                    boxes.append([x_min, y_min, x_max, y_max])
                                    
                                    # Convert dots pattern to Braille character class
                                    braille_class = self._dots_to_braille_class(dots)
                                    labels.append(braille_class)
                                    
        except Exception as e:
            print(f"Warning: Error parsing annotation {ann_path}: {e}")
        
        return boxes, labels
    
    def _dots_to_braille_class(self, dots):
        """Convert 6-dot pattern to Braille character class (1-26 for a-z)"""
        # Standard Braille alphabet patterns
        braille_patterns = {
            (1,0,0,0,0,0): 1,   # a
            (1,1,0,0,0,0): 2,   # b
            (1,0,0,1,0,0): 3,   # c
            (1,0,0,1,1,0): 4,   # d
            (1,0,0,0,1,0): 5,   # e
            (1,1,0,1,0,0): 6,   # f
            (1,1,0,1,1,0): 7,   # g
            (1,1,0,0,1,0): 8,   # h
            (0,1,0,1,0,0): 9,   # i
            (0,1,0,1,1,0): 10,  # j
            (1,0,1,0,0,0): 11,  # k
            (1,1,1,0,0,0): 12,  # l
            (1,0,1,1,0,0): 13,  # m
            (1,0,1,1,1,0): 14,  # n
            (1,0,1,0,1,0): 15,  # o
            (1,1,1,1,0,0): 16,  # p
            (1,1,1,1,1,0): 17,  # q
            (1,1,1,0,1,0): 18,  # r
            (0,1,1,1,0,0): 19,  # s
            (0,1,1,1,1,0): 20,  # t
            (1,0,1,0,0,1): 21,  # u
            (1,1,1,0,0,1): 22,  # v
            (0,1,0,1,1,1): 23,  # w
            (1,0,1,1,0,1): 24,  # x
            (1,0,1,1,1,1): 25,  # y
            (1,0,1,0,1,1): 26,  # z
        }
        
        dots_tuple = tuple(dots)
        return braille_patterns.get(dots_tuple, 1)  # Default to 'a' if pattern not found

    def get_num_classes(self):
        return self.num_classes

    def get_class_names(self):
        """Return class names compatible with your training pipeline"""
        class_names = ['__background__'] + [chr(ord('a') + i) for i in range(26)]
        print(f"Class names: {class_names}")
        print(f"Total class names: {len(class_names)}")
        return class_names
    
def collate_fn(batch):
    """Standard collate function that stacks tensors"""
    images = []
    labels = []
    boxes = []
    
    max_h = max(img.shape[1] for img, _, _ in batch)
    max_w = max(img.shape[2] for img, _, _ in batch)
    
    for img, label, box in batch:
        # Pad image to max size
        h, w = img.shape[1], img.shape[2]
        padded_img = torch.zeros(3, max_h, max_w)
        padded_img[:, :h, :w] = img
        
        images.append(padded_img)
        labels.append(label)
        boxes.append(box)
    
    # Stack into tensors
    images = torch.stack(images)
    return images, labels, boxes




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
