
import pathlib
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import os
import torchvision.transforms.functional as F

# Keep your original expected classes
EXPECTED_CLASSES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

class COCOData(Dataset):
    """
    Enhanced COCO dataset with preserved EXPECTED_CLASSES filtering
    """
    def __init__(self,
                 split_dir: pathlib.Path,
                 image_size=(700, 1024),
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
        self.image_ids = list(self.coco.imgs.keys())
        
        # Parameters
        self.image_size = image_size
        self.min_area = min_area
        self.max_detections = max_detections
        
        # PRESERVED FUNCTIONALITY: Map original COCO category IDs to contiguous 1..N
        # Filter categories to only include EXPECTED_CLASSES (your original logic)
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = [cat for cat in cats if cat['name'] in EXPECTED_CLASSES]  # Keep your filtering
        cats = sorted(cats, key=lambda x: x['id'])
        
        self.cat_id_to_contiguous = {cat['id']: i+1 for i, cat in enumerate(cats)}
        self.contiguous_to_cat_id = {v: k for k, v in self.cat_id_to_contiguous.items()}
        self.num_classes = len(cats) + 1  # +1 for background=0
        
        print(f"Dataset loaded: {len(self.image_ids)} images")
        print(f"Filtered to {len(cats)} classes from EXPECTED_CLASSES")
        print(f"Total classes (including background): {self.num_classes}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image info from COCO
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        
        # FIXED: Proper image path construction
        # Check both root directory and data subdirectory
        img_path = self.split_dir / img_info['file_name']
        if not img_path.exists():
            img_path = self.split_dir / "data" / img_info['file_name']
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_info['file_name']}")
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        original_size = img.size
        
        # FIXED: Load annotations from COCO (not .txt files!)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        
        for ann in anns:
            # Filter by area
            if ann['area'] < self.min_area:
                continue
            
            # Get bounding box in COCO format [x, y, width, height]
            x, y, w, h = ann['bbox']
            box = [x, y, x + w, y + h]  # Convert to [x1, y1, x2, y2]
            
            # PRESERVED: Use your original category filtering logic
            cat_id = ann['category_id']
            if cat_id in self.cat_id_to_contiguous:
                contiguous_id = self.cat_id_to_contiguous[cat_id]
                boxes.append(box)
                labels.append(contiguous_id)
        
        # Apply max detections limit
        if self.max_detections and len(boxes) > self.max_detections:
            boxes = boxes[:self.max_detections]
            labels = labels[:self.max_detections]
        
        # Resize image and adjust bounding boxes
        if self.image_size:
            img, boxes = self._resize_image_and_boxes(img, boxes, original_size)
        
        # Convert image to tensor
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        
        # Convert to tensors (handle empty annotations)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        
        return img_tensor, labels, boxes
    
    def _resize_image_and_boxes(self, img, boxes, original_size):
        """Resize image while maintaining aspect ratio and adjust bounding boxes"""
        target_w, target_h = self.image_size
        orig_w, orig_h = original_size
        
        # Calculate scale factors
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        scale = min(scale_w, scale_h)  # Maintain aspect ratio
        
        # Calculate new dimensions
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize image
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Create new image with target size (letterbox with padding)
        new_img = Image.new('RGB', (target_w, target_h), (128, 128, 128))
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Paste resized image onto padded canvas
        new_img.paste(img, (pad_x, pad_y))
        
        # Adjust bounding boxes
        adjusted_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Scale coordinates
            x1 = x1 * scale + pad_x
            y1 = y1 * scale + pad_y
            x2 = x2 * scale + pad_x
            y2 = y2 * scale + pad_y
            
            # Ensure boxes are within image bounds
            x1 = max(0, min(x1, target_w))
            y1 = max(0, min(y1, target_h))
            x2 = max(0, min(x2, target_w))
            y2 = max(0, min(y2, target_h))
            
            # Only keep valid boxes
            if x2 > x1 and y2 > y1:
                adjusted_boxes.append([x1, y1, x2, y2])
        
        return new_img, adjusted_boxes
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_class_names(self):
        """PRESERVED: Your original class names logic"""
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = [cat for cat in cats if cat['name'] in EXPECTED_CLASSES]  # Keep filtering
        cats = sorted(cats, key=lambda x: x['id'])
        
        # Debug output (preserved from your original)
        for c in cats:
            print(f"Class {c['id']}: {c['name']}")
        
        return ['__background__'] + [c['name'] for c in cats]


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

