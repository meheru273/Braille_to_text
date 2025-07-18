
import pathlib
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import os 
EXPECTED_CLASSES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

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
        cats = [cat for cat in cats if cat['name'] in EXPECTED_CLASSES]
        cats = sorted(cats, key=lambda x: x['id'])
        self.cat_id_to_contiguous = {cat['id']: i+1 for i, cat in enumerate(cats)}
        self.num_classes = len(cats) + 1  # +1 for background=0

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.coco.imgs[img_id]
        img_path = self.images_dir / info['file_name']
        
        # Load image (already resized and padded by Roboflow)
        img = Image.open(img_path).convert("RGB")
        
        # Simple conversion to tensor (no resizing/padding needed)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2,0,1)).float()
        
        # Load annotations (coordinates already match the processed image)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_contiguous[ann['category_id']])
        
        # Convert to tensors
        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            box_tensor = torch.zeros((0,4), dtype=torch.float32)
            label_tensor = torch.zeros((0,), dtype=torch.long)
        
        return img_tensor, label_tensor, box_tensor

        
    def get_num_classes(self):
        return self.num_classes

    def get_class_names(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = sorted(cats, key=lambda x: x['id'])
        for c in cats:
            print(f"Class {c['id']}: {c['name']}")
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

class DSBIData(torch.utils.data.Dataset):
    def __init__(self, root_dir, file_list, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        if file_list is None:
            # Auto-discover image files
            self.img_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        else:
            with open(file_list, 'r') as f:
                self.img_files = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_files[idx])
        img = Image.open(img_path).convert("RGB")

        # Load annotation for recto dots (assuming recto annotations)
        ann_path = img_path.replace('.jpg', '+recto.txt')
        boxes = []
        labels = []

        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    # Skip first three lines (angle, vertical lines, horizontal lines)
                    vertical_lines = list(map(int, lines[1].strip().split()))
                    horizontal_lines = list(map(int, lines[2].strip().split()))
                    
                    cell_lines = lines[3:]
                    for line in cell_lines:
                        parts = line.strip().split()
                        if len(parts) == 8:
                            row_num = int(parts[0])
                            col_num = int(parts[1])
                            dots = list(map(int, parts[2:]))
                            
                            # Calculate bounding box coordinates
                            # Note: row_num and col_num start from 1
                            if col_num <= len(vertical_lines) and row_num <= len(horizontal_lines):
                                x_min = vertical_lines[col_num-1]
                                x_max = vertical_lines[col_num] if col_num < len(vertical_lines) else vertical_lines[-1]
                                y_min = horizontal_lines[row_num-1]
                                y_max = horizontal_lines[row_num] if row_num < len(horizontal_lines) else horizontal_lines[-1]

                                boxes.append([x_min, y_min, x_max, y_max])
                                # Label 1 for Braille cell presence
                                labels.append(1)

        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,))
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms:
            img = self.transforms(img)
        return img, target

    def get_num_classes(self):
        print("number of classes :", self.num_classes)
        return self.num_classes

    def get_class_names(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = sorted(cats, key=lambda x: x['id'])
        class_names = ['__background__'] + [c['name'] for c in cats]
        print(f"Class names: {class_names}")
        print(f"Total class names: {len(class_names)}")
        return class_names


     





