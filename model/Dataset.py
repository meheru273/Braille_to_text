
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
                 image_size=(1200, 1800),
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
        print(f"Image ID: {img_id} | File: {img_path.name}")
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2,0,1)).float()
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        print(f"Total annotations: {len(anns)}")
        
        boxes, labels = [], []
        # Print only first 5 annotations for debugging
        for i, ann in enumerate(anns):
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_contiguous[ann['category_id']])
            
            if i < 5:
                print(f"  Annotation {i+1}: bbox={ann['bbox']}, category_id={ann['category_id']}, label={labels[-1]}")
            elif i == 5:
                print("  ... (more annotations omitted) ...")
        
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
