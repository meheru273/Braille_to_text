
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
    Enhanced COCO dataset with robust category ID handling
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
        ann_file = next((jf for jf in jsons if jf.name.lower().startswith(("train", "val", "valid", "annotations"))), jsons[0])
        
        print(f"Loading COCO annotations from: {ann_file}")
        self.coco = COCO(str(ann_file))
        self.images_dir = self.split_dir
        self.image_ids = list(self.coco.imgs.keys())
        
        # Configurable parameters
        self.image_size = image_size
        self.min_area = min_area
        self.max_detections = max_detections

        # Map ALL COCO category IDs to contiguous 1..N, filtering only expected classes
        all_cats = self.coco.loadCats(self.coco.getCatIds())
        valid_cats = [cat for cat in all_cats if cat['name'] in EXPECTED_CLASSES]
        print("total valid cats",len(valid_cats))
        if not valid_cats:
            raise ValueError("No valid categories found in the dataset matching expected classes")
        
        # Create mapping from original ID to contiguous ID
        sorted_cats = sorted(valid_cats, key=lambda x: x['id'])
        self.cat_id_to_contiguous = {}
        self.contiguous_to_name = {}
        
        for i, cat in enumerate(sorted_cats):
            self.cat_id_to_contiguous[cat['id']] = i + 1
            self.contiguous_to_name[i + 1] = cat['name']
        
        self.num_classes = len(sorted_cats) + 1  # +1 for background=0
        print(f"Mapped {len(self.cat_id_to_contiguous)} categories to contiguous IDs")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.coco.imgs[img_id]
        img_path = self.images_dir / info['file_name']
        
        # Handle potential path issues
        if not img_path.exists():
            # Try different naming variations
            possible_paths = [
                self.images_dir / info['file_name'],
                self.images_dir / Path(info['file_name']).name,
                self.images_dir / "images" / Path(info['file_name']).name,
                self.images_dir / "data" / Path(info['file_name']).name
            ]
            for p in possible_paths:
                if p.exists():
                    img_path = p
                    break
            else:
                raise FileNotFoundError(f"Image not found: {info['file_name']} in {self.images_dir}")
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes, labels = [], []
        skipped = 0
        
        for i, ann in enumerate(anns):
                
            # Handle missing category_id gracefully
            category_id = ann.get('category_id', None)
            if category_id is None or category_id not in self.cat_id_to_contiguous:
                skipped += 1
                continue
                
            # Convert bbox format from [x,y,w,h] to [x1,y1,x2,y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_contiguous[category_id])
        
        if skipped:
            print(f"Skipped {skipped} annotations in image {img_id} (invalid category or small area)")
        
        # Handle empty annotations
        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
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
    """Enhanced collate function with dynamic padding"""
    images, labels, boxes = zip(*batch)
    
    # Find max dimensions
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    padded_images = []
    for img in images:
        # Create padded image (C, H, W)
        padded_img = torch.zeros(3, max_h, max_w)
        h, w = img.shape[1], img.shape[2]
        padded_img[:, :h, :w] = img
        padded_images.append(padded_img)
    
    return torch.stack(padded_images), labels, boxes