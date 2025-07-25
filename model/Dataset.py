
import pathlib
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import os 
import re

EXPECTED_CLASSES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

class COCOData(Dataset):
    """
    Enhanced COCO dataset with proper preprocessing for Braille detection
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

