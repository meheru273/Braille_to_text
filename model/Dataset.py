
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
    COCOData with comprehensive debugging
    """
    def __init__(self,
                 split_dir: pathlib.Path,
                 image_size=(700, 1024),
                 min_area=2,
                 max_detections=None):
        
        print(f"OOO DEBUG: Initializing COCOData with split_dir: {split_dir}")
        
        self.split_dir = pathlib.Path(split_dir)
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Folder not found: {self.split_dir}")
        
        # Find annotation JSON
        jsons = list(self.split_dir.glob("*.json"))
        print(f"OOO DEBUG: Found JSON files: {[j.name for j in jsons]}")
        
        if not jsons:
            raise FileNotFoundError(f"No JSON annotation file in {self.split_dir}")
        ann_file = next((jf for jf in jsons if "annotation" in jf.name.lower()), jsons[0])
        
        print(f"OOO DEBUG: Loading COCO annotations from: {ann_file}")
        self.coco = COCO(str(ann_file))
        self.image_ids = list(self.coco.imgs.keys())
        print(f"OOO DEBUG: Found {len(self.image_ids)} image IDs: {self.image_ids[:5]}...")
        
        # Parameters
        self.image_size = image_size
        self.min_area = min_area
        self.max_detections = max_detections
        
        # Check available categories
        all_cats = self.coco.loadCats(self.coco.getCatIds())
        print(f"OOO DEBUG: All available categories: {[(cat['id'], cat['name']) for cat in all_cats[:10]]}")
        
        # Filter categories
        cats = [cat for cat in all_cats if cat['name'] in EXPECTED_CLASSES]
        cats = sorted(cats, key=lambda x: x['id'])
        print(f"OOO DEBUG: Filtered categories: {[(cat['id'], cat['name']) for cat in cats]}")
        
        if not cats:
            print("❌ WARNING: No categories found matching EXPECTED_CLASSES!")
            print(f"Available category names: {[cat['name'] for cat in all_cats]}")
        
        self.cat_id_to_contiguous = {cat['id']: i+1 for i, cat in enumerate(cats)}
        self.contiguous_to_cat_id = {v: k for k, v in self.cat_id_to_contiguous.items()}
        self.num_classes = len(cats) + 1  # +1 for background=0
        
        print(f"OOO DEBUG: Category mapping: {self.cat_id_to_contiguous}")
        print(f"OOO DEBUG: Number of classes: {self.num_classes}")
        
        # Check image file locations
        self._debug_image_locations()
        
        # Test load first few annotations
        self._debug_sample_annotations()
    
    def _debug_image_locations(self):
        """Debug image file locations"""
        print(f"OOO DEBUG: Checking image locations in {self.split_dir}")
        
        # Check root directory
        root_images = list(self.split_dir.glob("*.jpg"))
        print(f"OOO DEBUG: Images in root: {len(root_images)} files")
        if root_images:
            print(f"  Sample: {root_images[:3]}")
        
        # Check data subdirectory
        data_dir = self.split_dir / "data"
        if data_dir.exists():
            data_images = list(data_dir.glob("*.jpg"))
            print(f"OOO DEBUG: Images in data/: {len(data_images)} files")
            if data_images:
                print(f"  Sample: {data_images[:3]}")
        else:
            print(f"OOO DEBUG: No data/ subdirectory found")
    
    def _debug_sample_annotations(self):
        """Debug sample annotations"""
        print(f"OOO DEBUG: Testing sample annotations...")
        
        for i, img_id in enumerate(self.image_ids[:3]):
            print(f"OOO DEBUG: Image {i+1} (ID: {img_id})")
            
            # Get image info
            img_info = self.coco.loadImgs([img_id])[0]
            print(f"  Image info: {img_info}")
            
            # Get annotations
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            print(f"  Found {len(anns)} annotations")
            
            valid_anns = 0
            for ann in anns:
                if ann['category_id'] in self.cat_id_to_contiguous:
                    valid_anns += 1
            print(f"  Valid annotations (matching EXPECTED_CLASSES): {valid_anns}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        print(f"OOO DEBUG: Loading sample {idx}")
        
        # Get image info from COCO
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        print(f"OOO DEBUG: Image ID {img_id}, filename: {img_info['file_name']}")
        
        # Try to find image file
        img_path = self.split_dir / img_info['file_name']
        if not img_path.exists():
            img_path = self.split_dir / "data" / img_info['file_name']
        
        if not img_path.exists():
            print(f"❌ ERROR: Image not found at either location:")
            print(f"  - {self.split_dir / img_info['file_name']}")
            print(f"  - {self.split_dir / 'data' / img_info['file_name']}")
            raise FileNotFoundError(f"Image not found: {img_info['file_name']}")
        
        print(f"OOO DEBUG: Loading image from: {img_path}")
        
        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
            original_size = img.size
            print(f"OOO DEBUG: Image loaded successfully, size: {original_size}")
        except Exception as e:
            print(f"❌ ERROR: Failed to load image: {e}")
            raise
        
        # Load annotations from COCO
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        print(f"OOO DEBUG: Found {len(anns)} annotations for this image")
        
        boxes = []
        labels = []
        
        for i, ann in enumerate(anns):
            print(f"  Category ID: {ann['category_id']}, Area: {ann['area']}")
            
            # Get bounding box in COCO format [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Validate bounding box
            if w <= 0 or h <= 0:
                print(f"  Skipped: invalid box dimensions w={w}, h={h}")
                continue
            
            box = [x, y, x + w, y + h]  # Convert to [x1, y1, x2, y2]
            
            # Check category mapping
            cat_id = ann['category_id']
            if cat_id in self.cat_id_to_contiguous:
                contiguous_id = self.cat_id_to_contiguous[cat_id]
                boxes.append(box)
                labels.append(contiguous_id)
                print(f"  Added: bbox={box}, label={contiguous_id}")
            else:
                print(f"  Skipped: category {cat_id} not in mapping")
        
        print(f"OOO DEBUG: Final count - {len(boxes)} boxes, {len(labels)} labels")
        
        # Validate boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x1 >= x2 or y1 >= y2:
                print(f"❌ WARNING: Invalid box {i}: {box}")
            if x1 < 0 or y1 < 0 or x2 > original_size[0] or y2 > original_size[1]:
                print(f"❌ WARNING: Box {i} outside image bounds: {box}, image size: {original_size}")
        
        # Resize image and adjust bounding boxes
        if self.image_size:
            print(f"OOO DEBUG: Resizing from {original_size} to {self.image_size}")
            img, boxes = self._resize_image_and_boxes(img, boxes, original_size)
            print(f"OOO DEBUG: After resize - {len(boxes)} boxes remain")
        
        # Convert image to tensor
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # Check for invalid pixel values
        if np.isnan(img_np).any():
            print(f"❌ ERROR: NaN values in image tensor!")
        if np.isinf(img_np).any():
            print(f"❌ ERROR: Inf values in image tensor!")
        
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        print(f"OOO DEBUG: Image tensor shape: {img_tensor.shape}")
        
        # Convert to tensors (handle empty annotations)
        if len(boxes) == 0:
            print(f"⚠️ WARNING: No valid annotations for image {idx}")
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            
            # Check for invalid values in tensors
            if torch.isnan(boxes).any():
                print(f"❌ ERROR: NaN values in boxes tensor!")
            if torch.isinf(boxes).any():
                print(f"❌ ERROR: Inf values in boxes tensor!")
        
        print(f"OOO DEBUG: Final tensors - boxes: {boxes.shape}, labels: {labels.shape}")
        print(f"OOO DEBUG: Sample complete for image {idx}")
        
        return img_tensor, labels, boxes
    
    def _resize_image_and_boxes(self, img, boxes, original_size):
        """Resize with debugging"""
        target_w, target_h = self.image_size
        orig_w, orig_h = original_size
        
        print(f"OOO DEBUG: Resize - original: {orig_w}x{orig_h}, target: {target_w}x{target_h}")
        
        # Calculate scale factors
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        scale = min(scale_w, scale_h)  # Maintain aspect ratio
        
        print(f"OOO DEBUG: Scale factors - w: {scale_w:.3f}, h: {scale_h:.3f}, chosen: {scale:.3f}")
        
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
        
        print(f"OOO DEBUG: Padding - x: {pad_x}, y: {pad_y}")
        
        # Paste resized image onto padded canvas
        new_img.paste(img, (pad_x, pad_y))
        
        # Adjust bounding boxes
        adjusted_boxes = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            
            # Scale coordinates
            x1_new = x1 * scale + pad_x
            y1_new = y1 * scale + pad_y
            x2_new = x2 * scale + pad_x
            y2_new = y2 * scale + pad_y
            
            # Ensure boxes are within image bounds
            x1_final = max(0, min(x1_new, target_w))
            y1_final = max(0, min(y1_new, target_h))
            x2_final = max(0, min(x2_new, target_w))
            y2_final = max(0, min(y2_new, target_h))
            
            print(f"OOO DEBUG: Box {i} - original: {box}")
            print(f"                   scaled: [{x1_new:.1f}, {y1_new:.1f}, {x2_new:.1f}, {y2_new:.1f}]")
            print(f"                   final:  [{x1_final:.1f}, {y1_final:.1f}, {x2_final:.1f}, {y2_final:.1f}]")
            
            # Only keep valid boxes
            if x2_final > x1_final and y2_final > y1_final:
                adjusted_boxes.append([x1_final, y1_final, x2_final, y2_final])
            else:
                print(f"❌ WARNING: Box {i} became invalid after resize")
        
        return new_img, adjusted_boxes
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_class_names(self):
        """Preserved class names logic with debugging"""
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = [cat for cat in cats if cat['name'] in EXPECTED_CLASSES]
        cats = sorted(cats, key=lambda x: x['id'])
        
        print(f"OOO DEBUG: get_class_names() found {len(cats)} classes")
        for c in cats:
            print(f"  Class {c['id']}: {c['name']}")
        
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

