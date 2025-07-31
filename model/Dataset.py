import pathlib
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import os 
from pathlib import Path 
import json
import tempfile
from collections import defaultdict

# Use the EXPECTED_CLASSES from your uploaded file
EXPECTED_CLASSES = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
    'pattern_000001', 'pattern_000010', 'pattern_000011', 'pattern_000100', 
    'pattern_000101', 'pattern_000110', 'pattern_000111', 'pattern_001000', 
    'pattern_001001', 'pattern_001010', 'pattern_001011', 'pattern_001100', 
    'pattern_001101', 'pattern_001110', 'pattern_001111', 'pattern_010000', 
    'pattern_010001', 'pattern_010010', 'pattern_010011', 'pattern_010101', 
    'pattern_011000', 'pattern_011001', 'pattern_011010', 'pattern_011011', 
    'pattern_011101', 'pattern_011111', 'pattern_100001', 'pattern_100011', 
    'pattern_100101', 'pattern_100111', 'pattern_110001', 'pattern_110011', 
    'pattern_110101', 'pattern_110111', 'pattern_111011', 'pattern_111101', 
    'pattern_111111', 'space'
]

class COCOData(Dataset):
    """
    COCO dataset that dynamically maps classes found in the dataset to a contiguous range [0, N-1]
    based on their order in the EXPECTED_CLASSES list.
    This dataloader ensures:
    - Only classes present in EXPECTED_CLASSES are used
    - Class IDs are remapped to a contiguous range [0, num_classes_found-1]
    - The mapping respects the order defined in EXPECTED_CLASSES
    """
    def __init__(self,
                 split_dir: pathlib.Path,
                 image_size=(1600, 2000),
                 min_area=2,
                 max_detections=None,
                 verbose=True):
        self.split_dir = pathlib.Path(split_dir)
        self.verbose = verbose
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Folder not found: {self.split_dir}")

        # Find annotation JSON
        jsons = list(self.split_dir.glob("*.json"))
        if not jsons:
            raise FileNotFoundError(f"No JSON annotation file in {self.split_dir}")
        ann_file = None
        for json_file in jsons:
            if any(keyword in json_file.name.lower() for keyword in ["annotation", "coco"]):
                ann_file = json_file
                break
        if ann_file is None:
            ann_file = jsons[0]

        if self.verbose:
            print(f"Loading dataset from: {self.split_dir}")
            print(f"Using annotation file: {ann_file.name}")

        # Create clean annotation file
        self.clean_ann_file = self._create_clean_annotations(ann_file)

        # Load cleaned COCO data
        self.coco = COCO(str(self.clean_ann_file))
        self.images_dir = self.split_dir
        self.image_ids = list(self.coco.imgs.keys())

        # Parameters
        self.image_size = image_size
        self.min_area = min_area
        self.max_detections = max_detections

        # Setup class mapping based on EXPECTED_CLASSES order
        self._setup_class_mapping()

        if self.verbose:
            self._print_dataset_summary()

    def _create_clean_annotations(self, original_ann_file):
        """Create clean annotation file with only classes from EXPECTED_CLASSES"""
        if self.verbose:
            print(f"Creating clean annotation file...")

        with open(original_ann_file, 'r') as f:
            data = json.load(f)

        # Identify valid categories present in the dataset AND in EXPECTED_CLASSES
        # Preserve the order from EXPECTED_CLASSES
        valid_categories = []
        valid_cat_names_found = set(cat['name'] for cat in data['categories'] if cat['name'] in EXPECTED_CLASSES)
        
        # Add categories in the order they appear in EXPECTED_CLASSES
        for class_name in EXPECTED_CLASSES:
             if class_name in valid_cat_names_found:
                 # Find the original category dict
                 original_cat = next((cat for cat in data['categories'] if cat['name'] == class_name), None)
                 if original_cat:
                     valid_categories.append(original_cat)

        if self.verbose:
            original_cat_names = set(cat['name'] for cat in data['categories'])
            removed_names = original_cat_names - valid_cat_names_found
            if removed_names:
                print(f"  Removing classes not in EXPECTED_CLASSES: {sorted(removed_names)}")


        # Create mapping from old IDs to new sequential IDs (0-based)
        old_to_new_id = {}
        clean_categories = []
        for new_id, cat in enumerate(valid_categories):
            old_id = cat['id']
            old_to_new_id[old_id] = new_id
            clean_categories.append({
                'id': new_id,
                'name': cat['name'],
                'supercategory': cat.get('supercategory', 'braille')
            })

        # Filter and update annotations
        clean_annotations = []
        removed_count = 0
        for ann in data['annotations']:
            old_cat_id = ann['category_id']
            if old_cat_id in old_to_new_id:
                ann['category_id'] = old_to_new_id[old_cat_id]
                clean_annotations.append(ann)
            else:
                removed_count += 1

        # Create clean data
        clean_data = {
            'images': data['images'],
            'annotations': clean_annotations,
            'categories': clean_categories,
            'info': data.get('info', {}),
            'licenses': data.get('licenses', [])
        }

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(clean_data, temp_file, indent=2)
        temp_file.close()

        if self.verbose:
            print(f"  Final categories ({len(clean_categories)}): {[cat['name'] for cat in clean_categories]}")
            print(f"  Removed {removed_count} orphaned annotations")

        return temp_file.name

    def _setup_class_mapping(self):
        """Setup class mapping based on the cleaned categories"""
        all_cats = self.coco.loadCats(self.coco.getCatIds())
        if self.verbose:
            print(f"Setting up class mapping for {len(all_cats)} categories...")

        # These mappings are based on the cleaned categories (IDs 0 to N-1)
        self.cat_id_to_contiguous = {}  # Dataset ID (0 to N-1) -> Model class ID (0 to N-1) (Identity)
        self.contiguous_to_name = {}    # Model class ID (0 to N-1) -> Class name
        self.name_to_contiguous = {}    # Class name -> Model class ID (0 to N-1)

        # Since categories are already ordered and IDs are 0 to N-1, the mapping is straightforward
        for cat in all_cats:
            model_class_id = cat['id']  # This is the new sequential ID (0-based)
            class_name = cat['name']

            self.cat_id_to_contiguous[model_class_id] = model_class_id # Identity mapping
            self.contiguous_to_name[model_class_id] = class_name
            self.name_to_contiguous[class_name] = model_class_id

        self.num_classes = len(all_cats) # Dynamic number of classes found

        if self.verbose:
            print(f"  Class mapping complete: {self.num_classes} classes found in dataset and expected list")
            # Print first few mappings for verification
            print(f"  Example mappings (ID -> Name):")
            for i in range(min(5, self.num_classes)):
                print(f"    {i} -> {self.contiguous_to_name[i]}")
            if self.num_classes > 5:
                 print(f"    ... (and {self.num_classes - 5} more)")

    def _print_dataset_summary(self):
        """Print dataset summary"""
        print(f"\nDATASET SUMMARY:")
        print(f"  Split: {self.split_dir.name}")
        print(f"  Images: {len(self.image_ids)}")
        print(f"  Classes found in dataset and EXPECTED_CLASSES: {self.num_classes}")
        
        # Count annotations per class
        ann_counts = defaultdict(int)
        total_annotations = 0
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                cat_id = ann['category_id']
                # cat_id should now be the model class id (0 to N-1)
                if cat_id in self.contiguous_to_name: # Check if it's a valid mapped ID
                    class_name = self.contiguous_to_name[cat_id]
                    ann_counts[class_name] += 1
                    total_annotations += 1

        print(f"  Total annotations: {total_annotations}")

        # Show counts for classes in the order of EXPECTED_CLASSES
        print(f"\nCLASS ANNOTATION COUNTS (in EXPECTED_CLASSES order):")
        for class_name in EXPECTED_CLASSES:
            if class_name in self.name_to_contiguous: # If this class is in our dataset
                 model_id = self.name_to_contiguous[class_name]
                 count = ann_counts.get(class_name, 0)
                 status = "ok" if count > 0 else "not ok"
                 print(f"  {model_id:2d} ({class_name:<20}) : {count:4d} annotations {status}")
            # Optionally, you could also print classes NOT found in the dataset
            # else:
            #     print(f"     ({class_name:<20}) : Not present in this dataset")

        print(f"\nFINAL MAPPING: Classes mapped to IDs 0 to {self.num_classes - 1} based on EXPECTED_CLASSES order")


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.coco.imgs[img_id]
        img_path = self.images_dir / info['file_name']

        # Handle path issues (same as before)
        if not img_path.exists():
            possible_paths = [
                self.images_dir / info['file_name'],
                self.images_dir / Path(info['file_name']).name,
            ]
            for p in possible_paths:
                if p.exists():
                    img_path = p
                    break
            else:
                # Check if the dataset uses a 'data/' subdirectory structure
                data_subdir_path = self.images_dir / "data" / info['file_name']
                if data_subdir_path.exists():
                    img_path = data_subdir_path
                else:
                    raise FileNotFoundError(f"Image not found: {info['file_name']}")

        # Load image (same as before)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

        # Load annotations (same core logic, using updated mappings)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for ann in anns:
            category_id = ann.get('category_id', None)
            # Check if the category_id is valid (exists in our mapping)
            # After cleaning, category_id should be the new sequential ID (0 to N-1)
            if category_id is not None and category_id in self.cat_id_to_contiguous: 
                # Check minimum area
                if self.min_area and ann.get('area', 0) < self.min_area:
                    continue
                # Convert bbox [x,y,w,h] to [x1,y1,x2,y2]
                x, y, w, h = ann['bbox']
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h])
                # Use the identity mapping from cat_id_to_contiguous
                labels.append(self.cat_id_to_contiguous[category_id]) 
            # else: Annotation for an invalid/unmapped class, skip it

        # Limit detections (same as before)
        if self.max_detections and len(boxes) > self.max_detections:
            boxes = boxes[:self.max_detections]
            labels = labels[:self.max_detections]

        # Convert to tensors (same as before)
        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            label_tensor = torch.zeros((0,), dtype=torch.long)

        return img_tensor, label_tensor, box_tensor

    def get_num_classes(self):
        """Return the dynamic number of classes found in the dataset"""
        return self.num_classes

    def get_class_names(self):
        """Return the list of class names found, ordered by their model ID (0 to N-1)"""
        # Order names by their model ID
        names = [self.contiguous_to_name[i] for i in range(self.num_classes)]
        return names

    def get_class_mapping(self):
        """Return class mapping dictionaries"""
        return {
            'name_to_id': self.name_to_contiguous.copy(),
            'id_to_name': self.contiguous_to_name.copy()
        }

    def __del__(self):
        """Clean up temporary file"""
        if hasattr(self, 'clean_ann_file') and os.path.exists(self.clean_ann_file):
            try:
                os.unlink(self.clean_ann_file)
            except:
                pass

# collate_fn remains the same
def collate_fn(batch):
    """Collate function with padding"""
    images, labels, boxes = zip(*batch)
    # Find max dimensions
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    # Pad images
    padded_images = []
    for img in images:
        padded_img = torch.zeros(3, max_h, max_w, dtype=img.dtype)
        h, w = img.shape[1], img.shape[2]
        padded_img[:, :h, :w] = img
        padded_images.append(padded_img)
    image_batch = torch.stack(padded_images)
    return image_batch, labels, boxes
