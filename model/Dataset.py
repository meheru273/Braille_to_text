
import pathlib
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import os 



class DSBIData(torch.utils.data.Dataset):
    def __init__(self, root_dir, file_list, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
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






