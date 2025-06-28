import pathlib
import os
import logging
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from PIL import Image
from pycocotools.coco import COCO
import cv2

# Import existing functions from your modules
from inference import detections_from_network_output, render_detections_to_image
from FPN import FPN, normalize_batch
from Targets import generate_targets
from metrics import compute_metrics

logger = logging.getLogger(__name__)

class COCOData(Dataset):
    """
    Assumes split_dir contains:
      - image files (*.jpg, *.png, etc.)
      - one COCO JSON annotation file, e.g. annotations.coco.json
    """
    def __init__(self,
                 split_dir: pathlib.Path,
                 image_size=(512,512),
                 min_area=0,
                 max_detections=None):
        """
        Args:
            split_dir: path to folder with images and annotation JSON in the same directory
            image_size: (H, W) to resize images to
            min_area: filter out small GT boxes (< min_area in original pixels)
            max_detections: cap number of boxes per image
        """
        self.split_dir = pathlib.Path(split_dir)
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Folder not found: {self.split_dir}")
        
        # Find the JSON annotation file in split_dir
        jsons = list(self.split_dir.glob("*.json"))
        if not jsons:
            raise FileNotFoundError(f"No JSON annotation file in {self.split_dir}")
        
        ann_file = None
        for jf in jsons:
            if "annotation" in jf.name.lower():
                ann_file = jf
                break
        if ann_file is None:
            ann_file = jsons[0]
        
        print(f"Loading COCO annotations from: {ann_file}")
        self.coco = COCO(str(ann_file))

        # Images are directly in split_dir
        self.images_dir = self.split_dir
        self.image_ids = list(self.coco.imgs.keys())

        self.image_size = image_size
        self.transforms = [Resize(image_size)]
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
        file_name = info['file_name']
        img_path = self.images_dir / file_name
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = info['width'], info['height']
        
        # Resize
        for t in self.transforms:
            img = t(img)
        new_w, new_h = img.size  # PIL size: (width, height)
        
        # Convert to tensor - keep in 0-1 range for now
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2,0,1)).float()

        # Load annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            area = w * h
            if area < self.min_area:
                continue
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            # scale coords from original to resized
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            x1 *= scale_x; x2 *= scale_x
            y1 *= scale_y; y2 *= scale_y
            # clamp
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(new_w, x2); y2 = min(new_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            orig_cat = ann['category_id']
            labels.append(self.cat_id_to_contiguous[orig_cat])
            if self.max_detections and len(boxes) >= self.max_detections:
                break
        
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
        return ['__background__'] + [c['name'] for c in cats]


def collate_fn(batch):
    imgs, labels, boxes = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(labels), list(boxes)


def tensor_to_image(tensor):
    """Convert tensor to numpy array for visualization"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def _compute_loss(
    strides, classes, centernesses, boxes, class_targets, centerness_targets, box_targets
):
    batch_size = classes[0].shape[0]
    num_classes = classes[0].shape[-1]
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    box_loss_fn = torch.nn.L1Loss()
    cen_loss_fn = torch.nn.BCELoss()
    losses = []
    device = classes[0].device
    for idx in range(len(classes)):
        cls_t = class_targets[idx].to(device).view(batch_size, -1)
        cen_t = centerness_targets[idx].to(device).view(batch_size, -1)
        box_t = box_targets[idx].to(device).view(batch_size, -1, 4)
        cls_p = classes[idx].view(batch_size, -1, num_classes)
        box_p = boxes[idx].view(batch_size, -1, 4)
        cen_p = centernesses[idx].view(batch_size, -1)
        losses.append(cen_loss_fn(cen_p, cen_t))
        ls = cls_loss_fn(cls_p.view(-1, num_classes), cls_t.view(-1))
        losses.append(ls)
        for b in range(batch_size):
            mask = cls_t[b] > 0
            if mask.nonzero().sum() > 0:
                l = box_loss_fn(box_p[b][mask], box_t[b][mask]) * strides[idx]
                losses.append(l)
    return torch.stack(losses).mean()

def _render_targets_to_image(img: np.ndarray, box_labels: torch.Tensor):
    # Ensure img is contiguous uint8
    img = np.ascontiguousarray(img, dtype=np.uint8)
    for i in range(box_labels.shape[0]):
        x1, y1, x2, y2 = box_labels[i].tolist()
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)
    return img

def train(train_dir: pathlib.Path, val_dir: pathlib.Path, writer):
    """
    train_dir: path to dataset.coco/train/
    val_dir:   path to dataset.coco/val/
    """
    # Ensure paths exist and are Path objects
    train_dir = pathlib.Path(train_dir)
    val_dir = pathlib.Path(val_dir)

    
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir.absolute()}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir.absolute()}")
    
    print("Creating datasets...")
    train_dataset = COCOData(train_dir, image_size=(512,512), min_area=32)
    val_dataset   = COCOData(val_dir,   image_size=(512,512), min_area=32)

    num_classes = train_dataset.get_num_classes()
    class_names = train_dataset.get_class_names()
    logger.info(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")
    logger.info(f"Num classes (incl background=0): {num_classes}")

    # Reduce num_workers to 0 for debugging potential hanging issues
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              num_workers=3, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False,
                              num_workers=3, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    print("Creating model...")
    model = FPN(num_classes=num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint = 0

    for epoch in range(1, 51):
        logger.info(f"Epoch {epoch} start")
        model.train()
        for batch_idx, (x, class_labels, box_labels) in enumerate(train_loader):
            print(f"Processing batch {batch_idx}")  # Debug print
            optimizer.zero_grad()
            x = x.to(device)
            batch_norm = normalize_batch(x)
            cls_pred, cen_pred, box_pred = model(batch_norm)
            class_targets, centerness_targets, box_targets = generate_targets(
                x.shape, class_labels, box_labels, model.strides
            )
            loss = _compute_loss(
                model.strides, cls_pred, cen_pred, box_pred,
                class_targets, centerness_targets, box_targets
            )
            loss.backward()
            optimizer.step()
            if batch_idx % 20 == 0:
                logger.info(f"Epoch {epoch}, batch {batch_idx}/{len(train_loader)}, loss {loss.item():.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            for i, (x, class_labels, box_labels) in enumerate(val_loader):
                x = x.to(device)
                batch_norm = normalize_batch(x)
                cls_pred, cen_pred, box_pred = model(batch_norm)
                H, W = x.shape[2], x.shape[3]
                detections = detections_from_network_output(
                    H, W, cls_pred, cen_pred, box_pred, model.scales, model.strides
                )
                if i == 0:
                    img_vis = tensor_to_image(x[0])
                    render_detections_to_image(img_vis, detections[0])
                    _render_targets_to_image(img_vis, box_labels[0])
                    # Optionally log to TensorBoard

        ckpt_path = os.path.join(writer.log_dir, f"fcos_epoch{epoch}.pth")
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'class_names': class_names,
        }, ckpt_path)
        logger.info(f"Saved checkpoint {ckpt_path}")
        checkpoint += 1


# Usage:
if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    import pathlib
    train_dir = pathlib.Path(r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\dataset.coco\train")
    val_dir   = pathlib.Path(r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\dataset.coco\valid")
    writer = SummaryWriter("runs/fcos_custom")
    train(train_dir, val_dir, writer)