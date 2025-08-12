import torch 
import torch.nn.functional as F
import math
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np


class FocalLoss(nn.Module):
    """
    PROPERLY FIXED Focal Loss - normalizes by positive samples to prevent background dominance
    """
    def __init__(self, alpha=None, gamma=2.0, num_classes=26, reduction='positive_normalized'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        
        # Handle alpha initialization
        if isinstance(alpha, (list, np.ndarray, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif alpha == "auto":
            if num_classes is None:
                raise ValueError("num_classes must be provided for auto alpha calculation")
            self.alpha = "auto"
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        # Flatten inputs and targets for cross_entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of true class
        
        # Focal loss components
        focal_weight = (1.0 - pt) ** self.gamma
        
        # Apply alpha weighting
        if hasattr(self, 'alpha') and self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_tensor = self.alpha.to(targets.device)
                alpha_t = alpha_tensor[targets]
            else:
                alpha_t = self.alpha
            focal_weight *= alpha_t
        
        focal_loss = focal_weight * ce_loss
        
        # *** CRITICAL FIX: Normalize by positive samples ***
        if self.reduction == 'positive_normalized':
            pos_mask = targets > 0
            neg_mask = targets == 0
            
            num_pos = pos_mask.sum().item()
            num_neg = neg_mask.sum().item()
            
            if num_pos > 0:
                # Positive loss (full weight)
                pos_loss = focal_loss[pos_mask].sum()
                
                # Negative loss (controlled weight to prevent dominance)
                if num_neg > 0:
                    neg_loss = focal_loss[neg_mask].sum()
                    # Limit negative contribution - key insight!
                    neg_weight = min(1.0, num_pos / num_neg * 3.0)
                    neg_loss = neg_loss * neg_weight
                else:
                    neg_loss = 0.0
                
                # Normalize by positive samples
                return (pos_loss + neg_loss) / num_pos
            else:
                # Fallback if no positive samples
                return focal_loss.mean()
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def calculate_auto_alpha(self, dataset):
        num_classes = self.num_classes
        class_counts = torch.zeros(num_classes)
        total_samples = 0
    
        for *_, labels in dataset:
            labels = labels.view(-1).long()
            valid_labels = labels[(labels >= 0) & (labels < num_classes)]
            class_counts += torch.bincount(valid_labels, minlength=num_classes)
            total_samples += len(valid_labels)

        class_weights = total_samples / (num_classes * (class_counts + 1e-6))
        class_weights = torch.log(class_weights + 1.0)
        class_weights = torch.clamp(class_weights, min=0.1, max=3.0)
        self.alpha = class_weights
        print(f"Auto-calculated alpha weights: {self.alpha}")



def _compute_loss(
    classes, centerness, boxes, class_targets, centerness_targets, box_targets,
    focal_loss_fn, classification_weight=1.0, centerness_weight=1.0, regression_weight=1.0
):
    """
    Compute original FCOS losses (classification, centerness, regression)
    *** FIXED: Apply sigmoid to centerness predictions before BCE loss ***
    """
    device = classes[0].device
    num_classes = classes[0].shape[-1]
    
    classification_losses = []
    centerness_losses = []
    regression_losses = []
    
    box_loss_fn = nn.SmoothL1Loss(reduction='none')
    centerness_loss_fn = nn.BCELoss(reduction='none')
    
    for idx in range(len(classes)):
        cls_p = classes[idx]      # [B, H, W, num_classes]
        cent_p = centerness[idx]  # [B, H, W]
        box_p = boxes[idx]        # [B, H, W, 4]
        
        cls_t = class_targets[idx].to(device)      # [B, H, W]
        cent_t = centerness_targets[idx].to(device) # [B, H, W]
        box_t = box_targets[idx].to(device)        # [B, H, W, 4]
        
        # Flatten for loss computation
        cls_p_flat = cls_p.view(-1, num_classes)
        cls_t_flat = cls_t.view(-1).long()
        cent_p_flat = cent_p.view(-1)
        cent_t_flat = cent_t.view(-1)
        box_p_flat = box_p.view(-1, 4)
        box_t_flat = box_t.view(-1, 4)
        
        # Positive samples mask
        pos_mask = cls_t_flat > 0
        num_positives = pos_mask.sum().item()
        
        # Classification loss (Focal Loss)
        cls_loss = focal_loss_fn(cls_p_flat, cls_t_flat)
        if num_positives > 0:
            cls_loss = cls_loss / num_positives
        classification_losses.append(cls_loss)
        
        # *** CRITICAL FIX: Apply sigmoid to centerness predictions ***
        if num_positives > 0:
            # Apply sigmoid to convert logits to probabilities [0,1]
            cent_p_sigmoid = torch.sigmoid(cent_p_flat[pos_mask])
            cent_loss = centerness_loss_fn(cent_p_sigmoid, cent_t_flat[pos_mask])
            cent_loss = cent_loss.mean()
            centerness_losses.append(cent_loss)
            
            # Regression loss (only on positive samples, weighted by centerness)
            reg_loss = box_loss_fn(box_p_flat[pos_mask], box_t_flat[pos_mask])
            reg_loss = reg_loss.sum(dim=1)  # Sum over 4 coordinates
            reg_loss = (reg_loss * cent_t_flat[pos_mask]).mean()  # Weight by centerness targets
            regression_losses.append(reg_loss)
        else:
            centerness_losses.append(torch.tensor(0.0, device=device, requires_grad=True))
            regression_losses.append(torch.tensor(0.0, device=device, requires_grad=True))
    
    # Compute final losses
    total_cls_loss = torch.stack(classification_losses).mean() * classification_weight
    total_cent_loss = torch.stack(centerness_losses).mean() * centerness_weight
    total_reg_loss = torch.stack(regression_losses).mean() * regression_weight
    
    return total_cls_loss, total_cent_loss, total_reg_loss