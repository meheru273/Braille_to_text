import torch 
import torch.nn.functional as F
import math
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np



class FocalLoss(nn.Module):
    """Focal Loss with positive normalization"""
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=2, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Inputs: [N, num_classes]
        Targets: [N] (class indices)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha)[targets].to(inputs.device)
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss


def _compute_loss(
    classes, centerness, boxes, class_targets, centerness_targets, box_targets,
    focal_loss_fn, classification_weight=1.0, centerness_weight=1.0, regression_weight=1.0
):
    """
    Compute original FCOS losses (classification, centerness, regression)
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
        
        # Centerness loss (only on positive samples)
        if num_positives > 0:
            cent_loss = centerness_loss_fn(cent_p_flat[pos_mask], cent_t_flat[pos_mask])
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


