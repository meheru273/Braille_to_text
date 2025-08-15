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
    cls_pred_list, centerness_pred_list, box_pred_list,
    class_targets, centerness_targets, box_targets,
    focal_loss,
    classification_weight=1.0,
    centerness_weight=1.0,
    regression_weight=1.0
):
    """
    Updated loss computation for single detection layer
    
    Args:
        cls_pred_list: List with single element [B, H, W, num_classes]
        centerness_pred_list: List with single element [B, H, W] 
        box_pred_list: List with single element [B, H, W, 4]
        class_targets: List with single element [B, H, W]
        centerness_targets: List with single element [B, H, W]
        box_targets: List with single element [B, H, W, 4]
    """
    device = cls_pred_list[0].device
    
    total_cls_loss = torch.tensor(0.0, device=device)
    total_cent_loss = torch.tensor(0.0, device=device)
    total_reg_loss = torch.tensor(0.0, device=device)
    
    # Only one detection level now
    cls_pred = cls_pred_list[0]
    cent_pred = centerness_pred_list[0]
    reg_pred = box_pred_list[0]
    
    cls_target = class_targets[0].to(device).long()
    cent_target = centerness_targets[0].to(device).float()
    reg_target = box_targets[0].to(device).float()
    
    # Ensure shapes match
    if cls_pred.shape[:3] != cls_target.shape:
        raise ValueError(f"Classification shape mismatch: pred {cls_pred.shape[:3]} vs target {cls_target.shape}")
    
    if cent_pred.shape != cent_target.shape:
        raise ValueError(f"Centerness shape mismatch: pred {cent_pred.shape} vs target {cent_target.shape}")
    
    # Classification loss
    cls_pred_flat = cls_pred.view(-1, cls_pred.shape[-1])  # [B*H*W, num_classes]
    cls_target_flat = cls_target.view(-1)                  # [B*H*W]
    
    cls_loss = focal_loss(cls_pred_flat, cls_target_flat)
    total_cls_loss = cls_loss * classification_weight
    
    # Centerness and regression losses (only on positive samples)
    positive_mask = (cls_target > 0).float()
    num_positives = positive_mask.sum().item()
    
    if num_positives > 0:
        # Centerness loss
        cent_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            cent_pred[positive_mask > 0],
            cent_target[positive_mask > 0],
            reduction='mean'
        )
        total_cent_loss = cent_loss * centerness_weight
        
        # Regression loss  
        reg_pred_pos = reg_pred.view(-1, 4)[positive_mask.view(-1) > 0]
        reg_target_pos = reg_target.view(-1, 4)[positive_mask.view(-1) > 0]
        
        reg_loss = torch.nn.functional.smooth_l1_loss(
            reg_pred_pos, reg_target_pos, reduction='mean'
        )
        total_reg_loss = reg_loss * regression_weight
    else:
        # No positive samples - only classification loss
        total_cent_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
    
    return total_cls_loss, total_cent_loss, total_reg_loss