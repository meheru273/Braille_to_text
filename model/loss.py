import torch 
import torch.nn.functional as F
import math
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np

class SpatialAttentionLoss(nn.Module):
    """
    Spatial Attention Loss that encourages the model to focus on regions with actual objects
    and penalizes attention on background regions.
    """
    def __init__(self, temperature=1.0, weight=0.1):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        
    def forward(self, attention_maps: List[torch.Tensor], 
                class_targets: List[torch.Tensor], 
                box_targets: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            attention_maps: List of attention maps from different FPN levels [B, 1, H, W]
            class_targets: List of class targets for each FPN level [B, H, W]
            box_targets: List of box targets for each FPN level [B, H, W, 4]
        """
        total_loss = 0.0
        valid_levels = 0
        
        for att_map, cls_t, box_t in zip(attention_maps, class_targets, box_targets):
            # Create object mask (where class > 0)
            obj_mask = (cls_t > 0).float()  # [B, H, W]
            
            # Resize attention map if needed
            if att_map.shape[-2:] != obj_mask.shape[-2:]:
                att_map = F.interpolate(att_map, size=obj_mask.shape[-2:], 
                                      mode='bilinear', align_corners=False)
            
            # Squeeze attention map to [B, H, W]
            att_map = att_map.squeeze(1)
            
            # Normalize attention map with temperature
            att_map_norm = torch.sigmoid(att_map / self.temperature)
            
            # Compute spatial attention loss
            # Encourage high attention on object regions
            pos_loss = -torch.log(att_map_norm + 1e-6) * obj_mask
            
            # Penalize high attention on background
            neg_loss = -torch.log(1 - att_map_norm + 1e-6) * (1 - obj_mask)
            
            # Balance positive and negative losses
            num_pos = obj_mask.sum()
            num_neg = (1 - obj_mask).sum()
            
            if num_pos > 0:
                pos_loss = pos_loss.sum() / num_pos
                neg_loss = neg_loss.sum() / num_neg
                level_loss = pos_loss + neg_loss
                total_loss += level_loss
                valid_levels += 1
        
        return (total_loss / max(valid_levels, 1)) * self.weight

class CenterAttentionLoss(nn.Module):
    """
    Center Attention Loss that encourages attention to focus on object centers
    """
    def __init__(self, sigma=2.0, weight=0.05):
        super().__init__()
        self.sigma = sigma
        self.weight = weight
    
    def generate_center_heatmap(self, box_targets: torch.Tensor, class_targets: torch.Tensor, 
                               size: Tuple[int, int], stride: int) -> torch.Tensor:
        """Generate Gaussian heatmap centered at object centers"""
        B, H, W = class_targets.shape
        device = class_targets.device
        # Create coordinate grids
        y_coords = torch.arange(H, device=device).float()
        x_coords = torch.arange(W, device=device).float()
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        heatmap = torch.zeros(B, H, W, device=device)
        
        for b in range(B):
            # Find object locations
            obj_mask = class_targets[b] > 0
            if not obj_mask.any():
                continue
                
            # Get object centers from box targets
            obj_indices = torch.where(obj_mask)
            for i in range(len(obj_indices[0])):
                y, x = obj_indices[0][i], obj_indices[1][i]
                # FIXED: Convert distances to center coordinates
                left_dist = box_targets[b, y, x, 0] * stride
                top_dist = box_targets[b, y, x, 1] * stride
                right_dist = box_targets[b, y, x, 2] * stride
                bottom_dist = box_targets[b, y, x, 3] * stride
                
                # Calculate center in feature map coordinates
                center_x_feat = x
                center_y_feat = y
                
                # Generate Gaussian centered at feature location
                gaussian = torch.exp(-((x_grid - center_x_feat)**2 + (y_grid - center_y_feat)**2) / (2 * self.sigma**2))
                heatmap[b] = torch.maximum(heatmap[b], gaussian)
        
        return heatmap
    
    def forward(self, attention_maps: List[torch.Tensor],
                class_targets: List[torch.Tensor],
                box_targets: List[torch.Tensor],
                strides: List[int]) -> torch.Tensor:
        """Compute center attention loss"""
        total_loss = 0.0
        valid_levels = 0
        
        for level_idx, (att_map, cls_t, box_t) in enumerate(zip(attention_maps, class_targets, box_targets)):
            B, _, H, W = att_map.shape
            stride = strides[level_idx]
            
            # Generate center heatmap using the correct stride
            center_heatmap = self.generate_center_heatmap(box_t, cls_t, (H, W), stride)
            
            # Resize attention map if needed
            if att_map.shape[-2:] != center_heatmap.shape[-2:]:
                att_map = F.interpolate(att_map, size=center_heatmap.shape[-2:], 
                                      mode='bilinear', align_corners=False)
            
            # Normalize attention map
            att_map = torch.sigmoid(att_map.squeeze(1))
            
            # MSE loss between attention and center heatmap
            loss = F.mse_loss(att_map, center_heatmap, reduction='mean')
            
            total_loss += loss
            valid_levels += 1
        
        return (total_loss / max(valid_levels, 1)) * self.weight

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
    classes, boxes, class_targets, box_targets,
    focal_loss_fn, classification_weight=2.0, regression_weight=2.0,
    box_labels_by_batch=None, img_shape=None, strides=None,
):
    """SIMPLIFIED and CORRECTED loss computation"""

    device = classes[0].device
    num_classes = classes[0].shape[-1]
    box_loss_fn = torch.nn.SmoothL1Loss(reduction='none')

    classification_losses = []
    regression_losses = []

    for idx in range(len(classes)):
        cls_p = classes[idx]  # [B, H, W, num_classes]
        box_p = boxes[idx]    # [B, H, W, 4]

        cls_t = class_targets[idx].to(device)  # [B, H_target, W_target]
        box_t = box_targets[idx].to(device)   # [B, H_target, W_target, 4]

        # Ensure matching dimensions
        B, H_pred, W_pred = cls_p.shape[:3]
        
        if len(cls_t.shape) == 3:  # [B, H_target, W_target]
            B_t, H_target, W_target = cls_t.shape
        else:
            print(f"Warning: Unexpected target shape at level {idx}: {cls_t.shape}")
            continue

        # Resize targets to match predictions if needed
        if (H_pred, W_pred) != (H_target, W_target):
            cls_t = F.interpolate(cls_t.float().unsqueeze(1), size=(H_pred, W_pred),
                                  mode='nearest').squeeze(1).long()
            
            if len(box_t.shape) == 4:  # [B, H, W, 4]
                box_t = F.interpolate(box_t.permute(0, 3, 1, 2), size=(H_pred, W_pred),
                                      mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        cls_p_flat = cls_p.view(-1, num_classes).float()  # [B*H*W, num_classes]
        cls_t_flat = cls_t.view(-1).long()                # [B*H*W]
        box_p_flat = box_p.view(-1, 4).float()           # [B*H*W, 4]
        box_t_flat = box_t.view(-1, 4).float()           # [B*H*W, 4]

        if cls_p_flat.shape[0] != cls_t_flat.shape[0]:
            print(f"ERROR: Batch size mismatch at level {idx}")
            continue

        pos_mask = cls_t_flat > 0
        num_positives = pos_mask.sum().item()
        
        # *** FIXED: Use the corrected focal loss with positive normalization ***
        cls_loss = focal_loss_fn(cls_p_flat, cls_t_flat)
        classification_losses.append(cls_loss)

        # Regression loss (only on positive samples)
        if num_positives > 0:
            reg_loss = box_loss_fn(box_p_flat[pos_mask], box_t_flat[pos_mask]).mean()
            regression_losses.append(reg_loss)
        else:
            regression_losses.append(torch.tensor(0.0, device=device, requires_grad=True))

    # Compute final losses
    total_cls_loss = torch.stack(classification_losses).mean() if classification_losses else torch.tensor(0.0, device=device)
    total_reg_loss = torch.stack(regression_losses).mean() if regression_losses else torch.tensor(0.0, device=device)
    
    total_loss = (classification_weight * total_cls_loss +
                  regression_weight * total_reg_loss)

    return total_loss, total_cls_loss, total_reg_loss