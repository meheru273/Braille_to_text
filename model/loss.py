import torch 
import torch.nn.functional as F
import math
from Targets import generate_attention_targets

def _compute_loss(
    classes, centernesses, boxes, class_targets, centerness_targets, box_targets,
    focal_loss_fn, classification_weight=2.0, centerness_weight=1.0, regression_weight=2.0,
    # Optional attention parameters
    attention_maps=None, box_labels_by_batch=None, img_shape=None, strides=None,
    attention_weight=0.1
):
    """Fixed loss computation with optional attention supervision"""

    device = classes[0].device
    num_classes = classes[0].shape[-1]

    box_loss_fn = torch.nn.SmoothL1Loss(reduction='none')
    cen_loss_fn = torch.nn.BCELoss(reduction='none')

    classification_losses = []
    centerness_losses = []
    regression_losses = []

    for idx in range(len(classes)):
        cls_p = classes[idx]  # [B, H, W, num_classes]
        cen_p = centernesses[idx]  # [B, H, W]
        box_p = boxes[idx]    # [B, H, W, 4]

        cls_t = class_targets[idx].to(device)  # [B, H_target, W_target]
        cen_t = centerness_targets[idx].to(device)  # [B, H_target, W_target]
        box_t = box_targets[idx].to(device)  # [B, H_target, W_target, 4]

        # Resize targets to match model output if shapes don't match
        B, H_pred, W_pred = cls_p.shape[:3]
        B_t, H_target, W_target = cls_t.shape

        if (H_pred, W_pred) != (H_target, W_target):
            # Resize class targets using nearest neighbor
            cls_t = F.interpolate(cls_t.float().unsqueeze(1), size=(H_pred, W_pred),
                                  mode='nearest').squeeze(1).long()
            # Resize centerness targets using bilinear
            cen_t = F.interpolate(cen_t.unsqueeze(1), size=(H_pred, W_pred),
                                  mode='bilinear', align_corners=False).squeeze(1)

            # Resize box targets
            box_t = F.interpolate(box_t.permute(0, 3, 1, 2), size=(H_pred, W_pred),
                                  mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        # Flatten for loss computation
        cls_p_flat = cls_p.view(-1, num_classes)
        cls_t_flat = cls_t.view(-1)
        cen_p_flat = cen_p.view(-1)
        cen_t_flat = cen_t.view(-1)
        box_p_flat = box_p.view(-1, 4)
        box_t_flat = box_t.view(-1, 4)

        pos_mask = cls_t_flat > 0

        # Classification loss
        cls_loss = focal_loss_fn(cls_p_flat, cls_t_flat)
        classification_losses.append(cls_loss)

        # Centerness loss (only on positive samples)
        if pos_mask.sum() > 0:
            cen_loss = cen_loss_fn(cen_p_flat[pos_mask], cen_t_flat[pos_mask]).mean()
            centerness_losses.append(cen_loss)

            # Regression loss (only on positive samples)
            reg_loss = box_loss_fn(box_p_flat[pos_mask], box_t_flat[pos_mask]).mean()
            regression_losses.append(reg_loss)

    # Compute main losses
    total_cls_loss = torch.stack(classification_losses).mean() if classification_losses else torch.tensor(0.0, device=device)
    total_cen_loss = torch.stack(centerness_losses).mean() if centerness_losses else torch.tensor(0.0, device=device)
    total_reg_loss = torch.stack(regression_losses).mean() if regression_losses else torch.tensor(0.0, device=device)
    
    # Compute attention loss (optional)
    total_att_loss = compute_attention_loss(
        attention_maps, box_labels_by_batch, img_shape, strides, device
    )
    
    # Total loss
    total_loss = (classification_weight * total_cls_loss +
                  centerness_weight * total_cen_loss +
                  regression_weight * total_reg_loss +
                  attention_weight * total_att_loss)

    return total_loss, total_cls_loss, total_cen_loss, total_reg_loss, total_att_loss


def compute_attention_loss(attention_maps, box_labels_by_batch, img_shape, strides, device):
    """
    Compute attention supervision loss with proper tensor formatting
    """
    if attention_maps is None or box_labels_by_batch is None:
        return torch.tensor(0.0, device=device)
        
    # Generate attention targets
    attention_targets = generate_attention_targets(
        img_shape, box_labels_by_batch, strides
    )
        
    total_loss = 0.0
    valid_levels = 0
    
    for attention_map, attention_target in zip(attention_maps, attention_targets):
        if attention_map is not None and attention_target is not None:
            # Ensure same device
            attention_target = attention_target.to(attention_map.device)
            
            # ✅ FIX: Handle tensor dimensions properly
            if attention_map.shape != attention_target.shape:
                # Get target spatial dimensions from attention_map
                target_spatial_size = attention_map.shape[-2:]  # (H, W)
                
                # ✅ FIX: Ensure attention_target is 4D for interpolation
                if attention_target.dim() == 2:
                    # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
                    attention_target = attention_target.unsqueeze(0).unsqueeze(0)
                elif attention_target.dim() == 3:
                    # Add channel dimension: (B, H, W) -> (B, 1, H, W)
                    attention_target = attention_target.unsqueeze(1)
                
                # ✅ FIX: Use only spatial dimensions for output size
                attention_target = F.interpolate(
                    attention_target, 
                    size=target_spatial_size,  # Only (H, W)
                    mode='bilinear', 
                    align_corners=False
                )
                
                # ✅ FIX: Match attention_map dimensions
                if attention_map.dim() == 3:  # (B, H, W)
                    attention_target = attention_target.squeeze(1)  # Remove channel dim
                elif attention_map.dim() == 2:  # (H, W)
                    attention_target = attention_target.squeeze(0).squeeze(0)  # Remove batch & channel
            
            # MSE loss
            loss = F.mse_loss(attention_map, attention_target)
            total_loss += loss
            valid_levels += 1
    
    return total_loss / valid_levels if valid_levels > 0 else torch.tensor(0.0, device=device)
