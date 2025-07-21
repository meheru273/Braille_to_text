import torch 
import torch.nn.functional as F
import math

def _compute_loss(
    classes, boxes, class_targets, box_targets,
    focal_loss_fn, classification_weight=2.0, regression_weight=2.0,
     box_labels_by_batch=None, img_shape=None, strides=None,
):
    """Fixed loss computation with proper tensor dimension handling"""

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

        # CRITICAL: Ensure matching dimensions
        B, H_pred, W_pred = cls_p.shape[:3]
        
        if len(cls_t.shape) == 3:  # [B, H_target, W_target]
            B_t, H_target, W_target = cls_t.shape
        else:
            print(f"Warning: Unexpected target shape at level {idx}: {cls_t.shape}")
            continue

        # Resize targets to match predictions if needed
        if (H_pred, W_pred) != (H_target, W_target):
            print(f"Resizing targets from ({H_target}, {W_target}) to ({H_pred}, {W_pred})")
            
            # Resize class targets
            cls_t = F.interpolate(cls_t.float().unsqueeze(1), size=(H_pred, W_pred),
                                  mode='nearest').squeeze(1).long()
            
            # Resize box targets
            if len(box_t.shape) == 4:  # [B, H, W, 4]
                box_t = F.interpolate(box_t.permute(0, 3, 1, 2), size=(H_pred, W_pred),
                                      mode='bilinear', align_corners=False).permute(0, 2, 3, 1)


        cls_p_flat = cls_p.view(-1, num_classes).float()  # [B*H*W, num_classes]
        cls_t_flat = cls_t.view(-1).long()                # [B*H*W]
        box_p_flat = box_p.view(-1, 4).float()           # [B*H*W, 4]
        box_t_flat = box_t.view(-1, 4).float()           # [B*H*W, 4]

        # ✅ Debug: Check tensor shapes
        print(f"Level {idx} - cls_p_flat: {cls_p_flat.shape}, cls_t_flat: {cls_t_flat.shape}")
        
        # ✅ Ensure batch sizes match
        if cls_p_flat.shape[0] != cls_t_flat.shape[0]:
            print(f"ERROR: Batch size mismatch at level {idx}")
            print(f"Predictions: {cls_p_flat.shape}, Targets: {cls_t_flat.shape}")
            continue

        pos_mask = cls_t_flat > 0

        # Classification loss
        cls_loss = focal_loss_fn(cls_p_flat, cls_t_flat)
        classification_losses.append(cls_loss)

        # Regression loss (only on positive samples)
        if pos_mask.sum() > 0:
            reg_loss = box_loss_fn(box_p_flat[pos_mask], box_t_flat[pos_mask]).mean()
            regression_losses.append(reg_loss)

    # Compute final losses
    total_cls_loss = torch.stack(classification_losses).mean() if classification_losses else torch.tensor(0.0, device=device)
    total_reg_loss = torch.stack(regression_losses).mean() if regression_losses else torch.tensor(0.0, device=device)
    
    
    total_loss = (classification_weight * total_cls_loss +
                  regression_weight * total_reg_loss )

    return total_loss, total_cls_loss, total_reg_loss


