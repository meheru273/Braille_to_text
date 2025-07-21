import torch 
import torch.nn.functional as F
import math
from Targets import generate_attention_targets

def _compute_loss(
    classes, boxes, class_targets, box_targets,
    focal_loss_fn, classification_weight=2.0, regression_weight=2.0,
    # Optional attention parameters  
    attention_maps=None, box_labels_by_batch=None, img_shape=None, strides=None,
    attention_weight=0.1
):
    """Loss computation optimized for 2-head model with mixed precision"""

    device = classes[0].device
    num_classes = classes[0].shape[-1]

    # ✅ Only need regression loss (no centerness)
    box_loss_fn = torch.nn.SmoothL1Loss(reduction='none')
    
    # ✅ REMOVED: BCE loss (not needed for 2-head model)
    # cen_loss_fn = torch.nn.BCELoss(reduction='none')  # DELETE THIS

    classification_losses = []
    regression_losses = []

    for idx in range(len(classes)):
        cls_p = classes[idx]  # [B, H, W, num_classes]
        box_p = boxes[idx]    # [B, H, W, 4]

        cls_t = class_targets[idx].to(device)
        box_t = box_targets[idx].to(device)

        # Resize targets if needed (your existing logic)
        # ... target resizing code ...

        # ✅ CRITICAL: Cast to FP32 for mixed precision stability
        cls_p_flat = cls_p.view(-1, num_classes).float()  # Force FP32
        cls_t_flat = cls_t.view(-1)
        box_p_flat = box_p.view(-1, 4).float()           # Force FP32  
        box_t_flat = box_t.view(-1, 4).float()           # Force FP32

        pos_mask = cls_t_flat > 0

        # Classification loss using Focal Loss (already optimal for mixed precision)
        cls_loss = focal_loss_fn(cls_p_flat, cls_t_flat)
        classification_losses.append(cls_loss)

        # Regression loss (only on positive samples)
        if pos_mask.sum() > 0:
            reg_loss = box_loss_fn(box_p_flat[pos_mask], box_t_flat[pos_mask]).mean()
            regression_losses.append(reg_loss)

    # Compute final losses
    total_cls_loss = torch.stack(classification_losses).mean() if classification_losses else torch.tensor(0.0, device=device)
    total_reg_loss = torch.stack(regression_losses).mean() if regression_losses else torch.tensor(0.0, device=device)
    
    # Attention loss (optional)
    
    # Total loss
    total_loss = (classification_weight * total_cls_loss +
                  regression_weight * total_reg_loss)

    return total_loss, total_cls_loss, total_reg_loss, 

