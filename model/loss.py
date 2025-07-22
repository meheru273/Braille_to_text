import torch 
import torch.nn.functional as F
import math

import torch 
import torch.nn.functional as F
import math


def _compute_loss(
    classes, boxes, class_targets, box_targets,
    focal_loss_fn, classification_weight=2.0, regression_weight=2.0,
    box_labels_by_batch=None, img_shape=None, strides=None,
):


    # === INITIALIZATION CHECKS ===
    device = classes[0].device
    num_classes = classes[0].shape[-1]
    box_loss_fn = torch.nn.SmoothL1Loss(reduction='none')
    
    print(f"\n=== LOSS COMPUTATION DEBUG ===")
    print(f"Device: {device}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of pyramid levels: {len(classes)}")
    
    # Debug: Check if we have any predictions at all
    total_predictions = sum(cls_p.numel() for cls_p in classes)
    print(f"Total prediction elements across all levels: {total_predictions}")

    classification_losses = []
    regression_losses = []

    for idx in range(len(classes)):
        print(f"\n--- PROCESSING PYRAMID LEVEL {idx} ---")
        
        cls_p = classes[idx]  # [B, H, W, num_classes] - Classification predictions
        box_p = boxes[idx]    # [B, H, W, 4] - Bounding box predictions

        cls_t = class_targets[idx].to(device)  # [B, H_target, W_target] - Classification targets
        box_t = box_targets[idx].to(device)   # [B, H_target, W_target, 4] - Box targets

        # === SHAPE VALIDATION ===
        print(f"Prediction shapes: cls_p={cls_p.shape}, box_p={box_p.shape}")
        print(f"Target shapes: cls_t={cls_t.shape}, box_t={box_t.shape}")
        
        # Check for NaN or infinite values in predictions
        if torch.isnan(cls_p).any() or torch.isinf(cls_p).any():
            print(f"⚠️  WARNING: NaN/Inf detected in classification predictions at level {idx}")
        if torch.isnan(box_p).any() or torch.isinf(box_p).any():
            print(f"⚠️  WARNING: NaN/Inf detected in box predictions at level {idx}")

        # ✅ CRITICAL: Ensure matching dimensions
        B, H_pred, W_pred = cls_p.shape[:3]
        
        if len(cls_t.shape) == 3:  # [B, H_target, W_target]
            B_t, H_target, W_target = cls_t.shape
        else:
            print(f"❌ ERROR: Unexpected target shape at level {idx}: {cls_t.shape}")
            continue

        # === TARGET RESIZING (IF NEEDED) ===
        if (H_pred, W_pred) != (H_target, W_target):
            print(f"🔄 Resizing targets from ({H_target}, {W_target}) to ({H_pred}, {W_pred})")
            
            # Resize class targets
            cls_t = F.interpolate(cls_t.float().unsqueeze(1), size=(H_pred, W_pred),
                                  mode='nearest').squeeze(1).long()
            
            # Resize box targets
            if len(box_t.shape) == 4:  # [B, H, W, 4]
                box_t = F.interpolate(box_t.permute(0, 3, 1, 2), size=(H_pred, W_pred),
                                      mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        # === FLATTEN TENSORS FOR LOSS COMPUTATION ===
        cls_p_flat = cls_p.view(-1, num_classes).float()  # [B*H*W, num_classes]
        cls_t_flat = cls_t.view(-1).long()                # [B*H*W]
        box_p_flat = box_p.view(-1, 4).float()           # [B*H*W, 4]
        box_t_flat = box_t.view(-1, 4).float()           # [B*H*W, 4]

        print(f"Flattened shapes: cls_p_flat={cls_p_flat.shape}, cls_t_flat={cls_t_flat.shape}")
        
        # ✅ Ensure batch sizes match
        if cls_p_flat.shape[0] != cls_t_flat.shape[0]:
            print(f"❌ ERROR: Batch size mismatch at level {idx}")
            print(f"Predictions: {cls_p_flat.shape}, Targets: {cls_t_flat.shape}")
            continue

        # === POSITIVE SAMPLE ANALYSIS ===
        pos_mask = cls_t_flat > 0  # Foreground pixels (class > 0)
        neg_mask = cls_t_flat == 0  # Background pixels (class = 0)
        
        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()
        total_pixels = cls_t_flat.numel()
        
        print(f"Sample distribution:")
        print(f"  Positive samples: {num_pos} ({num_pos/total_pixels*100:.2f}%)")
        print(f"  Negative samples: {num_neg} ({num_neg/total_pixels*100:.2f}%)")
        print(f"  Total pixels: {total_pixels}")
        
        # Check target class distribution
        unique_classes = torch.unique(cls_t_flat)
        print(f"  Target classes present: {unique_classes.tolist()}")
        
        if num_pos == 0:
            print(f"⚠️  WARNING: No positive samples at level {idx} - model won't learn objects!")

        # === CLASSIFICATION LOSS COMPUTATION ===
        print(f"Computing classification loss using focal loss...")
        
        # Check prediction score ranges before loss
        cls_probs = torch.softmax(cls_p_flat, dim=1)
        max_pred_prob = cls_probs.max().item()
        min_pred_prob = cls_probs.min().item()
        print(f"  Prediction probability range: {min_pred_prob:.4f} - {max_pred_prob:.4f}")
        
        cls_loss = focal_loss_fn(cls_p_flat, cls_t_flat)
        
        # Verify classification loss is reasonable
        if torch.isnan(cls_loss) or torch.isinf(cls_loss):
            print(f"❌ ERROR: Invalid classification loss at level {idx}: {cls_loss}")
            continue
        else:
            cls_loss_value = cls_loss.item() if cls_loss.dim() == 0 else cls_loss.mean().item()
            print(f"✅ Classification loss at level {idx}: {cls_loss_value:.6f}")
            
        classification_losses.append(cls_loss)

        # === REGRESSION LOSS COMPUTATION ===
        if num_pos > 0:
            print(f"Computing regression loss on {num_pos} positive samples...")
            
            # Extract positive predictions and targets
            pos_box_pred = box_p_flat[pos_mask]  # [num_pos, 4]
            pos_box_target = box_t_flat[pos_mask]  # [num_pos, 4]
            
            print(f"  Positive box predictions shape: {pos_box_pred.shape}")
            print(f"  Box prediction range: {pos_box_pred.min().item():.4f} - {pos_box_pred.max().item():.4f}")
            print(f"  Box target range: {pos_box_target.min().item():.4f} - {pos_box_target.max().item():.4f}")
            
            # Check for invalid box values
            if torch.isnan(pos_box_pred).any() or torch.isinf(pos_box_pred).any():
                print(f"⚠️  WARNING: NaN/Inf in positive box predictions")
            if torch.isnan(pos_box_target).any() or torch.isinf(pos_box_target).any():
                print(f"⚠️  WARNING: NaN/Inf in positive box targets")
            
            reg_loss = box_loss_fn(pos_box_pred, pos_box_target).mean()
            
            if torch.isnan(reg_loss) or torch.isinf(reg_loss):
                print(f"❌ ERROR: Invalid regression loss at level {idx}: {reg_loss}")
            else:
                print(f"✅ Regression loss at level {idx}: {reg_loss.item():.6f}")
                regression_losses.append(reg_loss)
        else:
            print(f"⚠️  No positive samples - skipping regression loss at level {idx}")

    # === FINAL LOSS AGGREGATION ===
    print(f"\n--- FINAL LOSS COMPUTATION ---")
    print(f"Classification losses from {len(classification_losses)} levels")
    print(f"Regression losses from {len(regression_losses)} levels")
    
    # Check if we have any losses to compute
    if not classification_losses:
        print(f"❌ CRITICAL: No classification losses computed!")
        total_cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        total_cls_loss = torch.stack(classification_losses).mean()
        print(f"✅ Mean classification loss: {total_cls_loss.item():.6f}")
    
    if not regression_losses:
        print(f"⚠️  WARNING: No regression losses computed - model won't learn localization!")
        total_reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        total_reg_loss = torch.stack(regression_losses).mean()
        print(f"✅ Mean regression loss: {total_reg_loss.item():.6f}")
    
    # Compute weighted total loss
    total_loss = (classification_weight * total_cls_loss + regression_weight * total_reg_loss)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Classification loss (weight={classification_weight}): {total_cls_loss.item():.6f}")
    print(f"Regression loss (weight={regression_weight}): {total_reg_loss.item():.6f}")
    print(f"Total weighted loss: {total_loss.item():.6f}")
    
    # === GRADIENT FLOW CHECK ===
    if total_loss.requires_grad:
        print(f"✅ Loss requires gradients - backprop should work")
    else:
        print(f"❌ CRITICAL: Loss doesn't require gradients - no learning will occur!")
    
    # Check if loss is reasonable for learning
    if total_loss.item() == 0.0:
        print(f"❌ CRITICAL: Zero total loss - model won't learn anything!")
    elif total_loss.item() > 100.0:
        print(f"⚠️  WARNING: Very high loss ({total_loss.item():.2f}) - check for gradient explosion")
    elif total_loss.item() < 0.001:
        print(f"⚠️  WARNING: Very low loss ({total_loss.item():.6f}) - model might have converged or data issues")
    else:
        print(f"✅ Loss value appears reasonable for training")

    return total_loss, total_cls_loss, total_reg_loss
