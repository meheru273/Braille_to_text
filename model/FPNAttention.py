import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
from torchvision.ops import DeformConv2d
from Attention import (CoordinateAttention, CBAM)
from loss import _compute_loss, SpatialAttentionLoss, CenterAttentionLoss, FocalLoss


class DeformableConvBlock(nn.Module):
    """Deformable convolution block with proper offset generation"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        
        # Offset convolution - generates 2 * kernel_size^2 * groups offsets
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size * groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
        # Deformable convolution
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        
        # Initialize offset conv to zero (important for stable training)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
    
    def forward(self, x):
        # Generate offsets
        offset = self.offset_conv(x)
        
        # Apply deformable convolution
        return self.deform_conv(x, offset)


class FPN(nn.Module):
    """
    Enhanced FPN with integrated attention mechanisms for Braille detection
    """
    def __init__(self, num_classes=26, use_coord: bool = False, use_cbam: bool = True, use_deform: bool = False):
        super().__init__()
        
        from BackBone import BackBone
        self.backbone = BackBone()
        
        self.num_classes = num_classes
        self.use_coord = use_coord
        self.use_cbam = use_cbam
        self.use_deform = use_deform
        
        self.strides = [2, 4, 8, 16, 32]
        self.scales = nn.Parameter(torch.tensor([16.,32.,64.,128.,256.]))
        
        # Feature channels from backbone
        backbone_channels = [64, 128, 256, 512]
        fpn_channels = 256
        
        # Lateral connections with optional coordinate attention
        self.lateral_convs = nn.ModuleList()
        self.lateral_attention = nn.ModuleList()
        
        for in_ch in backbone_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, fpn_channels, kernel_size=1, bias=False)
            )
            
            # Add coordinate attention to lateral connections if enabled
            if self.use_coord:
                self.lateral_attention.append(
                    CoordinateAttention(fpn_channels, reduction=16)
                )
            else:
                # Identity module when coordinate attention is disabled
                self.lateral_attention.append(nn.Identity())
        
        # Extra FPN level
        self.extra_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, stride=2, padding=1),
                nn.GroupNorm(32, fpn_channels),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Multi-scale feature fusion module with optional CBAM
        self.feature_fusion = nn.ModuleList()
        for _ in range(len(self.strides)):
            if self.use_cbam:
                # With CBAM attention
                self.feature_fusion.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                        nn.GroupNorm(32, fpn_channels),
                        nn.ReLU(inplace=True),
                        CBAM(fpn_channels, reduction=16)
                    )
                )
            else:
                # Without CBAM attention
                self.feature_fusion.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                        nn.GroupNorm(32, fpn_channels),
                        nn.ReLU(inplace=True)
                    )
                )
        
        # Spatial attention branches for loss computation
        self.spatial_attention = nn.ModuleList()
        for _ in range(len(self.strides)):
            self.spatial_attention.append(
                nn.Sequential(
                    nn.Conv2d(fpn_channels, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 1, 1),
                )
            )
        
        # Enhanced detection heads with optional deformable convolutions
        self.classification_head = self._make_head(fpn_channels, 128, use_deform=self.use_deform)
        self.regression_head = self._make_head(fpn_channels, 128, use_deform=self.use_deform)
        
        # Output layers
        self.classification_to_class = nn.Conv2d(128, self.num_classes, 3, padding=1)
        self.regression_to_bbox = nn.Conv2d(128, 4, 3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
        # Print configuration
        self._print_config()
    
    def _print_config(self):
        """Print the configuration of attention mechanisms"""
        print(f"FPN Configuration:")
        print(f"  - Coordinate Attention: {'Enabled' if self.use_coord else 'Disabled'}")
        print(f"  - CBAM Attention: {'Enabled' if self.use_cbam else 'Disabled'}")
        print(f"  - Deformable Convolution: {'Enabled' if self.use_deform else 'Disabled'}")
        print(f"  - Number of Classes: {self.num_classes}")
    
    def _make_head(self, in_channels: int, out_channels: int, 
                   use_deform: bool = False) -> nn.Module:
        """Create detection head with optional deformable convolution"""
        layers = []
        
        # First layer
        layers.extend([
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        ])
        
        # Second layer (with optional deformable convolution)
        if use_deform:
            layers.append(DeformableConvBlock(in_channels, in_channels, 3, 1, 1))
            layers.extend([
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True)
            ])
        else:
            layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True)
            ])
        
        # Third layer
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(32, out_channels // 4), out_channels),
            nn.ReLU(inplace=True)
        ])
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize classification head with focal loss prior
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.classification_to_class.bias, bias_value)
        
        # Initialize regression head
        nn.init.normal_(self.regression_to_bbox.weight, std=0.01)
        nn.init.constant_(self.regression_to_bbox.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        # Backbone forward
        backbone_features = self.backbone(x)
        c2, c3, c4, c5 = backbone_features
        
        # Build FPN with optional coordinate attention
        laterals = []
        for i, (conv, att, feat) in enumerate(zip(self.lateral_convs, 
                                                  self.lateral_attention,
                                                  [c2, c3, c4, c5])):
            lateral = conv(feat)          
            # Apply coordinate attention 
            lateral = att(lateral)  # This handles both enabled and disabled cases
            laterals.append(lateral)
        
        # Top-down pathway
        p5 = laterals[3]
        p4 = laterals[2] + F.interpolate(p5, size=laterals[2].shape[2:], 
                                        mode='bilinear', align_corners=False)
        p3 = laterals[1] + F.interpolate(p4, size=laterals[1].shape[2:], 
                                        mode='bilinear', align_corners=False)
        p2 = laterals[0] + F.interpolate(p3, size=laterals[0].shape[2:], 
                                        mode='bilinear', align_corners=False)
        
        # Extra level
        p6 = self.extra_convs[0](p5)
        
        # Apply feature fusion with optional CBAM
        fpn_features = [p2, p3, p4, p5, p6]
        fused_features = []
        attention_maps = []
        
        for i, (feat, fusion) in enumerate(zip(fpn_features, self.feature_fusion)):
            # Feature fusion already handles CBAM based on use_cbam flag in __init__
            fused = fusion(feat)
            fused_features.append(fused)
            
            # Generate spatial attention maps
            att_map = self.spatial_attention[i](fused)
            attention_maps.append(att_map)
        
        # Detection heads (already handle deformable convolution based on use_deform flag)
        classes_by_feature = []
        regression_by_feature = []
        
        for scale, feat in zip(self.scales, fused_features):
            # Classification
            cls_feat = self.classification_head(feat)
            classes = self.classification_to_class(cls_feat)
            
            # Regression
            reg_feat = self.regression_head(feat)
            bbox_pred = torch.exp(self.regression_to_bbox(reg_feat)) * scale
            
            # Reshape outputs
            classes = classes.permute(0, 2, 3, 1).contiguous()
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
            
            classes_by_feature.append(classes)
            regression_by_feature.append(bbox_pred)
        
        return classes_by_feature, regression_by_feature, attention_maps


def compute_loss_with_attention(
    classes, boxes, class_targets, box_targets,
    focal_loss_fn, attention_maps=None,
    classification_weight=2.0, regression_weight=2.0,
    spatial_attention_weight=0.1, center_attention_weight=0.05,
    box_labels_by_batch=None, img_shape=None, strides=None
):
    """Enhanced loss computation with attention losses"""
    
    # Get device from input tensors
    device = classes[0].device if len(classes) > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute base losses (classification and regression)
    base_loss, cls_loss, reg_loss = _compute_loss(
        classes, boxes, class_targets, box_targets,
        focal_loss_fn, classification_weight, regression_weight,
        box_labels_by_batch, img_shape, strides
    )
    
    # Compute attention losses if attention maps are provided
    if attention_maps is not None:
        # Ensure attention maps are on the correct device
        attention_maps = [att_map.to(device) for att_map in attention_maps]
        
        # Ensure targets are on the correct device
        if isinstance(class_targets, list):
            class_targets = [target.to(device) if isinstance(target, torch.Tensor) else target for target in class_targets]
        elif isinstance(class_targets, torch.Tensor):
            class_targets = class_targets.to(device)
            
        if isinstance(box_targets, list):
            box_targets = [target.to(device) if isinstance(target, torch.Tensor) else target for target in box_targets]
        elif isinstance(box_targets, torch.Tensor):
            box_targets = box_targets.to(device)
        
        # Spatial attention loss
        spatial_loss_fn = SpatialAttentionLoss(weight=spatial_attention_weight)
        spatial_loss_fn = spatial_loss_fn.to(device)  # Ensure loss function is on correct device
        spatial_loss = spatial_loss_fn(attention_maps, class_targets, box_targets)
        
        # Center attention loss
        center_loss_fn = CenterAttentionLoss(weight=center_attention_weight)
        center_loss_fn = center_loss_fn.to(device)  # Ensure loss function is on correct device
        center_loss = center_loss_fn(attention_maps, class_targets, box_targets)
        
        # Total loss
        total_loss = base_loss + spatial_loss + center_loss
        
        return total_loss, cls_loss, reg_loss, spatial_loss, center_loss
    
    return base_loss, cls_loss, reg_loss, None, None


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """Proper normalization for training and inference consistency"""
    if x.max() > 1.0:
        x = x / 255.0
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1, 3, 1, 1)
    
    x = (x - mean) / std
    return x