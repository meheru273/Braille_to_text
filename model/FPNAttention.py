import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
from Attention import (CoordinateAttention, CBAM, PositionAwareAttention)


class FPN(nn.Module):
    """
    Modified FPN with centerness branch for FCOS compatibility
    """
    def __init__(self, num_classes=2, use_coord: bool = False, use_cbam: bool = True, use_pos: bool = True):
        super().__init__()
        
        from BackBone import BackBone
        self.backbone = BackBone()
        
        self.num_classes = num_classes  # Changed to 2 to match FCOS (BACKGROUND, CAR)
        self.use_coord = use_coord
        self.use_cbam = use_cbam
        self.use_pos = use_pos
        
        # FCOS-compatible strides and scales
        self.strides = [8, 16, 32, 64, 128]  # Original FCOS strides
        self.scales = nn.Parameter(torch.tensor([8., 16., 32., 64., 128.]))
        
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
            
            if self.use_coord:
                self.lateral_attention.append(
                    CoordinateAttention(fpn_channels, reduction=16)
                )
            else:
                self.lateral_attention.append(nn.Identity())
        
        # Extra FPN levels (P6, P7)
        self.extra_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, stride=2, padding=1),
                nn.GroupNorm(32, fpn_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, stride=2, padding=1), 
                nn.GroupNorm(32, fpn_channels),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Multi-scale feature fusion with optional CBAM
        self.feature_fusion = nn.ModuleList()
        for _ in range(len(self.strides)):
            if self.use_cbam:
                self.feature_fusion.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                        nn.GroupNorm(32, fpn_channels),
                        nn.ReLU(inplace=True),
                        CBAM(fpn_channels, reduction=16)
                    )
                )
            else:
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
        
        # FCOS-style detection heads
        self.classification_head = nn.Sequential(
            self._convgn(fpn_channels, fpn_channels),
            nn.ReLU(inplace=True),
            self._convgn(fpn_channels, fpn_channels),
            nn.ReLU(inplace=True),
            self._convgn(fpn_channels, fpn_channels),
            nn.ReLU(inplace=True),
            self._convgn(fpn_channels, fpn_channels),
            nn.ReLU(inplace=True),
        )
        
        self.regression_head = nn.Sequential(
            self._convgn(fpn_channels, fpn_channels),
            nn.ReLU(inplace=True),
            self._convgn(fpn_channels, fpn_channels),
            nn.ReLU(inplace=True),
            self._convgn(fpn_channels, fpn_channels),
            nn.ReLU(inplace=True),
            self._convgn(fpn_channels, fpn_channels),
            nn.ReLU(inplace=True),
        )
        
        # Output layers - FCOS style
        self.classification_to_class = nn.Sequential(self._convgn(fpn_channels, self.num_classes))
        self.classification_to_centerness = nn.Sequential(self._convgn(fpn_channels, 1))
        self.regression_to_bbox = nn.Sequential(self._convgn(fpn_channels, 4))
        
        # Initialize weights FCOS style
        self._initialize_weights_fcos_style()
        
        # Print configuration
        self._print_config()
    
    def _convgn(self, in_channels: int, out_channels: int, kernel_size=3, padding=1, stride=1) -> nn.Module:
        """FCOS-style conv + group norm"""
        return nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
        )
    
    def _print_config(self):
        """Print the configuration"""
        print(f"FPN Configuration (FCOS Compatible):")
        print(f"  - Coordinate Attention: {'Enabled' if self.use_coord else 'Disabled'}")
        print(f"  - CBAM Attention: {'Enabled' if self.use_cbam else 'Disabled'}")
        print(f"  - Position-Aware Attention: {'Enabled' if self.use_pos else 'Disabled'}")
        print(f"  - Number of Classes: {self.num_classes}")
        print(f"  - Strides: {self.strides}")
        print(f"  - Scales: {self.scales.data.tolist()}")
    
    def _initialize_weights_fcos_style(self):
        """Initialize weights like original FCOS"""
        for modules in [
            self.regression_head,
            self.classification_to_centerness,
            self.classification_to_class,
            self.classification_head,
            self.lateral_convs,
            self.extra_convs,
        ]:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    torch.nn.init.normal_(module.weight, std=0.01)
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.GroupNorm):
                    torch.nn.init.constant_(module.weight, 1)
                    torch.nn.init.constant_(module.bias, 0)
        
        # Special initialization for classification head (focal loss prior)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for module in self.classification_to_class.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.constant_(module.bias, bias_value)
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass returning FCOS-compatible format:
        Returns: (classes_by_feature, centerness_by_feature, reg_by_feature, attention_maps)
        """
        # Backbone forward
        backbone_features = self.backbone(x)
        c2, c3, c4, c5 = backbone_features
        
        # Build FPN with optional coordinate attention
        laterals = []
        for i, (conv, att, feat) in enumerate(zip(self.lateral_convs, 
                                                  self.lateral_attention,
                                                  [c2, c3, c4, c5])):
            lateral = conv(feat)
            lateral = att(lateral)  # Apply attention (or identity)
            laterals.append(lateral)
        
        # Top-down pathway
        p5 = laterals[3]
        p4 = laterals[2] + F.interpolate(p5, size=laterals[2].shape[2:], 
                                        mode='bilinear', align_corners=True)
        p3 = laterals[1] + F.interpolate(p4, size=laterals[1].shape[2:], 
                                        mode='bilinear', align_corners=True)
        p2 = laterals[0] + F.interpolate(p3, size=laterals[0].shape[2:], 
                                        mode='bilinear', align_corners=True)
        
        # Extra levels
        p6 = self.extra_convs[0](p5)
        p7 = self.extra_convs[1](p6)
        
        # Skip P2, use P3-P7 like original FCOS
        fpn_features = [p3, p4, p5, p6, p7]
        
        # Apply feature fusion with optional CBAM
        fused_features = []
        attention_maps = []
        
        for i, (feat, fusion) in enumerate(zip(fpn_features, self.feature_fusion)):
            fused = fusion(feat)
            fused_features.append(fused)
            
            # Generate spatial attention maps
            att_map = self.spatial_attention[i](fused)
            attention_maps.append(att_map)
        
        # FCOS-style detection heads
        classes_by_feature = []
        centerness_by_feature = []
        reg_by_feature = []
        
        for scale, feat in zip(self.scales, fused_features):
            # Classification branch
            classification = self.classification_head(feat)
            classes = self.classification_to_class(classification).sigmoid()
            centerness = self.classification_to_centerness(classification).sigmoid()
            
            # Regression branch
            reg = torch.exp(self.regression_head(feat)) * scale
            
            # Reshape to FCOS format: B[C]HW -> BHW[C]
            classes = classes.permute(0, 2, 3, 1).contiguous()
            centerness = centerness.permute(0, 2, 3, 1).contiguous().squeeze(3)
            reg = reg.permute(0, 2, 3, 1).contiguous()
            
            classes_by_feature.append(classes)
            centerness_by_feature.append(centerness)
            reg_by_feature.append(reg)
        
        return classes_by_feature, centerness_by_feature, reg_by_feature, attention_maps


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """
    FCOS-compatible normalization
    """
    # Handle both normalized and unnormalized inputs
    if x.max() > 1.0:
        x = x / 255.0
    
    # FCOS normalization (matching original)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    
    x = (x - mean) / std
    return x