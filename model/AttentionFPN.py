import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
from Attention import TransformerSelfAttention2D, TransformerCrossAttention2D


class FPNTransformerFusion(nn.Module):
    """
    Cross-attention fusion for FPN levels
    Each level attends to all other levels
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Single cross-attention module for all level interactions
        self.cross_attn = TransformerCrossAttention2D(d_model, num_heads, dropout)
        
        # Self-attention for each level
        self.self_attn = TransformerSelfAttention2D(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, 1),
            nn.GroupNorm(32, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model * 2, d_model, 1),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.GroupNorm(32, d_model)
        
    def forward(self, features):
        """
        features: List of [B, C, H_i, W_i] feature maps
        Returns: List of enhanced feature maps
        """
        if not isinstance(features, list) or len(features) == 1:
            return features
        
        enhanced_features = []
        
        for i, query_feat in enumerate(features):
            # Start with self-attention
            enhanced_feat = self.self_attn(query_feat)
            
            # Cross-attention with other levels
            for j, key_feat in enumerate(features):
                if i != j:
                    # Resize key_feat to match query_feat spatial dimensions
                    if key_feat.shape[2:] != query_feat.shape[2:]:
                        key_feat_resized = F.interpolate(
                            key_feat, size=query_feat.shape[2:], 
                            mode='bilinear', align_corners=False
                        )
                    else:
                        key_feat_resized = key_feat
                    
                    # Apply cross-attention with reduced weight
                    cross_attention = self.cross_attn(enhanced_feat, key_feat_resized)
                    enhanced_feat = enhanced_feat + 0.1 * cross_attention
            
            # Apply FFN
            residual = enhanced_feat
            enhanced_feat = self.layer_norm(enhanced_feat)
            enhanced_feat = self.ffn(enhanced_feat) + residual
            
            enhanced_features.append(enhanced_feat)
        
        return enhanced_features


class FPN(nn.Module):
    """
    Enhanced FPN with integrated attention mechanisms for Braille detection
    Compatible with original FCOS inference and target generation
    """
    def __init__(self, num_classes=26, use_fpn_att: bool = True):
        super().__init__()
        
        from BackBone import BackBone
        self.backbone = BackBone()
        
        self.num_classes = num_classes
        self.use_fpn_att = use_fpn_att

        # FCOS strides for compatibility
        self.strides = [8, 16, 32, 64, 128]
        self.scales = nn.Parameter(torch.tensor([8.,16.,32.,64.,128.]))
        
        # Feature channels from backbone
        backbone_channels = [64, 128, 256, 512]
        fpn_channels = 256
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList()
        for in_ch in backbone_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, fpn_channels, kernel_size=1, bias=False)
            )
        
        # Extra FPN levels
        self.extra_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, stride=2, padding=1),
                nn.GroupNorm(32, fpn_channels),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Feature fusion layers
        if self.use_fpn_att:
            # Use cross-attention fusion between FPN levels
            self.fpn_cross_attention_fusion = FPNTransformerFusion(fpn_channels, num_heads=8)
        else:
            # Use regular CNN fusion
            self.feature_fusion = nn.ModuleList()
            for _ in range(len(self.strides)):
                self.feature_fusion.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                        nn.GroupNorm(32, fpn_channels),
                        nn.ReLU(inplace=True)
                    )
                )  
           
        # Detection heads
        self.classification_head = self._make_head(fpn_channels, 128)
        self.regression_head = self._make_head(fpn_channels, 128)
        self.centerness_head = self._make_head(fpn_channels, 128)
        
        # Output layers
        self.classification_to_class = nn.Conv2d(128, self.num_classes, 3, padding=1)
        self.regression_to_bbox = nn.Conv2d(128, 4, 3, padding=1)
        self.centerness_to_centerness = nn.Conv2d(128, 1, 3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
        # Print configuration
        self._print_config()
    
    def _print_config(self):
        """Print the configuration of attention mechanisms"""
        print(f"FPN Configuration:")
        print(f"  - FPN Cross-Attention: {self.use_fpn_att}")
        print(f"  - Centerness Branch: Enabled (FCOS Compatible)")
        print(f"  - Number of Classes: {self.num_classes}")
        print(f"  - Strides: {self.strides}")
    
    def _make_head(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create detection head layers"""
        layers = []
        
        # First layer
        layers.extend([
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        ])
        
        # Second layer
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
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize output layers
        # Initialize classification head with focal loss prior
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.classification_to_class.bias, bias_value)
        nn.init.normal_(self.classification_to_class.weight, std=0.01)
            
        # Initialize regression head
        nn.init.normal_(self.regression_to_bbox.weight, std=0.01)
        nn.init.constant_(self.regression_to_bbox.bias, 0)
        
        # Initialize centerness head
        nn.init.normal_(self.centerness_to_centerness.weight, std=0.01)
        nn.init.constant_(self.centerness_to_centerness.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns (classes, centerness, regression, attention_maps) for FCOS compatibility
        """
        # Backbone forward
        backbone_features = self.backbone(x)
        c2, c3, c4, c5 = backbone_features
        
        # Lateral connections
        laterals = []
        for i, (conv, feat) in enumerate(zip(self.lateral_convs, [c2, c3, c4, c5])):
            lateral = conv(feat)
            laterals.append(lateral)
        
        # Top-down pathway
        p5 = laterals[3]
        p4 = laterals[2] + F.interpolate(p5, size=laterals[2].shape[2:], 
                                        mode='bilinear', align_corners=False)
        p3 = laterals[1] + F.interpolate(p4, size=laterals[1].shape[2:], 
                                        mode='bilinear', align_corners=False)
        p2 = laterals[0] + F.interpolate(p3, size=laterals[0].shape[2:], 
                                        mode='bilinear', align_corners=False)
        
        # Extra levels
        p6 = self.extra_convs[0](p5)
        p7 = F.max_pool2d(p6, kernel_size=3, stride=2, padding=1)
        
        # FPN features - note we use P3-P7 to match FCOS levels
        fpn_features = [p3, p4, p5, p6, p7]  # Match original FCOS levels
        
        # Feature fusion with cross-attention or regular CNN
        if self.use_fpn_att:
            # Use cross-attention fusion between FPN levels
            fused_features = self.fpn_cross_attention_fusion(fpn_features)
        else:
            # Use regular CNN fusion
            fused_features = []
            for feat, fusion in zip(fpn_features, self.feature_fusion):
                fused = fusion(feat)
                fused_features.append(fused)
        
        # For attention maps (can be empty list if not using attention)
        attention_maps = []
        
        # Detection predictions
        classes_by_feature = []
        centerness_by_feature = []
        regression_by_feature = []
        
        for scale, feat in zip(self.scales, fused_features):
            # Use CNN-based heads
            cls_feat = self.classification_head(feat)
            cls_out = torch.sigmoid(self.classification_to_class(cls_feat))
            
            cent_feat = self.centerness_head(feat)
            # CHANGE: Output logits instead of sigmoid for BCEWithLogitsLoss compatibility
            cent_out = self.centerness_to_centerness(cent_feat)  # Remove torch.sigmoid()
            
            reg_feat = self.regression_head(feat)
            reg_out = torch.exp(self.regression_to_bbox(reg_feat)) * scale
            
            # Reshape to match FCOS format: B[C]HW -> BHW[C]
            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
            cent_out = cent_out.permute(0, 2, 3, 1).contiguous().squeeze(3)
            reg_out = reg_out.permute(0, 2, 3, 1).contiguous()
            
            classes_by_feature.append(cls_out)
            centerness_by_feature.append(cent_out)
            regression_by_feature.append(reg_out)
        
        return classes_by_feature, centerness_by_feature, regression_by_feature, attention_maps


# Helper function to normalize batch
def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """Proper normalization for training and inference consistency"""
    if x.max() > 1.0:
        x = x / 255.0
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1, 3, 1, 1)
    
    x = (x - mean) / std
    return x