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


class FeatureFusionModule(nn.Module):
    """
    Module to fuse multi-scale features into a single feature map
    """
    def __init__(self, in_channels=256, out_channels=256, target_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_size = target_size
        
        # Channel attention for weighting different scales
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 5, in_channels * 5 // 4, 1),  # 5 FPN levels
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 5 // 4, 5, 1),  # Output weights for 5 levels
            nn.Sigmoid()
        )
        
        # Final fusion layer
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 5, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        """
        features: List of [B, C, H_i, W_i] feature maps
        Returns: Single fused feature map [B, C, H, W]
        """
        # Determine target size (use the largest feature map or specified size)
        if self.target_size is None:
            target_size = max([feat.shape[2:] for feat in features])
        else:
            target_size = self.target_size
        
        # Resize all features to target size
        resized_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                resized_feat = F.interpolate(
                    feat, size=target_size, 
                    mode='bilinear', align_corners=False
                )
            else:
                resized_feat = feat
            resized_features.append(resized_feat)
        
        # Concatenate all features
        concat_features = torch.cat(resized_features, dim=1)  # [B, C*5, H, W]
        
        # Generate attention weights for each scale
        attention_weights = self.channel_attention(concat_features)  # [B, 5, 1, 1]
        
        # Apply attention weights to each scale
        weighted_features = []
        for i, feat in enumerate(resized_features):
            weight = attention_weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
            weighted_feat = feat * weight
            weighted_features.append(weighted_feat)
        
        # Concatenate weighted features and fuse
        weighted_concat = torch.cat(weighted_features, dim=1)
        fused_feature = self.fusion_conv(weighted_concat)
        
        return fused_feature


class FPN(nn.Module):
    """
    Enhanced FPN with single detection layer for Braille detection
    Addresses zero positive samples issue by fusing all scales
    """
    def __init__(self, num_classes=26, use_fpn_att: bool = True, fusion_target_size=(64, 64)):
        super().__init__()
        
        from BackBone import BackBone
        self.backbone = BackBone()
        
        self.num_classes = num_classes
        self.use_fpn_att = use_fpn_att
        self.fusion_target_size = fusion_target_size

        # Single stride for the fused detection layer
        self.stride = 8  # You can adjust this based on your needs
        self.scale = nn.Parameter(torch.tensor(8.0))  # Single scale parameter
        
        # Keep strides for backward compatibility (only one element now)
        self.strides = [8]  # Single stride in a list for compatibility
        self.scales = nn.Parameter(torch.tensor([8.0]))  # Single scale in a list
        
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
        
        # Feature fusion layers (optional - for FPN level enhancement)
        if self.use_fpn_att:
            # Use cross-attention fusion between FPN levels
            self.fpn_cross_attention_fusion = FPNTransformerFusion(fpn_channels, num_heads=8)
        else:
            # Use regular CNN fusion
            self.feature_fusion = nn.ModuleList()
            for _ in range(5):  # 5 FPN levels
                self.feature_fusion.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                        nn.GroupNorm(32, fpn_channels),
                        nn.ReLU(inplace=True)
                    )
                )
        
        # NEW: Multi-scale feature fusion module
        self.feature_fusion_module = FeatureFusionModule(
            in_channels=fpn_channels, 
            out_channels=fpn_channels,
            target_size=fusion_target_size
        )
        
        # Single detection head (instead of multiple heads for different scales)
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
        print(f"  - Single Detection Layer: Enabled")
        print(f"  - Fusion Target Size: {self.fusion_target_size}")
        print(f"  - Number of Classes: {self.num_classes}")
        print(f"  - Single Stride: {self.strides}")  # Show as list for compatibility
    
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Returns single detection outputs instead of multi-scale lists
        Returns (classes, centerness, regression, attention_maps)
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
        
        # FPN features
        fpn_features = [p2, p3, p4, p5, p6]  # All 5 levels
        
        # Optional: Feature enhancement at FPN level
        if self.use_fpn_att:
            # Use cross-attention fusion between FPN levels
            enhanced_features = self.fpn_cross_attention_fusion(fpn_features)
        else:
            # Use regular CNN fusion
            enhanced_features = []
            for feat, fusion in zip(fpn_features, self.feature_fusion):
                fused = fusion(feat)
                enhanced_features.append(fused)
        
        # NEW: Fuse all scales into single feature map
        fused_feature = self.feature_fusion_module(enhanced_features)
        
        # For attention maps (can be empty list if not using attention)
        attention_maps = []
        
        # Single detection head predictions
        cls_feat = self.classification_head(fused_feature)
        cls_out = torch.sigmoid(self.classification_to_class(cls_feat))
        
        cent_feat = self.centerness_head(fused_feature)
        cent_out = self.centerness_to_centerness(cent_feat)  # Logits for BCEWithLogitsLoss
        
        reg_feat = self.regression_head(fused_feature)
        reg_out = torch.exp(self.regression_to_bbox(reg_feat)) * self.scales[0]
        
        # Reshape to match expected format: B[C]HW -> BHW[C]
        cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
        cent_out = cent_out.permute(0, 2, 3, 1).contiguous().squeeze(3)
        reg_out = reg_out.permute(0, 2, 3, 1).contiguous()
        
        return cls_out, cent_out, reg_out, attention_maps


# Helper function to normalize batch
def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """Proper normalization for training and inference consistency"""
    if x.max() > 1.0:
        x = x / 255.0
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1, 3, 1, 1)
    
    x = (x - mean) / std
    return x