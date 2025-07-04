import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from typing import List, Tuple

# Braille character classes
BRAILLE_CLASSES = [
    "BACKGROUND",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z"
]

def ConvNorm(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    num_groups = min(32, in_channels // 4) if in_channels >= 32 else max(1, in_channels // 2)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.GroupNorm(num_groups, out_channels)
    )

def _upsample(x: torch.Tensor, size) -> torch.Tensor:
    """Upsample tensor using bilinear interpolation"""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=28):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class FPN(nn.Module):
    """
    FIXED: Optimized FPN for small Braille character detection
    """
    
    def __init__(self, num_classes=None):
        super(FPN, self).__init__()
        
        from BackBone import BackBone
        self.backbone = BackBone(pretrained=True)
        
        if num_classes is None:
            num_classes = len(BRAILLE_CLASSES)
        
        from CBAM import CBAM
        
        self.num_classes = num_classes
        
        # FIXED: Consistent configuration for 5 levels
        self.strides = [4, 8, 16, 32, 64]
        self.scales = nn.Parameter(torch.tensor([6.0, 4.0, 4.0, 3.0, 2.0]))

        
        # FIXED: 5 CBAM modules for 5 FPN levels
        self.fpn_cbam = nn.ModuleList([
            CBAM(256, reduction=8) for _ in range(5)  # P2, P3, P4, P5, P6
        ])
        
        # Lateral connections for ResNet-50 channels
        self.lateral_convs = nn.ModuleList([
            self._make_lateral_conv(256, 256),   # C2 -> P2
            self._make_lateral_conv(512, 256),   # C3 -> P3
            self._make_lateral_conv(1024, 256),  # C4 -> P4  
            self._make_lateral_conv(2048, 256),  # C5 -> P5
        ])
        
        # Extra FPN level
        self.extra_convs = nn.ModuleList([
            ConvNorm(256, 256, stride=2),  # P5 -> P6
        ])
        
        # REMOVED: Unused p1_conv
        
        # Enhanced detection heads
        self.classification_head = nn.Sequential(
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 128),  # Reduce to 128 channels
            nn.ReLU(inplace=True),
        )
        
        # Output layers
        self.classification_to_class = nn.Conv2d(128, self.num_classes, kernel_size=3, padding=1)
        self.classification_to_centerness = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        
        # Regression head
        self.regression_head = nn.Sequential(
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 128),
            nn.ReLU(inplace=True),
        )
        
        self.regression_to_bbox = nn.Conv2d(128, 4, kernel_size=3, padding=1)
        
        # Loss function
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, num_classes=num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_lateral_conv(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create lateral connection layer"""
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def _initialize_weights(self):
        """Improved weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # FIXED: Less negative bias for better non-background prediction
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.classification_to_class.bias, bias_value)
        
        # Initialize centerness and regression outputs
        nn.init.normal_(self.classification_to_centerness.weight, std=0.01)
        nn.init.constant_(self.classification_to_centerness.bias, 0)
        nn.init.normal_(self.regression_to_bbox.weight, std=0.01)
        nn.init.constant_(self.regression_to_bbox.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        # Backbone forward
        backbone_features = self.backbone(x)   
        c2, c3, c4, c5 = backbone_features
        
        # Build FPN pyramid
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + _upsample(p5, c4.shape[2:])
        p3 = self.lateral_convs[1](c3) + _upsample(p4, c3.shape[2:])
        p2 = self.lateral_convs[0](c2) + _upsample(p3, c2.shape[2:])
        
        # FIXED: Apply CBAM to all levels including P6
        p2 = self.fpn_cbam[0](p2)  # Fine details
        p3 = self.fpn_cbam[1](p3)  # Main Braille level
        p4 = self.fpn_cbam[2](p4)  # Larger characters
        p5 = self.fpn_cbam[3](p5)  # Character groups
        
        # Add P6 level with CBAM
        p6 = self.extra_convs[0](p5)
        p6 = self.fpn_cbam[4](p6)  # FIXED: Apply CBAM to P6
        
        # All 5 FPN levels
        fpn_features = [p2, p3, p4, p5, p6]
        
        # Apply detection heads
        classes_by_feature = []
        centerness_by_feature = []
        regression_by_feature = []
        
        for scale, fpn_feature in zip(self.scales, fpn_features):
            # Classification branch
            cls_features = self.classification_head(fpn_feature)
            classes = self.classification_to_class(cls_features)
            centerness = torch.sigmoid(self.classification_to_centerness(cls_features))
            
            # FIXED: Regression branch without torch.exp
            reg_features = self.regression_head(fpn_feature)
            bbox_pred = F.relu(self.regression_to_bbox(reg_features)) * scale
            
            # Reshape outputs: B[C]HW -> BHW[C]
            classes = classes.permute(0, 2, 3, 1).contiguous()
            centerness = centerness.permute(0, 2, 3, 1).contiguous().squeeze(-1)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
            
            classes_by_feature.append(classes)
            centerness_by_feature.append(centerness)
            regression_by_feature.append(bbox_pred)
        
        return classes_by_feature, centerness_by_feature, regression_by_feature

def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """Proper normalization for training and inference consistency"""
    if x.max() > 1.0:
        x = x / 255.0
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1, 3, 1, 1)
    
    x = (x - mean) / std
    return x
