import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from typing import List, Tuple
import numpy as np

from Attention import CoordinateAttention
from torchvision.ops import DeformConv2d
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
    """
    Focal Loss for addressing class imbalance in multi-class classification.
    Supports manual alpha weights or automatic inverse-frequency weighting.
    """
    def __init__(self, alpha=None, gamma=2.0,num_classes=27, reduction='mean'):
       
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
            self.alpha = "auto"  # Placeholder for deferred calculation
        else:
            self.alpha = alpha  # Scalar or None

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
                alpha_t = alpha_tensor[targets]  # Gather per-class weights
            else:
                alpha_t = self.alpha  # Scalar alpha
            focal_weight *= alpha_t
        
        focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
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
            # Flatten labels and convert to integers
            labels = labels.view(-1).long()  # Flattens and ensures 1D
            # Filter out invalid indices (e.g., ignore background class if needed)
            valid_labels = labels[(labels >= 0) & (labels < num_classes)]
            class_counts += torch.bincount(valid_labels, minlength=num_classes)
            total_samples += len(valid_labels)

        # Compute weights (same as before)
        class_weights = total_samples / (num_classes * (class_counts + 1e-6))
        class_weights = torch.log(class_weights + 1.0)
        class_weights = torch.clamp(class_weights, min=0.1, max=3.0)
        self.alpha = class_weights
        print(f"Auto-calculated alpha weights: {self.alpha}")

class FPN(nn.Module):
    """
    Optimized FPN for small Braille character detection
    """
    
    def __init__(self, num_classes=27):
        super(FPN, self).__init__()
        
        self.use_checkpoint = True
        
        from BackBone import BackBone
        self.backbone = BackBone()
        
        if num_classes is None:
            num_classes = len(BRAILLE_CLASSES)
            
        self.num_classes = num_classes
        
        # FIXED: Consistent configuration for 5 levels
        self.strides = [2,4, 8, 16, 32]
        self.scales = nn.Parameter(torch.tensor([0.5,1.0,2.0,4.0,8]))


        # self.coord_attention = nn.ModuleList([   
        #     CoordinateAttention(256, reduction=16),  # P2 (highest resolution)
        #     CoordinateAttention(256, reduction=16),  # P5 (deepest features)
        # ])
           
        # Lateral connections for ResNet-50 channels
        self.lateral_convs = nn.ModuleList([
            self._make_lateral_conv(64, 256),   
            self._make_lateral_conv(128, 256),   # C3 -> P3
            self._make_lateral_conv(256, 256),  # C4 -> P4  
            self._make_lateral_conv(512, 256),  # C5 -> P5
        ])
        
        # Extra FPN level
        self.extra_convs = nn.ModuleList([
            ConvNorm(256, 256, stride=2),  # P5 -> P6
        ])
        
        
        # Enhanced detection heads
        self.classification_head = nn.Sequential(
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 128),  # Reduce to 128 channels
            nn.ReLU(inplace=True),
        )
        
        # Output layers
        self.classification_to_class = nn.Conv2d(128, self.num_classes, kernel_size=3, padding=1)
    
        
        # Regression head
        self.regression_head = nn.Sequential(
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 128),
            nn.ReLU(inplace=True),
        )
        
        self.regression_to_bbox = nn.Conv2d(128, 4, kernel_size=3, padding=1)
        
        # Loss function
        self.focal_loss = FocalLoss(alpha=0.5, gamma=1.5, num_classes=num_classes)
        
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
        
        # Initialize regression outputs
        nn.init.normal_(self.regression_to_bbox.weight, std=0.01)
        nn.init.constant_(self.regression_to_bbox.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        # Backbone forward
        backbone_features = self.backbone(x)   
        c2, c3, c4, c5 = backbone_features
        
         # Build FPN pyramid (top-down)
        p5 = self.lateral_convs[3](c5)
        
        p4 = self.lateral_convs[2](c4) + _upsample(p5, c4.shape[2:])
        p3 = self.lateral_convs[1](c3) + _upsample(p4, c3.shape[2:])
        p2 = self.lateral_convs[0](c2) + _upsample(p3, c2.shape[2:])    
        # Generate P6 level
        p6 = self.extra_convs[0](p5)
        # All 5 FPN levels in correct order
        fpn_features = [p2, p3, p4, p5, p6]
        
        # Apply detection heads
        classes_by_feature = []
        regression_by_feature = []
        
        for scale, fpn_feature in zip(self.scales, fpn_features):
            # Classification branch
            cls_features = self.classification_head(fpn_feature)
            classes = self.classification_to_class(cls_features)
            
            # FIXED: Regression branch 
            reg_features = self.regression_head(fpn_feature)
            bbox_pred = torch.exp(self.regression_to_bbox(reg_features)) * scale
            
            # Reshape outputs: B[C]HW -> BHW[C]
            classes = classes.permute(0, 2, 3, 1).contiguous()
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
            
            classes_by_feature.append(classes)
            regression_by_feature.append(bbox_pred)
            
        
        
        return classes_by_feature, regression_by_feature

def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """Proper normalization for training and inference consistency"""
    if x.max() > 1.0:
        x = x / 255.0
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1, 3, 1, 1)
    
    x = (x - mean) / std
    return x