import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from typing import List, Tuple
from torchvision.transforms import Normalize

# Braille character classes - 64 possible combinations (2^6) plus background
BRAILLE_CLASSES = [
    "BACKGROUND",
    # Basic Braille characters (Grade 1)
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z"
]

def ConvNorm(in_channels: int, out_channels: int, kernel_size=3, padding=1, stride=1) -> nn.Module:
    """
    Convolution with GroupNorm - adapted for Braille detection
    """
    num_groups = min(32, in_channels // 2) if in_channels >= 32 else in_channels
    num_groups = max(1, num_groups)  # Ensure at least 1 group
    return nn.Sequential(
        nn.GroupNorm(num_groups, in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
    )

def _upsample(x: torch.Tensor, size) -> torch.Tensor:
    """Upsample tensor to specified size using bilinear interpolation."""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)

class FPN(nn.Module):
    """
    Feature Pyramid Network adapted for Braille character detection
    """
    
    def __init__(self, num_classes=None):
        super(FPN, self).__init__()
        
        # Import your backbone (assuming it returns C2, C3, C4, C5 features)
        from BackBone import BackBone
        self.backbone = BackBone()
        
        if num_classes is None:
            num_classes = len(BRAILLE_CLASSES)
        
        self.num_classes = num_classes
        self.strides = [8, 16, 32, 64, 128]  # FPN level strides
        
        # Learnable scales for each FPN level (smaller for Braille characters)
        self.scales = nn.Parameter(torch.tensor([4.0, 8.0, 16.0, 32.0, 64.0]))
        
        # FPN lateral connections (matching your backbone channel sizes)
        # BackBone outputs: c2(64), c3(128), c4(256), c5(512)
        self.lateral_convs = nn.ModuleList([
            self._make_lateral_conv(64, 256),   # C2 -> P2
            self._make_lateral_conv(128, 256),  # C3 -> P3
            self._make_lateral_conv(256, 256),  # C4 -> P4  
            self._make_lateral_conv(512, 256),  # C5 -> P5
        ])
        
        # Extra FPN levels
        self.extra_convs = nn.ModuleList([
            ConvNorm(256, 256, stride=2),  # P5 -> P6
            ConvNorm(256, 256, stride=2),  # P6 -> P7
        ])
        
        # Classification head - specialized for Braille
        self.classification_head = nn.Sequential(
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
        )
        
        # Output layers
        self.classification_to_class = ConvNorm(256, self.num_classes)
        self.classification_to_centerness = ConvNorm(256, 1)
        
        # Regression head - for Braille character bounding boxes
        self.regression_head = nn.Sequential(
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
            ConvNorm(256, 256),
            nn.ReLU(inplace=True),
        )
        
        self.regression_to_bbox = ConvNorm(256, 4)  # [left, top, right, bottom] distances
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_lateral_conv(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create lateral connection layer"""
        return ConvNorm(in_channels, out_channels, kernel_size=1, padding=0)
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for modules in [
            self.lateral_convs,
            self.extra_convs,
            self.classification_head,
            self.regression_head,
            self.classification_to_class,
            self.classification_to_centerness,
            self.regression_to_bbox,
        ]:
            for module in modules:
                for layer in module.modules():
                    if isinstance(layer, nn.Conv2d):
                        nn.init.normal_(layer.weight, std=0.01)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
        
        # Special initialization for classification (focal loss style)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if hasattr(self.classification_to_class, '__getitem__'):
            # Sequential module
            for module in self.classification_to_class:
                if isinstance(module, nn.Conv2d) and module.bias is not None:
                    nn.init.constant_(module.bias, bias_value)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        
        # BackBone returns 4 feature maps: c2, c3, c4, c5
        backbone_features = self.backbone(x)   
        c2, c3, c4, c5 = backbone_features
        
        # Build FPN pyramid
        p5 = self.lateral_convs[3](c5)  # C5 -> P5
        p4 = self.lateral_convs[2](c4) + _upsample(p5, c4.shape[2:])  # C4 + P5_up -> P4
        p3 = self.lateral_convs[1](c3) + _upsample(p4, c3.shape[2:])  # C3 + P4_up -> P3
        p2 = self.lateral_convs[0](c2) + _upsample(p3, c2.shape[2:])  # C2 + P3_up -> P2
        
        # Extra levels
        p6 = self.extra_convs[0](p5)
        p7 = self.extra_convs[1](p6)
        
        # Use P3-P7 instead of P2-P6 for better performance with small objects
        fpn_features = [p3, p4, p5, p6, p7]
        
        # Apply detection heads
        classes_by_feature = []
        centerness_by_feature = []
        regression_by_feature = []
        
        for scale, fpn_feature in zip(self.scales, fpn_features):
            # Classification branch
            cls_features = self.classification_head(fpn_feature)
            classes = self.classification_to_class(cls_features).sigmoid()
            centerness = self.classification_to_centerness(cls_features).sigmoid()
            
            # Regression branch
            reg_features = self.regression_head(fpn_feature)
            bbox_pred = torch.exp(self.regression_to_bbox(reg_features)) * scale
            
            # Reshape outputs: B[C]HW -> BHW[C]
            classes = classes.permute(0, 2, 3, 1).contiguous()
            centerness = centerness.permute(0, 2, 3, 1).contiguous().squeeze(-1)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
            
            classes_by_feature.append(classes)
            centerness_by_feature.append(centerness)
            regression_by_feature.append(bbox_pred)
        
        return classes_by_feature, centerness_by_feature, regression_by_feature

def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Fixed normalization function - normalize each image in-place
    """
    # Convert from 0-1 range to 0-255 range if needed
    if x.max() <= 1.0:
        x = x * 255.0
    
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1, 3, 1, 1) * 255.0
    std = torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1, 3, 1, 1) * 255.0
    
    x = (x - mean) / std
    return x