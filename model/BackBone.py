import torch
import torch.nn as nn
import torchvision.models as models

class BackBone(nn.Module):
    """
    Pretrained ResNet-50 backbone for feature extraction
    Returns feature maps: c2, c3, c4, c5 corresponding to layers 1–4
    Channels: 256, 512, 1024, 2048
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # Output: 256 channels
        self.layer2 = resnet.layer2  # Output: 512 channels  
        self.layer3 = resnet.layer3  # Output: 1024 channels
        self.layer4 = resnet.layer4  # Output: 2048 channels
        
        # Freeze early layers for stability (optional)
        if pretrained:
            self._freeze_early_layers()
    
    def _freeze_early_layers(self):
        """Freeze conv1 and layer1 for stable training"""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # x: [B,3,H,W]
        x = self.conv1(x)   # [B,64,H/2,W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [B,64,H/4,W/4]
        
        c2 = self.layer1(x) # [B,256,H/4,W/4]
        c3 = self.layer2(c2)# [B,512,H/8,W/8]
        c4 = self.layer3(c3)# [B,1024,H/16,W/16]
        c5 = self.layer4(c4)# [B,2048,H/32,W/32]
        
        return c2, c3, c4, c5
