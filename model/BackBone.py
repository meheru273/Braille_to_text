import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add residual connection
        out += identity
        out = F.relu(out)

        return out


class BackBone(nn.Module):
    """
    ResNet-34 backbone specifically for FPN
    """
    def __init__(self):
        super(BackBone, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)   # 1/4 resolution
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)  # 1/8 resolution
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)  # 1/16 resolution
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)  # 1/32 resolution
        
        # Output channels for FPN
        self.out_channels = [64, 128, 256, 512]  # For c2, c3, c4, c5

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        # Create downsample layer if needed
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers - extract features at different scales
        c2 = self.layer1(x)  # 1/4 resolution, 64 channels
        c3 = self.layer2(c2)  # 1/8 resolution, 128 channels
        c4 = self.layer3(c3)  # 1/16 resolution, 256 channels
        c5 = self.layer4(c4)  # 1/32 resolution, 512 channels

       
        return [c2,c3, c4, c5]
