import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BackBone(nn.Module):
    """
    ResNet-18 backbone implemented from scratch.
    Returns feature maps: c2, c3, c4, c5 corresponding to layers 1–4.
    Channels: 64, 128, 256, 512.
    """
    def __init__(self, pretrained=False):
        super().__init__()
        # Initial conv: same as torchvision: 7x7 stride 2, then BN+ReLU, then 3x3 maxpool stride 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Build layers
        # layer1: 2 BasicBlocks, in_planes=64, planes=64, stride=1
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        # layer2: in=64, out=128, first block stride=2
        self.layer2 = self._make_layer(64*BasicBlock.expansion, 128, blocks=2, stride=2)
        # layer3: in=128, out=256, first stride=2
        self.layer3 = self._make_layer(128*BasicBlock.expansion, 256, blocks=2, stride=2)
        # layer4: in=256, out=512, first stride=2
        self.layer4 = self._make_layer(256*BasicBlock.expansion, 512, blocks=2, stride=2)
        # (Optionally load pretrained weights into matching layers if desired.)
        if pretrained:
            # if you have a state_dict for ResNet18, load matching keys here
            # e.g., from torchvision.models.resnet18(pretrained=True).state_dict()
            # but this is left to the user
            pass

    def _make_layer(self, in_planes, planes, blocks, stride):
        downsample = None
        if stride != 1 or in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride, downsample))
        in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B,3,H,W]
        x = self.conv1(x)   # [B,64,H/2,W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [B,64,H/4,W/4]
        c2 = self.layer1(x) # [B,64, H/4, W/4]
        c3 = self.layer2(c2)# [B,128,H/8, W/8]
        c4 = self.layer3(c3)# [B,256,H/16,W/16]
        c5 = self.layer4(c4)# [B,512,H/32,W/32]
        return c2, c3, c4, c5
