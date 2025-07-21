import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_map))
        return x * attention

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Apply channel attention first
        x = self.channel_attention(x)
        # Then apply spatial attention
        y = self.spatial_attention(x)
        return y

class SE_Attention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SE_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction*2, in_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        se = self.avg_pool(x)
        se = self.fc(se)
        return x * se
    
class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, 1)
        self.conv_h = nn.Conv2d(mip, channels, 1)
        self.conv_w = nn.Conv2d(mip, channels, 1)
    
    def forward(self, x):
        n, c, h, w = x.size()
        
        # Memory-efficient implementation
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = F.relu(self.conv1(y))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # Compute attention weights separately to save memory
        att_h = torch.sigmoid(self.conv_h(x_h))
        att_w = torch.sigmoid(self.conv_w(x_w))
        
        # Apply attention in-place to save memory
        x = x * att_h
        x = x * att_w
        
        return x
