import torch
import torch.nn as nn

from torch.nn import functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, padding_mode='reflect', bias=False),# 'refelct'为对称填充
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.layer(x)
    
class DownSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # 下采样不采用最大池化，避免特征丢失太多
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.layer(x)
    
    
class UpSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 3, padding=1),# 降低通道数
        )
    
    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat([out, feature_map], dim=1)
        
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 64)
        self.d1 = DownSample(64)
        self.c2 = ConvBlock(64, 128)
        self.d2 = DownSample(128)
        self.c3 = ConvBlock(128, 256)
        self.d3 = DownSample(256)
        self.c4 = ConvBlock(256, 512)
        self.d4 = DownSample(512)
        self.c5 = ConvBlock(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = ConvBlock(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = ConvBlock(512, 256)
        self.u3 = UpSample(256)
        self.c8 = ConvBlock(256, 128)
        self.u4 = UpSample(128)
        self.c9 = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, 3, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        R6 = self.c6(self.u1(R5, R4))
        R7 = self.c7(self.u2(R6, R3))
        R8 = self.c8(self.u3(R7, R2))
        R9 = self.c9(self.u4(R8, R1))
        return self.sigmoid(self.out(R9))
    
    
if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    model = UNet()
    print(model(x).shape) 
    
        
        