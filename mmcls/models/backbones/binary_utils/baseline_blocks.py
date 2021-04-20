import torch
import torch.nn as nn
from .binary_convs import BLConv2d
from .binary_functions import RPRelu, LearnableBias

class BaselineBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(BaselineBlock, self).__init__()
        self.conv1 = BLConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.conv2 = BLConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinear1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        identity = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonlinear2(out)
        out += identity

        return out
