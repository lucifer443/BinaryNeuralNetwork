import torch
import torch.nn as nn
from .binary_convs import BLConv2d
from .binary_functions import FeaExpand


class MultiFea_3_1_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_3_1_Block, self).__init__()
        self.fexpand1 = FeaExpand(expansion=3, mode='1')
        self.conv1 = BLConv2d(in_channels * 3, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.fexpand2 = FeaExpand(expansion=3, mode='1')
        self.conv2 = BLConv2d(out_channels * 3, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        residual = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.nonlinear1(out)

        residual = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.nonlinear2(out)

        return out


class MultiFea_3_1c_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_3_1c_Block, self).__init__()
        self.fexpand1 = FeaExpand(expansion=3, mode='1c')
        self.conv1 = BLConv2d(in_channels * 3, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.fexpand2 = FeaExpand(expansion=3, mode='1c')
        self.conv2 = BLConv2d(out_channels * 3, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        residual = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.nonlinear1(out)

        residual = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.nonlinear2(out)

        return out


class MultiFea_2_2_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_2_2_Block, self).__init__()
        self.fexpand1 = FeaExpand(expansion=2, mode='2')
        self.conv1 = BLConv2d(in_channels * 2, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.fexpand2 = FeaExpand(expansion=2, mode='2')
        self.conv2 = BLConv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        residual = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.nonlinear1(out)

        residual = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.nonlinear2(out)

        return out
