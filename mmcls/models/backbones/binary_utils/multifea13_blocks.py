import torch
import torch.nn as nn
from .binary_convs import BLConv2d
from .binary_functions import act_name_map, FeaExpand


class MultiFea_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, nonlinear='hardtanh', fea_num=1, mode='1', **kwargs):
        super(MultiFea_Block, self).__init__()
        self.fea_num = fea_num
        self.fexpand1 = FeaExpand(expansion=fea_num, mode=mode)
        self.conv1 = BLConv2d(in_channels * fea_num, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.fexpand2 = FeaExpand(expansion=fea_num, mode=mode)
        self.conv2 = BLConv2d(out_channels * fea_num, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if nonlinear == 'prelu':
            self.nonlinear1 = nn.PReLU(out_channels)
            self.nonlinear2 = nn.PReLU(out_channels)
        else:
            self.nonlinear1 = act_name_map[nonlinear](inplace=True)
            self.nonlinear2 = act_name_map[nonlinear](inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear2(out)

        return out


class MultiFea13_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, nonlinear='prelu', fea_num=1, mode='1', **kwargs):
        super(MultiFea13_Block, self).__init__()
        self.fea_num = fea_num
        self.fexpand1 = FeaExpand(expansion=fea_num, mode=mode)
        self.conv1 = BLConv2d(in_channels * fea_num, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.fexpand2 = FeaExpand(expansion=fea_num, mode=mode)
        self.conv2 = BLConv2d(out_channels * fea_num, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if nonlinear == 'prelu':
            self.nonlinear1 = nn.PReLU(out_channels)
            self.nonlinear2 = nn.PReLU(out_channels)
        else:
            self.nonlinear1 = act_name_map[nonlinear](inplace=True)
            self.nonlinear2 = act_name_map[nonlinear](inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.nonlinear1(out)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.nonlinear2(out)
        out = self.bn2(out)
        out += identity

        return out


class MultiFea13clip_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, fea_num=1, mode='1', **kwargs):
        super(MultiFea13clip_Block, self).__init__()
        self.fea_num = fea_num
        self.fexpand1 = FeaExpand(expansion=fea_num, mode=mode)
        self.conv1 = BLConv2d(in_channels * fea_num, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear11 = nn.PReLU(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.fexpand2 = FeaExpand(expansion=fea_num, mode=mode)
        self.nonlinear12 = nn.Hardtanh(inplace=True)
        self.conv2 = BLConv2d(out_channels * fea_num, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.nonlinear21 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.nonlinear11(out)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear12(out)

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.nonlinear21(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear22(out)

        return out