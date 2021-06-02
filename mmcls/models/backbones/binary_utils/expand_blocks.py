import torch
import torch.nn as nn
from .binary_convs import IRConv2dnew, RAConv2d ,IRConv2d_bias ,IRConv2d_bias_x2,IRConv2d_bias_x2x,STEConv2d,StrongBaselineConv2d
from .binary_functions import RPRelu, LearnableBias, LearnableScale, AttentionScale,Expandx


class ExpandBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, Expand_num=1,**kwargs):
        super(ExpandBlock, self).__init__()
        self.expand1 = Expandx(Expand_num=Expand_num, in_channels=in_channels)
        self.conv1 = IRConv2dnew(in_channels*Expand_num, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = IRConv2dnew(out_channels*Expand_num, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.expand2 = Expandx(Expand_num=Expand_num, in_channels=out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.expand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.nonlinear(out)

        residual = out
        out = self.expand2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.nonlinear(out)

        return out