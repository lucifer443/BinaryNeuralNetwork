import torch
import torch.nn as nn
from .binary_convs import IRConv2d, RAConv2d ,IRConv2d_bias ,IRConv2d_bias_x2,IRConv2d_bias_x2x,STEConv2d,StrongBaselineConv2d
from .binary_functions import RPRelu, LearnableBias, LearnableScale, AttentionScale

class RANetBlockAlb(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, gbi=0,**kwargs):
        super(RANetBlockAlb, self).__init__()

        self.rebias1 = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.rebias2 = nn.Parameter(torch.zeros(1),requires_grad=True)
        #self.rebias1 = nn.Parameter(torch.zeros(1),requires_grad=True)
        #self.gbi=gbi
        self.conv1 = RAConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.Mish()
        #self.rebias2 = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.conv2 = RAConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.Mish()

        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = x+self.rebias1
        out = self.conv1(out)
        out = self.bn1(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu1(out)

        residual = out
        out = out+self.rebias2
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.prelu2(out)
        return out


class RANetBlockA_fl(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(RANetBlockA_fl, self).__init__()

        #self.move1 = LearnableBias(in_channels)
        self.conv1 = RAConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.Mish()

        #self.move2 = LearnableBias(out_channels)
        self.conv2 = RAConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.Mish()

        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        #out = self.move1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu1(out)

        residual = out
        #out = self.move2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.prelu2(out)
        return out