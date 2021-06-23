import torch
import torch.nn as nn
from ..binary_utils.binary_convs import BLConv2d, STEConv2d, BConvWS2d
from ..binary_utils.binary_functions import LearnableScale


class Baseline11Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline11Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = BLConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = BLConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.nonlinear1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        identity = out
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.nonlinear2(out)
        out += identity

        return out


class Baseline11sBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline11sBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = BLConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.scale1 = LearnableScale(out_channels)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = BLConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.scale2 = LearnableScale(out_channels)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.scale1(out)
        out = self.nonlinear1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        identity = out
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.scale2(out)
        out = self.nonlinear2(out)
        out += identity

        return out


class BaselineStrongBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(BaselineStrongBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = STEConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, clip=1.25, **kwargs)
        self.scale1 = LearnableScale(out_channels)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = STEConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, clip=1.25, **kwargs)
        self.scale2 = LearnableScale(out_channels)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.scale1(out)
        out = self.nonlinear1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        identity = out
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.scale2(out)
        out = self.nonlinear2(out)
        out += identity

        return out


class Baseline12Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline12Block, self).__init__()
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


class Baseline13Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline13Block, self).__init__()
        self.conv1 = BLConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = BLConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.nonlinear1(out)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        identity = out
        out = self.conv2(out)
        out = self.nonlinear2(out)
        out = self.bn2(out)
        out += identity

        return out


class Baseline13pBlock(nn.Module):
    """ add a prelu after each block """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline13pBlock, self).__init__()
        self.conv1 = BLConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear12 = nn.PReLU(out_channels, init=1.0)
        self.conv2 = BLConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = nn.PReLU(out_channels, init=1.0)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.nonlinear1(out)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear12(out)

        identity = out
        out = self.conv2(out)
        out = self.nonlinear2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear22(out)

        return out


class Baseline13sBlock(nn.Module):
    """ add a learnable scale after each block """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline13sBlock, self).__init__()
        self.conv1 = BLConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear12 = LearnableScale(out_channels)
        self.conv2 = BLConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = LearnableScale(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.nonlinear1(out)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear12(out)

        identity = out
        out = self.conv2(out)
        out = self.nonlinear2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear22(out)

        return out


class Baseline14Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline14Block, self).__init__()
        self.conv1 = BLConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = BLConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.nonlinear1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.bn1(out)

        identity = out
        out = self.conv2(out)
        out = self.nonlinear2(out)
        out += identity
        out = self.bn2(out)

        return out


class Baseline15Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline15Block, self).__init__()
        self.conv1 = BLConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.conv2 = BLConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn11(out)
        out = self.nonlinear1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.bn12(out)

        identity = out
        out = self.conv2(out)
        out = self.bn21(out)
        out = self.nonlinear2(out)
        out += identity
        out = self.bn22(out)

        return out


class Baseline21Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline21Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = BLConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = BLConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)

        identity = out
        out = self.bn2(out)
        out = self.conv2(out)
        out += identity
        out = self.nonlinear2(out)

        return out


class Baseline22Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline22Block, self).__init__()
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
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)

        identity = out
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear2(out)

        return out


class Baseline23Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline23Block, self).__init__()
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
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.bn1(out)
        out = self.nonlinear1(out)

        identity = out
        out = self.conv2(out)
        out += identity
        out = self.bn2(out)
        out = self.nonlinear2(out)

        return out


class Baseline24Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline24Block, self).__init__()
        self.conv1 = BLConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = BLConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)
        out = self.bn1(out)

        identity = out
        out = self.conv2(out)
        out += identity
        out = self.nonlinear2(out)
        out = self.bn2(out)

        return out


class Baseline13clipBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline13clipBlock, self).__init__()
        self.conv1 = BLConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.clip1 = nn.Hardtanh(inplace=True)
        self.conv2 = BLConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.clip2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.nonlinear1(out)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.clip1(out)

        identity = out
        out = self.conv2(out)
        out = self.nonlinear2(out)
        out = self.bn2(out)
        out += identity
        out = self.clip2(out)

        return out



    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline12wsBlock, self).__init__()
        self.conv1 = BConvWS2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.conv2 = BConvWS2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
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


class Baseline13wsBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline13wsBlock, self).__init__()
        self.conv1 = BConvWS2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True, **kwargs)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = BConvWS2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, **kwargs)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.nonlinear1(out)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        identity = out
        out = self.conv2(out)
        out = self.nonlinear2(out)
        out = self.bn2(out)
        out += identity

        return out


class Baseline14wsBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(Baseline14wsBlock, self).__init__()
        self.conv1 = BConvWS2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True, **kwargs)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = BConvWS2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, **kwargs)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.nonlinear1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.bn1(out)

        identity = out
        out = self.conv2(out)
        out = self.nonlinear2(out)
        out += identity
        out = self.bn2(out)

        return out
