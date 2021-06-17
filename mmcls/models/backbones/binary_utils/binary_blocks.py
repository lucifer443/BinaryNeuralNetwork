import torch
import torch.nn as nn
from .binary_convs import IRConv2d, RAConv2d, STEConv2d
from .binary_functions import RPRelu, LearnableBias, LearnableScale, AttentionScale


class IRNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetBlock, self).__init__()
        self.conv1 = IRConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.nonlinear(out)

        residual = out
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.nonlinear(out)

        return out


class RANetBlockA(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(RANetBlockA, self).__init__()

        self.move1 = LearnableBias(in_channels)
        self.conv1 = RAConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = RPRelu(out_channels)

        self.move2 = LearnableBias(out_channels)
        self.conv2 = RAConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = RPRelu(out_channels)

        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.move1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu1(out)

        residual = out
        out = self.move2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.prelu2(out)
        return out


class RANetBlockB(nn.Module):
    def __init__(self, inplanes, planes, stride=1, **kwargs):
        super(RANetBlockB, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.move1 = LearnableBias(inplanes)
        self.binary_3x3 = RAConv2d(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(inplanes)

        self.prelu1 = RPRelu(inplanes)

        self.move2 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = RAConv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.binary_pw_down1 = RAConv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False,
                                            **kwargs)
            self.binary_pw_down2 = RAConv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False,
                                            **kwargs)
            self.bn2_1 = nn.BatchNorm2d(inplanes)
            self.bn2_2 = nn.BatchNorm2d(inplanes)

        self.prelu2 = RPRelu(planes)

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):

        out1 = self.move1(x)

        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.prelu1(out1)

        out2 = self.move2(out1)

        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.bn2(out2)
            out2 += out1

        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.prelu2(out2)

        return out2
    

class StrongBaselineBlock(nn.Module):
    """Strong baseline block from real-to-binary net"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(StrongBaselineBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = STEConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.scale1 = LearnableScale(out_channels)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = STEConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
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


class RealToBinaryBlock(nn.Module):
    """base block from real-to-binary net"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(RealToBinaryBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = STEConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.scale1 = AttentionScale(out_channels)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = STEConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.scale2 = AttentionScale(out_channels)
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


class RealToBinaryFPBlock(nn.Module):
    """base block from real-to-binary net"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(RealToBinaryFPBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.scale1 = AttentionScale(out_channels)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.scale2 = AttentionScale(out_channels)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels
        self.approx_sign = nn.Tanh()

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.approx_sign(out)
        out = self.conv1(out)
        out = self.scale1(out)
        out = self.nonlinear1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        identity = out
        out = self.bn2(out)
        out = self.approx_sign(out)
        out = self.conv2(out)
        out = self.scale2(out)
        out = self.nonlinear2(out)
        out += identity

        return out
