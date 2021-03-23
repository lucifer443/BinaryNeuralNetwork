import torch
import torch.nn as nn
from .binary_convs import IRConv2d, RAConv2d
from .binary_functions import RPRelu, LearnableBias

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

# group sign
# 多分支conv，将原来的一个conv拆分成四个conv，每个conv的输入经过不同阈值的sign
class IRNetG4Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetG4Block, self).__init__()

        self.conv11 = IRConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.conv12 = IRConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.conv13 = IRConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.conv14 = IRConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)

        self.bn11 = nn.BatchNorm2d(out_channels)
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.bn13 = nn.BatchNorm2d(out_channels)
        self.bn14 = nn.BatchNorm2d(out_channels)

        self.conv21 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.conv22 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.conv23 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.conv24 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)

        self.bn21 = nn.BatchNorm2d(out_channels)
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.bn23 = nn.BatchNorm2d(out_channels)
        self.bn24 = nn.BatchNorm2d(out_channels)

        self.nonlinear = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        alpha = torch.max(abs(x))
        alpha1 = alpha / 5   # 1/5
        alpha2 = alpha1 * 3  # 3/5

        x1 = x + alpha2
        x2 = x + alpha1
        x3 = x - alpha1
        x4 = x - alpha2

        out1 = self.conv11(x1)
        out2 = self.conv12(x2)
        out3 = self.conv13(x3)
        out4 = self.conv14(x4)

        out1 = self.bn11(out1)
        out2 = self.bn12(out2)
        out3 = self.bn13(out3)
        out4 = self.bn14(out4)
        out = out1 + out2 + out3 + out4

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.nonlinear(out)

        identity = out
        alpha = torch.max(abs(out))
        alpha1 = alpha / 5   # 1/5
        alpha2 = alpha1 * 3  # 3/5

        x1 = out + alpha2
        x2 = out + alpha1
        x3 = out - alpha1
        x4 = out - alpha2

        out1 = self.conv21(x1)
        out2 = self.conv22(x2)
        out3 = self.conv23(x3)
        out4 = self.conv24(x4)

        out1 = self.bn21(out1)
        out2 = self.bn22(out2)
        out3 = self.bn23(out3)
        out4 = self.bn24(out4)
        out = out1 + out2 + out3 + out4

        out += identity
        out = self.nonlinear(out)

        return out

# hierarchical conv
# 分层conv，一个block的输出是两个conv输出的拼接的结果
class IRNetH1Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetH1Block, self).__init__()
        self.conv1 = IRConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)

        if self.downsample is not None:
            identity = self.downsample(x)
        out1 += identity

        out1 = self.nonlinear(out1)

        identity = out1
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)

        out2 += identity
        out2 = self.nonlinear(out2)

        out = torch.cat((out1, out2), dim=1)
        out = torch.reshape(out, (out.shape[0], out.shape[1] // 2, 2, out.shape[2], out.shape[3]))
        out = torch.sum(out, dim=2)

        return out