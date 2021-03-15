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

    
class CM1Block(nn.Module):
    expansion = 1
    cout_kernel_map = {64: (3, 1), 128: (3, 1), 256: (5, 2), 512: (7, 3)}

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(CM1Block, self).__init__()
        kernel_size, padding = self.cout_kernel_map[out_channels]
        self.conv1 = IRConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = IRConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
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
    
    def ede(self, k, t):
        for m in self.modules():
            if isinstance(m, IRConv2d):
                m.k = k
                m.t = t
#         print(f"k: {k},  t: {t}")
    
    
class CM2Block(nn.Module):
    expansion = 1
    cout_kernel_map = {64: (3, 1), 128: (3, 1), 256: (5, 2), 512: (7, 3)}

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(CM2Block, self).__init__()
        kernel_size, padding = self.cout_kernel_map[out_channels]
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = IRConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = IRConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels
        self.lateral_conv = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.nonlinear(out)
        out = self.conv1(out)
        

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual


        residual = out
        out = self.bn2(out)
        out = self.nonlinear(out)
        out = self.conv2(out)

        out += residual
        
        return out

    
class CM3Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(CM3Block, self).__init__()
        self.conv1 = IRConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels
        if self.downsample is None:
            self.lateral_conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels))
        self.lateral_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.is_train = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = self.lateral_conv1(residual)
        out += residual

        out = self.nonlinear(out)

        residual = out
        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.lateral_conv2(residual)
        out += residual
        out = self.nonlinear(out)

        return out
    
    def ede(self, k, t):
        for m in self.modules():
            if isinstance(m, IRConv2d):
                m.k = k
                m.t = t
                
    def train(self, mode=True):
        super(CM3Block, self).train(mode)
        self.is_train = mode
    

class ReActBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(ReActBlock, self).__init__()

        self.move1 = LearnableBias(in_channels)
        self.conv_3x3 = RAConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.rprelu1 = RPRelu(in_channels)

        self.move2 = LearnableBias(in_channels)

        if in_channels == out_channels:
            self.conv_1x1 = RAConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1_down1 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.conv_1x1_down2 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2_1 = nn.BatchNorm2d(in_channels)
            self.bn2_2 = nn.BatchNorm2d(in_channels)

        self.rprelu2 = RPRelu(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # 3x3 conv
        out1 = self.move1(x)

        out1 = self.conv_3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 += x

        out1 = self.rprelu1(out1)

        # 1x1 conv
        out2 = self.move2(out1)

        if self.in_channels == self.out_channels:
            out2 = self.conv_1x1(out2)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.in_channels * 2 == self.out_channels

            out2_1 = self.conv_1x1_down1(out2)
            out2_2 = self.conv_1x1_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.rprelu2(out2)

        return out2