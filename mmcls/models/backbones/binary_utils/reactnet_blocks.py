import torch
import torch.nn as nn
from .binary_convs import IRConv2d, RAConv2d
from .binary_functions import RPRelu, LearnableBias
    

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


class ReActFATBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(ReActFATBlock, self).__init__()

        self.move1 = LearnableBias(in_channels)
        self.conv_3x3 = RAFATConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.rprelu1 = RPRelu(in_channels)

        self.move2 = LearnableBias(in_channels)

        if in_channels == out_channels:
            self.conv_1x1 = RAFATConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1_down1 = RAFATConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.conv_1x1_down2 = RAFATConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
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