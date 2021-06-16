import torch
import torch.nn as nn
from .binary_convs import IRConv2dnew, RAConv2d ,IRConv2d_bias ,IRConv2d_bias_x2,IRConv2d_bias_x2x,BLConv2d,StrongBaselineConv2d
from .binary_functions import RPRelu, LearnableBias, LearnableScale, AttentionScale,Expandx,GPRPRelu

class RPStrongBaselineBlock(nn.Module):
    """Strong baseline block from real-to-binary net"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, Expand_num=1,rpgroup=1,**kwargs):
        super(RPStrongBaselineBlock, self).__init__()

        if rpgroup == 1:
            self.rpmode = 1
        elif rpgroup == 2:
            self.rpmode = out_channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = BLConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.scale1 = LearnableScale(out_channels)
        self.nonlinear1 = RPRelu(self.rpmode)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = BLConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.scale2 = LearnableScale(out_channels)
        self.nonlinear2 = RPRelu(self.rpmode)
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

class RANetBlockA(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, Expand_num=1,rpgroup=1,gp=1,**kwargs):
        super(RANetBlockA, self).__init__()
        if rpgroup == 1:
            self.rpmode = out_channels
            self.prelu1 = RPRelu(self.rpmode)
            self.prelu2 = RPRelu(self.rpmode)
        elif rpgroup == 2:
            self.rpmode = out_channels
            if out_channels ==256:
                self.prelu1 = GPRPRelu(self.rpmode,gp=gp)
                self.prelu2 = GPRPRelu(self.rpmode,gp=gp)
            elif out_channels ==512 :
                self.prelu1 = GPRPRelu(self.rpmode,gp=64)
                self.prelu2 = GPRPRelu(self.rpmode,gp=64)
            else:
                self.prelu1 = RPRelu(self.rpmode)
                self.prelu2 = RPRelu(self.rpmode)
        elif rpgroup == 3:
            self.rpmode = out_channels
            if out_channels == 512:
                self.prelu1 = GPRPRelu(self.rpmode,gp=gp)
                self.prelu2 = GPRPRelu(self.rpmode,gp=gp)
            else:
                self.prelu1 = RPRelu(out_channels)
                self.prelu2 = RPRelu(out_channels)

        self.conv1 = RAConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = RAConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
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
        out = self.prelu1(out)

        identity = out
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.prelu2(out)
        return out

class Baseline15Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,Expand_num=1,rpgroup=1,gp=1, **kwargs):
        super(Baseline15Block, self).__init__()
        if rpgroup == 1:
            self.rpmode = 1
        elif rpgroup == 2:
            self.rpmode = out_channels
        self.conv1 = RAConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = RPRelu(self.rpmode)
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.conv2 = RAConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = RPRelu(self.rpmode)
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


class Baseline13_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, Expand_num=1,  rpgroup=1,gp=1,**kwargs):
        super(Baseline13_Block, self).__init__()
        self.out_channels = out_channels
        if rpgroup == 1:
            self.rpmode = 1
        elif rpgroup == 2:
            self.rpmode = out_channels

        self.stride = stride
        self.downsample = downsample
        self.Expand_num = Expand_num
        self.fexpand1 = Expandx(Expand_num=Expand_num, in_channels=in_channels)
        self.conv1 = BLConv2d(in_channels * Expand_num, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        #self.nonlinear11 = LearnableScale(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear12 = RPRelu(self.rpmode)
        self.fexpand2 = Expandx(Expand_num=Expand_num, in_channels=out_channels)
        self.conv2 = BLConv2d(out_channels * Expand_num, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        #self.nonlinear21 = LearnableScale(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = RPRelu(self.rpmode)


    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)   
        #out = self.nonlinear11(out)
        out = self.nonlinear12(out)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        #out = self.nonlinear21(out)
        out = self.nonlinear22(out)
        out = self.bn2(out)
        out += identity


        return out