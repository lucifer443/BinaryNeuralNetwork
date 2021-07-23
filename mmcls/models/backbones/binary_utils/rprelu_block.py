import torch
import torch.nn as nn
from .binary_convs import IRConv2dnew, RAConv2d ,IRConv2d_bias ,IRConv2d_bias_x2,IRConv2d_bias_x2x,BLConv2d,StrongBaselineConv2d
from .binary_functions import RPRelu, LearnableBias, LearnableScale, AttentionScale,Expandx,GPRPRelu,MGPRPRelu,GPLearnableBias

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
            self.move1 = LearnableBias(in_channels)
            self.move2 = LearnableBias(out_channels)
        elif rpgroup == 2:
            self.rpmode = out_channels
            if out_channels ==256 and downsample is not None:
                self.prelu1 = GPRPRelu(self.rpmode,gp=gp)
                self.prelu2 = RPRelu(self.rpmode)
                self.move1 = GPLearnableBias(in_channels,gp=gp)
                self.move2 = LearnableBias(out_channels)
            elif out_channels ==512 :
                self.prelu1 = GPRPRelu(self.rpmode,gp=64)
                self.prelu2 = GPRPRelu(self.rpmode,gp=64)
                self.move1 = GPLearnableBias(in_channels,gp=gp)
                self.move2 = GPLearnableBias(out_channels,gp=gp)
            else:
                self.prelu1 = RPRelu(self.rpmode)
                self.prelu2 = RPRelu(self.rpmode)
                self.move1 = LearnableBias(in_channels)
                self.move2 = LearnableBias(out_channels)
        elif rpgroup == 3:
            self.rpmode = out_channels
            if out_channels == 512:
                self.prelu1 = GPRPRelu(self.rpmode,gp=gp)
                self.prelu2 = GPRPRelu(self.rpmode,gp=gp)
            else:
                self.prelu1 = RPRelu(out_channels)
                self.prelu2 = RPRelu(out_channels)
        elif rpgroup == 4:
            self.rpmode = out_channels
            self.prelu1 = nn.PReLU(self.rpmode)
            self.prelu2 = nn.PReLU(self.rpmode)

        
        self.conv1 = RAConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = RAConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x
        x = self.move1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.prelu1(out)

        identity = out
        
        out = self.move2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.prelu2(out)
        return out

class RANetBlockB(nn.Module):
    def __init__(self, inplanes, planes, stride=1, Expand_num=1,rpgroup=1,gp=1,**kwargs):
        super(RANetBlockB, self).__init__()
        #norm_layer = nn.BatchNorm2d
        if rpgroup == 1:
            self.prelu1 = RPRelu(inplanes)
            self.prelu2 = RPRelu(planes)
            #self.move1 = LearnableBias(inplanes)
            #self.move2 = LearnableBias(inplanes)
        elif rpgroup == 2:
            if planes == 51200:
                self.prelu1 = GPRPRelu(inplanes,gp=gp)
                self.prelu2 = GPRPRelu(planes,gp=gp)
            elif planes == 1024 :
                if inplanes != planes:
                    # self.prelu1 = RPRelu(inplanes)
                    # self.prelu2 = GPRPRelu(planes,gp=gp)
                    # self.move1 = LearnableBias(inplanes)
                    # self.move2 = GPLearnableBias(inplanes,gp=gp)
                    self.prelu1 = GPRPRelu(inplanes,gp=gp)
                    self.prelu2 = GPRPRelu(planes,gp=gp)
                    #self.move1 = GPLearnableBias(inplanes,gp=gp//2)
                    #self.move2 = GPLearnableBias(inplanes,gp=gp//2)
                else:
                    self.prelu1 = GPRPRelu(inplanes,gp=gp)
                    self.prelu2 = GPRPRelu(planes,gp=gp)
                    #self.move1 = GPLearnableBias(inplanes,gp=gp)
                    #self.move2 = GPLearnableBias(inplanes,gp=gp)
            else:
                self.prelu1 = RPRelu(inplanes)
                self.prelu2 = RPRelu(planes)
                #self.move1 = LearnableBias(inplanes)
                #self.move2 = LearnableBias(inplanes)

        
        self.binary_3x3 = RAConv2d(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(inplanes)


        

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

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):

        #out1 = self.move1(x)

        out1 = x+0.5
        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.prelu1(out1)

        #out2 = self.move2(out1)
        out2 = out1+0.5

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
        self.stride = stride
        self.downsample = downsample
        self.Expand_num = Expand_num
        self.fexpand1 = Expandx(Expand_num=Expand_num, in_channels=in_channels)
        self.conv1 = RAConv2d(in_channels * Expand_num, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        #self.nonlinear11 = LearnableScale(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        #self.nonlinear12 = RPRelu(self.rpmode)
        self.nonlinear11 = nn.PReLU(out_channels) 
        self.fexpand2 = Expandx(Expand_num=Expand_num, in_channels=out_channels)
        self.conv2 = RAConv2d(out_channels * Expand_num, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        #self.nonlinear21 = LearnableScale(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #self.nonlinear22 = RPRelu(self.rpmode)


    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)   
        out = self.nonlinear11(out)
        #out = self.nonlinear12(out)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        #out = self.nonlinear21(out)
        #out = self.nonlinear22(out)
        out = self.bn2(out)
        out += identity


        return out
