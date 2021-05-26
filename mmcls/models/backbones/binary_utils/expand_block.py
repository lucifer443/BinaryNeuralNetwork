import torch
import torch.nn as nn
from .binary_convs import IRConv2d, RAConv2d, STEConv2d
from .binary_functions import RPRelu, LearnableBias, LearnableScale, AttentionScale, BiasExpand



class EpBlockA(nn.Module):
    ratio = 2

    def __init__(self, inplanes, planes, stride=1, **kwargs):
        super(EpBlockA, self).__init__()
        self.expand1 = BiasExpand(self.ratio)
        if planes != inplanes:
            self.expand2 = BiasExpand(planes // inplanes)
            self.conv1 = RAConv2d(inplanes * planes // inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False,
                                  **kwargs)
        else:
            self.conv1 = RAConv2d(inplanes * self.ratio, planes, kernel_size=1, stride=1, padding=0, bias=False,
                                  **kwargs)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = RAConv2d(planes * self.ratio, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                              **kwargs)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = RAConv2d(planes * self.ratio, planes, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.bn3 = nn.BatchNorm2d(planes)

        self.nonlinear = nn.Hardtanh(inplace=True)

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):

        if self.inplanes == self.planes:
            identity = x
            out1 = self.expand1(x)
        else:
            identity = torch.cat([x] * (self.planes // self.inplanes), dim=1)
            out1 = self.expand2(x)
        out1 = self.conv1(out1)
        out1 = self.bn1(out1)
        out1 = identity + out1
        out1 = self.nonlinear(out1)

        if self.stride == 2:
            identity = self.pooling(out1)
        else:
            identity = out1

        out2 = self.expand1(out1)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        out2 = identity + out2
        out2 = self.nonlinear(out2)

        out3 = self.expand1(out2)
        out3 = self.conv3(out3)
        out3 = self.bn3(out3)
        out3 = identity + out3
        out3 = self.nonlinear(out3)
        return out3

class EpBlockB(nn.Module):
    ratio = 2

    def __init__(self, inplanes, planes, stride=1, **kwargs):
        super(EpBlockB, self).__init__()
        self.expand1 = BiasExpand(self.ratio)
        if planes != inplanes:
            self.expand2 = BiasExpand(planes // inplanes)
            self.conv1 = RAConv2d(inplanes * planes // inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False,
                                  **kwargs)
        else:
            self.conv1 = RAConv2d(inplanes * self.ratio, planes, kernel_size=1, stride=1, padding=0, bias=False,
                                  **kwargs)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = RAConv2d(planes * self.ratio, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                              **kwargs)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = RAConv2d(planes * self.ratio, planes, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.bn3 = nn.BatchNorm2d(planes)

        self.nonlinear = nn.Hardtanh(inplace=True)

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):

        if self.inplanes == self.planes:
            identity = x
            out1 = self.expand1(x)
        else:
            identity = torch.cat([x] * (self.planes // self.inplanes), dim=1)
            out1 = self.expand2(x)
        out1 = self.conv1(out1)
        out1 = self.bn1(out1)
        out1 = identity + out1
        out1 = self.nonlinear(out1)

        if self.stride == 2:
            identity = self.pooling(out1)
        else:
            identity = out1

        out2 = self.expand1(out1)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        out2 = identity + out2
        out2 = self.nonlinear(out2)

        out3 = self.expand1(out2)
        out3 = self.conv3(out3)
        out3 = self.bn3(out3)
        out3 = identity + out3
        out3 = self.nonlinear(out3)
        return out3