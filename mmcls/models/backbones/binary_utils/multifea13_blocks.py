import torch
import torch.nn as nn
from .binary_convs import BLConv2d, BConvWS2d
from .binary_functions import act_name_map, FeaExpand


class MultiFea_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 nonlinear=('identity', 'hardtanh'), fea_num=1, mode='1', thres=None, **kwargs):
        super(MultiFea_Block, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample
        self.fea_num = fea_num

        self.fexpand1 = FeaExpand(expansion=fea_num, mode=mode, in_channels=in_channels, thres=thres)
        self.conv1 = BLConv2d(in_channels * fea_num, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear11 = self._build_act(nonlinear[0])
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear12 = self._build_act(nonlinear[1])
        self.fexpand2 = FeaExpand(expansion=fea_num, mode=mode, in_channels=out_channels, thres=thres)
        self.conv2 = BLConv2d(out_channels * fea_num, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.nonlinear21 = self._build_act(nonlinear[0])
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = self._build_act(nonlinear[1])


    def _build_act(self, act_name):
        if act_name == 'identity':
            return nn.Sequential()
        elif act_name == 'abs':
            return torch.abs
        elif act_name == 'prelu':
            return nn.PReLU(self.out_channels)
        else:
            return act_name_map[act_name]()


    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.nonlinear11(out)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear12(out)

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.nonlinear21(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear22(out)

        return out
