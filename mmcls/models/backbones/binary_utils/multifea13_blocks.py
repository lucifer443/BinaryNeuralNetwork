import torch
import torch.nn as nn
from .binary_convs import BLConv2d, BConvWS2d
from .binary_functions import act_name_map, FeaExpand, LearnableScale, RPRelu, CfgLayer, DPReLU, NPReLU


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
        elif act_name == 'rprelu':
            return RPRelu(self.out_channels)
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


class MF1Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, nonlinear=('prelu', 'identity'), **kwargs):
        super(MF1Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fea_num = 2

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)
        self.fexpand1 = FeaExpand(expansion=self.fea_num, mode='5', thres=(-0.55, 0.55))
        self.conv_3x3 = BLConv2d(in_channels * self.fea_num, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear11 = self._build_act(nonlinear[0], in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.nonlinear12 = self._build_act(nonlinear[1], in_channels)
        self.fexpand2 = FeaExpand(expansion=self.fea_num, mode='5', thres=(-0.55, 0.55))
        self.conv_1x1 = BLConv2d(in_channels * self.fea_num, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.nonlinear21 = self._build_act(nonlinear[0], out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = self._build_act(nonlinear[1], out_channels)

    def _build_act(self, act_name, channels):
        if act_name == 'identity':
            return nn.Sequential()
        elif act_name == 'abs':
            return torch.abs
        elif act_name == 'prelu':
            return nn.PReLU(channels)
        elif act_name == 'prelu_pi=1':
            return nn.PReLU(channels, init=1.0)
        elif act_name == 'prelu_one_pi=1':
            return nn.PReLU(1, init=1.0)
        elif act_name == 'rprelu':
            return RPRelu(channels)
        elif act_name == 'rprelu_pi=1':
            return RPRelu(channels, prelu_init=1.0)
        elif act_name == 'bp':
            return CfgLayer(channels, config='bp', prelu_init=1.0)
        elif act_name == 'pb':
            return CfgLayer(channels, config='pb', prelu_init=1.0)
        elif act_name == 'bias':
            return CfgLayer(channels, config='b')
        elif act_name == 'bpbpb':
            return CfgLayer(channels, config='bpbpb', prelu_init=1.0)
        elif act_name == 'dprelu':
            return DPReLU(channels)
        elif act_name == 'nprelu':
            return NPReLU(channels)
        elif act_name == 'scale':
            return LearnableScale(channels)
        elif act_name == 'scale_one':
            return LearnableScale(1)
        else:
            return act_name_map[act_name]()

    def forward(self, x):
        # conv 3x3 (降采样)
        if self.stride == 2:
            identity = self.pooling(x)
        else:
            identity = x
        out1 = self.fexpand1(x)
        out1 = self.conv_3x3(out1)
        out1 = self.nonlinear11(out1)
        out1 = self.bn1(out1)
        out1 += identity
        out1 = self.nonlinear12(out1)

        # conv 1x1 (升维)
        if self.in_channels == self.out_channels:
            identity = out1
        else:
            assert self.in_channels * 2 == self.out_channels
            identity = torch.cat([out1] * 2, dim=1)
        out2 = self.fexpand2(out1)
        out2 = self.conv_1x1(out2)
        out2 = self.nonlinear21(out2)
        out2 = self.bn2(out2)
        out2 += identity
        out2 = self.nonlinear22(out2)

        return out2


class MF11Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(MF11Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fea_num = 1

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)
        self.fexpand1 = FeaExpand(expansion=self.fea_num, mode='1', thres=(-0.55, 0.55))
        self.conv_3x3 = BLConv2d(in_channels * self.fea_num, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear1 = nn.PReLU(in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.fexpand2 = FeaExpand(expansion=self.fea_num, mode='1', thres=(-0.55, 0.55))
        self.conv_1x1 = BLConv2d(in_channels * self.fea_num, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # conv 3x3 (降采样)
        if self.stride == 2:
            identity = self.pooling(x)
        else:
            identity = x
        out1 = self.fexpand1(x)
        out1 = self.conv_3x3(out1)
        out1 = self.nonlinear1(out1)
        out1 = self.bn1(out1)
        out1 += identity

        # conv 1x1 (升维)
        if self.in_channels == self.out_channels:
            identity = out1
        else:
            assert self.in_channels * 2 == self.out_channels
            identity = torch.cat([out1] * 2, dim=1)
        out2 = self.fexpand2(out1)
        out2 = self.conv_1x1(out2)
        out2 = self.nonlinear2(out2)
        out2 = self.bn2(out2)
        out2 += identity

        return out2


# class MF1Block(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1, nonlinear=('prelu', 'identity'), **kwargs):
#         super(MF1Block, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.stride = stride
#         self.fea_num = 2

#         if self.stride == 2:
#             self.pooling = nn.AvgPool2d(2, 2)
#         self.fexpand1 = FeaExpand(expansion=self.fea_num, mode='5', thres=(-0.55, 0.55))
#         self.conv_3x3 = BLConv2d(in_channels * self.fea_num, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
#         self.nonlinear1 = self._build_act(nonlinear[0], in_channels)
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.nonlinear12 = self._build_act(nonlinear[1], in_channels)
#         self.fexpand2 = FeaExpand(expansion=self.fea_num, mode='5', thres=(-0.55, 0.55))
#         self.conv_1x1 = BLConv2d(in_channels * self.fea_num, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
#         self.nonlinear2 = self._build_act(nonlinear[0], out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.nonlinear22 = self._build_act(nonlinear[1], out_channels)

#     def _build_act(self, act_name, channels):
#         if act_name == 'identity':
#             return nn.Sequential()
#         elif act_name == 'abs':
#             return torch.abs
#         elif act_name == 'prelu':
#             return nn.PReLU(channels)
#         elif act_name == 'rprelu':
#             return RPRelu(channels)
#         elif act_name == 'rprelu_pi=1':
#             return RPRelu(channels, prelu_init=1.0)
#         else:
#             return act_name_map[act_name]()

#     def forward(self, x):
#         # conv 3x3 (降采样)
#         if self.stride == 2:
#             identity = self.pooling(x)
#         else:
#             identity = x
#         out1 = self.fexpand1(x)
#         out1 = self.conv_3x3(out1)
#         out1 = self.nonlinear1(out1)
#         out1 = self.bn1(out1)
#         out1 += identity
#         out1 = self.nonlinear11(out1)

#         # conv 1x1 (升维)
#         if self.in_channels == self.out_channels:
#             identity = out1
#         else:
#             assert self.in_channels * 2 == self.out_channels
#             identity = torch.cat([out1] * 2, dim=1)
#         out2 = self.fexpand2(out1)
#         out2 = self.conv_1x1(out2)
#         out2 = self.nonlinear2(out2)
#         out2 = self.bn2(out2)
#         out2 += identity
#         out2 = self.nonlinear22(out2)

#         return out2