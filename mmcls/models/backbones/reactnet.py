import logging

import torch.nn as nn

from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner import load_checkpoint

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .binary_utils.reactnet_blocks import ReActBlock


class firstconv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(firstconv3x3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)

        return out


@BACKBONES.register_module()
class ReActNet(BaseBackbone):
    # stage_out_channels
    out_chn = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2

    def __init__(self,
                 binary_type=(True, True)):
        super(ReActNet, self).__init__()
        self.feature = nn.ModuleList()
        for i in range(len(self.out_chn)):
            if i == 0:
                # 首层conv stride == 2
                self.feature.append(firstconv3x3(3, self.out_chn[i], stride=2))
            elif self.out_chn[i-1] != self.out_chn[i] and self.out_chn[i] != 64:
                # 输入输出通道数不同，需要升维
                # 除了第一个dw的stride为2，其余stage的stride都为1
                self.feature.append(
                    ReActBlock(self.out_chn[i-1], self.out_chn[i], stride=2, binary_type=binary_type))
            else:
                self.feature.append(
                    ReActBlock(self.out_chn[i-1], self.out_chn[i], stride=1, binary_type=binary_type))
    
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x):
        for i, block in enumerate(self.feature):
            x = block(x)

        return x

