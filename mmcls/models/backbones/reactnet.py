import torch.nn as nn
import logging
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner import load_checkpoint

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .resnet import ResLayer
from .binary_utils.binary_blocks import IRNetBlock, RANetBlockA, CM1Block, CM2Block, CM3Block,RANetBlock_A

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2
class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)

        return out
@BACKBONES.register_module()
class reactnet_A(BaseBackbone):
    def __init__(self, binary_type=(True, True)):
        super(reactnet_A, self).__init__()
        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], stride=2))
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(RANetBlock_A(stage_out_channel[i-1], stage_out_channel[i], stride=2,binary_type=binary_type))
            else:
                self.feature.append(RANetBlock_A(stage_out_channel[i-1], stage_out_channel[i], stride=1,binary_type=binary_type))


    def forward(self, x):
        for i, block in enumerate(self.feature):
            x = block(x)
        return x

