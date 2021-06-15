import logging

import torch.nn as nn

from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner import load_checkpoint

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .binary_utils.multifea13_blocks import MF1Block, MF11Block


@BACKBONES.register_module()
class MFNet(nn.Module):
    """MobileNet architecture"""

    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, stride.
    arch_settings = {
        'mf_1': (MF1Block, [[64, 2, 2], [128, 3, 2], [256, 7, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_2': (MF1Block, [[64, 2, 2], [128, 2, 2], [256, 3, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_2_c2': (MF11Block, [[90, 2, 2], [180, 2, 2], [360, 3, 2], [720, 1, 1], [1440, 1, 2]]),
    }

    def __init__(self,
                 arch,
                 binary_type=(True, False),
                 block_act=('prelu', 'identity'),
                 binary_type_cfg=None,
                 out_indices=(4,),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 stem_conv_ks=7,
                 in_channels=3,
                 stem_channels=32,
                 norm_eval=False,
                 with_cp=False, **kwargs):
        super(MFNet, self).__init__()
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 5):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 5). But received {index}')

        # if frozen_stages not in range(-1, 5):
        #     raise ValueError('frozen_stages must be in range(-1, 8). '
        #                      f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = stem_channels

        self.stem_conv = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=stem_conv_ks,
            stride=2,
            padding=stem_conv_ks // 2,
            bias=False)
        self.stem_act = nn.PReLU(stem_channels)
        self.stem_bn = nn.BatchNorm2d(stem_channels)

        self.layers = []

        self.block, self.layers_cfg = self.arch_settings[arch]
        for i, layer_cfg in enumerate(self.layers_cfg):
            if binary_type_cfg:
                stage_binary_type = binary_type_cfg[i]
            else:
                stage_binary_type = binary_type
            out_channels, num_blocks, stride = layer_cfg
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                binary_type=stage_binary_type,
                nonlinear=block_act,
                **kwargs)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)
        
        self._freeze_stages()

    def make_layer(self, out_channels, num_blocks, stride, binary_type=(True, False), **kwargs):
        layers = []
        for i in range(num_blocks):
            if isinstance(binary_type[0], bool):
                block_binary_type = binary_type
            elif isinstance(binary_type[0], tuple):
                block_binary_type = binary_type[i]
            if i >= 1:
                stride = 1
            layers.append(
                self.block(
                    self.in_channels,
                    out_channels,
                    stride,
                    binary_type=block_binary_type,
                    **kwargs))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

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
        x = self.stem_conv(x)
        x = self.stem_act(x)
        x = self.stem_bn(x)

        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)

        return x

    def _freeze_stages(self):
        if self.frozen_stages != -1:
            for param in self.stem_conv.parameters():
                param.requires_grad = False
            for param in self.stem_bn.parameters():
                param.requires_grad = False
            for param in self.stem_act.parameters():
                param.requires_grad = False
            for i in self.frozen_stages:
                layer = getattr(self, f'layer{i}')
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(MFNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
