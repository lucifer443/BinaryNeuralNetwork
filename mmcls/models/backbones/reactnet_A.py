import torch.nn as nn
import logging
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner import load_checkpoint

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .resnet import ResLayer
from .binary_utils.rprelu_block import RANetBlockB

def build_act(name):
    name_map = {'hardtanh': nn.Hardtanh, 'relu': nn.ReLU, 'prelu': nn.PReLU}
    if name.lower() not in name_map:
        raise ValueError(f'Unknown activation function : {name}')
    return name_map[name]

@BACKBONES.register_module()
class MobileArch(nn.Module):
    """MobileNet architecture"""

    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, stride.
    arch_settings = {'ReActNet-A': (RANetBlockB, [[64, 1, 1], [128, 2, 2], [256, 2, 2], [512, 6, 2], [1024, 2, 2]]),
                     }

    def __init__(self,
                 arch,
                 out_indices=(4,),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 stem_act=None,
                 stem_channels=32,
                 norm_eval=False,
                 with_cp=False, 
                 Expand_num = 1,
                 rpgroup = 1,
                 gp = 1,
                 binary_type=(True, True),
                 **kwargs):
        super(MobileArch, self).__init__()
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 5):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 5). But received {index}')

        if frozen_stages not in range(-1, 5):
            raise ValueError('frozen_stages must be in range(-1, 8). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = stem_channels

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        activation = build_act(stem_act) if stem_act else None
        if activation:
            if activation == nn.PReLU:
                self.stem_act = activation(stem_channels)
            else:
                self.stem_act = activation(inplace=True)
        else:
            self.stem_act = None
        self.layers = []

        self.block, self.layers_cfg = self.arch_settings[arch]
        for i, layer_cfg in enumerate(self.layers_cfg):
            out_channels, num_blocks, stride = layer_cfg
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,                 
                Expand_num = Expand_num,
                rpgroup =  rpgroup,
                gp = gp,
                binary_type = binary_type,
                **kwargs)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

    def make_layer(self, out_channels, num_blocks, stride,Expand_num, rpgroup, gp, **kwargs):
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(
                self.block(
                    self.in_channels,
                    out_channels,
                    stride,
                    Expand_num,
                    rpgroup,
                    gp,
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
        x = self.conv1(x)
        if self.stem_act:
            x = self.stem_act(x)

        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)

        return x

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileArch, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()