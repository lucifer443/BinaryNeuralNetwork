import torch.nn as nn
import logging
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner import load_checkpoint

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .resnet import ResLayer
from .binary_utils.binary_blocks import IRNetBlock, RANetBlockA, RANetBlockB, StrongBaselineBlock, RealToBinaryBlock, RealToBinaryFPBlock
from .binary_utils.expand_block import *

def build_act(name):
    name_map = {'hardtanh': nn.Hardtanh, 'relu': nn.ReLU, 'prelu': nn.PReLU}
    if name.lower() not in name_map:
        raise ValueError(f'Unknown activation function : {name}')
    return name_map[name]

@BACKBONES.register_module()
class ResArch(BaseBackbone):

    arch_settings = {
        "IRNet-18": (IRNetBlock, (2, 2, 2, 2)),
        "IRNet-34": (IRNetBlock, (3, 4, 6, 3)),
        "ReActNet-18": (RANetBlockA, (2, 2, 2, 2)),
        "ReActNet-34": (RANetBlockA, (3, 4, 6, 3)),
        "StrongBaseline": (StrongBaselineBlock, (2, 2, 2, 2)),
        "Real2Bi": (RealToBinaryBlock, (2, 2, 2, 2)),
        "Real2BiFP": (RealToBinaryFPBlock, (2, 2, 2, 2))
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3,),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 binary_type=(True, True),
                 stem_act=None,
                 zero_init_residual=False):
        super(ResArch, self).__init__()
        if arch not in self.arch_settings:
            raise KeyError(f'invalid arch type {arch} for resnet')
        self.arch = arch
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[arch]
        self.stage_blocks = stage_blocks[:num_stages]
        # self.expansion = get_expansion(self.block, expansion)
        self.expansion = 1 if not expansion else expansion

        self.activation = build_act(stem_act) if stem_act else None
    
        self._make_stem_layer(in_channels, stem_channels, self.activation)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                binary_type=binary_type,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            _in_channels = _out_channels
            _out_channels *= 2
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
            
        self.extra_downsample = nn.AvgPool2d(kernel_size=2, stride=2) if strides[-1] == 1 else None

        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels, activation):
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            if activation:
                if activation == nn.PReLU:
                    self.stem_act = activation(stem_channels)
                else:
                    self.stem_act = activation(inplace=True)
            else:
                self.stem_act = None
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    constant_init(m.norm2, 0)

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            if self.stem_act:
                x = self.stem_act(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
#             print(x.shape)
            x = res_layer(x)
            if i == 3 and self.extra_downsample:
                x = self.extra_downsample(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResArch, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_module()
class MobileArch(nn.Module):
    """MobileNet architecture"""

    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, stride.
    arch_settings = {'ReActNet-A': (RANetBlockB, [[64, 1, 1], [128, 2, 2], [256, 2, 2], [512, 6, 2], [1024, 2, 2]]),
                     'ExpandNet-1': (EpBlockA, [[64, 2, 2], [128, 3, 2], [256, 7, 2], [1024, 1, 2]]),
                     'ExpandNet-2': (EpBlockA, [[48, 1, 2], [96, 2, 2], [192, 3, 2], [768, 1, 2]]),
                     'ExpandNet-3': (EpBlockB, [[64, 1, 2], [128, 2, 2], [256, 3, 2], [1024, 1, 2]]),
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
                 with_cp=False, **kwargs):
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
                stride=stride, **kwargs)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

    def make_layer(self, out_channels, num_blocks, stride, **kwargs):
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(
                self.block(
                    self.in_channels,
                    out_channels,
                    stride, **kwargs))
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
