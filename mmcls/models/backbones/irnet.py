import math
import torch.nn as nn
import logging
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner import load_checkpoint

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .resnet import ResLayer
from .binary_utils.irnet_blocks import (IRNetBlock, IRNetH1Block, IRNetH2Block, IRNetH1aBlock,
                                        IRNetG2Block, IRNetG3Block, IRNetG4Block, IRNetG5Block,
                                        IRNetG6Block, IRNetG7Block, IRNetG8Block,
                                        IRNetG3nBlock,
                                        IRNetG3swBlock,
                                        IRNetGB4Block, IRNetGBa4Block,
                                        IRNetShiftBlock,
                                       )

def build_act(name):
    name_map = {'hardtanh': nn.Hardtanh, 'relu': nn.ReLU}
    if name.lower() not in name_map:
        raise ValueError(f'Unknown activation function : {name}')
    return name_map[name]

@BACKBONES.register_module()
class IRNet(BaseBackbone):

    arch_settings = {
        "irnet_r18": (IRNetBlock, (2, 2, 2, 2)),
        "irnet_g2_r18": (IRNetG2Block, (2, 2, 2, 2)),
        "irnet_g3_r18": (IRNetG3Block, (2, 2, 2, 2)),
        "irnet_g3n_r18": (IRNetG3nBlock, (2, 2, 2, 2)),
        "irnet_g4_r18": (IRNetG4Block, (2, 2, 2, 2)),
        "irnet_g5_r18": (IRNetG5Block, (2, 2, 2, 2)),
        "irnet_g6_r18": (IRNetG6Block, (2, 2, 2, 2)),
        "irnet_g7_r18": (IRNetG7Block, (2, 2, 2, 2)),
        "irnet_g8_r18": (IRNetG8Block, (2, 2, 2, 2)),
        "irnet_h1_r18": (IRNetH1Block, (2, 2, 2, 2)),
        "irnet_h1a_r18": (IRNetH1aBlock, (2, 2, 2, 2)),
        "irnet_h2_r18": (IRNetH2Block, (2, 2, 2, 2)),
        "irnet_g3sw_r18": (IRNetG3swBlock, (2, 2, 2, 2)),
        "irnet_gb4_r18": (IRNetGB4Block, (2, 2, 2, 2)),
        "irnet_gba4_r18": (IRNetGBa4Block, (2, 2, 2, 2)),
        "irnet_shift_r18": (IRNetShiftBlock, (2, 2, 2, 2)),
    }

    def __init__(self,
                 arch,
                 group_stages=None,
                 shift=0.0,
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
        super(IRNet, self).__init__()
        if arch not in self.arch_settings:
            raise KeyError(f'invalid arch type {arch} for irnet')
        self.arch = arch
        # 多分支conv必须指定需要多分支的stage(形参group_stages不能为None)，否则不会对任何stage进行多分支操作
        # 多分支conv的stem_channels和base_channels使用默认的64即可
        # 每个stage的具体通道数会根据分支个数自动调整，以保证总conv计算量与不进行分支的conv一致
        # 注意需要手动调整config中head的in_channels数值，以匹配实际的输出通道数
        self.group_stages=group_stages
        if self.group_stages:
            self.groups = int(arch[7:8]) # 多分支conv的分支个数由arch中g后面的数字指定
        self.shift = shift # sign阈值的正向偏移量，实际sign函数为sign(x + shift)
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
            real_block = self.block
            if group_stages:
                if i not in self.group_stages:
                    ''' 当前stage不拆成多分支 '''
                    real_block = IRNetBlock
                elif i == self.group_stages[0]:
                    ''' 根据分支数减少通道数，保证总计算量不变 '''
                    _out_channels = int(_out_channels / math.sqrt(self.groups))
            res_layer = self.make_res_layer(
                block=real_block,
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
                norm_cfg=norm_cfg,
                shift=self.shift)
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
            self.stem_act = activation(inplace=True) if activation else None
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
        super(IRNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


class MobileArch(nn.Module):
    pass
