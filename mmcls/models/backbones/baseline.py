import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone

from .baseline_blocks.baseline_blocks import (
    Baseline11Block, Baseline12Block, Baseline13Block, Baseline14Block, Baseline15Block,
    Baseline21Block, Baseline22Block, Baseline23Block, Baseline24Block,
    Baseline11sBlock, BaselineStrongBlock,
    Baseline13clipBlock)
from .baseline_blocks.baseline_ste_blocks import (
    Baseline11STEBlock, Baseline12STEBlock, Baseline13STEBlock, Baseline14STEBlock, Baseline15STEBlock,
    Baseline21STEBlock, Baseline22STEBlock,)
from .binary_utils.multifea_blocks import (
    MultiFea_1_Block,
    MultiFea_2_1_Block, MultiFea_3_1_Block, MultiFea_4_1_Block, MultiFea_5_1_Block,
    MultiFea_6_1_Block, MultiFea_7_1_Block, MultiFea_10_1_Block,
    MultiFea_2_2_Block, MultiFea_3_1c_Block,
    MultiFea13_3_1_Block,
    MultiFea13_3_3_Block, MultiFea13_3_3n_Block, MultiFea13_3_3c_Block, MultiFea13_3_3nc_Block,
    MultiFea_3_4_Block, MultiFea_3_4c_Block,)


def build_act(name):
    name_map = {
        'hardtanh': nn.Hardtanh,
        'relu': nn.ReLU,
        'prelu': nn.PReLU,
    }
    if name.lower() not in name_map:
        raise ValueError(f'Unknown activation function : {name}')
    return name_map[name]

def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 group_cfg=None,
                 branch_cfg=None,
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


@BACKBONES.register_module()
class Baseline(BaseBackbone):
    """ResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`_ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.

    Example:
        >>> from mmcls.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        'baseline_11': (Baseline11Block, (2, 2, 2, 2)),
        'baseline_12': (Baseline12Block, (2, 2, 2, 2)),
        'baseline_13': (Baseline13Block, (2, 2, 2, 2)),
        'baseline_13clip': (Baseline13clipBlock, (2, 2, 2, 2)),
        'baseline_14': (Baseline14Block, (2, 2, 2, 2)),
        'baseline_15': (Baseline15Block, (2, 2, 2, 2)),
        'baseline_11_ste': (Baseline11STEBlock, (2, 2, 2, 2)),
        'baseline_12_ste': (Baseline12STEBlock, (2, 2, 2, 2)),
        'baseline_13_ste': (Baseline13STEBlock, (2, 2, 2, 2)),
        'baseline_14_ste': (Baseline14STEBlock, (2, 2, 2, 2)),
        'baseline_15_ste': (Baseline15STEBlock, (2, 2, 2, 2)),
        'baseline_21_ste': (Baseline21STEBlock, (2, 2, 2, 2)),
        'baseline_22_ste': (Baseline22STEBlock, (2, 2, 2, 2)),
        'baseline_21': (Baseline21Block, (2, 2, 2, 2)),
        'baseline_22': (Baseline22Block, (2, 2, 2, 2)),
        'baseline_23': (Baseline23Block, (2, 2, 2, 2)),
        'baseline_24': (Baseline24Block, (2, 2, 2, 2)),
        'baseline_11s': (Baseline11sBlock, (2, 2, 2, 2)),
        'baseline_strong': (BaselineStrongBlock, (2, 2, 2, 2)),
        'mf_1': (MultiFea_1_Block, (2, 2, 2, 2)),
        'mf_2_1': (MultiFea_2_1_Block, (2, 2, 2, 2)),
        'mf_3_1': (MultiFea_3_1_Block, (2, 2, 2, 2)),
        'mf_4_1': (MultiFea_4_1_Block, (2, 2, 2, 2)),
        'mf_5_1': (MultiFea_5_1_Block, (2, 2, 2, 2)),
        'mf_6_1': (MultiFea_6_1_Block, (2, 2, 2, 2)),
        'mf_7_1': (MultiFea_7_1_Block, (2, 2, 2, 2)),
        'mf_10_1': (MultiFea_10_1_Block, (2, 2, 2, 2)),
        'mf_3_1c': (MultiFea_3_1c_Block, (2, 2, 2, 2)),
        'mf_2_2': (MultiFea_2_2_Block, (2, 2, 2, 2)),
        'mf13_3_1': (MultiFea13_3_1_Block, (2, 2, 2, 2)),
        'mf13_3_3': (MultiFea13_3_3_Block, (2, 2, 2, 2)),
        'mf13_3_3n': (MultiFea13_3_3n_Block, (2, 2, 2, 2)),
        'mf13_3_3c': (MultiFea13_3_3c_Block, (2, 2, 2, 2)),
        'mf13_3_3nc': (MultiFea13_3_3nc_Block, (2, 2, 2, 2)),
        'mf_3_4': (MultiFea_3_4_Block, (2, 2, 2, 2)),
        'mf_3_4c': (MultiFea_3_4c_Block, (2, 2, 2, 2)),
    }

    def __init__(self,
                 arch,
                 stage_setting=None,
                 binary_type=(True, True),
                 stem_act='relu',
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False,):
        super(Baseline, self).__init__()
        if arch not in self.arch_settings:
            raise KeyError(f'invalid arch type {arch} for baseline')
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

        # set stage_blocks from user config
        if stage_setting:
            self.stage_blocks = stage_setting
        # set stem activation method
        self.activation = build_act(stem_act) if stem_act else None

        self._make_stem_layer(in_channels, stem_channels)

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
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                binary_type=binary_type,)
            _in_channels = _out_channels
            _out_channels *= 2
            layer_name = f'stage{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
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
            if self.activation == nn.PReLU:
                self.stem_act = self.activation(self.stem_channels)
            else:
                self.stem_act = self.activation() if self.activation else None
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
            m = getattr(self, f'stage{i}')
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
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(Baseline, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
