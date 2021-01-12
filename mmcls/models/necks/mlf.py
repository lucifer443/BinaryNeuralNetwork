import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import NECKS
from mmcv.cnn import constant_init, kaiming_init
from mmcls.models.backbones.binary_utils.binary_convs import IRConv2d, ANDConv2d


@NECKS.register_module()
class MultiLevelFuse(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling.
    We do not use `squeeze` as it will also remove the batch dimension
    when the tensor has a batch dimension of size 1, which can lead to
    unexpected errors.
    """

    act_conv = {"XNOR": IRConv2d, "AND": ANDConv2d}
    def __init__(self, in_channels, out_channels, conv_type="XNOR"):
        super(MultiLevelFuse, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.lateral_convs = nn.ModuleList()
        conv = self.act_conv[conv_type]
        for ic in in_channels:
            conv_bn = nn.Sequential(
                conv(ic, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.lateral_convs.append(conv_bn)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            n, _, h, w = inputs[0].shape
            outs = []
            for i, inp in enumerate(inputs):
                out = self.lateral_convs[i](inp)
                if i:
                    out = F.interpolate(out, size=[h, w], mode='bilinear')
                outs.append(out)
            outs = self.gap(sum(outs))
            outs = outs.view(n, -1)
        else:
            raise TypeError('neck inputs should be tuple')
        return outs
