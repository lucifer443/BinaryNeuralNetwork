from ..builder import BACKBONES
from .base_backbone import BaseBackbone
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

@BACKBONES.register_module()
class BiReal18(BaseBackbone):
    """binary resnet18 from xiayangyang"""

    def __init__(self, model_path, norm_eval=False, **kwargs):
        super(BiReal18, self).__init__()
        import sys
        sys.path.append(model_path)
        if "resnet" in model_path:
            from birealnet import birealnet18
            self.base_model = birealnet18().cuda()
        else:
            from reactnet import reactnet
            self.base_model = reactnet().cuda()

    def forward(self, x):
        if x.device.type == 'cpu':
            x = x.cuda()
        features = self.base_model.forward(x)
        return  features
