from ..builder import BACKBONES
from .base_backbone import BaseBackbone
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

@BACKBONES.register_module()
class BinaryResNet18(BaseBackbone):

    def __init__(self, model_path, norm_eval=False, **kwargs):
        super(BinaryResNet18, self).__init__()
        import sys
        sys.path.append(model_path+"/modules")
        sys.path.append(model_path)
        from resnet import resnet18
        self.base_model = resnet18(**kwargs).cuda()

    def forward(self, x):
        if x.device.type == 'cpu':
            x = x.cuda()
        features = self.base_model.forward(x)
        return  features

@BACKBONES.register_module()
class BinaryResNet34(BaseBackbone):

    def __init__(self, model_path, norm_eval=False, **kwargs):
        super(BinaryResNet34, self).__init__()
        import sys
        sys.path.append(model_path+"/modules")
        sys.path.append(model_path)
        from resnet import resnet34
        self.base_model = resnet34(**kwargs).cuda()

    def forward(self, x):
        if x.device.type == 'cpu':
            x = x.cuda()
        features = self.base_model.forward(x)
        return  features