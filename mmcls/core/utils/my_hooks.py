import torch
import math
from mmcv.runner import HOOKS, Hook
from mmcls.models.backbones.binary_utils.binary_convs import BLSTEConv2d


@HOOKS.register_module()
class WeightClipHook(Hook):

    def __init__(self, clip=1.5):
        assert clip > 0
        self.clip = clip
    
    def after_iter(self, runner):
        for m in runner.model.modules():
            if isinstance(m, BLSTEConv2d):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        with torch.no_grad():
                            param.set_(param.clamp(-1.5, 1.5))
