import torch
import math
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class EDEHook(Hook):

    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        
    def before_epoch(self, runner):
        k, t = get_kt(runner.epoch, self.total_epoch)
        for m in runner.model.modules():
            if hasattr(m, 'ede'):
                m.ede(k, t)


def get_kt(epoch, total_epoch, tmin=0.1, tmax=10):
    t = tmin * torch.pow(torch.tensor([10], dtype=torch.float), epoch/total_epoch * torch.log(torch.tensor([tmax/tmin], dtype=torch.float)) / math.log(10))
    k = torch.clamp_min(torch.tensor([1])/t, 1.0)
    return k.cuda(), t.cuda()