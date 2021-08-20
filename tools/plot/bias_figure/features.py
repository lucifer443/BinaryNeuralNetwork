import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mmcls.apis import inference_model, init_model
from mmcls.models.backbones.binary_utils.binary_convs import RAConv2d
from mmcls.models.backbones.binary_utils.binary_functions import *


class Features(object):
    def __init__(self, config, ckpt):
        self.RAconv_in = []
        self.RAconv_out = []
        self.dprelu_in = []
        self.dprelu_out = []
        self.bn_in = []
        self.bn_out = []
        self.fexpand_in = []
        self.fexpand_out = []
        self.model = init_model(config, ckpt, device='cuda')
        for m in self.model.modules():
            # print(type(m))
            if isinstance(m, RAConv2d):
                m.register_forward_hook(hook=self.get_RAconv_inout)
            # if isinstance(m, DPReLU):
            #     m.register_forward_hook(hook=self.get_dprelu_inout)

            # if isinstance(m, nn.BatchNorm2d):
            #     m.register_forward_hook(hook=self.get_bn_inout)
            # if isinstance(m, FeaExpand):
            #     m.register_forward_hook(hook=self.get_fexpand_inout)
    
    def get_RAconv_inout(self, module, fea_in, fea_out):
        self.RAconv_in.append(fea_in[0].cpu())
        self.RAconv_out.append(fea_out.cpu())
    
    def get_dprelu_inout(self, module, fea_in, fea_out):
        self.dprelu_in.append(fea_in[0].cpu())
        self.dprelu_out.append(fea_out.cpu())

    def get_bn_inout(self, module, fea_in, fea_out):
        self.bn_in.append(fea_in[0].cpu())
        self.bn_out.append(fea_out.cpu())
    
    def get_fexpand_inout(self, module, fea_in, fea_out):
        self.fexpand_in.append(fea_in[0].cpu())
        self.fexpand_out.append(fea_out.cpu())
    
    def save_features(self, img_path):
        result = inference_model(self.model, img_path)
        print(f'RAconv: {len(self.RAconv_in)}')
        # print(f'dprelu: {len(self.dprelu_in)}')
        # print(f'bn: {len(self.bn_in)}')
        # print(f'fexpand: {len(self.fexpand_in)}')

