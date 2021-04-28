from argparse import ArgumentParser
from mmcls.apis import inference_model, init_model, show_result_pyplot

import torch
import torch.nn as nn
from mmcls.models.backbones.binary_utils.binary_convs import *
from mmcls.models.backbones.binary_utils.binary_blocks import *
from mmcls.models.backbones.binary_utils.binary_functions import *

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math
import os

names = []
features = []

irconv_fea_in = []
irconv_fea_out = []

raconv_fea_in = []
raconv_fea_out = []

blconv_fea_in = []
blconv_fea_out = []

def get_features(module, fea_in, fea_out):
    global names
    global features
    names.append(module.__class__.__name__)
    features.append(fea_out)

def get_irconv_inout(module, fea_in, fea_out):
    global irconv_fea_in
    global irconv_fea_out
    # print(type(fea_in), type(fea_out))
    # fea_in is a tuple, fea_out is a tensor
    # print(module, fea_in[0].max())
    irconv_fea_in.append(fea_in[0].cpu())
    irconv_fea_out.append(fea_out.cpu())

def get_raconv_inout(module, fea_in, fea_out):
    global raconv_fea_in
    global raconv_fea_out
    raconv_fea_in.append(fea_in[0].cpu())
    raconv_fea_out.append(fea_out.cpu())

def get_blconv_inout(module, fea_in, fea_out):
    global blconv_fea_in
    global blconv_fea_out
    blconv_fea_in.append(fea_in[0].cpu())
    blconv_fea_out.append(fea_out.cpu())

def compute_ratio(fea):
    fea = fea.sign().flatten()
    diff = fea @ torch.ones(fea.shape).T
    total = fea.numel()
    return diff / total

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # assume the checkpoint file is the same name of config file
    # and in the dir checkpoint/
    arch_name = args.config.split('/')[-1].rstrip('.py')
    img_name = args.img.split('/')[-1].rstrip('.JPEG')

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # add my own hooks
    for m in model.modules():
        # print(type(m))
        if isinstance(m, IRConv2d_bias_x2x):
            m.register_forward_hook(hook=get_irconv_inout)
        if isinstance(m, RAConv2d):
            m.register_forward_hook(hook=get_raconv_inout)

    # test a single image
    result = inference_model(model, args.img)

    # plot the results
    if 'irnet' in arch_name :
        conv_num = 16
        fea_in = irconv_fea_in
    elif 'reactnet' in arch_name:
        conv_num = 31
        fea_in = raconv_fea_in
    elif 'baseline' in arch_name:
        conv_num = 16
        fea_in = blconv_fea_in
    else:
        print('arch not support')
        exit()
    
    ncols = 8
    nrows = math.ceil(conv_num / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    index = 0
    for ax, fea in zip(axs.flat, fea_in):
        print(f'plotting img {index}...')
        ratio_list = [compute_ratio(fea_c).item() for fea_c in fea[0]]
        ax.hist(ratio_list, bins=20, range=(-1, 1))
        ax.set_title(f'conv_{index} = {compute_ratio(fea):.3f}, total = {sum([abs(a) for a in ratio_list]):.3f}')
        ax.grid()
        index += 1

    print('saving...')
    path = f'./work_dirs/plot/ratio_channel/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.savefig(f'./work_dirs/plot/ratio_channel/{arch_name}_ratio_channel_{img_name}.jpg')

                                                                                                                                                                                                                                                                                                                                                                      
if __name__ == '__main__':
    main()