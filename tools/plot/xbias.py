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

def compute_inner_error_v3c(float_fea, bin_fea):
    '''
    loat_fea和bin_fea各是一组(C, H, W)的特征图
    该实现是v2的向量版本
    '''
    #breakpoint()
    chn = float_fea.shape[0]
    float_fea = float_fea.reshape(chn, -1)
    bin_fea = bin_fea.reshape(chn, -1)
    dim = float_fea.shape[1]

    #float_matrix = 
    #bin_matrix = 
    error = (float_fea.reshape(chn, dim, 1) - float_fea.reshape(chn, 1, dim) - (bin_fea.reshape(chn, dim, 1) - bin_fea.reshape(chn, 1, dim))).abs().sum() / (float_fea.reshape(chn, dim, 1) - float_fea.reshape(chn, 1, dim)).numel()

    return error

def compute_inner_error_v3(float_fea, bin_fea):
    '''float_fea和bin_fea都是一个pytorch tensor'''
    float_fea = float_fea.flatten()
    bin_fea = bin_fea.flatten()
    dim = float_fea.shape[0]

    float_matrix = float_fea.reshape(dim, 1) - float_fea.reshape(1, dim)
    bin_matrix = bin_fea.reshape(dim, 1) - bin_fea.reshape(1, dim)

    error = (float_matrix - bin_matrix).abs().sum() / float_matrix.numel()

    return error


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
    parser.add_argument('--img', help='Image file',default='data/imagenet/val/n01484850/ILSVRC2012_val_00002338.JPEG')
    parser.add_argument('--config', help='Config file',default='configs/baseline/rprelu_group/react_a/adreact_a_rprelu_step1.py')
    parser.add_argument('--checkpoint', help='checkpoint file',default='work_dirs/rprelu/react_a/adreact_rprelu_nobias_step1/latest.pth')
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
        #exit()
    
    fea = raconv_fea_in
    #chn = fea.shape[0]
    error_list=[]
    for i in range(-20,21):
        print(i)
        error = 0
        bias  =i/10
        for j in range(31):
            #print(j)
            error+=compute_inner_error_v3c(fea[j][0],(fea[j][0]+bias).sign())
        error_list.append(error/31)
    print(error_list)
    plt.figure()
    plt.title(error_list.index(min(error_list))*0.1-2)
    plt.plot(np.arange(-2, 2.1, 0.1),error_list)
    plt.grid()


    
    #plt.savefig(f'./work_dirs/plot/ratio_channel/{arch_name}_ratio_channel_{img_name}.jpg')
    plt.savefig('/workspace/S/jiangfei/BinaryNeuralNetwork_debug/tools/plot/nobiastotal.jpg')

                                                                                                                                                                                                                                                                                                                                                                      
if __name__ == '__main__':
    main()