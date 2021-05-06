from argparse import ArgumentParser
from mmcls.apis import inference_model, init_model, show_result_pyplot

import torch
import torch.nn as nn
from mmcls.models.backbones.binary_utils.binary_convs import RAConv2d, IRConv2d, BLConv2d
from mmcls.models.backbones.binary_utils.binary_functions import LearnableBias, RANetActSign, RANetWSign

import matplotlib.pyplot as plt
import numpy as np
import matplotlib


blconv_in = []
blconv_out = []
bn_in = []
bn_out = []
prelu_in = []
prelu_out = []
maxpool_out = []

def get_blconv_inout(module, fea_in, fea_out):
    global blconv_in
    global blconv_out
    # print(type(fea_in), type(fea_out))
    # fea_in is a tuple, fea_out is a tensor
    blconv_in.append(fea_in[0].cpu())
    blconv_out.append(fea_out.cpu())

def get_bn_inout(module, fea_in, fea_out):
    global bn_in
    global bn_out
    bn_in.append(fea_in[0].cpu())
    bn_out.append(fea_out.cpu())

def get_prelu_inout(module, fea_in, fea_out):
    global prelu_in
    global prelu_out
    prelu_in.append(fea_in[0].cpu())
    prelu_out.append(fea_out.cpu())

def get_maxpool_out(module, fea_in, fea_out):
    global maxpool_out
    maxpool_out.append(fea_out.cpu())

def main():
    global blconv_in
    global blconv_out
    global bn_in
    global bn_out
    global prelu_in
    global prelu_out
    global maxpool_out
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    config = 'configs/zzp/baseline/baseline_prelu/baseline_prelu_11_b32x8/baseline_prelu_11_b32x8_step1.py'
    ckpt = 'work_dir/baseline/baseline_prelu/baseline_prelu_11_b32x8/baseline_prelu_11_b32x8_step1/epoch_75.pth'
    model = init_model(config, ckpt, device=args.device)

    # add my own hooks
    for m in model.modules():
        # print(type(m))
        if isinstance(m, BLConv2d):
            m.register_forward_hook(hook=get_blconv_inout)
        if isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(hook=get_bn_inout)
        if isinstance(m, nn.PReLU):
            m.register_forward_hook(hook=get_prelu_inout)
        if isinstance(m, nn.MaxPool2d):
            m.register_forward_hook(hook=get_maxpool_out)

    # test a single image
    result = inference_model(model, args.img)

    # remove features for fist conv
    # print(len(prelu_in), len(prelu_out))
    # print(len(bn_in))
    bn_idx = [0, 7, 12, 17]
    bn_in = [bn_in[i] for i in range(len(bn_in)) if i not in bn_idx]
    prelu_out = prelu_out[1:]
    sum_out = bn_in[1:]
    sum_out.append(torch.ones_like(bn_in[0]))
    assert len(bn_in) == 16
    assert len(blconv_in) == 16
    assert len(blconv_out) == 16
    assert len(prelu_out) == 16
    assert len(sum_out) == 16

    # # show the results
    fig, axs = plt.subplots(5, 16, figsize=(64, 15))
    axs = axs.flat
    for i in range(16):
        print(f'plotting feature {i}...')
        axs[i].set_title(f'bn_in_{i}')
        axs[i].hist(bn_in[i].flatten().numpy(), bins=100, alpha=0.7)
        axs[i].grid()
        axs[i+16].set_title(f'conv_in_{i}')
        axs[i+16].hist(blconv_in[i].flatten().numpy(), bins=100, alpha=0.7)
        axs[i+16].grid()
        axs[i+32].set_title(f'conv_out_{i}')
        axs[i+32].hist(blconv_out[i].flatten().numpy(), bins=100, alpha=0.7)
        axs[i+32].grid()
        axs[i+48].set_title(f'prelu_out_{i}')
        axs[i+48].hist(prelu_out[i].flatten().numpy(), bins=100, alpha=0.7)
        axs[i+48].grid()
        axs[i+64].set_title(f'sum_out_{i}')
        axs[i+64].hist(sum_out[i].flatten().numpy(), bins=100, alpha=0.7)
        axs[i+64].grid()
    fig.savefig('./work_dir/plot/feature_hist/feature_hist_baseline_prelu_11.jpg')


if __name__ == '__main__':
    main()
