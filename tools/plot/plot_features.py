from argparse import ArgumentParser
from mmcls.apis import inference_model, init_model, show_result_pyplot

import torch
import torch.nn as nn
from mmcls.models.backbones.binary_utils.binary_convs import RAConv2d, IRConv2d
from mmcls.models.backbones.binary_utils.binary_functions import LearnableBias, RANetActSign, RANetWSign

import matplotlib.pyplot as plt
import numpy as np
import matplotlib


irconv_in = []
irconv_out = []
bn_out = []
hardtanh_in = []
hardtanh_out = []
maxpool_out = []

def get_irconv_inout(module, fea_in, fea_out):
    global irconv_in
    global irconv_out
    # print(type(fea_in), type(fea_out))
    # fea_in is a tuple, fea_out is a tensor
    irconv_in.append(fea_in[0].cpu())
    irconv_out.append(fea_out.cpu())

def get_bn_out(module, fea_in, fea_out):
    global bn_out
    bn_out.append(fea_out.cpu())

def get_hardtanh_inout(module, fea_in, fea_out):
    global hardtanh_in
    global hardtanh_out
    hardtanh_in.append(fea_in[0].cpu())
    hardtanh_out.append(fea_out.cpu())

def get_maxpool_out(module, fea_in, fea_out):
    global maxpool_out
    maxpool_out.append(fea_out.cpu())

def main():
    global irconv_in
    global irconv_out
    global bn_out
    global hardtanh_in
    global hardtanh_out
    global maxpool_out
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # add my own hooks
    for m in model.modules():
        # print(type(m))
        if isinstance(m, IRConv2d):
            m.register_forward_hook(hook=get_irconv_inout)
        if isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(hook=get_bn_out)
        if isinstance(m, nn.Hardtanh):
            m.register_forward_hook(hook=get_hardtanh_inout)
        if isinstance(m, nn.MaxPool2d):
            m.register_forward_hook(hook=get_maxpool_out)

    # test a single image
    result = inference_model(model, args.img)

    # remove features for fist conv
    print(len(hardtanh_in), len(hardtanh_out))
    bn_idx = [0, 7, 12, 17]
    bn_out = [bn_out[i] for i in range(len(bn_out)) if i not in bn_idx]
    shift_in = hardtanh_in[:-1]
    shift_in[0] = maxpool_out[0]
    sum_out = hardtanh_in[1:]
    hardtanh_out = hardtanh_out[1:]
    sign_out = [fea.sign() for fea in irconv_in]
    assert len(shift_in) == 16
    assert len(irconv_in) == 16
    assert len(sign_out) == 16
    assert len(irconv_out) == 16
    assert len(bn_out) == 16
    assert len(sum_out) == 16
    assert len(hardtanh_out) == 16

    # show the results
    plt.figure(figsize=(64, 21))
    for i in range(16):
        print(f'plotting feature {i}...')
        plt.subplot(7, 16, i+1)
        plt.title(f'shift_in_{i}')
        plt.imshow(shift_in[i][0][0].numpy())
        plt.grid()

        plt.subplot(7, 16, i+17)
        plt.title(f'sign_in_{i}')
        plt.imshow(irconv_in[i][0][0].numpy())
        plt.grid()

        plt.subplot(7, 16, i+33)
        plt.title(f'sign_out_{i}')
        plt.imshow(sign_out[i][0][0].numpy())
        plt.grid()

        plt.subplot(7, 16, i+49)
        plt.title(f'conv_out_{i}')
        plt.imshow(irconv_out[i][0][0].numpy())
        plt.grid()

        plt.subplot(7, 16, i+65)
        plt.title(f'bn_out_{i}')
        plt.imshow(bn_out[i][0][0].numpy())
        plt.grid()

        plt.subplot(7, 16, i+81)
        plt.title(f'sum_out_{i}')
        plt.imshow(sum_out[i][0][0].numpy())
        plt.grid()

        plt.subplot(7, 16, i+97)
        plt.title(f'hardtanh_out_{i}')
        plt.imshow(hardtanh_out[i][0][0].numpy())
        plt.grid()

    plt.savefig(f'./work_dir/plot/feature/temp.jpg')


if __name__ == '__main__':
    main()
