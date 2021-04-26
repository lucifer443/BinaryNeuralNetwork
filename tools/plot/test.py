from argparse import ArgumentParser
from mmcls.apis import inference_model, init_model, show_result_pyplot

import torch
import torch.nn as nn
from mmcls.models.backbones.binary_utils.binary_convs import RAConv2d, IRConv2d
from mmcls.models.backbones.binary_utils.baseline_blocks import BLConv2d
from mmcls.models.backbones.binary_utils.binary_functions import LearnableBias, RANetActSign, RANetWSign

import matplotlib.pyplot as plt
import numpy as np
import matplotlib


names = []
features = []

irconv_fea_in = []
irconv_fea_out = []

raconv_fea_in = []
raconv_fea_out = []

blconv_fea_in = []
blconv_fea_out = []

conv_fea_in = []
conv_fea_out = []

bn_fea_in = []
bn_fea_out = []
bn_params = []

relu_fea_in = []
relu_fea_out = []

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
    # print(type(fea_in), type(fea_out))
    # fea_in is a tuple, fea_out is a tensor
    # print(module, fea_in[0].max())
    raconv_fea_in.append(fea_in[0].cpu())
    raconv_fea_out.append(fea_out.cpu())

def get_blconv_inout(module, fea_in, fea_out):
    global blconv_fea_in
    global blconv_fea_out
    blconv_fea_in.append(fea_in[0].cpu())
    blconv_fea_out.append(fea_out.cpu())

def get_conv_inout(module, fea_in, fea_out):
    global conv_fea_in
    global conv_fea_out
    conv_fea_in.append(fea_in[0].cpu())
    conv_fea_out.append(fea_out.cpu())

def get_bn_inout(module, fea_in, fea_out):
    global bn_fea_in
    global bn_fea_out
    bn_fea_in.append(fea_in[0].cpu())
    bn_fea_out.append(fea_out.cpu())

def get_relu_inout(module, fea_in, fea_out):
    global relu_fea_in
    global relu_fea_out
    relu_fea_in.append(fea_in[0].cpu())
    relu_fea_out.append(fea_out.cpu())

def compute_ratio(fea):
    fea = fea.sign().flatten()
    diff = fea @ torch.ones(fea.shape).T
    total = fea.numel()
    return diff / total

def main():
    global bn_params
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # assume the checkpoint file is the same name of config file
    # and in the dir checkpoint/
    arch_name = args.config.split('/')[-1].rstrip('.py')
    img_name = args.img.split('/')[-1].rstrip('.JPEG')

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, f'checkpoints/{arch_name}.pth', device=args.device)

    # add my own hooks
    for m in model.modules():
        # print(type(m))
        if isinstance(m, IRConv2d):
            m.register_forward_hook(hook=get_irconv_inout)
        if isinstance(m, RAConv2d):
            m.register_forward_hook(hook=get_raconv_inout)
        if isinstance(m, BLConv2d):
            m.register_forward_hook(hook=get_blconv_inout)
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(hook=get_conv_inout)
        if isinstance(m, nn.BatchNorm2d):
            bn_params.append(list(m.named_parameters()))
            m.register_forward_hook(hook=get_bn_inout)
        if isinstance(m, nn.ReLU):
            m.register_forward_hook(hook=get_relu_inout)

    # test a single image
    result = inference_model(model, args.img)

    # show the results
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
    
    fig, axs = plt.subplots(8, 8, figsize=(32, 24))
    index = 0
    for ax, fea in zip(axs.flat, blconv_fea_in[0][0]):
        print(f'plotting img {index}...')
        ax.imshow(fea.sign().numpy(), vmin=-1, vmax=1)
        ax.set_title(f'channel = {index}, bn_bias = {bn_params[0][1][1][index]: .3f}')
        index += 1
    # breakpoint()

    print('saving...')
    fig.savefig(f'./work_dir/plot/features/{arch_name}_blconv0_in_{img_name}.jpg')

                                                                                                                                                                                                                                                                                                                                                                      
if __name__ == '__main__':
    main()
