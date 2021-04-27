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


def main():
    parser = ArgumentParser()
    parser.add_argument('dir', help='directory for all epoches')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # assume the checkpoint file is the same name of config file
    # and in the dir checkpoint/
    arch_name = args.dir.split('/')[-1].rstrip('/')

    bn_0_bias = [0] * 64
    bn_1_bias = [0] * 64
    for i in range(75):
        print(f'saving {i}...')
        ckpt = torch.load(f'{args.dir}/epoch_{i + 1}.pth')
        for j in range(64):
            if not isinstance(bn_0_bias[j], list):
                bn_0_bias[j] = []
                bn_1_bias[j] = []
            bn_0_bias[j].append(ckpt['state_dict']['backbone.bn1.bias'][j])
            bn_1_bias[j].append(ckpt['state_dict']['backbone.stage1.0.bn1.bias'][j])

        # breakpoint()

    fig, axs = plt.subplots(8, 8, figsize=(32, 24))
    print('plotting bn_0_bias...')
    for ax, bias in zip(axs.flat, bn_0_bias):
        ax.plot(np.arange(75), bias)
        ax.grid()
    print('plotting bn_1_bias...')
    for ax, bias in zip(axs.flat, bn_1_bias):
        ax.plot(np.arange(75), bias)
    fig.savefig(f'./work_dir/plot/stem_bn_bias/{arch_name}_bn_0_1.jpg')

                                                                                                                                                                                                                                                                                                                                                                      
if __name__ == '__main__':
    main()
