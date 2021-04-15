from argparse import ArgumentParser
from mmcls.apis import inference_model, init_model, show_result_pyplot

import torch
import torch.nn as nn
from mmcls.models.backbones.binary_utils.binary_convs import RAConv2d, IRConv2d
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

def main():
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
        # if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d,
        #                   nn.PReLU, nn.Hardtanh,
        #                   LearnableBias, RANetActSign, RANetWSign, RAConv2d,
        #                   IRConv2d,)):
        #     m.register_forward_hook(hook=get_features)
        if isinstance(m, IRConv2d):
            m.register_forward_hook(hook=get_irconv_inout)
        if isinstance(m, RAConv2d):
            m.register_forward_hook(hook=get_raconv_inout)

    # test a single image
    result = inference_model(model, args.img)

    # show the results
    for name, feature in zip(names, features):
        pass
        # print(name)
    # first_conv = features[0][0].flatten().cpu().numpy()
    # plt.hist(first_conv, bins=1000, alpha = 0.4)
    # plt.savefig('./h1_1000.jpg')
    # bn1 = features[1][0].flatten().cpu().numpy()
    # plt.hist(bn1, bins=1000, alpha = 0.5)
    # plt.savefig('./h2_1000.jpg')
    # r1 = features[2].flatten().cpu().numpy()
    # plt.hist(r1, bins=1000, alpha = 0.5)
    # conv1 = features[5][0].flatten().cpu().numpy()
    # plt.hist(conv1, bins=1000, alpha = 0.5)
    # bn11 = features[6].flatten().cpu().numpy()
    # plt.hist(bn11, bins=1000, alpha = 0.5)
    # plt.grid()
    # plt.savefig('./h3_1000.jpg')

    '''
    for i in range(len(irconv_fea_in)):
        print(f'plotting feature {i}...', irconv_fea_in[i].max())
        plt.cla()
        plt.hist(irconv_fea_in[i].flatten().cpu().numpy(), bins=100, alpha=1.0)
        plt.grid()
        plt.savefig(f'./fea_in_{i}.jpg')
    '''
    if 'irnet' in arch_name:
    # num of 1 / num of -1
        ratio = []
        for fea in irconv_fea_in:
            fea = fea.sign().flatten()
            diff = fea @ torch.ones(fea.shape).T
            total = fea.numel()
            # ratio.append((total + diff) / (total - diff))
            ratio.append(diff / total)
    
        sum = 0
        for i in range(16):
            if ratio[i] > 3.5:
                ratio[i] = 3.5
            sum += abs(ratio[i])
        print(sum)
        x = np.arange(0, 16)
        plt.plot(x, ratio, 'o-')
        plt.yticks(np.arange(-1, 1.2, 0.2))
        plt.grid()
        plt.savefig(f'./work_dir/plot/{arch_name}_ratio1.jpg')
    elif 'reactnet' in arch_name:
        ratio = []
        for fea in raconv_fea_in:
            fea = fea.sign().flatten()
            diff = fea @ torch.ones(fea.shape).T
            total = fea.numel()
            # ratio.append((total + diff) / (total - diff))
            ratio.append(diff / total)
    
        sum = 0
        for i in range(len(raconv_fea_in)):
            # if ratio[i] > 3.5:
            #     ratio[i] = 3.5
            sum += abs(ratio[i])
        print(sum)
        x = np.arange(0, 31)
        plt.plot(x, ratio, 'o-')
        plt.yticks(np.arange(-1, 1.2, 0.2))
        plt.grid()
        plt.savefig(f'./work_dir/plot/{arch_name}_ratio_{img_name}.jpg')



if __name__ == '__main__':
    main()
