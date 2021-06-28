from argparse import ArgumentParser
from mmcls.apis import inference_model, init_model, show_result_pyplot

import torch
import torch.nn as nn
from mmcls.models.backbones.binary_utils.binary_convs import RAConv2d, IRConv2d
from mmcls.models.backbones.binary_utils.binary_functions import RPRelu

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
    #parser.add_argument('img', help='Image file')
    #parser.add_argument('config', help='Config file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # assume the checkpoint file is the same name of config file
    # and in the dir checkpoint/
    #arch_name = args.config.split('/')[-1].rstrip('.py')
    #img_name = args.img.split('/')[-1].rstrip('.JPEG')
    img = 'data/imagenet/val/n01530575/ILSVRC2012_val_00033092.JPEG'
    # build the model from a config file and a checkpoint file
    model = init_model(f'configs/baseline/rprelu_group/react_a/adreact_a_rprelu_step1.py', f'work_dirs/rprelu/react_a/adreact_rprelu_step1/epoch_75.pth', device=args.device)
    #breakpoint()

    # add my own hooks
    for m in model.modules():
        # print(type(m))
        # if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d,
        #                   nn.PReLU, nn.Hardtanh,
        #                   LearnableBias, RANetActSign, RANetWSign, RAConv2d,
        #                   IRConv2d,)):
        #     m.register_forward_hook(hook=get_features)
        if isinstance(m, RPRelu):
            m.register_forward_hook(hook=get_irconv_inout)
        # if isinstance(m, RAConv2d):
        #     m.register_forward_hook(hook=get_raconv_inout)

    # test a single image
    cos = []
    result = inference_model(model, img)
    for x in irconv_fea_in:
         chanel = x.size()[1]
         cos_t = []
         for i in range(0,chanel-1):
             for j in range(i+1,chanel):
                 tens1 = torch.flatten(x[:,i,:,:])
                 tens2 = torch.flatten(x[:,j,:,:])
                 rela =torch.cosine_similarity(tens1,tens2,dim=0)
                 cos_t.append(rela)
         cos_t = np.array(cos_t)
         cos_t = abs(cos_t)
         k  = []
         #k1 = np.sum(cos_t>0.5)
         k2 = np.sum(cos_t>0.75)
         #k3 = np.sum(cos_t>0.9)
         #k.append(k1)
         k.append(k2)
         #k.append(k3)
         cos.append(k)
    f=open('tools/plot/cos3.txt','a')
    f.write('\n')
    f.write(str(cos))
    f.close()

if __name__ == '__main__':
    main()