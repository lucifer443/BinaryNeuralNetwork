from argparse import ArgumentParser
from mmcls.apis import inference_model, init_model, show_result_pyplot

import torch.nn as nn
from mmcls.models.backbones.binary_utils.binary_convs import RAConv2d
from mmcls.models.backbones.binary_utils.binary_functions import LearnableBias, RANetActSign, RANetWSign

import matplotlib.pyplot as plt
import numpy as np
import matplotlib


names = []
features = []

def get_features(module, fea_in, fea_out):
    global names
    global features
    names.append(module.__class__.__name__)
    features.append(fea_out)

def main():
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
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.PReLU,
                          LearnableBias, RANetActSign, RANetWSign, RAConv2d,)):
            m.register_forward_hook(hook=get_features)

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
    r1 = features[2].flatten().cpu().numpy()
    plt.hist(r1, bins=1000, alpha = 0.5)
    # conv1 = features[5][0].flatten().cpu().numpy()
    # plt.hist(conv1, bins=1000, alpha = 0.5)
    bn11 = features[6].flatten().cpu().numpy()
    plt.hist(bn11, bins=1000, alpha = 0.5)
    plt.grid()
    plt.savefig('./h3_1000.jpg')


if __name__ == '__main__':
    main()
