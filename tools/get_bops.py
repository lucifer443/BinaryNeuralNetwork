import argparse

from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info

from mmcls.models import build_classifier
from mmcls.models.backbones.binary_utils.binary_convs import BaseBinaryConv2d
import torch

TOTAL_BOPS = 0

def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args

def count_bops(module, fea_in, fea_out):
    global TOTAL_BOPS
    groups = module.groups
    kh, kw = module.kernel_size
    in_channels = module.in_channels
    n, c, h, w = fea_out.shape
    bops = (n * c * h * w) * in_channels // groups * kh * kw
    TOTAL_BOPS += bops
    return None

def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_classifier(cfg.model)
    model.cuda()
    model.eval()
    for m in model.modules():
        if isinstance(m, BaseBinaryConv2d):
            m.register_forward_hook(hook=count_bops)

    if hasattr(model, 'extract_feat'):
        model.forward = model.simple_test
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape, input_constructor=lambda x: {'img':torch.rand((1, *x)).cuda()}, print_per_layer_stat=False)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n',
          'Bops: %.3f GBOPs\n' % (TOTAL_BOPS/1e9),
          f'Flops: {flops}\n Params: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
