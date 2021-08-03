import torch
import torch.nn as nn
from .binary_convs import BLConv2d
# from .binary_functions import FeaExpand
from .feature_expand import FeaExpand


class MultiFea_1_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, n=1, **kwargs):
        super(MultiFea_1_Block, self).__init__()
        self.n = n
        self.conv1 = BLConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.conv2 = BLConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)

        identity = out
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear2(out)

        return out


class MultiFea_n_1_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, n=1, **kwargs):
        super(MultiFea_n_1_Block, self).__init__()
        self.n = n
        self.fexpand1 = FeaExpand(expansion=n, mode='1')
        self.conv1 = BLConv2d(in_channels * n, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.fexpand2 = FeaExpand(expansion=n, mode='1')
        self.conv2 = BLConv2d(out_channels * n, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear2(out)

        return out


class MultiFea_2_1_Block(MultiFea_n_1_Block):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_2_1_Block, self).__init__(in_channels, out_channels, stride, downsample, n=2, **kwargs)


class MultiFea_3_1_Block(MultiFea_n_1_Block):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_3_1_Block, self).__init__(in_channels, out_channels, stride, downsample, n=3, **kwargs)


class MultiFea_4_1_Block(MultiFea_n_1_Block):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_4_1_Block, self).__init__(in_channels, out_channels, stride, downsample, n=4, **kwargs)


class MultiFea_5_1_Block(MultiFea_n_1_Block):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_5_1_Block, self).__init__(in_channels, out_channels, stride, downsample, n=5, **kwargs)


class MultiFea_6_1_Block(MultiFea_n_1_Block):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_6_1_Block, self).__init__(in_channels, out_channels, stride, downsample, n=6, **kwargs)


class MultiFea_7_1_Block(MultiFea_n_1_Block):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_7_1_Block, self).__init__(in_channels, out_channels, stride, downsample, n=7, **kwargs)


class MultiFea_10_1_Block(MultiFea_n_1_Block):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_10_1_Block, self).__init__(in_channels, out_channels, stride, downsample, n=10, **kwargs)


class MultiFea_3_1c_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_3_1c_Block, self).__init__()
        self.fexpand1 = FeaExpand(expansion=3, mode='1c')
        self.conv1 = BLConv2d(in_channels * 3, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.fexpand2 = FeaExpand(expansion=3, mode='1c')
        self.conv2 = BLConv2d(out_channels * 3, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear2(out)

        return out


class MultiFea_2_2_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_2_2_Block, self).__init__()
        self.fexpand1 = FeaExpand(expansion=2, mode='2')
        self.conv1 = BLConv2d(in_channels * 2, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.fexpand2 = FeaExpand(expansion=2, mode='2')
        self.conv2 = BLConv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear2(out)

        return out



    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, n=1, **kwargs):
        super(MultiFea13_n_1_Block, self).__init__()
        self.n = n
        self.fexpand1 = FeaExpand(expansion=n, mode='1')
        self.conv1 = BLConv2d(in_channels * n, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear1 = nn.PReLU(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.fexpand2 = FeaExpand(expansion=n, mode='1')
        self.conv2 = BLConv2d(out_channels * n, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.nonlinear1(out)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.nonlinear2(out)
        out = self.bn2(out)
        out += identity

        return out



    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea13_3_1_Block, self).__init__(in_channels, out_channels, stride, downsample, n=3, **kwargs)



    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea13_3_3nc_Block, self).__init__(in_channels, out_channels, stride, downsample, n=3, **kwargs)


class MultiFea_n_4_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, n=1, **kwargs):
        super(MultiFea_n_4_Block, self).__init__()
        self.n = n
        self.fexpand1 = FeaExpand(expansion=n, in_channels=in_channels, mode='4')
        self.conv1 = BLConv2d(in_channels * n, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.fexpand2 = FeaExpand(expansion=n, in_channels=out_channels, mode='4')
        self.conv2 = BLConv2d(out_channels * n, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear2(out)

        return out


class MultiFea_3_4_Block(MultiFea_n_4_Block):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_3_4_Block, self).__init__(in_channels, out_channels, stride, downsample, n=3, **kwargs)


class MultiFea_n_4c_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, n=1, **kwargs):
        super(MultiFea_n_4c_Block, self).__init__()
        self.n = n
        self.fexpand1 = FeaExpand(expansion=n, in_channels=in_channels, mode='4c')
        self.conv1 = BLConv2d(in_channels * n, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.fexpand2 = FeaExpand(expansion=n, in_channels=out_channels, mode='4c')
        self.conv2 = BLConv2d(out_channels * n, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear2(out)

        return out


class MultiFea_3_4c_Block(MultiFea_n_4c_Block):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_3_4c_Block, self).__init__(in_channels, out_channels, stride, downsample, n=3, **kwargs)


class MultiFea_n_6_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, n=1, **kwargs):
        super(MultiFea_n_6_Block, self).__init__()
        self.n = n
        self.fexpand1 = FeaExpand(expansion=n, mode='6')
        self.conv1 = BLConv2d(in_channels * n, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.fexpand2 = FeaExpand(expansion=n, mode='6')
        self.conv2 = BLConv2d(out_channels * n, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear2(out)

        return out


class MultiFea_3_6_Block(MultiFea_n_6_Block):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_3_6_Block, self).__init__(in_channels, out_channels, stride, downsample, n=3, **kwargs)


class MultiFea_n_6n_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, n=1, **kwargs):
        super(MultiFea_n_6n_Block, self).__init__()
        self.n = n
        self.fexpand1 = FeaExpand(expansion=n, mode='6n')
        self.conv1 = BLConv2d(in_channels * n, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.fexpand2 = FeaExpand(expansion=n, mode='6n')
        self.conv2 = BLConv2d(out_channels * n, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear2(out)

        return out


class MultiFea_3_6n_Block(MultiFea_n_6n_Block):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_3_6n_Block, self).__init__(in_channels, out_channels, stride, downsample, n=3, **kwargs)


class MultiFea_n_6c_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, n=1, **kwargs):
        super(MultiFea_n_6c_Block, self).__init__()
        self.n = n
        self.fexpand1 = FeaExpand(expansion=n, mode='6c')
        self.conv1 = BLConv2d(in_channels * n, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.fexpand2 = FeaExpand(expansion=n, mode='6c')
        self.conv2 = BLConv2d(out_channels * n, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear2(out)

        return out


class MultiFea_3_6c_Block(MultiFea_n_6c_Block):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_3_6c_Block, self).__init__(in_channels, out_channels, stride, downsample, n=3, **kwargs)


class MultiFea_n_6nc_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, n=1, **kwargs):
        super(MultiFea_n_6nc_Block, self).__init__()
        self.n = n
        self.fexpand1 = FeaExpand(expansion=n, mode='6nc')
        self.conv1 = BLConv2d(in_channels * n, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.fexpand2 = FeaExpand(expansion=n, mode='6nc')
        self.conv2 = BLConv2d(out_channels * n, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear1(out)

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear2(out)

        return out


class MultiFea_3_6nc_Block(MultiFea_n_6nc_Block):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(MultiFea_3_6nc_Block, self).__init__(in_channels, out_channels, stride, downsample, n=3, **kwargs)
