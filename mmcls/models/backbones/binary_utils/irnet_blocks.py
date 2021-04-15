import torch
import torch.nn as nn
from .binary_convs import IRConv2d, RAConv2d, IRG3swConv2d
from .binary_functions import RPRelu, LearnableBias

class IRNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetBlock, self).__init__()
        self.conv1 = IRConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.conv2 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.nonlinear1(out)

        residual = out
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.nonlinear2(out)

        return out


# group sign
# 多分支conv，将原来的一个conv拆分成几个conv，每个conv的输入经过不同阈值的sign
class IRNetGBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, n=0, **kwargs):
        super(IRNetGBlock, self).__init__()

        self.groups = n
        self.alpha = []
        self.conv1 = nn.ModuleList()
        self.bn1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()

        self.nonlinear = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        for i in range(self.groups):
            self.alpha.append(-1 + (i + 1) * 2 / (self.groups + 1))
            self.conv1.append(IRConv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False, **kwargs))
            self.bn1.append(nn.BatchNorm2d(self.out_channels))
            self.conv2.append(IRConv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs))
            self.bn2.append(nn.BatchNorm2d(self.out_channels))

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        x_max = torch.max(abs(x))
        x_list = []
        for i in range(self.groups):
            a = self.alpha[i] * x_max
            x_list.append(x + a)

        out = 0
        for i in range(self.groups):
            x_list[i] = self.conv1[i](x_list[i])
            x_list[i] = self.bn1[i](x_list[i])
            out += x_list[i]

        out += identity
        out = self.nonlinear(out)

        identity = out
        x_max = torch.max(abs(out))
        x_list = []
        for i in range(self.groups):
            a = self.alpha[i] * x_max
            x_list.append(out + a)

        out = 0
        for i in range(self.groups):
            x_list[i] = self.conv2[i](x_list[i])
            x_list[i] = self.bn2[i](x_list[i])
            out += x_list[i]

        out += identity
        out = self.nonlinear(out)

        return out


class IRNetG2Block(IRNetGBlock):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetG2Block, self).__init__(in_channels, out_channels, stride, downsample, n=2, **kwargs)


class IRNetG3Block(IRNetGBlock):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetG3Block, self).__init__(in_channels, out_channels, stride, downsample, n=3, **kwargs)


# 使用分支conv，但所有分支的sign阈值都是0
# 用来和channel数是64的多分支不同阈值sign的conv做对比
class IRNetG3nBlock(IRNetGBlock):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetG3nBlock, self).__init__(in_channels, out_channels, stride, downsample, n=3, **kwargs)

        for i in range(3):
            self.alpha[i] = 0


class IRNetG4Block(IRNetGBlock):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetG4Block, self).__init__(in_channels, out_channels, stride, downsample, n=4, **kwargs)


class IRNetG5Block(IRNetGBlock):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetG5Block, self).__init__(in_channels, out_channels, stride, downsample, n=5, **kwargs)


class IRNetG6Block(IRNetGBlock):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetG6Block, self).__init__(in_channels, out_channels, stride, downsample, n=6, **kwargs)


class IRNetG7Block(IRNetGBlock):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetG7Block, self).__init__(in_channels, out_channels, stride, downsample, n=7, **kwargs)


class IRNetG8Block(IRNetGBlock):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetG8Block, self).__init__(in_channels, out_channels, stride, downsample, n=8, **kwargs)


class IRNetG3swBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetG3swBlock, self).__init__()
        self.conv1 = IRG3swConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = IRG3swConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.nonlinear(out)

        residual = out
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.nonlinear(out)

        return out


class IRNetGBBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, n=0, **kwargs):
        super(IRNetGBBlock, self).__init__()

        self.groups = n
        self.alpha = []
        self.conv1 = nn.ModuleList()
        self.bn1 = nn.ModuleList()

        self.nonlinear = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_in_channels = in_channels // self.groups
        self.group_out_channels = out_channels // self.groups

        for i in range(self.groups):
            self.alpha.append(-1 + 2 / (self.groups + 1) * (i + 1))
            for j in range(self.groups):
                self.conv1.append(IRConv2d(self.group_in_channels, self.group_out_channels, kernel_size=3,
                                           stride=self.stride, padding=1, bias=False, **kwargs))
                self.bn1.append(nn.BatchNorm2d(self.group_out_channels))
        
        self.conv2 = IRConv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        x_group = x.split(self.in_channels // self.groups, dim=1)
        x_group_list = [0] * self.groups
        assert len(x_group) == self.groups
        for i in range(self.groups):
            x_max = torch.max(abs(x_group[i]))
            x_list = []
            for j in range(self.groups):
                a = self.alpha[i] * x_max
                x_list.append(x_group[i] + a)

            for j in range(self.groups):
                x_list[j] = self.conv1[self.groups * i + j](x_list[j])
                x_list[j] = self.bn1[self.groups * i + j](x_list[j])
                x_group_list[i] += x_list[j]
        
        out = torch.cat(x_group_list, dim=1)
        out += identity
        out = self.nonlinear(out)

        identity = out
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.nonlinear(out)

        return out


class IRNetGB4Block(IRNetGBBlock):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetGB4Block, self).__init__(in_channels, out_channels, stride, downsample, n=4, **kwargs)


class IRNetGBaBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, n=0, **kwargs):
        super(IRNetGBaBlock, self).__init__()

        self.groups = n
        self.alpha = []
        self.conv1 = nn.ModuleList()
        self.bn1 = nn.ModuleList()

        self.nonlinear = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        for i in range(self.groups):
            self.alpha.append(-1 + (i + 1) * 2 / (self.groups + 1))
            self.conv1.append(IRConv2d(self.in_channels, self.out_channels, kernel_size=3,
                                       stride=self.stride, padding=1, bias=False, groups = self.groups, **kwargs))
            self.bn1.append(nn.BatchNorm2d(self.out_channels))
        
        self.conv2 = IRConv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        x_max = torch.max(abs(x)).detach()
        x_list = []
        for i in range(self.groups):
            a = self.alpha[i] * x_max
            x_list.append(x + a)

        out = 0
        for i in range(self.groups):
            x_list[i] = self.conv1[i](x_list[i])
            x_list[i] = self.bn1[i](x_list[i])
            out += x_list[i]

        out += identity
        out = self.nonlinear(out)

        identity = out
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.nonlinear(out)

        return out


class IRNetGBa4Block(IRNetGBaBlock):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetGBa4Block, self).__init__(in_channels, out_channels, stride, downsample, n=4, **kwargs)


class IRNetGBbBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, g=1, b=1, **kwargs):
        super(IRNetGBbBlock, self).__init__()

        self.groups = g
        self.branches = b
        self.alpha = []
        self.conv1 = nn.ModuleList()
        self.bn1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()

        self.nonlinear11 = nn.Hardtanh(inplace=True)
        self.nonlinear12 = nn.Hardtanh(inplace=True)
        self.nonlinear21 = nn.Hardtanh(inplace=True)
        self.nonlinear22 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        for i in range(self.branches):
            self.alpha.append(-1 + (i + 1) * 2 / (self.branches + 1))
            self.conv1.append(IRConv2d(self.in_channels, self.out_channels, kernel_size=3,
                                       stride=self.stride, padding=1, bias=False, groups=self.groups, **kwargs))
            self.bn1.append(nn.BatchNorm2d(self.out_channels))
            self.conv2.append(IRConv2d(self.out_channels, self.out_channels, kernel_size=3,
                                       stride=1, padding=1, bias=False, groups=self.groups, **kwargs))
            self.bn2.append(nn.BatchNorm2d(self.out_channels))
        self.conv1_1x1 = IRConv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.bn1_1x1 = nn.BatchNorm2d(self.out_channels)
        self.conv2_1x1 = IRConv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.bn2_1x1 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        # 3x3 muti-branch group conv
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        x_max = torch.max(abs(x)).detach()
        x_list = []
        for i in range(self.branches):
            a = self.alpha[i] * x_max
            x_list.append(x + a)

        out = 0
        for i in range(self.branches):
            x_list[i] = self.conv1[i](x_list[i])
            x_list[i] = self.bn1[i](x_list[i])
            out += x_list[i]

        out += identity
        out = self.nonlinear11(out)

        # 1x1 conv
        identity = out
        out = self.conv1_1x1(out)
        out = self.bn1_1x1(out)

        out += identity
        out = self.nonlinear12(out)

        # 3x3 muti-branch group conv
        identity = out

        out_max = torch.max(abs(out)).detach()
        out_list = []
        for i in range(self.branches):
            a = self.alpha[i] * out_max
            out_list.append(out + a)

        out = 0
        for i in range(self.branches):
            out_list[i] = self.conv2[i](out_list[i])
            out_list[i] = self.bn2[i](out_list[i])
            out += out_list[i]

        out += identity
        out = self.nonlinear21(out)

        # 1x1 conv
        identity = out
        out = self.conv2_1x1(out)
        out = self.bn2_1x1(out)

        out += identity
        out = self.nonlinear22(out)

        return out
    

# class IRNetGBb4Block(IRNetGBbBlock):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
#         super(IRNetGBb4Block, self).__init__(in_channels, out_channels, stride, downsample, groups, branches, **kwargs)


# hierarchical conv
# 分层conv，一个block的输出是两个conv输出的拼接的结果
class IRNetH1Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetH1Block, self).__init__()
        self.conv1 = IRConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)

        if self.downsample is not None:
            identity = self.downsample(x)
        out1 += identity

        out1 = self.nonlinear(out1)

        identity = out1
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)

        out2 += identity
        out2 = self.nonlinear(out2)

        out = torch.cat((out1, out2), dim=1)
        out = torch.reshape(out, (out.shape[0], out.shape[1] // 2, 2, out.shape[2], out.shape[3]))
        out = torch.sum(out, dim=2)

        return out


# hierarchical conv
# 分层conv，一个block的输出是两个conv输出的拼接的结果
class IRNetH1aBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetH1aBlock, self).__init__()
        self.conv1 = IRConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)

        if self.downsample is not None:
            identity = self.downsample(x)
        out1 += identity

        out1 = self.nonlinear(out1)

        identity = out1
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)

        out2 += identity
        out2 = self.nonlinear(out2)

        out = torch.cat((out1, out2), dim=1)
        out = torch.reshape(out, (out.shape[0], out.shape[1] // 2, 2, out.shape[2], out.shape[3]))
        out = torch.sum(out, dim=2)
        out = out / 2

        return out


# hierarchical conv
# 分层conv，一个block的输出是两个conv输出的拼接的结果
class IRNetH2Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(IRNetH2Block, self).__init__()
        self.conv1 = IRConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)

        if self.downsample is not None:
            identity = self.downsample(x)
        out1 += identity

        out1 = self.nonlinear(out1)

        identity = out1
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)

        out2 += identity
        out2 = self.nonlinear(out2)

        identity = out2
        out3 = self.conv3(out2)
        out3 = self.bn2(out3)

        out3 += identity
        out3 = self.nonlinear(out3)

        out = torch.cat((out1, out2, out3), dim=1)
        out = torch.reshape(out, (out.shape[0], out.shape[1] // 3, 3, out.shape[2], out.shape[3]))
        out = torch.sum(out, dim=2)
        out = out / 3

        return out


class IRNetShiftBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, shift=0.0, ratio=None, **kwargs):
        super(IRNetShiftBlock, self).__init__()
        self.conv1 = IRConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.conv2 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels
        self.shift = shift
        self.ratio = ratio
        self.ratio_map = {
            0.3: 2.5,
        }

    def forward(self, x):
        residual = x

        if self.ratio:
            self.shift = self.ratio * x.std() * self.ratio_map[self.ratio] + x.mean()
        x = x + self.shift

        fea = x.sign().flatten()
        diff = fea @ torch.ones(fea.shape).T.cuda()
        total = fea.numel()
        print(diff / total)
        
        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.nonlinear1(out)

        residual = out
        if self.ratio:
            self.shift = self.ratio * out.std() * self.ratio_map[self.ratio] + out.mean()
        out = out + self.shift
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.nonlinear2(out)

        return out


class IRNetShiftHalfBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, shift=0.0, ratio=None, **kwargs):
        super(IRNetShiftHalfBlock, self).__init__()
        self.conv1 = IRConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.conv2 = IRConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels
        self.shift = shift
        self.ratio = ratio
        self.ratio_map = {
            0.3: 2.5,
        }

    def forward(self, x):
        residual = x

        if self.ratio:
            self.shift = self.ratio * x.std() * self.ratio_map[self.ratio] + x.mean()
        x = x + self.shift
        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.nonlinear1(out)

        residual = out
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.nonlinear2(out)

        return out