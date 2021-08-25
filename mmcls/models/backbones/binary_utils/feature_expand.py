from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from .binary_functions import LearnableBias, LearnableScale
from .binary_convs import BLConv2d

from scipy.stats import norm


class FeaExpand(nn.Module):
    """expand feature map

    mode:
        1: 根据特征图绝对值最大值均匀选择阈值
        1c: 1的基础上分通道
        1c-m: 根据特征图的最大值和最小值,分通道计算阈值
        1nc-m: 根据特征图的最大值和最小值，分输入分通道计算阈值
        2: 仅限于2张特征图，第1张不变，第2张绝对值小的映射为+1，绝对值大的映射为-1
        3: 根据均值方差选择阈值（一个batch中的所有图片计算一个均值和方差）
        3n: 3的基础上分输入，每个输入图片计算自己的均值方差
        3c: 3的基础上分通道计算均值方差（类似bn）
        3nc: 3的基础上既分输入也分通道计算均值方差
        4: 使用1的值初始化的可学习的阈值
        4c: 4的基础上分通道
        4g*: 4的基础上分组（是4和4c的折中）
        4s: 可学习的是阈值的系数，而不是阈值本身
        4s-a: 可学习系数使用sigmoid函数计算，多个阈值共用一个系数
        4sc-a: 4s-a的分通道版本
        4s-a-n: 4s-a的每个阈值拥有自己的可学习系数的版本
        4sc-b: 输入特征图先经过分通道的可学习scale，再使用固定阈值
        5: 手动设置阈值
        5re：专门针对expand为2的情况，将负阈值得到的结果乘-1
        5-3：在5的基础上增加一份特征图，其中绝对值小的数映射为+1，绝对值大的映射为-1
        5-mean: 以特征图的均值为轴对称应用手动设置的阈值
        6: 按照数值的个数均匀选择阈值，由直方图计算得到
        7: 根据输入计算的自适应阈值
        8: 使用conv进行通道数扩增
        8b: 在8的基础上增加bn层
        8ab: 在8的基础上增加bn层和激活层，顺序为先激活后bn
        8ba: 在8的基础上增加bn层和激活层，顺序为先bn后激活
        82: 仅限于2张特征图，第1张不变，第2张使用conv计算得到
        8bin：使用二值conv进行通道数扩增
    """
    def __init__(self, expansion=3, mode='1', in_channels=None, thres=None):
        super(FeaExpand, self).__init__()
        self.expansion = expansion
        self.mode = mode
        if '1' in self.mode:
            self.alpha = [-1 + (i + 1) * 2 / (expansion + 1) for i in range(expansion)]

        elif '3' == self.mode or '3n' == self.mode or '3c' == self.mode or '3nc' == self.mode or '6' in self.mode:
            self.alpha = [(i + 1) / (expansion + 1) for i in range(expansion)]
            self.ppf_alpha = [norm.ppf(alpha) for alpha in self.alpha]

        elif '4' == self.mode:
            self.move = nn.ModuleList()
            for i in range(expansion):
                if thres == None:
                    alpha = -1 + (i + 1) * 2 / (expansion + 1)
                else:
                    alpha = thres[i]
                self.move.append(LearnableBias(channels=1, init=alpha))

        elif '4c' == self.mode:
            self.move = nn.ModuleList()
            for i in range(expansion):
                alpha = -1 + (i + 1) * 2 / (expansion + 1)
                self.move.append(LearnableBias(in_channels, init=alpha))
        
        elif '4g' in self.mode:
            groups = int(self.mode.split('g')[-1])
            self.move = nn.ModuleList()
            for i in range(expansion):
                alpha = -1 + (i + 1) * 2 / (expansion + 1)
                self.move.append(LearnableBias(in_channels, init=alpha, groups=groups))
        
        elif '4-1' == self.mode:
            self.move = nn.ModuleList()
            for i in range(expansion):
                alpha = -1 + (i + 1) * 2 / (expansion + 1)
                self.move.append(LearnableBias(channels=1, init=alpha))
        
        elif '4s-a' == self.mode:
            assert len(thres) == expansion
            self.thres = thres
            self.thres_alpha = nn.Parameter(torch.zeros(1, 1, 1, 1), requires_grad=True)
        
        elif '4sc-a' == self.mode:
            assert len(thres) == expansion
            self.thres = thres
            self.thres_alpha = nn.Parameter(torch.zeros(1, in_channels, 1, 1), requires_grad=True)
        
        elif '4s-a-n' == self.mode:
            assert len(thres) == expansion
            self.thres = thres
            self.thres_alpha = nn.ParameterList()
            for i in range(expansion):
                self.thres_alpha.append(nn.Parameter(torch.zeros(1, 1, 1, 1), requires_grad=True))
        
        elif '4sc-b' == self.mode:
            assert in_channels != None
            assert len(thres) == expansion
            self.thres = thres
            self.scale = LearnableScale(in_channels)

        elif '5' == self.mode or '5re' == self.mode:
            assert len(thres) == expansion
            self.thres = thres
        
        elif '5-3' == self.mode or '5-mean' == self.mode:
            assert len(thres) == 2
            self.thres = thres
        
        elif '7' == self.mode:
            self.scale = nn.Parameter(torch.ones(n, c, h, w), requires_grad=True)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear1 = nn.Linear(in_channels, in_channels, bias=True)
            self.linear1 = nn.Linear(in_channels, in_channels, bias=True)
            self.tanh = nn.Tanh()
        
        elif '8' == self.mode:
            out_channels = in_channels * expansion
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        
        elif '8b' == self.mode:
            out_channels = in_channels * expansion
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
        
        elif '8ab' == self.mode or '8ba' == self.mode:
            out_channels = in_channels * expansion
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.activate = nn.PReLU(out_channels)
            self.bn = nn.BatchNorm2d(out_channels)
        
        elif '82' == self.mode:
            assert expansion == 2
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)
        
        elif '8bin' == self.mode:
            out_channels = in_channels * expansion
            self.conv = BLConv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False, binary_type=(True, False))


    def bin_id_to_thres(self, bins, bin_id, low, high):
        interval = (high - low) / bins
        thres_low = interval / 2 + low
        thres_high = high - interval / 2
        thres = [thres_low + id * interval for id in bin_id]
        while 1:
            if len(thres) < 3:
                thres.append(thres_high)
            else:
                break
        # breakpoint()
        return torch.tensor(thres).cuda()

    def compute_thres(self, fea, bins=20):
        hist = fea.histc(bins=bins)
        fea_num = fea.numel()
        alpha_index = 0
        sum = 0
        bin_id = []
        for i in range(bins):
            if alpha_index >= self.expansion:
                break
            sum = sum + hist[i]
            if sum >= fea_num * self.alpha[alpha_index]:
                bin_id.append(i)
                alpha_index += 1
        thres = self.bin_id_to_thres(bins, bin_id, fea.min(), fea.max())
        return thres

    def forward(self, x):
        if self.expansion == 1 and self.mode != '5':
            return x

        out = []
        if '1' == self.mode:
            x_max = x.abs().max()
            out = [x + alpha * x_max for alpha in self.alpha]

        elif '1c' == self.mode:
            # amin is supported by pytorch 1.8.1
            x_max = x.abs().amax(dim=(0, 2, 3), keepdim=True)
            out = [x + alpha * x_max for alpha in self.alpha]

        elif '1c-m' == self.mode:
            # amin and amax are supported by pytorch 1.8.1
            x_min = x.amin(dim=(0, 2, 3), keepdim=True)
            x_max = x.amax(dim=(0, 2, 3), keepdim=True)
            out = [x + ((alpha + 1) / 2 * (x_max - x_min) + x_min) for alpha in self.alpha]

        elif '1nc-m' == self.mode:
            # amin and amax are supported by pytorch 1.8.1
            x_min = x.amin(dim=(2, 3), keepdim=True)
            x_max = x.amax(dim=(2, 3), keepdim=True)
            out = [x + ((alpha + 1) / 2 * (x_max - x_min) + x_min) for alpha in self.alpha]

        elif '2' == self.mode:
            bias = x.abs().max() / 2
            out.append(x)
            out.append(-x.abs() + bias)

        elif '3' == self.mode:
            mean = x.mean().item()
            std = x.std().item()
            out = [x + (a * std + mean) for a in self.ppf_alpha]

        elif '3n' == self.mode:
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
            out = [x + (a * std + mean) for a in self.ppf_alpha]

        elif '3c' == self.mode:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            std = x.std(dim=(0, 2, 3), keepdim=True)
            out = [x + (a * std + mean) for a in self.ppf_alpha]

        elif '3nc' == self.mode:
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True)
            out = [x + (a * std + mean) for a in self.ppf_alpha]

        elif '4' == self.mode or '4c' == self.mode:
            out = [self.move[i](x) for i in range(self.expansion)]
        
        elif '4s-a' == self.mode or '4sc-a' == self.mode:
            # thres = thres * 2 * sigmoid(alpha)
            # alpha is the learnalbe parameter
            thres_scale = torch.sigmoid(self.thres_alpha) * 2
            out = [x + t * thres_scale for t in self.thres]
        
        elif '4s-a-n' == self.mode:
            thres_scale = [torch.sigmoid(ta) * 2 for ta in self.thres_alpha]
            out = [x + t * s for t, s in zip(self.thres, thres_scale)]
        
        elif '4sc-b' == self.mode:
            x = self.scale(x)
            out = [x + t for t in self.thres]

        elif '5' == self.mode:
            out = [x + t for t in self.thres]
        
        elif '5re' == self.mode:
            assert self.expansion == 2
            out = [x + t for t in self.thres]
            out[1] *= -1
        
        elif '5-3' == self.mode:
            out = [x + t for t in self.thres]
            out.append(x.abs() * -1 + abs(self.thres[0]))
        
        elif '5-mean' == self.mode:
            mean = x.mean(dim=(2, 3), keepdim=True)
            out = [x - mean + t for t in self.thres]

        elif '6' == self.mode:
            thres = self.compute_thres(x)
            out = [x + t for t in thres]

        elif '6n' == self.mode:
            n = x.size(0)
            thres_n = [self.compute_thres(x_n) for x_n in x]
            thres_n_tensor = torch.stack(thres_n)
            thres_n_tensor = thres_n_tensor.T.reshape((self.expansion, n, 1, 1, 1))
            out = [x + t for t in thres_n_tensor]
        
        elif '6c' == self.mode:
            c = x.size(1)
            thres_c = [self.compute_thres(x_c) for x_c in x.transpose(0, 1)]
            thres_c_tensor = torch.stack(thres_c)
            thres_c_tensor = thres_c_tensor.T.reshape((self.expansion, 1, c, 1, 1))
            out = [x + t for t in thres_c_tensor]
        
        elif '6nc' == self.mode:
            n, c, h, w = x.shape
            thres_nc = [self.compute_thres(x_nc) for x_nc in x.reshape(n * c , h, w)]
            thres_nc_tensor = torch.stack(thres_nc)
            thres_nc_tensor = thres_nc_tensor.T.reshape((self.expansion, n, c, 1, 1))
            out = [x + t for t in thres_nc_tensor]
        
        elif '72' == self.mode:
            bias = x * self.scale
            bias = self.avgpool(bias)
            bias = self.linear1(bias)
            bias = self.tanh(bias)
        
        elif '8' == self.mode or '8bin' == self.mode:
            out = self.conv(x)
            return out
        
        elif '8b' == self.mode:
            out = self.conv(x)
            out = self.bn(out)
            return out
        
        elif '8ab' == self.mode:
            out = self.conv(x)
            out = self.activate(out)
            out = self.bn(out)
            return out
        
        elif '8ba' == self.mode:
            out = self.conv(x)
            out = self.bn(out)
            out = self.activate(out)
            return out

        elif '82' == self.mode:
            out.append(x)
            out2 = self.conv(x)
            out.append(out2)
        
        return torch.cat(out, dim=1)