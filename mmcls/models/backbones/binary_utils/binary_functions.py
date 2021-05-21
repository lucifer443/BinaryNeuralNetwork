from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import norm


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

act_name_map = {
        'hardtanh': nn.Hardtanh,
        'relu': nn.ReLU,
        'prelu_one': nn.PReLU,
        'mish': Mish,
    }


class IRNetSign(Function):
    """Sign function from IR-Net, which can add EDE progress"""
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class RANetActSign(nn.Module):
    """ReActNet's activation sign function"""
    def __init__(self):
        super(RANetActSign, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class RANetWSign(nn.Module):
    """ReActNet's weight sign function"""
    def __init__(self, clip=1):
        super(RANetWSign, self).__init__()
        self.clip = clip

    def forward(self, x):
        binary_weights_no_grad = torch.sign(x)
        cliped_weights = torch.clamp(x, -self.clip, self.clip)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        return binary_weights


class STESign(nn.Module):
    """a sign function using STE"""
    def __init__(self, clip=1):
        super(STESign, self).__init__()
        assert clip > 0
        self.clip = clip

    def forward(self, x):
        out_no_grad = torch.sign(x)
        cliped_out = torch.clamp(x, -self.clip, self.clip)
        out = out_no_grad.detach() - cliped_out.detach() + cliped_out

        return out


class RPRelu(nn.Module):
    """RPRelu form ReActNet"""
    def __init__(self, in_channels, **kwargs):
        super(RPRelu, self).__init__()
        self.bias1 = LearnableBias(in_channels)
        self.prelu = nn.PReLU(in_channels)
        self.bias2 = LearnableBias(in_channels)

    def forward(self, x):
        x = self.bias1(x)
        x = self.prelu(x)
        x = self.bias2(x)
        return x


class LearnableBias(nn.Module):
    def __init__(self, channels, init=0):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.ones(1, channels, 1, 1) * init, requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class LearnableScale(nn.Module):
    hw_settings = {
        64: 56,
        128: 28,
        256: 14,
        512: 7,
    }
    def __init__(self, channels):
        super(LearnableScale, self).__init__()
        self.channels = channels
        self.height = self.hw_settings[channels]
        self.width = self.hw_settings[channels]
        self.alpha = nn.Parameter(torch.ones(1, self.channels, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1, 1, self.height, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, self.width), requires_grad=True)

    def forward(self, x):
        out = x * self.alpha.expand_as(x)
        out = out * self.beta.expand_as(x)
        out = out * self.gamma.expand_as(x)

        return out


class FeaExpand(nn.Module):
    """expand feature map

    mode:
        1: 根据特征图绝对值最大值均匀选择阈值
        1c: 1的基础上分通道
        2: 仅限于2张特征图，第1张不变，第2张绝对值小的映射为+1，绝对值大的映射为-1
        3: 根据均值方差选择阈值（一个batch中的所有图片计算一个均值和方差）
        3n: 3的基础上分输入，每个输入图片计算自己的均值方差
        3c: 3的基础上分通道计算均值方差（类似bn）
        3nc: 3的基础上既分输入也分通道计算均值方差
        4: 使用1的值初始化的可学习的阈值
        4c: 4的基础上分通道
        5:
        6: 按照数值的个数均匀选择阈值，由直方图计算得到
    """
    def __init__(self, expansion=3, mode='1', in_channels=None):
        super(FeaExpand, self).__init__()
        self.expansion = expansion
        self.mode = mode
        if '1' in self.mode:
            self.alpha = [-1 + (i + 1) * 2 / (expansion + 1) for i in range(expansion)]

        elif '3' in self.mode or '6' in self.mode:
            self.alpha = [(i + 1) / (expansion + 1) for i in range(expansion)]
            self.ppf_alpha = [norm.ppf(alpha) for alpha in self.alpha]

        elif '4' == self.mode:
            self.move = nn.ModuleList()
            for i in range(expansion):
                alpha = -1 + (i + 1) * 2 / (expansion + 1)
                self.move.append(LearnableBias(channels=1, init=alpha))

        elif '4c' == self.mode:
            self.move = nn.ModuleList()
            for i in range(expansion):
                alpha = -1 + (i + 1) * 2 / (expansion + 1)
                self.move.append(LearnableBias(in_channels, init=alpha))

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
        if self.expansion == 1:
            return x

        out = []
        if self.mode == '1':
            x_max = x.abs().max()
            out = [x + alpha * x_max for alpha in self.alpha]

        elif self.mode == '1c':
            x_max = x.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            out = [x + alpha * x_max for alpha in self.alpha]

        elif self.mode == '2':
            bias = x.abs().max() / 2
            out.append(x)
            out.append(-x.abs() + bias)

        elif self.mode == '3':
            mean = x.mean().item()
            std = x.std().item()
            out = [x + (a * std + mean) for a in self.ppf_alpha]

        elif self.mode == '3n':
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
            out = [x + (a * std + mean) for a in self.ppf_alpha]

        elif self.mode == '3c':
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            std = x.std(dim=(0, 2, 3), keepdim=True)
            out = [x + (a * std + mean) for a in self.ppf_alpha]

        elif self.mode == '3nc':
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True)
            out = [x + (a * std + mean) for a in self.ppf_alpha]

        elif self.mode == '4' or self.mode == '4c':
            out = [self.move[i](x) for i in range(self.expansion)]

        elif self.mode == '6':
            thres = self.compute_thres(x)
            out = [x + t for t in thres]

        elif self.mode == '6n':
            n = x.size(0)
            thres_n = [self.compute_thres(x_n) for x_n in x]
            thres_n_tensor = torch.stack(thres_n)
            thres_n_tensor = thres_n_tensor.T.reshape((self.expansion, n, 1, 1, 1))
            out = [x + t for t in thres_n_tensor]
        
        elif self.mode == '6c':
            c = x.size(1)
            thres_c = [self.compute_thres(x_c) for x_c in x.transpose(0, 1)]
            thres_c_tensor = torch.stack(thres_c)
            thres_c_tensor = thres_c_tensor.T.reshape((self.expansion, 1, c, 1, 1))
            out = [x + t for t in thres_c_tensor]
        
        elif self.mode == '6nc':
            n, c, h, w = x.shape
            thres_nc = [self.compute_thres(x_nc) for x_nc in x.reshape(n * c , h, w)]
            thres_nc_tensor = torch.stack(thres_nc)
            thres_nc_tensor = thres_nc_tensor.T.reshape((self.expansion, n, c, 1, 1))
            out = [x + t for t in thres_nc_tensor]

        return torch.cat(out, dim=1)