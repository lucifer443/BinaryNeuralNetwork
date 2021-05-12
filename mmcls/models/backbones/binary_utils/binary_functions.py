from torch.autograd import Function
import torch
import torch.nn as nn

from scipy.stats import norm

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
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

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
        3: 根据均值方差选择阈值
    """
    def __init__(self, expansion=3, mode='1'):
        super(FeaExpand, self).__init__()
        self.mode = mode
        if '1' in self.mode:
            self.alpha = []
            for i in range(expansion):
                self.alpha.append(-1 + (i + 1) * 2 / (expansion + 1))
        elif '3' in self.mode:
            self.alpha = []
            for i in range(expansion):
                self.alpha.append((i + 1) / (expansion + 1))

    def forward(self, x):
        out = []
        if self.mode == '1':
            x_max = x.abs().max()
            for alpha in self.alpha:
                out.append(x + alpha * x_max)
        elif self.mode == '1c':
            x_max = x.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            for alpha in self.alpha:
                out.append(x + alpha * x_max)
        elif self.mode == '2':
            bias = x.abs().max() / 2
            out.append(x)
            out.append(-x.abs() + bias)
        elif self.mode == '3':
            mean = x.mean().item()
            std = x.std().item()
            for alpha in self.alpha:
                out.append(x + norm.ppf(alpha, loc=mean, scale=std))

        return torch.cat(out, dim=1)