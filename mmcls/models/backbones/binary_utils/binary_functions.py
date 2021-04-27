from torch.autograd import Function
import torch
import torch.nn as nn


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
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
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


class RPRelu(nn.Module):
    """RPRelu form ReActNet"""
    def __init__(self, in_channels, **kwargs):
        super(RPRelu, self).__init__()
        self.bias1 = LearnableBias(in_channels)
        self.bias2 = LearnableBias(in_channels)
        self.prelu = nn.PReLU(in_channels)

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
    """scale from XNOR Net++"""
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