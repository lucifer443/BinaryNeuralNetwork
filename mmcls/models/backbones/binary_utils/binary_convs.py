import torch.nn as nn
import torch.nn.functional as F
from .binary_functions import IRNetSign, RANetActSign, RANetWSign
import torch
import math


class BaseBinaryConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), **kwargs):
        super(BaseBinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mode = binary_type

    def binary_weight(self, x):
        pass

    def binary_input(self, w):
        pass

    def forward(self, input):
        x = self.binary_input(input) if self.mode[0] else input
        w = self.binary_weight(self.weight) if self.mode[1] else self.weight
        output =  F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class IRConv2d(BaseBinaryConv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), **kwargs):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binary_type, **kwargs)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def binary_input(self, x):
        return IRNetSign().apply(x, self.k, self.t)

    def binary_weight(self, w):
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()
        bw = IRNetSign().apply(bw, self.k, self.t)
        return bw * sw

    def ede(self, k, t):
        self.k = k
        self.t = t


class RAConv2d(BaseBinaryConv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), **kwargs):
        super(RAConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binary_type, **kwargs)
        self.sign_a = RANetActSign()
        self.sign_w = RANetWSign()

    def binary_input(self, x):
        return self.sign_a(x)

    def binary_weight(self, w):
        bw = self.sign_w(w)
        sw = torch.mean(torch.mean(torch.mean(abs(w),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True).detach()
        return bw * sw


class STEConv2d(BaseBinaryConv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), **kwargs):
        super(STEConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binary_type, **kwargs)
        self.sign_a = RANetWSign(clip=1.25)
        self.sign_w = RANetWSign(clip=1.25)

    def binary_input(self, x):
        return self.sign_a(x)

    def binary_weight(self, w):
        return self.sign_w(w)


class ANDConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ANDConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input):
        w = self.weight
        a = input
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        bw = (IRNetSign().apply(bw, self.k, self.t) + 1.0) / 2.0
        ba = (IRNetSign().apply(a, self.k, self.t) + 1.0) / 2.0
        bw = bw * sw
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output


class CorrBiasConv2d(RAConv2d):
    hw_settings = {
        64: 56,
        128: 28,
        256: 14,
        512: 7,
    }
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), **kwargs):
        super(CorrBiasConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                       binary_type, **kwargs)
        self.register_buffer('bias_w', torch.zeros(1, 1, self.hw_settings[out_channels], 1))
        self.register_buffer('bias_h', torch.zeros(1, 1, 1, self.hw_settings[out_channels]))
        self.register_buffer('bias_c', torch.zeros(1, self.hw_settings[out_channels], 1, 1))
        self.alpha = 0.99

    def forward(self, input):
        x = self.binary_input(input) if self.mode[0] else input
        w = self.binary_weight(self.weight) if self.mode[1] else self.weight
        boutput =  F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.training:
            foutput =  F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            err = foutput - boutput
            bias_w = torch.mean(err, dim=[0, 1, 2], keepdim=True)
            bias_h = torch.mean(err, dim=[0, 1, 3], keepdim=True)
            bias_c = torch.mean(err, dim=[0, 2, 3], keepdim=True)
            self.bias_w[:] = self.bias_w[:] * self.alpha + bias_w * (1-self.alpha)
            self.bias_h[:] = self.bias_h[:] * self.alpha + bias_h * (1-self.alpha)
            self.bias_c[:] = self.bias_c[:] * self.alpha + bias_c * (1-self.alpha)
            output = boutput + bias_w + bias_h + bias_c
        else:
            output = boutput + self.bias_w + self.bias_h + self.bias_c
        return output