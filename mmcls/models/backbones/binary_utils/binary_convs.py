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


class IRConv2d_bias(BaseBinaryConv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), **kwargs):
        super(IRConv2d_bias, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binary_type, **kwargs)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.bias_buffer=0

    def float_weight(self,w):
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()
        bw = (self.k)*torch.tanh((self.t)*bw)
        return bw * sw

    def float_x(self,x):
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        return out3

    
    def ede(self, k, t):
        self.k = k
        self.t = t
   

    def forward(self, input):

        floatx = self.float_x(input)
        float_w = self.float_weight(self.weight)
        output = F.conv2d(floatx, float_w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return(output)        


class IRConv2d_bias_x2(BaseBinaryConv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), **kwargs):
        super(IRConv2d_bias_x2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binary_type, **kwargs)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.sign = RANetActSign()
        bias_bi = torch.zeros(1,out_channels,1,1)
        self.aerfa = 0.999
        self.register_buffer('bias_bi',bias_bi)

    def binary_input(self, x):
        return self.sign(x)

    def binary_weight(self, w):
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()
        bw = IRNetSign().apply(bw, self.k, self.t)
        return bw * sw
    
    def float_weight(self,w):
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()
        bw = (self.k)*torch.tanh((self.t)*bw)
        return bw * sw

    def float_x(self,x):
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        return out3

    
    def ede(self, k, t):
        self.k = k
        self.t = t
   

    def forward(self, input):
        x = self.binary_input(input) if self.mode[0] else input
        w = self.binary_weight(self.weight) if self.mode[1] else self.weight
        floatx = self.float_x(input)
        float_w = self.float_weight(self.weight)
        output1 =  F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        
        if self.training is True:
            output2 =  F.conv2d(floatx, float_w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            with torch.no_grad():
                err = output2 - output1
                mybias = err.mean(dim=0,keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True)
                #mybias = err.mean(dim=0,keepdim=True)
                self.bias_bi[:] = self.bias_bi[:]*self.aerfa +mybias*(1-self.aerfa)
                output =output1 + mybias
            return(output)        
        else:
            output = F.conv2d(floatx, float_w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return(output)        

class IRConv2d_bias_x2x(BaseBinaryConv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), **kwargs):
        super(IRConv2d_bias_x2x, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binary_type, **kwargs)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.sign = RANetActSign()
        bias_bi = torch.zeros(1,out_channels,1,1)
        self.aerfa = 0.999
        self.register_buffer('bias_bi',bias_bi)

    def binary_input(self, x):
        return self.sign(x)

    def binary_weight(self, w):
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()
        bw = IRNetSign().apply(bw, self.k, self.t)
        return bw * sw
    
    def float_weight(self,w):
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()
        bw = (self.k)*torch.tanh((self.t)*bw)
        return bw * sw

    def float_x(self,x):
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        return out3

    
    def ede(self, k, t):
        self.k = k
        self.t = t
   

    def forward(self, input):
        x = self.binary_input(input) if self.mode[0] else input
        w = self.binary_weight(self.weight) if self.mode[1] else self.weight
        floatx = self.float_x(input)
        float_w = self.float_weight(self.weight)
        output1 =  F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        
        if self.training is True:
            output2 =  F.conv2d(floatx, float_w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            with torch.no_grad():
                err = output2 - output1
                #print(err.shape)
                mybias = err.mean(dim=0,keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True)
                #mybias = err.mean(dim=0,keepdim=True)
                self.bias_bi[:] = self.bias_bi[:]*self.aerfa +mybias*(1-self.aerfa)
                output =output1 + mybias
            return(output)        
        else:
            output = output1 + self.bias_bi[:]
            return(output)        