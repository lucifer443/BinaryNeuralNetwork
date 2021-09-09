from torch.autograd import Function
import torch
import torch.nn as nn
import random

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


class biasadd(Function):
    """Sign function from IR-Net, which can add EDE progress"""
    @staticmethod
    def forward(ctx, input, b):
        ctx.save_for_backward(input, b)
        out = input + b
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, b = ctx.saved_tensors
        grad_input = grad_output
        mask = (input+b).sign()*grad_output<0
        mask0 = (input+b).sign()*grad_output>0
        dif = mask.float()*grad_input
        mask1 = dif>0
        mask2 = dif<0
        grad0 = mask0.float().sum(dim=[0,2,3],keepdim=True)
        grad1 = mask1.float().sum(dim=[0,2,3],keepdim=True)
        grad2 = mask2.float().sum(dim=[0,2,3],keepdim=True)

        g0 = (grad0>grad1)*(grad0>grad2)
        g1 = (grad1>grad0)*(grad1>grad2)
        g2 = (grad2>grad0)*(grad2>grad1)


        grad_b = 0*g0.float()+g1.float()-g2.float()
 
        return grad_input, grad_b

class biasadd22(Function):

    @staticmethod
    def forward(ctx, input, b):
        ctx.save_for_backward(input, b)
        out = input + b
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, b = ctx.saved_tensors
        grad_input = grad_output
        mask = (input+b).sign()*grad_output<0
        dif = mask.float()*grad_input
        mask1 = dif>0
        mask2 = dif<0

        maskf = grad_output>0
        maskz = grad_output<0

        grad0 =(maskf.float().sum(dim=[0,2,3],keepdim=True))>(maskz.float().sum(dim=[0,2,3],keepdim=True)) #正梯度多
        grad1 =(maskf.float().sum(dim=[0,2,3],keepdim=True))<(maskz.float().sum(dim=[0,2,3],keepdim=True)) #负梯度多
        
        grad2 = grad0*((mask1.float().sum(dim=[0,2,3],keepdim=True))>(mask2.float().sum(dim=[0,2,3],keepdim=True))) #往正
        grad3 = grad0*((mask1.float().sum(dim=[0,2,3],keepdim=True))<(mask2.float().sum(dim=[0,2,3],keepdim=True))) #不动
        
        grad4 = grad1*((mask1.float().sum(dim=[0,2,3],keepdim=True))>(mask2.float().sum(dim=[0,2,3],keepdim=True))) #不动
        grad5 = grad1*((mask1.float().sum(dim=[0,2,3],keepdim=True))<(mask2.float().sum(dim=[0,2,3],keepdim=True)))  #往负
 



        grad_b = grad2.float()+0*grad3.float()+0*grad4.float()-grad5.float()
 
        return grad_input, grad_b

class biasaddtry(Function):
    """Sign function from IR-Net, which can add EDE progress"""
    @staticmethod
    def forward(ctx, input, b,st):
        ctx.save_for_backward(input, b,st)
        out = input + b
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, b,st = ctx.saved_tensors
        grad_input = grad_output
        bi = (input+b).sign()
        mask1 = grad_input<0
        mask2 = grad_input>0
        biz = bi>0
        bif = bi<0
        
        mask3 = mask1*biz
        mask4 = mask2*bif
        realbi = bi-2*(mask3.float())+2*(mask4.float())

        
        maez = (realbi-(input+b+st).sign()).abs().sum(dim=[0,2,3],keepdim=True) 
        maef = (realbi-(input+b-st).sign()).abs().sum(dim=[0,2,3],keepdim=True)
        mae0 = (realbi-bi).abs().sum(dim=[0,2,3],keepdim=True)

        gradz = (maez<maef)*(maez<mae0)
        gradf = (maef<maez)*(maef<mae0)
        grad0 = (mae0<maez)*(mae0<maef)
        grad_b = gradz.float()-gradf.float()+0*grad0.float()


 
        return grad_input, grad_b,None
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

class GPRPRelu(nn.Module):
    """RPRelu form ReActNet"""
    def __init__(self, in_channels,gp=1,**kwargs):
        super(GPRPRelu, self).__init__()
        self.gp = gp
        self.in_channels = in_channels
        self.gprelu = RPRelu(self.in_channels//gp)


    def forward(self, x):
        x_sp = x.chunk(self.gp,1)
        out_sp = []
        for i in range(self.gp):
            out_sp.append(self.gprelu(x_sp[i]))
        out = torch.cat(out_sp,dim=1)
        return out

class MGPRPRelu(nn.Module):
    """RPRelu form ReActNet"""
    def __init__(self, in_channels,gp=1,**kwargs):
        super(MGPRPRelu, self).__init__()
        self.gp = gp
        self.bias1 = LearnableBias(in_channels)
        self.bias2 = LearnableBias(in_channels)
        self.in_channels = in_channels
        self.gprelu = nn.PReLU(self.in_channels//gp)


    def forward(self, x):
        x = self.bias1(x)
        x_sp = x.chunk(self.gp,1)
        out_sp = []
        for i in range(self.gp):
            out_sp.append(self.gprelu(x_sp[i]))
        out = torch.cat(out_sp,dim=1)
        out = self.bias2(out)
        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class scalebias(nn.Module):
    def __init__(self,out_chn):
        super(scalebias,self).__init__()
        self.scale = nn.Parameter(torch.ones(1,out_chn,1,1),requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
    
    def forward(self,x):
        out = x*self.scale.expand_as(x)
        out = out + self.bias.expand_as(x)
        return out 

class GPLearnableBias(nn.Module):
    def __init__(self, out_chn,gp=1):
        super(GPLearnableBias, self).__init__()
        self.gp = gp
        self.bias = nn.Parameter(torch.zeros(1,out_chn//gp,1,1), requires_grad=True)

    def forward(self, x):
        out = x + (self.bias.repeat(1,self.gp,1,1)).expand_as(x)
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

class selfBias(nn.Module):
    def __init__(self):
        super(selfBias, self).__init__()
        self.af1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.af2 = nn.Parameter(torch.ones(1), requires_grad=True)
        #self.af2 = nn.Parameter(0.1, requires_grad=True)

    def forward(self, x):
        mask1 = x<-0.5
        mask2 = x<= 0
        mask3 = x<0.5
        out1 = (x) * mask1.type(torch.float32) + (x-self.af1*(1-x*x)) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32)+ (x-self.af2*(1-x*x))*(1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + (x) * (1- mask3.type(torch.float32))
        return out3

class AttentionScale(nn.Module):
    """attention scale from Real-To-Binary Net"""
    ##缺少relu和sigmod
    scale_factor = 8
    def __init__(self, channels):
        super(AttentionScale, self).__init__()
        self.fc1 = nn.Linear(channels, channels // self.scale_factor)
        self.fc2 = nn.Linear(channels // self.scale_factor, channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        x = self.pool(inputs)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return inputs * x[:, :, None, None]

class Expandx(nn.Module):

    aerfa3_settings =[[-0.5,0,0.5],[-0.618,0,0.618],[-0.9,0,0.9],[-0.9,0,0.1],
                     [-0.3,0,0.7],[-0.1,0,0.9],[-0.1,0,0.1],[-0.33,0,0.33]]
    
    aerfa2_settings =[[-0.5,0.5],[-0.618,0.618],[-0.9,0.9],[-0.9,0.1],
                     [-0.3,0.7],[-0.1,0.9],[-0.1,0.1],[-0.33,0.33]]
    aerfa_settings = [-0.33,0.33]
    def __init__(self, Expand_num=1,in_channels=None,):
        super(Expandx, self).__init__()
        self.Expand_num = Expand_num


    def forward(self, x):
        if self.Expand_num == 1:
            return x

        out = []
        if self.Expand_num==2:
            #dex = random.randint(0,7)
            out = [x + alpha  for alpha in self.aerfa_settings]

        elif self.Expand_num==3:
            dex = random.randint(0,7)
            out = [x + alpha  for alpha in self.aerfa3_settings[dex]]    
        return torch.cat(out, dim=1)