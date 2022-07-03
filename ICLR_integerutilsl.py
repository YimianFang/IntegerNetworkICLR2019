import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

class IntConv2dTrans(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=5, stride=2, padding=2, output_padding=1,
                    if_relu=True, if_fltdvd=False, wBit=8, actBit=8):
        super(IntConv2dTrans, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride, padding, output_padding)
        self.relu = nn.ReLU()
        self.c = nn.Parameter(torch.zeros([1, out_channel, 1, 1]))
        # torch.nn.init.uniform_(self.c, a=1.0, b=10.0)
        torch.nn.init.constant_(self.c, 2.0)
        self.if_relu = if_relu
        self.if_fltdvd = if_fltdvd
        self.K = wBit
        self.L = actBit

    def forward(self, x):
        x = F.conv_transpose2d(x, 
                            weight=wSTE_Trans.apply(self.deconv.weight, self.K), 
                            bias=bSTE.apply(self.deconv.bias, self.K), 
                            stride=self.deconv.stride, 
                            padding=self.deconv.padding, 
                            output_padding=self.deconv.output_padding)
        if self.if_relu:
            x = self.simu_intdvd(x, cSTE.apply(self.c, self.K))
            x = qrelu.apply(x, self.L)
        elif self.if_fltdvd:
            x = x / cSTE.apply(self.c, self.K)
        else:
            x = self.simu_intdvd(x, cSTE.apply(self.c, self.K))

        return x
    
    def simu_intdvd(self, a, b):
        if self.training:
            return a / b
        else:
            b_2 = torch.div(b, 2, rounding_mode='trunc')
            return torch.div((a + b_2), b, rounding_mode='trunc')

class IntConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=5, stride=2, padding=2,
                    if_relu=True, if_fltdvd=False, wBit=8, actBit=8):
        super(IntConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding)
        self.relu = nn.ReLU()
        self.c = nn.Parameter(torch.zeros([1, out_channel, 1, 1]))
        # torch.nn.init.uniform_(self.c, a=1.0, b=10.0)
        torch.nn.init.constant_(self.c, 2.0)
        self.if_relu = if_relu
        self.if_fltdvd = if_fltdvd
        self.K = wBit
        self.L = actBit

    def forward(self, x):
        x = F.conv2d(x, 
                    weight=wSTE.apply(self.conv.weight, self.K), 
                    bias=bSTE.apply(self.conv.bias, self.K), 
                    stride=self.conv.stride, 
                    padding=self.conv.padding)
        if self.if_relu:
            x = self.simu_intdvd(x, cSTE.apply(self.c, self.K))
            x = qrelu.apply(x, self.L)
        elif self.if_fltdvd:
            x = x / cSTE.apply(self.c, self.K)
        else:
            x = self.simu_intdvd(x, cSTE.apply(self.c, self.K))

        return x

    """
    training: float point devide
    inference: integer devide
    """    
    def simu_intdvd(self, a, b):
        if self.training:
            return a / b
        else:
            b_2 = torch.div(b, 2, rounding_mode='trunc')
            return torch.div((a + b_2), b, rounding_mode='trunc')

class wSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k=8):
        input_cpy = copy.deepcopy(input.data)
        n_upper = 2 ** (k - 1) - 1
        n_lower = - n_upper - 1
        input_cpy_re = input_cpy.reshape(input_cpy.shape[0], -1)
        scale = F.relu(torch.max(input_cpy_re.max(-1)[0] / n_upper, input_cpy_re.min(-1)[0] / n_lower).reshape(-1, 1, 1, 1) - 1e-20) + 1e-20
        ctx.save_for_backward(scale)
        input_out = torch.round(input_cpy / scale)
        return input_out ###return是tensor不会改变self.weight的值，如果还是Parameter会连接在一起, 但是梯度都会传播.

    @staticmethod
    def backward(ctx, grad_output):
        scale, = ctx.saved_tensors
        grad_output /= scale
        # print("wSTE")
        # print(grad_output)
        return grad_output, None, None, None

class wSTE_Trans(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, K=8):
        input_cpy = copy.deepcopy(input.data)
        n_upper = 2 ** (K - 1) - 1
        n_lower = - n_upper - 1
        input_cpy_re = input_cpy.permute(1, 0, 2, 3).reshape(input_cpy.shape[1], -1)
        scale = F.relu(torch.max(input_cpy_re.max(-1)[0] / n_upper, input_cpy_re.min(-1)[0] / n_lower).reshape(1, -1, 1, 1) - 1e-20) + 1e-20
        ctx.save_for_backward(scale)
        input_out = torch.round(input_cpy / scale)
        return input_out ###return是tensor不会改变self.weight的值，如果还是Parameter会连接在一起, 但是梯度都会传播.

    @staticmethod
    def backward(ctx, grad_output):
        scale, = ctx.saved_tensors
        grad_output /= scale
        # print("wSTE_Trans")
        # print(grad_output)
        return grad_output, None, None, None

class bSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, K=8):
        input_cpy = copy.deepcopy(input.data)
        scale = torch.tensor(2 ** K)
        ctx.save_for_backward(scale)
        input_out = torch.round(input_cpy * scale)
        return input_out ###return是tensor不会改变self.weight的值，如果还是Parameter会连接在一起, 但是梯度都会传播.

    @staticmethod
    def backward(ctx, grad_output):
        scale, = ctx.saved_tensors
        grad_output *= scale
        # print("bSTE")
        # print(grad_output)
        return grad_output, None, None, None

class cSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, K=8):
        c_cpy = copy.deepcopy(c.data)
        c_thd = np.sqrt(1 + 10 ** -40)
        m = F.relu(c_cpy - c_thd) + c_thd
        r_c = m ** 2 - (10 ** -40)
        scale = torch.tensor(2 ** K)
        ctx.save_for_backward((c_cpy >= c_thd), m * scale)
        input_out = torch.round(r_c * scale)
        return input_out ###return是tensor不会改变self.weight的值，如果还是Parameter会连接在一起, 但是梯度都会传播.

    @staticmethod
    def backward(ctx, grad_output):
        idx_c, m_scale, = ctx.saved_tensors
        idx = idx_c | (grad_output < 0)
        grad_output *= 2 * m_scale * idx
        # print("cSTE")
        # print(grad_output)
        return grad_output, None, None, None

class qrelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, L=8):
        ctx.upper = 2 ** L - 1
        alpha = 0.9943258522851727 # (1/4*Gamma(1/4))**4
        grad_scale = torch.exp((-alpha**4) * torch.abs(2.0 / ctx.upper * input - 1) ** 4)
        ctx.save_for_backward(input, grad_scale)
        input_out = torch.round(torch.clip(input, 0, ctx.upper))
        return input_out

    @staticmethod
    def backward(ctx, grad_output):
        input, grad_scale, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_sub = grad_scale * grad_output.clone()
        grad_input[input < 0] = grad_sub[input < 0]
        grad_input[input > ctx.upper] = grad_sub[input > ctx.upper]
        # print("qrelu")
        # print(grad_output)
        return grad_input, None, None, None


class Bitparm(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)

class BitEstimator(nn.Module):
    '''
    Estimate bit
    '''
    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)
        
    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)