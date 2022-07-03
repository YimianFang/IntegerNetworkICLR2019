from re import L
from ICLR_integerutilsl import IntConv2dTrans, IntConv2d, BitEstimator
import torch
import torch.nn as nn
import math
import os
import numpy as np

def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))

def load_resume_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0

def load_pretrain_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        for k in pretrained_dict:
            if "deconv" in k and "weight" in k:
                pretrained_dict[k] = torch.flip(pretrained_dict[k].permute(1, 0, 2, 3), [2,3])
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0

class Encoder(nn.Module):

    def __init__(self, out_channel_N=128, out_channel_M=192):
        super(Encoder, self).__init__()
        self.ConvLayer1 = IntConv2d(3, out_channel_N, kernel=5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.ConvLayer1.conv.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.ConvLayer1.conv.bias.data, 0.01)
        self.ConvLayer2 = IntConv2d(out_channel_N, out_channel_N, kernel=5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.ConvLayer2.conv.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.ConvLayer2.conv.bias.data, 0.01)
        self.ConvLayer3 = IntConv2d(out_channel_N, out_channel_N, kernel=5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.ConvLayer3.conv.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.ConvLayer3.conv.bias.data, 0.01)
        self.ConvLayer4 = IntConv2d(out_channel_N, out_channel_M, kernel=5, stride=2, padding=2, if_relu=False, if_fltdvd=True)
        torch.nn.init.xavier_normal_(self.ConvLayer4.conv.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.ConvLayer4.conv.bias.data, 0.01)

    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        return self.ConvLayer4(x)

class priorEncoder(nn.Module):

    def __init__(self, out_channel_N=128, out_channel_M=192):
        super(priorEncoder, self).__init__()
        self.ConvLayer1 = IntConv2d(out_channel_M, out_channel_N, kernel=3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.ConvLayer1.conv.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.ConvLayer1.conv.bias.data, 0.01)
        self.ConvLayer2 = IntConv2d(out_channel_N, out_channel_N, kernel=5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.ConvLayer2.conv.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.ConvLayer2.conv.bias.data, 0.01)
        self.ConvLayer3 = IntConv2d(out_channel_N, out_channel_N, kernel=5, stride=2, padding=2, if_relu=False, if_fltdvd=True)
        torch.nn.init.xavier_normal_(self.ConvLayer3.conv.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.ConvLayer3.conv.bias.data, 0.01)

    def forward(self, x):
        x = torch.abs(x)
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        return self.ConvLayer3(x)

class priorDecoder(nn.Module):

    def __init__(self, out_channel_N=128, out_channel_M=192, s_min=0.11, s_max=256, level=256):
        super(priorDecoder, self).__init__()
        self.DeconvLayer1 = IntConv2dTrans(out_channel_N, out_channel_N, kernel=5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.DeconvLayer1.deconv.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.DeconvLayer1.deconv.bias.data, 0.01)
        self.DeconvLayer2 = IntConv2dTrans(out_channel_N, out_channel_N, kernel=5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.DeconvLayer2.deconv.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.DeconvLayer2.deconv.bias.data, 0.01)
        # self.DeconvLayer3 = IntConv2dTrans(out_channel_N, out_channel_M, kernel=3, stride=1, padding=1, output_padding=0, if_relu=False, if_fltdvd=True)
        self.DeconvLayer3 = IntConv2dTrans(out_channel_N, out_channel_M, kernel=3, stride=1, padding=1, output_padding=0, if_relu=False, if_fltdvd=True)
        torch.nn.init.xavier_normal_(self.DeconvLayer3.deconv.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.DeconvLayer3.deconv.bias.data, 0.01)
        self.log_min = np.log(s_min)
        self.log_max = np.log(s_max)
        self.L = level

    def forward(self, x):
        x = self.DeconvLayer1(x)
        x = self.DeconvLayer2(x)
        x = self.DeconvLayer3(x)
        # x = self.sigma(x)
        x = torch.exp(x)
        return x
    
    def sigma(self, x):
        return torch.exp(self.log_min + x * (self.log_max - self.log_min) / (self.L - 1))

class Decoder(nn.Module):

    def __init__(self, out_channel_N=128, out_channel_M=192):
        super(Decoder, self).__init__()
        self.DeconvLayer1 = IntConv2dTrans(out_channel_M, out_channel_N, kernel=5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.DeconvLayer1.deconv.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.DeconvLayer1.deconv.bias.data, 0.01)
        self.DeconvLayer2 = IntConv2dTrans(out_channel_N, out_channel_N, kernel=5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.DeconvLayer2.deconv.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.DeconvLayer2.deconv.bias.data, 0.01)
        self.DeconvLayer3 = IntConv2dTrans(out_channel_N, out_channel_N, kernel=5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.DeconvLayer3.deconv.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.DeconvLayer3.deconv.bias.data, 0.01)
        self.DeconvLayer4 = IntConv2dTrans(out_channel_N, 3, kernel=5, stride=2, padding=2, output_padding=1, if_relu=False, if_fltdvd=True)
        torch.nn.init.xavier_normal_(self.DeconvLayer4.deconv.weight.data, (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.DeconvLayer4.deconv.bias.data, 0.01)
        torch.nn.init.constant_(self.DeconvLayer4.c, 8.0)

    def forward(self, x):
        x = self.DeconvLayer1(x)
        x = self.DeconvLayer2(x)
        x = self.DeconvLayer3(x)
        x = self.DeconvLayer4(x)
        return x

class IntegerModel(nn.Module):
    def __init__(self, out_channel_N=128, out_channel_M=192):
        super(IntegerModel, self).__init__()
        self.Encoder = Encoder(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Decoder(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = priorEncoder(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = priorDecoder(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 64, input_image.size(3) // 64).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
        feature = self.Encoder(input_image)
        batch_size = feature.size()[0]

        z = self.priorEncoder(feature)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
        recon_sigma = self.priorDecoder(compressed_z)
        feature_renorm = feature
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        recon_image = self.Decoder(compressed_feature_renorm)
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        def feature_probs_based_sigma(feature, sigma):
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-10, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z
        return clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp