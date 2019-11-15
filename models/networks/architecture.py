"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
from models.networks.reflect_conv import Conv2d

ConvLayer = Conv2d

# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = ConvLayer(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = ConvLayer(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = ConvLayer(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(ConvLayer(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(ConvLayer(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class BooruNet(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        model = torch.hub.load('RF5/danbooru-pretrained', 'resnet18')[0]
        self.slice1 = model[:4]
        self.slice2 = model[4]
        self.slice3 = model[5]
        self.slice4 = model[6]
        self.slice5 = model[7]
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class LadderNet(torch.nn.Module):
    def __init__(self, nf):
        super().__init__()
        def convlayer(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
            return nn.Sequential(
                SynchronizedBatchNorm2d(in_channels),
                Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
                nn.LeakyReLU(0.2)
            )
        def basicblock(in_channels, out_channels, kernel_size=3, bias=False):
            return nn.Sequential(
                convlayer(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=bias),
                convlayer(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=bias)
            )
        self.base = convlayer(5, nf, 3, 1, 1)   # output H: 256
        self.down_3 = convlayer(nf, nf, 3, 1, 1)  # 256
        self.down_2 = convlayer(nf, nf, 3, 2, 1)    # 128
        self.down_1 = convlayer(nf, nf, 3, 2, 1)    # 64
        self.down_0 = convlayer(nf, nf, 3, 2, 1)   # 32
        self.L_middle_1 = convlayer(nf, nf, 3, 2, 1)  # 16
        self.L_middle_0 = convlayer(nf, nf, 3, 1, 1)  # 16
        self.head_0 = convlayer(nf, nf, 3, 2, 1)    # 8

    def forward(self, x):
        x = self.base(x)
        down_3 = self.down_3(x)
        down_2 = self.down_2(down_3)
        down_1 = self.down_1(down_2)
        down_0 = self.down_0(down_1)
        L_middle_1 = self.L_middle_1(down_0)
        L_middle_0 = self.L_middle_0(L_middle_1)
        head_0 = self.head_0(L_middle_0)
        return down_3, down_2, down_1, down_0, L_middle_1, L_middle_0, head_0
