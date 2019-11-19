"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.architecture import VGG19, BooruNet


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class BooruLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(BooruLoss, self).__init__()
        self.net = BooruNet().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_net, y_net = self.net(x), self.net(y)
        loss = 0
        for i in range(len(x_net)):
            loss += self.weights[i] * self.criterion(x_net[i], y_net[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# Loss on HSV Color Space

def rgb2hsv(image):
    """
    :param image: NCHW torch.Tensor (RGB) -1 ~ 1
    :return: NCHW torch.Tensor (HSV)  0 ~ 1
    """
    img = image * 0.5 + 0.5
    hue = torch.Tensor(image.shape[0], image.shape[2], image.shape[3]).to(image.device)
    # max over dimension 1 (C), [0] for get max value. ([1] -> max index)
    # B
    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + 1e-07))[
        img[:, 2] == img.max(1)[0]]
    # G
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + 1e-07))[
        img[:, 1] == img.max(1)[0]]
    # R
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + 1e-07))[
        img[:, 0] == img.max(1)[0]]) % 6

    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue = hue / 6

    saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + 1e-07)
    saturation[img.max(1)[0] == 0] = 0

    value = img.max(1)[0]
    return torch.stack((hue, saturation, value), dim=1)


class HSVTVLoss(nn.Module):
    """
    Only calculates total variation on HUE channel.
    """
    def forward(self, fake_image, sketch):

        def tv_loss(hsv_image):
            img = hsv_image[:, [0, ], :, :]
            h1 = img[:, :, 1:, :]
            h2 = img[:, :, :-1, :]
            w1 = img[:, :, :, 1:]
            w2 = img[:, :, :, :-1]
            vert = F.interpolate((h1 - h2) ** 2, size=img.shape[2:4], mode='bilinear', align_corners=True)
            horiz = F.interpolate((w1 - w2) ** 2, size=img.shape[2:4], mode='bilinear', align_corners=True)
            loss = vert + horiz
            return loss

        hsv = rgb2hsv(fake_image)
        loss = tv_loss(hsv)
        mask = (sketch > 0).float()  # Do not apply TV loss on the sketch
        return (loss * mask).mean()


class HighSVLoss(nn.Module):
    def forward(self, fake_image, sketch):
        hsv = rgb2hsv(fake_image)
        val = hsv[:, [2, ], :, :]
        hue = hsv[:, [1, ], :, :]
        v_margin = ((val < 0.1) * (sketch > 0)).float()
        h_margin = ((hue < 0.1) * (sketch > 0)).float()
        v_loss = ((1 - val) ** 2)
        h_loss = ((1 - hue) ** 2)
        loss = v_loss * v_margin + h_loss * h_margin
        return loss.mean()

