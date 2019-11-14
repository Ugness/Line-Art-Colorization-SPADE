import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.base_network import BaseNetwork
from models.networks.reflect_conv import Conv2d

ConvLayer = Conv2d
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d

class SketchFeatureExtractor(BaseNetwork):
    def __init__(self, opt):
        self.opt = opt
        super(SketchFeatureExtractor, self).__init__()

        def convlayer(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
            return nn.Sequential(
                SynchronizedBatchNorm2d(in_channels),
                Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
                nn.LeakyReLU(0.2)
            )

        model = torch.hub.load('RF5/danbooru-pretrained', 'resnet18')
        self.model = nn.Sequential(convlayer(5, 64, 7, 2, padding=3, bias=False), model[0][1:], convlayer(512, 128, 1))

    def forward(self, images):  # return (N, 128, H, W)
        return self.model(images)
