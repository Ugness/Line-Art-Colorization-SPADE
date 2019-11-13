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
        modules = list()
        modules.append(convlayer(1, 64, 3, 1, padding=1))
        modules.append(convlayer(64, 128, 3, 1, padding=1))
        modules.append(convlayer(128, 128, 3, 1, padding=1))
        modules.append(convlayer(128, 128, 3, 1, padding=1))

        self.model = nn.Sequential(*modules)

    def forward(self, images):  # return (N, 128, H, W)
        return self.model(images)
