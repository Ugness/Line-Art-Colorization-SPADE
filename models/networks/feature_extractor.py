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
        model = torch.hub.load('RF5/danbooru-pretrained', 'resnet18')
        self.model = model[0]

    def forward(self, images):  # return (N, 512, H, W)
        return self.model(images.repeat(1, 3, 1, 1))
