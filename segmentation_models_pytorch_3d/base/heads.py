import torch.nn as nn
from .modules import Activation


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.Upsample(scale_factor=upsampling, mode='trilinear') if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv3d, upsampling, activation)
