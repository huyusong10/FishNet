# Author: hys

import torch
from torch import nn
from torch.nn import functional as F

from functools import partial

from .utils import Conv2dStaticSamePadding

def get_conv():  
    return Conv2dStaticSamePadding

class Canary(nn.Module):

    def __init__(self, inc, ouc, kernel_size, stride, expand_ratio=6, se=False, bn_mom = 0.99, bn_eps = 0.001):
        super().__init__()
        self.inc = inc
        self.ouc = ouc
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.se = se
        self._bn_mom = 1 - bn_mom
        self._bn_eps = bn_eps
        self.conv2d = get_conv()

        expc = inc * self.expand_ratio
        if self.expand_ratio != 1:
            self._expand_conv = self.conv2d(in_channels=inc, out_channels=expc, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=expc, momentum=self._bn_mom, eps=self._bn_eps)

        # depthwise conv
        self._depthwise_conv = self.conv2d(
            in_channels=expc, out_channels=expc, groups=expc,
            kernel_size=kernel_size, stride=stride, bias=False
        )
        self._bn1 = nn.BatchNorm2d(num_features=expc, momentum=self._bn_mom, eps=self._bn_eps)

        # se attention
        if self.se:
            self._se_reduce = self.conv2d(in_channels=expc, out_channels=expc//8, kernel_size=1)
            self._se_expand = self.conv2d(in_channels=expc//8, out_channels=expc, kernel_size=1)

        # pointwise conv
        self._pointwise_conv = self.conv2d(
            in_channels=expc, out_channels=ouc, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(num_features=ouc, momentum=self._bn_mom, eps=self._bn_eps)

        self._activate = nn.ReLU()

    def forward(self, inputs):
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            # x = self._activate(x)
        
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._activate(x)

        if self.se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._activate(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._pointwise_conv(x)
        x = self._bn2(x)

        if self.inc == self.ouc and self.stride == 1:
            x = x + inputs
        return x

class FishNet(nn.Module):

    def __init__(self, num_points, bn_mom = 0.99, bn_eps=0.001):
        super().__init__()
        self.conv2d = get_conv()

        repeats = [1, 2, 3, 3, 4, 1]
        strides = [2, 2, 2, 2, 2, 1]
        kernel_size = [5, 3, 3, 3, 3, 3]
        channels = [3, 32, 56, 80, 112, 160, 192]
        se = [False, False, True, True, True, True]

        self._blocks = []
        for idx in range(6):
            block = Canary(channels[idx], channels[idx+1], kernel_size=kernel_size[idx], stride=strides[idx])
            self._blocks.append(block)
            for _ in range(repeats[idx] - 1):
                block = Canary(channels[idx+1], channels[idx+1], se=se[idx], kernel_size=kernel_size[idx], stride=1)
                self._blocks.append(block)
        self._blocks = nn.Sequential(*self._blocks)

        head_chann = 640
        self._conv_head = self.conv2d(channels[-1], head_chann, kernel_size=1, bias=False)
        self._bn_head = nn.BatchNorm2d(num_features=head_chann, momentum=bn_mom, eps=bn_eps)

        self._regressor = self.conv2d(head_chann, num_points, kernel_size=1, bias=False)
        self._avgpool = nn.AdaptiveAvgPool2d(1)

        self._activate = nn.ReLU()

    def forward(self, inputs):
        # extract feature from backbone
        x = self._blocks(inputs)
        # expand dimension
        x = self._activate(self._bn_head(self._conv_head(x)))
        # regress
        x = self._regressor(x)
        x = self._avgpool(x).flatten(start_dim=1)
    
        return x