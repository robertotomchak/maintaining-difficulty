"""
    Defines all models used in this project
"""

import os
import glob
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image
from json import dump
from tqdm import tqdm
from datetime import datetime



# L2 normalization to normalize output of network
class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)
    

# L2 pooling layer, used in Inception layers
class L2Pool(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs["divisor_override"] = 1
        self.pool = nn.AvgPool2d(*args, **kwargs)
    def forward(self, x):
        return torch.sqrt(self.pool(x ** 2))


# creates a mobilenetV3 small backbone
def MobileNetV3SmallBackbone(out_size, pre_trained=True):
    if pre_trained:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.mobilenet_v3_small(weights = weights)
    # changing the classifier layer accordingly
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=576, out_features=out_size),
        L2NormalizationLayer()
    )
    return model

# creates a efficientnetb0 backbone
def EfficientNetB0Backbone(out_size, pre_trained=True):
    if pre_trained:
        weights = "DEFAULT"
    else:
        weights = None
    model = models.efficientnet_b0(weights = weights)
    # changing the classifier layer accordingly
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=out_size),
        L2NormalizationLayer()
    )
    return model

# creates an efficientnetb0 backbone, similar to one found in kaggle
# https://www.kaggle.com/code/quyraichi/siamese-network-with-efficentnetb0-backbone
def KaggleEfficientNetB0Backbone(out_size, pre_trained=True):
    if pre_trained:
        weights = "DEFAULT"
    else:
        weights = None
    model = models.efficientnet_b0(weights = weights)
    # changing the classifier layer accordingly
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=512, out_features=256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=256, out_features=out_size),
        L2NormalizationLayer()
    )
    return model



# defining an Inception Layer
class InceptionLayer(nn.Module):
    def __init__(self, in_channels, conv1x1, reduce3x3, conv3x3, reduce5x5, conv5x5, pool_fn, pool_proj, stride=1):
        super(InceptionLayer, self).__init__()
        if conv1x1 == 0:
            self.branch1x1 = None
        else:
            self.branch1x1 = nn.Sequential(
                nn.Conv2d(in_channels, conv1x1, kernel_size=1, bias=False),
                nn.BatchNorm2d(conv1x1),
                nn.ReLU()
            )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce3x3, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduce3x3),
            nn.ReLU(),
            nn.Conv2d(reduce3x3, conv3x3, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(conv3x3),
            nn.ReLU()
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce5x5, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduce5x5),
            nn.ReLU(),
            nn.Conv2d(reduce5x5, conv5x5, kernel_size=5, padding=2, stride=stride, bias=False),
            nn.BatchNorm2d(conv5x5),
            nn.ReLU()
        )

        self.branch_pool = nn.Sequential(
            pool_fn(kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU()
        )

    def forward(self, x):
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)
        if self.branch1x1:
            b1 = self.branch1x1(x)
            return torch.cat([b1, b2, b3, b4], 1)
        return torch.cat([b2, b3, b4], 1)
    

# defining the NN2 architecture
# input shape must be 3 x 224 x 224
class NN2(nn.Module):
    def __init__(self, output_shape):
        super(NN2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0)
        )

        # simple enough that doest not require the Inception Module
        self.inception2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inception3 = nn.Sequential(
            Models.InceptionLayer(192, 64, 96, 128, 16, 32, nn.MaxPool2d, 32),
            Models.InceptionLayer(256, 64, 96, 128, 32, 64, Models.L2Pool, 64),
            Models.InceptionLayer(320, 0, 128, 256, 32, 64, nn.MaxPool2d, 320, stride=2),
        )

        self.inception4 = nn.Sequential(
            Models.InceptionLayer(640, 256, 96, 192, 32, 64, Models.L2Pool, 128),
            Models.InceptionLayer(640, 224, 112, 224, 32, 64, Models.L2Pool, 128),
            Models.InceptionLayer(640, 192, 128, 256, 32, 64, Models.L2Pool, 128),
            Models.InceptionLayer(640, 160, 144, 288, 32, 64, Models.L2Pool, 128),
            Models.InceptionLayer(640, 0, 160, 256, 64, 128, nn.MaxPool2d, 640, stride=2)
        )

        self.inception5 = nn.Sequential(
            Models.InceptionLayer(1024, 384, 192, 384, 48, 128, Models.L2Pool, 128),
            Models.InceptionLayer(1024, 384, 182, 384, 48, 128, nn.MaxPool2d, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fully_conn = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, output_shape),
            Models.L2NormalizationLayer()
        )

        self.apply(init_weights)


    def forward(self, x):
        x = self.conv(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.fully_conn(x)
        return x
    


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)



# calculates size of output of convolutions of the Hachuli layer
def size_conv_output(input_shape, num_kernels):
    w, h = input_shape[0], input_shape[0]
    # first pool layer
    w, h = w // 2, h // 2
    # second conv
    w, h = w - 2, h - 2
    # second pool
    w, h = w // 2, h // 2
    # third conv
    w, h = w - 2, h - 2
    # third pool
    w, h = w // 2, h // 2
    return w * h * num_kernels


# defines the hochuli network
# mostly for some basic testing
class Hochuli(nn.Module):
    def __init__(self, input_shape, last_output):
        super(Hochuli, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.flatten = nn.Flatten()
        self.fully_connected = nn.Sequential(
            nn.Linear(size_conv_output(input_shape, 64), last_output), 
            L2NormalizationLayer()
        )

    def forward(self, x):
        x = self.convolution(x)
        x = self.flatten(x)
        x = self.fully_connected(x)
        return x


