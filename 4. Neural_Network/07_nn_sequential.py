# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 08//07//2022//
"""
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1)  # First convolution layer
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # First pool layer with kernel size 2
        # self.conv2 = nn.Conv2d(32, 32, 5, padding=2)  # Second convolution layer
        # self.maxpool2 = nn.MaxPool2d(2)  # Second pool layer
        # self.conv3 = nn.Conv2d(32, 64, 5, padding=2)  # third convolution layer
        # self.maxpool3 = nn.MaxPool2d(2)
        # self.flatten = nn.Flatten()  # a flatten layer, laying the data size to 64x4x4 = 1024
        # self.linear1 = nn.Linear(1024, 64)
        # self.linear2 = nn.Linear(64, 10)

        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        output = self.model1(x)
        return output

tudui = Tudui()
print(tudui)

input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)

writer = SummaryWriter('logs_seq')
writer.add_graph(tudui, input)
writer.close()
