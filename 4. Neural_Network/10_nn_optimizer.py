# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 09//07//2022//
"""
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

dataset = torchvision.datasets.CIFAR10('./data', download=True, train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
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

loss = nn.CrossEntropyLoss()
tudui = Tudui()
for data in dataloader:
    img, label = data
    outputs = tudui(img)
    loss_result = loss(outputs, label)
    loss_result.backward()
    print(loss_result)
