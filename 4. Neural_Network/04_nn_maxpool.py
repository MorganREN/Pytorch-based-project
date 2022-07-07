# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 08//07//2022//
"""
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

dataset = datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = nn.AvgPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


tudui = Tudui()

writer = SummaryWriter('logs_maxpool')
step = 0
for data in dataloader:
    imgs, label = data
    writer.add_images('input', imgs, step)
    output = tudui(imgs)
    writer.add_images('output', output, step)
    step += 1
writer.close()

# Role of pool: Keep the main features and reduce the data size