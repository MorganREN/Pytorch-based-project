# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 07//07//2022//
"""
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(torch.nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, a):
        x = self.conv1(a)
        return x

tudui = Tudui()

writer = SummaryWriter('logs')
step = 0
for data in dataloader:
    imgs, label = data
    output = tudui(imgs)
    writer.add_images('input', imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('output', output, step)
    step += 1

writer.close()
