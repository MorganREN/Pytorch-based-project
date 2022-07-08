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
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = nn.Linear(196608, 10)  # 196608 features for input, 10 features for output

    def forward(self, intput):
        return self.linear1(intput)

tudui = Tudui()
writer = SummaryWriter('logs_sigmoid')
step = 0
for data in dataloader:
    imgs, label = data
    print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    output = torch.flatten(imgs)
    print(output.shape)  # torch.Size([1, 1, 1, 196608])
    output = tudui(output)
    print(output.shape)
writer.close()

