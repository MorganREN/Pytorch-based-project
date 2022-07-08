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

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))

dataset = datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, intput):
        return self.sigmoid1(intput)


tudui = Tudui()
writer = SummaryWriter('logs_sigmoid')
step = 0
for data in dataloader:
    imgs, label = data
    writer.add_images('input', imgs, step)
    output = tudui(imgs)
    writer.add_images('output', output, step)
    step += 1
writer.close()