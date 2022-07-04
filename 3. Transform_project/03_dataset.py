# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 03//07//2022//
"""
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# Download the dataset
train_set = torchvision.datasets.CIFAR10('./dataset', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10('./dataset', train=False, transform=dataset_transform, download=True)

# print(train_set.classes)
# print(len(train_set))
# print(len(test_set))
writer = SummaryWriter('p10')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_set', img, i)

writer.close()