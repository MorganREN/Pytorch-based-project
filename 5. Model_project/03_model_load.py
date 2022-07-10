# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 10//07//2022//
"""
import torch
import torchvision

# Load method:

# Method 1
model1 = torch.load('vgg16_method1.pth')
print(model1)

# Method 2
model2 = torch.load('vgg16_method2.pth')
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(model2)
print(vgg16)