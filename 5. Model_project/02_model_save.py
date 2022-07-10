# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 10//07//2022//
"""
import torchvision
import torch

vgg16 = torchvision.models.vgg16(pretrained=False)

# Save method:

# Method 1
torch.save(vgg16, 'vgg16_method1.pth')

# Method 2 (Recommended)
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')