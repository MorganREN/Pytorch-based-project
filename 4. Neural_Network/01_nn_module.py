# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 04//07//2022//
"""
import torch
from torch import nn

class Tudui(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

tudui = Tudui()
x = torch.tensor(1.0)
output = tudui.forward(x)
print(output)