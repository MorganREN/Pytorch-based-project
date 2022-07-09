# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 09//07//2022//
"""
import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

loss1 = L1Loss(reduction='mean')
result1 = loss1(input, target)

loss2 = MSELoss()
result2 = loss2(input, target)

print(result1)
print(result2)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = CrossEntropyLoss()
result3 = loss_cross(x, y)
print(result3)