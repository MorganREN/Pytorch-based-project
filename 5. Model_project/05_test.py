# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 12//07//2022//
"""
import torch
import torchvision
from PIL import Image
from torch import nn

img_path = './dog.jpg'
img = Image.open(img_path)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

img = transform(img)

class Mohan(nn.Module):
    def __init__(self):
        super(Mohan, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


model = torch.load('./mohan_19.pth', map_location=torch.device('cpu'))
img = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(img)
print(output.argmax(1))
