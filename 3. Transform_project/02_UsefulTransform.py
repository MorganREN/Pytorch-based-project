# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 03//07//2022//
"""
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs')
img = Image.open('./helloMe.jpeg')

# ToTensor using
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('ToTensor', img_tensor)

# Normalize using: input[channel] = (input[channel] - mean[channel]) / std[channel]
trans_norms = transforms.Normalize([0.1, 0.1, 0.1], [0.1, 0.1, 0.1])
img_norm = trans_norms(img_tensor)
writer.add_image('Normalization', img_norm)

# Resize using
trans_resize = transforms.Resize((512, 512))  # set the new size
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> toTensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image('Resize', img_resize)

# Use Compose (1. Resize; 2. ToTensor)
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize = trans_compose(img)
writer.add_image('Resize', img_resize, 1)

# RandomCrop using
trans_random = transforms.RandomCrop((300, 400))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(20):
    img_crop = trans_compose_2(img)
    writer.add_image('RandomCrop', img_crop, i)

writer.close()