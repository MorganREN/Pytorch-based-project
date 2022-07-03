# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 19//06//2022//
"""
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

# Create a directory named "logs", storing the source of the execution
writer = SummaryWriter("logs")
img_path = '../practice_data/train/ants_image/0013035.jpg'
img_PIL = Image.open(img_path)
img_np = np.array(img_PIL)

writer.add_image('test', img_np, 1, dataformats='HWC')

# function: y = 2x
for i in range(100):
    writer.add_scalar('y = x', 2*i, i)  # the second parameter is y, and the third parameter is x

writer.close()

# After the logs been created, use command:
# tensorboard --logdir=logs --port=xxxx
# to open the file logs in the localhost:xxxx

'''
note 1: delete the logs file if you want to add an absolutely new scalar, or tensorboard will helps to fit the model
'''
