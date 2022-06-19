# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 18//06//2022//
"""
from torch.utils.data import Dataset
from PIL import Image
import os


class myData(Dataset):  # Inherited the Dataset class

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = str(self.root_dir) + "/" + str(self.label_dir)  # concatenate the directory path
        self.img_path = os.listdir(self.path)  # the list storing the whole files

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = str(self.path) + "/" + str(img_name)  # concatenate the directory path
        img = Image.open(img_item_path)  # open the image in the PIL Image way
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = 'hymenoptera_data/train'
label_ant = 'ants'
label_bee = 'bees'

ants_dataset = myData(root_dir, label_ant)
print(len(ants_dataset))
print(ants_dataset[5])

bees_dataset = myData(root_dir, label_bee)
print(len(bees_dataset))
print(bees_dataset[5])

train_data = ants_dataset + bees_dataset
print(len(train_data))


