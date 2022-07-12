# -*- coding:utf-8 -*-
"""
Author: mohanren
Date: 11//07//2022//
"""
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Mohan import *

data_train = torchvision.datasets.CIFAR10('./data', download=True, train=True, transform=torchvision.transforms.ToTensor())
data_test = torchvision.datasets.CIFAR10('./data', download=True, train=False, transform=torchvision.transforms.ToTensor())

data_train_size = len(data_train)
data_test_size = len(data_test)

train_loader = DataLoader(dataset=data_train, batch_size=64)
test_loader = DataLoader(dataset=data_test, batch_size=64)

# Create nn model
mohan = Mohan()

# Create Loss Function (Classification)
loss_fn = nn.CrossEntropyLoss()

# Create Optimizer
learning_rate = 1e-2
optim = torch.optim.SGD(mohan.parameters(), lr=learning_rate)

# Train Parameter set
total_train_step = 0
total_test_step = 0
epoch = 10

# Add tensorboard
writer = SummaryWriter('./logs')

for i in range(5):
    print("---------- Current epoch: {} ----------".format(i+1))
    # Train
    for data in train_loader:
        imgs, labels = data
        outputs = mohan(imgs)
        # Get the loss
        loss = loss_fn(outputs, labels)
        # Optimize the model
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            writer.add_scalar('train_loss', loss.item(), total_train_step)
            print("Training No. {}, Loss No. {} ".format(total_train_step, loss.item()))

    # Test start
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            outputs = mohan(imgs)
            loss = loss_fn(outputs, labels)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy += accuracy

    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy / data_test_size, total_test_step)
    torch.save(mohan, 'mohan_{}.pth'.format(i))
    print("The total loss in the testing dataset is {}".format(total_test_loss))
    print("Total accuracy: {} \n".format(total_accuracy / data_test_size))
    total_test_step += 1

writer.close()



