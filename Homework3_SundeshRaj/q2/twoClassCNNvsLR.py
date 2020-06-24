# -*- coding: utf-8 -*-
"""
Created on Thu May  7 00:14:58 2020

@author: sundesh raj
"""

from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
#from data_loader import data_loader
from loadTwoClassCifar import TwoClassCifar10
from mlp_model import ConvNet, LogisticRegression
from train_test import train, test, lr_train, lr_test
from parameter_setting import parameter_setting

args = parameter_setting()

print(f"{datetime.now().ctime()} - Start Loading Dataset...")
train_data = TwoClassCifar10(args.root, train=True)
test_data = TwoClassCifar10(args.root, train=False)
train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False, num_workers=0)
#train_loader, test_loader, train_data, test_data = data_loader(args)
print(f"{datetime.now().ctime()} - Finish Loading Dataset")

print(
    f"{datetime.now().ctime()} - Start Creating Net, Criterion, Optimizer and Scheduler..."
)
conv_net = ConvNet(args.input_channel, 2)
lr_model = LogisticRegression(args.image_cifar10_width*args.image_cifar10_height)
conv_criterion = nn.CrossEntropyLoss()
lr_criterion = nn.BCEWithLogitsLoss()
conv_optimizer = optim.SGD(conv_net.parameters(),
                           args.learningRate,
                           momentum=args.momentum,
                           weight_decay=args.wd)
lr_optimizer = optim.SGD(lr_model.parameters(),
                         args.learningRate,
                         momentum=args.momentum,
                         weight_decay=args.wd)
conv_scheduler = optim.lr_scheduler.CosineAnnealingLR(conv_optimizer,
                                                      len(train_loader) *
                                                      args.epoch_number)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(lr_optimizer,
                                                    len(train_loader) *
                                                    args.epoch_number)
print(
    f"{datetime.now().ctime()} - Finish Creating Net, Criterion, Optimizer and Scheduler"
)

print(f"{datetime.now().ctime()} - Start Training...")
print(
    f"Traing dataset: {len(train_data)}, iteration: {len(train_loader)}"
)
print(
    f"Testing dataset: {len(test_data)}, iteration: {len(test_loader)}")
print(
    f"Epochs: {args.epoch_number}, Batch Size: {args.batch_size}, LR:{args.learningRate}",
    end='\n\n')

conv_loss = []
conv_acc = []
for epoch in range(args.epoch_number):
    train(train_loader, conv_net, conv_criterion, conv_optimizer, conv_scheduler, args, epoch)
    loss, acc =test(test_loader, conv_net, conv_criterion, epoch)
    conv_loss.append(loss)
    conv_acc.append(acc)
print(f"{datetime.now().ctime()} - Finish Training")

lr_loss = []
lr_acc = []
for epoch in range(args.epoch_number):
    lr_train(train_loader, lr_model, lr_criterion, lr_optimizer, lr_scheduler, args, epoch)
    loss, acc = lr_test(test_loader, lr_model, lr_criterion, epoch+1)
    lr_loss.append(loss)
    lr_acc.append(acc)
print(f"{datetime.now().ctime()} - Finish Training")

plt.plot(conv_loss, label="ConvNet")
plt.plot(lr_loss, label="LR")
plt.title("Loss Comparison")
plt.legend()
plt.grid()
plt.show()

plt.plot(conv_acc, label="ConvNet")
plt.plot(lr_acc, label="LR")
plt.title("Acc Comparison")
plt.legend()
plt.grid()
plt.show()