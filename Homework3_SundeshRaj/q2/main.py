## Do NOT modify the code in this file
from __future__ import print_function
import torch
import numpy as np
import os
from torch import nn, optim
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from parameter_setting import parameter_setting
from data_loader import data_loader
from mlp_model import MLP, ConvNet, OneLayer
from train_test import train, test

## parameters
args = parameter_setting()

## data loaders for training and testing;
train_loader, test_loader, train_data, test_data = data_loader(args)

## load args based on model and dataset
if args.model == "mlp":
    network = MLP(args.image_cifar10_width*args.image_cifar10_height, args.class_number_cifar10)
elif args.model == "cnn":
    network = ConvNet(args.input_channel, args.class_number_cifar10)
elif args.model == "onelayer":
    network = OneLayer(args.image_fashion_mnist_width*args.image_fashion_mnist_height, args.class_number_fashion_mnist)
    
## optimizer used to implement weight update using gradient descent
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(),
                      args.lr,
                      momentum=args.momentum,
                      weight_decay=args.wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                 len(train_loader) *
                                                 args.epoch_number)

## train the model
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

for epoch_idx in range(args.epoch_number):
    train_loss_ret, train_acc_ret = train(train_loader, network, criterion, optimizer, scheduler, args, epoch_idx)
    train_loss_list.append(train_loss_ret)
    train_acc_list.append(train_acc_ret)

    test_loss_ret, test_acc_ret = test(test_loader, network, criterion, epoch_idx)
    test_loss_list.append(test_loss_ret)
    test_acc_list.append(test_acc_ret)

## plot loss
x_axis = np.arange(args.epoch_number)
loss_fig = plt.figure()
plt.plot(x_axis, train_loss_list, 'r')
plt.plot(x_axis, test_loss_list, 'b')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(("train", "test"))
os.makedirs(args.output_folder, exist_ok=True)
loss_fig.savefig(os.path.join(args.output_folder, args.dataset_name + "fash_wPlot" + "_loss.pdf"))
plt.close(loss_fig)

## plot accuracy
acc_fig = plt.figure()
plt.plot(x_axis, train_acc_list, 'r')
plt.plot(x_axis, test_acc_list, 'b')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(("train", "test"))
os.makedirs(args.output_folder, exist_ok=True)
acc_fig.savefig(os.path.join(args.output_folder, args.dataset_name + "fash_wPlot" + "_accuracy.pdf"))
plt.close(acc_fig)

## save loss and accuracy to the file
loss_acc_np = np.arange(args.epoch_number).reshape((-1, 1))
loss_acc_np = np.append(loss_acc_np, np.array(train_loss_list).reshape((-1, 1)), axis=1)
loss_acc_np = np.append(loss_acc_np, np.array(test_loss_list).reshape((-1, 1)), axis=1)
loss_acc_np = np.append(loss_acc_np, np.array(train_acc_list).reshape((-1, 1)), axis=1)
loss_acc_np = np.append(loss_acc_np, np.array(test_acc_list).reshape((-1, 1)), axis=1)
os.makedirs(args.output_folder, exist_ok=True)
np.savetxt(os.path.join(args.output_folder,args.dataset_name + "fash_wPlot" + "_loss_acc.txt"), loss_acc_np)