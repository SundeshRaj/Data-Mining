# -*- coding: utf-8 -*-
"""
Created on Thu May  7 20:40:32 2020

@author: sundesh raj
"""

import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from loadTwoClassCifar import TwoClassCifar10
from parameter_setting import parameter_setting
from mlp_model import ConvNet, LogisticRegression

args = parameter_setting()

test_dataset = TwoClassCifar10(args.root, train=False)

conv_net = ConvNet(args.input_channel, 2)
lr_model = LogisticRegression(args.image_cifar10_width*args.image_cifar10_height)
conv_net.load_state_dict(torch.load("model/20205721119_0.500000.pth")) # saved best resulting model
lr_model.load_state_dict(torch.load("model/20205721150_0.592500.pth")) # saved best resulting model

conv_preds = []
lr_preds = []
targets = []
with torch.no_grad():
    for image, label in test_dataset:
        image.unsqueeze_(0)
        conv_pred = conv_net(image)
        lr_pred = lr_model(image)
        conv_pred = torch.max(torch.softmax(conv_pred, dim=1),
                              dim=1)[0].squeeze()
        lr_pred = torch.sigmoid(lr_pred).squeeze()
        conv_preds.append(conv_pred.item())
        lr_preds.append(lr_pred.item())
        targets.append(label)

fpr, tpr, thresholds = metrics.roc_curve(targets, conv_preds)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ConvNet ROC')
plt.title("ConvNet ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

fpr, tpr, thresholds = metrics.roc_curve(targets, lr_preds)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ConvNet ROC')
plt.title("Logistic Regression ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()
plt.show()