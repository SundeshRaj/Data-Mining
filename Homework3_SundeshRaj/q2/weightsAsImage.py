# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:59:07 2020

@author: sundesh raj
"""
import torch
from mlp_model import OneLayer
from parameter_setting import parameter_setting
import matplotlib.pyplot as plt

label_dict = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

args = parameter_setting()

model = OneLayer(args.image_fashion_mnist_width*args.image_fashion_mnist_height, args.defNumClasses)
model.load_state_dict(torch.load("model_weight_image/20205722337_0.765900.pth"))

weight = model.linear.weight.data.numpy()

for idx in range(10):
    vec = weight[idx]
    plt.imshow(vec.reshape(28, 28))
    plt.title(f"Label {idx}: {label_dict[idx]}")
    plt.show()