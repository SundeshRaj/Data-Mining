# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:39:35 2020

@author: sundesh raj
"""

import numpy as np
from logisticRegression import LogisticRegression

if __name__ == '__main__':
    
    mu_1 = [1, 0]
    mu_2 = [0, 1.5]
    sigma_1 = [[1, 0.75], [0.75, 1]]
    sigma_2 = [[1, 0.75], [0.75, 1]]
    
    train_x_0 = np.random.multivariate_normal(mu_1, sigma_1, size=500)
    train_x_1 = np.random.multivariate_normal(mu_2, sigma_2, size=500)
    train_y_0 = np.zeros(500)
    train_y_1 = np.ones(500)
    test_x_0 = np.random.multivariate_normal(mu_1, sigma_1, size=250)
    test_x_1 = np.random.multivariate_normal(mu_2, sigma_2, size=250)
    test_y_0 = np.zeros(250)
    test_y_1 = np.ones(250)
    
    train_x = np.concatenate((train_x_0, train_x_1), axis=0)
    train_y = np.concatenate((train_y_0, train_y_1), axis=0)
    test_x = np.concatenate((test_x_0, test_x_1), axis=0)
    test_y = np.concatenate((test_y_0, test_y_1), axis=0)
    
    modes = ["batch", "online"]
    lrs = [1, 0.1, 0.01, 0.001]
    for mode in modes:
        for lr in lrs:
            model = LogisticRegression(mode=mode, lr=lr)
            model.fit(train_x, train_y)
            preds = model.predict(test_x)
            print(
                f"Mode: {mode}, LR: {lr}, Iteration: {model.best_iter}, Accuracy: {model.accuracy(preds, test_y)}",
                end="\n\n")
            model.plot_boundary(test_x, test_y)
            model.plot_loss_history()
            model.plot_grad_history()