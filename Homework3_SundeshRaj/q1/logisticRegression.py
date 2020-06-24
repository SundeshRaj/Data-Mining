# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:53:47 2020

@author: sundesh raj
"""

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self,
                 mode="batch",
                 lr=0.001,
                 predict_threshold=0.5,
                 grad_norm_tol=0.001,
                 max_iter=100000):
        self._check_init(mode, lr, predict_threshold, grad_norm_tol, max_iter)
        self._mode = mode
        self._lr = lr
        self._predict_threshold = predict_threshold
        self._grad_norm_tol = grad_norm_tol
        self._max_iter = max_iter
        self._weight = None
        self._bias = 0
        self._loss_history = []
        self._norm_history = []
        self._best_iter = 0

    def fit(self, inputs, targets):
        num_features = inputs.shape[1]
        self._weight = np.random.randn(num_features)
        if self._mode == "batch":
            for idx in range(self._max_iter):
                probs = self._forward(inputs)
                loss = self._cross_entropy(probs, targets)
                grad_weight, grad_bias = self._backward(inputs, targets)
                norm = self._grad_norm(grad_weight, grad_bias)
                self._loss_history.append(loss)
                self._norm_history.append(norm)
                print(
                    f"\rIteration: {idx+1}/{self._max_iter}, Loss: {loss:.5f}, Norm: {norm:.5f}/{self._grad_norm_tol:.5f}",
                    end="")
                if norm >= self._grad_norm_tol:
                    self._update_weight(grad_weight, grad_bias)
                else:
                    break
            self._best_iter = idx + 1
        else:
            for idx in range(self._max_iter):
                for data, label in zip(inputs, targets):
                    probs = self._forward(data)
                    loss = self._cross_entropy(probs, label)
                    grad_weight, grad_bias = self._backward(data, label)
                    norm = self._grad_norm(grad_weight, grad_bias)
                    self._loss_history.append(loss)
                    self._norm_history.append(norm)
                    print(
                        f"\rIter: {idx+1}/{self._max_iter}, Loss: {loss:.5f}, Norm: {norm:.5f}/{self._grad_norm_tol:.5f}",
                        end="")
                    if norm >= self._grad_norm_tol:
                        self._update_weight(grad_weight, grad_bias)
                    else:
                        break
            self._best_iter = idx + 1
        print()

    def predict(self, inputs):
        preds = self._forward(inputs)
        preds[preds >= self._predict_threshold] = 1
        preds[preds < self._predict_threshold] = 0
        return preds

    def plot_boundary(self, inputs, targets):
        x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
        y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        preds = self.predict(np.c_[xx.ravel(), yy.ravel()])
        preds = preds.reshape(xx.shape)
        plt.plot()
        plt.contourf(xx, yy, preds, alpha=0.4)
        for idx in np.unique(targets):
            plt.scatter(inputs[targets == idx][:, 0],
                        inputs[targets == idx][:, 1])
        plt.title("Logistic Regression Boundary")
        plt.show()

    def plot_loss_history(self):
        plt.plot(self._loss_history)
        plt.title("Loss History")
        plt.show()

    def plot_grad_history(self):
        plt.plot(self._norm_history)
        plt.title("Gradient Norm History")
        plt.show()

    def accuracy(self, preds, targets, eps=1e-7):
        correct = (preds == targets).sum()
        total = len(targets)
        return format(correct / (total + eps), ".5f")

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias

    @property
    def best_iter(self):
        return self._best_iter

    def _sigmoid(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def _cross_entropy(self, probs, targets):
        return (-(targets * np.log(probs) +
                  (1 - targets) * np.log(1 - probs))).mean()

    def _forward(self, inputs):
        return self._sigmoid(np.dot(inputs, self._weight) + self._bias)

    def _backward(self, inputs, targets):
        grad_weight = (inputs.T * (self._forward(inputs) - targets))
        grad_bias = (self._forward(inputs) - targets)
        if self._mode == "batch":
            grad_weight = grad_weight.mean(1)
            grad_bias = grad_bias.mean()
        return grad_weight, grad_bias

    def _update_weight(self, grad_weight, grad_bias):
        self._weight -= self._lr * grad_weight
        self._bias -= self._lr * grad_bias

    def _grad_norm(self, grad_weight, grad_bias):
        grad = np.hstack((grad_bias, grad_weight))
        norm = np.linalg.norm(grad, ord=1)
        return norm

    def _check_init(self, mode, lr, predict_threshold, grad_norm_tol,
                    max_iter):
        assert mode in ["batch", "online"]
        assert lr > 0
        assert predict_threshold > 0 and predict_threshold < 1
        assert grad_norm_tol > 0
        assert max_iter > 0