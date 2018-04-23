# -*- coding: utf-8 -*-

import numpy as np


class LinerRegression(object):

    def __init__(self, learning_rate=0.01, max_iter=100, seed=None):
        np.random.seed(seed)
        self.lr = learning_rate
        self.max_iter = max_iter
        self.w = np.random.normal(1, 0.1)
        self.b = np.random.normal(1, 0.1)
        self.loss_arr = []

    def fit(self, x, y):
        self.x = x
        self.y = y
        for i in range(self.max_iter):
            self._train_step()
            self.loss_arr.append(self.loss())
            # print('loss: \t{:.3}'.format(self.loss()))
            # print('w: \t{:.3}'.format(self.w))
            # print('b: \t{:.3}'.format(self.b))

    def _f(self, x, w, b):
        return x * w + b

    def predict(self, x=None):
        if x is None:
            x = self.x
        y_pred = self._f(x, self.w, self.b)
        return y_pred

    def loss(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred = self.predict(self.x)
        return np.mean((y_true - y_pred)**2)

    def _calc_gradient(self):
        d_w = np.mean((self.x * self.w + self.b - self.y) * self.x)
        d_b = np.mean(self.x * self.w + self.b - self.y)
        return d_w, d_b

    def _train_step(self):
        d_w, d_b = self._calc_gradient()
        self.w = self.w - self.lr * d_w
        self.b = self.b - self.lr * d_b
        return self.w, self.b
