# -*- coding: utf-8 -*-

import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd

def loadData(filename):
    df = pd.read_table(filename, '\t', header=None)
    return np.array(df.loc[:,0:1]), np.array(df.loc[:,2])
    
def showData(X, y, w=None, b=None):
    plt.scatter(x=X[:,0], y=X[:,1], c=y)
    if (w is None) or (b is None):
        pass
    else:
        a = -w[1,0] / w[0,0]
        b = -b / w[1,0]
        plt.plot(X[:,0], a * X[:,0]+b, c='red')

def init_w_b(shape_w, shape_b, seed):
    np.random.seed(seed)
    w = np.random.rand(shape_w[0], shape_w[1])
    b = np.random.rand(shape_b[0])+0.01
    return w, b
    
def forward(X, w, b):
    z = np.dot(X, w) + b
    a = 1.0 / (1.0 + np.exp(-z))
    return a

def cost_func(y_, y):
    y_ = y_.flatten()
    y = y.flatten()
    #cost = np.average( -(y*np.log(y_) + (1-y)*np.log(1-y_)) )
    cost = np.sum(np.power(y_ - y, 2)) / y.shape[0]
    return cost

def train(maxloop, lr, X, y, w, b):
    m = X.shape[0]
    y = y.reshape((m,1))
    for i in range(maxloop):
        a = forward(X, w, b)
        d_z = a - y
        d_w = np.dot(X.T, d_z) / m
        d_b = np.sum(d_z) / m
        w = w - lr * d_w
        b = b - lr * d_b
        y_ = a
        #print(cost_func(y_, y))
        print(calc_accuarcy(y_, y))
    return w, b

def harden(y_, sepNum):
    y_[y_ > sepNum] = 1
    y_[y_ <= sepNum] = 0
    return y_

def calc_accuarcy(y_, y):
    y = y.flatten()
    y_ = y_.flatten()
    y_ = harden(y_, 0.5)
    correctNum = len(y_[y_ == y])
    return float(correctNum) / y.shape[0]

# ------------ main -------------- #
X, y = loadData('./data/testSet.txt')
showData(X, y)

w, b = init_w_b([2,1], [1], seed=314)
a = forward(X, w, b)
y_ = a.flatten()
cost = cost_func(y_, y)
w, b = train(100, 0.1, X, y, w, b)
y_ = forward(X, w, b)
y_ = harden(y_, 0.5)

showData(X, y_, w, b)
