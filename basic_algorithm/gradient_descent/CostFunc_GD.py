# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


def hypo(X, theta):
    return theta * X[:,0]

def costFunc(X, y, theta, h=hypo):
    m = np.shape(X)[0]
    diff = h(X, theta) - y
    cost_list = np.power(diff, 2)
    return np.sum(cost_list) / (2 * float(m))

# gradient descent
def gradientDescent(X, y, alpa=0.1, h=hypo):
    m = np.shape(X)[0]
    theta = 0.0
    thetaArr = []
    maxCycles = 1000
    for k in range(maxCycles):
        theta = theta - alpa * ( np.sum((h(X, theta) - y) * X[:, 0]) / float(m) )
        thetaArr.append(theta)
    return theta, thetaArr

def showData(X, y):
    plt.title('x-y data')
    plt.scatter(X[:, 0], y)
    plt.show()

def showTheta_Cost(theta_list, cFunc=costFunc):
    cost_list = []
    for theta in theta_list:
        cost = cFunc(X, y, theta, h=hypo)
        cost_list.append(cost)
    plt.title('theta - cost')
    plt.scatter(theta_list, cost_list)
    plt.show()

# generate data
np.random.seed(314)
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=10)
showData(X, y)

# plot the cost with different theta (model parameter)
cost_list = []
theta_list = np.arange(0, 50, 1)
for theta in theta_list:
    cost = costFunc(X, y, theta, h=hypo)
    cost_list.append(cost)
showTheta_Cost(theta_list, cFunc=costFunc)

# optimize cost-function by using gradient descent algorithm
theta, thetaArr = gradientDescent(X, y)
showTheta_Cost(thetaArr, cFunc=costFunc)

# plot the final result (best model paramter)
plt.title('best regression model')
plt.scatter(X[:, 0], y, c='b')
plt.plot(X[:, 0], hypo(X, theta), 'r')
