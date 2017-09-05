# -*- coding: utf-8 -*-

from sklearn import datasets, svm
import matplotlib.pyplot as plt
import numpy as np

def showData(X, y):
    X = np.mat(X)
    y = np.mat(y)
    plt.scatter(X[:,0], X[:,1])
    plt.show()

# 生成随机数
np.random.seed(0)
X = np.r_[ np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2] ]
y = [0] * 20 + [1] * 20

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

plt.plot(xx, yy, 'k--')
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

#clf.score([[5, 5], [5, 6], [-5, -5], [-6, -6]], [1, 0, 0, 0])

#showData(X, None)