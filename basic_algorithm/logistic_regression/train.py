# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import data_helper
from logistic_regression import *


# data generation
x, y = data_helper.generate_data(seed=272)
x_train, y_train, x_test, y_test = data_helper.train_test_split(x, y)

# visualize data
# plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='.')
# plt.show()
# plt.scatter(x_test[:,0], x_test[:,1], c=y_test, marker='.')
# plt.show()

# data normalization
x_train = (x_train - np.min(x_train, axis=0)) / (np.max(x_train, axis=0) - np.min(x_train, axis=0))
x_test = (x_test - np.min(x_test, axis=0)) / (np.max(x_test, axis=0) - np.min(x_test, axis=0))

# Logistic regression classifier
clf = LogisticRegression(learning_rate=0.1, max_iter=500, seed=272)
clf.fit(x_train, y_train)

# plot the result
split_boundary_func = lambda x: (-clf.b - clf.w[0] * x) / clf.w[1]
xx = np.arange(0.1, 0.6, 0.1)
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='.')
plt.plot(xx, split_boundary_func(xx), c='red')
plt.show()

# loss on test set
y_test_pred = clf.predict(x_test)
y_test_pred_proba = clf.predict_proba(x_test)
print(clf.score(y_test, y_test_pred))
print(clf.loss(y_test, y_test_pred_proba))
# print(y_test_pred_proba)
