# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(314)
data_size_1 = 100
x1_1 = np.random.normal(loc=5.0, scale=1.0, size=data_size_1)
x2_1 = np.random.normal(loc=4.0, scale=1.0, size=data_size_1)
y_1 = [0 for _ in range(data_size_1)]

data_size_2 = 200
x1_2 = np.random.normal(loc=10.0, scale=2.0, size=data_size_2)
x2_2 = np.random.normal(loc=8.0, scale=2.0, size=data_size_2)
y_2 = [1 for _ in range(data_size_2)]

x1 = np.concatenate((x1_1, x1_2), axis=0)
x2 = np.concatenate((x2_1, x2_2), axis=0)
x = np.vstack((x1, x2))
y = np.concatenate((y_1, y_2), axis=0)

plt.scatter(x1, x2, c=y)
plt.show()
