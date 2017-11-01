# -*- coding: utf-8 -*-

import numpy as np

def func(x):
    return np.power(x, 2)

def d_func(x):
    return 2.0 * x

learning_rate = 0.1
max_loop = 30

x_init = np.random.rand() * 100
x = x_init
lr = 0.1
for i in range(max_loop):
    d_f_x = 2.0 * x
    x = x - learning_rate * d_func(x)
    print(x)

print('initial x =', x_init)
print('arg min f(x) of x =', x)
print('f(x) =', func(x))
