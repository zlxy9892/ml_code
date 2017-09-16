# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def showData(x, y):
    plt.scatter(x, y)
    plt.show()

# create data
np.random.seed(314)
x = np.random.rand(100).astype(np.float32)
y = x * 0.1 + 0.3
showData(x, y)

### create tensorflow structure start ###
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y_pred = weights * x + biases
loss = tf.reduce_mean(tf.square(y_pred - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
### create tensorflow structure end ###

session = tf.Session()
session.run(init)

for step in range(201):
    session.run(train)
    if step % 20 == 0:
        print(step, session.run(weights), session.run(biases))

session.close()
