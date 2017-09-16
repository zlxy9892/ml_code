# -*- coding: utf-8 -*-

# design a neural network

import numpy as np
import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    z = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = z
    else:
        outputs = activation_function(z)
    return outputs

np.random.seed(314)
X_train = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, X_train.shape)
y_train = np.square(X_train) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
y_pred = add_layer(layer1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - y_pred),
                     reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss=loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: X_train, ys: y_train}))
