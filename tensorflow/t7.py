# -*- coding: utf-8 -*-

# tensorflow board

import numpy as np
import tensorflow as tf


def add_layer(layername, inputs, in_size, out_size, activation_function=None):
    with tf.name_scope(layername):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layername+'/weights', weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layername+'/biases', biases)
        with tf.name_scope('Z'):
            z = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            outputs = z
        else:
            outputs = activation_function(z)
        tf.summary.histogram(layername+'/outputs', outputs)
        return outputs

# create data
np.random.seed(314)
X_train = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, X_train.shape)
y_train = np.square(X_train) - 0.5 + noise

# define the placeholder for inputs to network
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layers
layer1 = add_layer('layer_1', xs, 1, 10, activation_function=tf.nn.relu)
y_pred = add_layer('layer_2', layer1, 10, 1, activation_function=None)

# define the error
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - y_pred), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
    
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss=loss)

# start train
init = tf.global_variables_initializer()
sess = tf.Session()

# tensor board
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('D:/logs', sess.graph)

sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs:X_train, ys:y_train})
    if i % 50 == 0:
        res = sess.run(merged, feed_dict={xs:X_train, ys:y_train})
        writer.add_summary(res, i)

writer.close()
sess.close()













