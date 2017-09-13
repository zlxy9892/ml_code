# -*- coding: utf-8 -*-

# classification

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(initial_value=tf.random_normal([in_size, out_size]))
    biases = tf.Variable(initial_value=tf.zeros([1, out_size])+0.1)
    z = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = z
    else:
        outputs = activation_function(z)
    return outputs

# define placeholder for inputs network
xs = tf.placeholder(tf.float32, [None, 784]) # 784 = 28*28
ys = tf.placeholder(tf.float32, [None, 10])  # 0-9

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# the error between prediction and true value
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction)),
                               reduction_indices=[1])
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 50 == 0:
        pass

















