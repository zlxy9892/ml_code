# -*- coding: utf-8 -*-

# convolutional neural network

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# create weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)

# create biases
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])    # 28*28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, shape=[-1,28,28,1])

## conv1 layer ##
# [patch_xsize, patch_ysize, number of input channels, number of output channels]
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    # output size: 28*28*32
h_pool1 = max_pool_2x2(h_conv1)                             # output size: 14*14*32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)    # output size: 14*14*64
h_pool2 = max_pool_2x2(h_conv2)                             # output size: 7*7*64

## func1 layer ##
#add a fully-connected layer with 1024 neurons to allow processing on the entire image.
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64]) # [n_samples, 7,7,64] -> [n_samples, 7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)    # reduce overfitting

## func2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the erro between prediction and true value
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)), reduction_indices=[1])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

# start session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={xs: batch[0], ys: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={xs:batch[0], ys:batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={xs:mnist.test.images, ys:mnist.test.labels, keep_prob:1.0}))















