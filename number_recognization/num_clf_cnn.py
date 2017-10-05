# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


### hyper parameters ###
IMG_X = 28
IMG_Y = 28
INPUT_DIM = IMG_X * IMG_Y
OUTPUT_DIM = 10
LR = 1e-4
MAX_LOOP = 10000
BATCH_SIZE = 50
KEEP_PROB = 0.5
### hyper parameters ###

# load data
mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

def get_batch_data(X, y, batch_size):
    ix = np.random.randint(0, len(X), batch_size)
    X_batch = X[ix]
    y_batch = y[ix]
    return X_batch, y_batch

# generate weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_value=initial)

# generate bias
def bias_variable(shape):
    initial = tf.constant(value=0.1, shape=shape)
    return tf.Variable(initial_value=initial)

# convolution layer
def conv_2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding='SAME')

# pooling layer
def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def train_model(sess):
    sess.run(tf.global_variables_initializer())
    for i in range(MAX_LOOP):
        X_batch, y_batch = get_batch_data(X_train, y_train, BATCH_SIZE)
        sess.run(train_step, feed_dict={xs: X_batch, ys: y_batch, keep_prob: KEEP_PROB})
        if i % 100 == 0:
            print('正在训练: %5.1f' % (100*float(i)/MAX_LOOP), '%')
            print('loss:\t', sess.run(loss, feed_dict={xs: X_batch, ys: y_batch, keep_prob: 1.0}))
            print('train accurary:\t', sess.run(accuracy, feed_dict={xs: X_batch, ys: y_batch, keep_prob: 1.0}))

# ---------------------------- main ---------------------------- #
# define placeholder
xs = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_DIM])
ys = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_DIM])
keep_prob = tf.placeholder(dtype=tf.float32)
x_img = tf.reshape(xs, shape=[-1, IMG_X, IMG_Y, 1])

### --- build the convolution neural network --- ###
# conv layer 1
W_conv_1 = weight_variable(shape=[5,5,1,32])
b_conv_1 = bias_variable(shape=[32])
h_conv_1 = tf.nn.relu(conv_2d(x_img, W_conv_1) + b_conv_1)      # out [-1,28,28,32]
h_pool_1 = max_pool_2x2(h_conv_1)                               # out [-1,14,14,32]

# conv layer 2
W_conv_2 = weight_variable(shape=[5,5,32,64])
b_conv_2 = bias_variable(shape=[64])
h_conv_2 = tf.nn.relu(conv_2d(h_pool_1, W_conv_2) + b_conv_2)   # out [-1,14,14,64]
h_pool_2 = max_pool_2x2(h_conv_2)                               # out [-1,7,7,64]

# fully-connected layer 1
W_fc_1 = weight_variable(shape=[7*7*64, 1024])
b_fc_1 = bias_variable(shape=[1024])
h_pool_2_flatten = tf.reshape(h_pool_2, shape=[-1, 7*7*64])
h_fc_1 = tf.nn.relu(tf.matmul(h_pool_2_flatten, W_fc_1) + b_fc_1)
h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_prob=keep_prob)

# fully-connected layer 2 (out layer)
W_fc_2 = weight_variable(shape=[1024, OUTPUT_DIM])
b_fc_2 = bias_variable(shape=[OUTPUT_DIM])
h_fc_2 = tf.nn.softmax(tf.matmul(h_fc_1_drop, W_fc_2) + b_fc_2)
y_pred = h_fc_2
### --- build the convolution neural network --- ###

# loss & train
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=y_pred))
train_step = tf.train.AdamOptimizer(LR).minimize(loss)

# evaluation
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

# start session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    train_model(sess)
    
    #saver.restore(sess, "./Model/cnn_model.ckpt")
    #saver.save(sess, 'Model/cnn_model.ckpt')
    
    import img_proc
    for num in range(10):
        #filename = 'test_data/0_28x28.jpg'
        filename = 'test_data/' + str(num) + '.jpg'
        digit_test = img_proc.getImgAsMatFromFile(filename)
        digit_test = digit_test.reshape((-1))
        pred = sess.run(y_pred, feed_dict={xs: digit_test[np.newaxis, :], keep_prob: 1.0})
        print('----------', num, '----------')
        print('predict:\t', pred)
        print('predict_number:\t', np.argmax(pred, 1))

print('\n--- DONE! ---')

import os
os.system('pause')
