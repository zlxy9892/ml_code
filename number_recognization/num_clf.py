# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

### Hyperparameters ###
IMG_X = 28
IMG_Y = 28
INPUT_DIM = IMG_X * IMG_Y
OUTPUT_DIM = 10
LR = 0.1
MAX_LOOP = 20000
BATCH_SIZE = 100
### Hyperparameters ###

# load data
mnist = input_data.read_data_sets('D:/data/ml_data/MNIST_data', one_hot=True)
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

def get_batch_data(X, y, batch_size):
    ix = np.random.randint(0, len(X), batch_size)
    X_batch = X[ix]
    y_batch = y[ix]
    return X_batch, y_batch

# build a nueral network layer
def nn_layer(inputs, in_dim, out_dim, act=None):
    weights = tf.Variable(tf.random_normal(shape=[in_dim, out_dim]), dtype=tf.float32)
    biases = tf.Variable(tf.zeros(shape=[out_dim]) + 0.1)
    z = tf.matmul(inputs, weights) + biases
    if act is None:
        return z
    else:
        return act(z)
    
# set placeholder
xs = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_DIM])
ys = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_DIM])

# build the neural network
hidden_layer_1 = nn_layer(xs, INPUT_DIM, OUTPUT_DIM, act=tf.nn.softmax)
y_pred = hidden_layer_1

# loss and train
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=y_pred))
train_step = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

# evaluate model
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# training the model
#with tf.Session() as sess:
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(MAX_LOOP):
    X_train_batch, y_train_batch = get_batch_data(X_train, y_train, BATCH_SIZE)
    #X_train_batch, y_train_batch = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={xs: X_train_batch, ys: y_train_batch})
    if i % 50 == 0:
        print('train error:\t', sess.run(loss, feed_dict={xs: X_train, ys: y_train}))
        print('train accurary:\t', sess.run(accuracy, feed_dict={xs: X_train, ys: y_train}))
        print('test error:\t', sess.run(loss, feed_dict={xs: X_test, ys: y_test}))
        print('test accurary:\t', sess.run(accuracy, feed_dict={xs: X_test, ys: y_test}))
        print('-----------------------------------')
print('---------------FINAL-------------')
print('train error:\t', sess.run(loss, feed_dict={xs: X_train, ys: y_train}))
print('train accurary:\t', sess.run(accuracy, feed_dict={xs: X_train, ys: y_train}))
print('test error:\t', sess.run(loss, feed_dict={xs: X_test, ys: y_test}))
print('test accurary:\t', sess.run(accuracy, feed_dict={xs: X_test, ys: y_test}))
print('---------------FINAL-------------')


import img_proc
filename = 'my_9_28x28.jpg'
#img_proc.resizeImg('E:/data/ml_data/my_9.jpg', 'my_9_28x28.jpg', 28, 28)
digit_test = img_proc.getImgAsMatFromFile(filename)
digit_test = digit_test.reshape((-1))

print('predict:\t', sess.run(y_pred, feed_dict={xs: digit_test[np.newaxis, :]}))

print('\n--- DONE! ---')

